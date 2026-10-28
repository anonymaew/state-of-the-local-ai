#!/usr/bin/env python3
"""Interleaved A/B: llama.cpp + MTP  vs  oMLX + oQ4e-MTP.

The two configs were previously measured in *different sessions* (13.90 vs 15.65) on a machine
that drifts up to 25% within a session, so that comparison is not trustworthy. This script
alternates the two engines round-robin, one model resident at a time, and reports the per-round
ratio (which cancels drift) as well as the raw numbers.

Usage: .venv/bin/python benchmarks/text/ab_mtp.py [rounds]

Weights resolve by alias from the Hub cache (`qwen38-27b-gguf-q4km`, `qwen38-27b-mtp`). Set
LLAMA_GGUF / MTP_GGUF to override with explicit paths.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "common"))
from model_store import resolve  # pyright: ignore[reportMissingImports]  # noqa: E402 - repo-local helper

LLAMA = os.path.join(ROOT, "src/llama.cpp/build/bin/llama-server")
OMLX = os.path.join(ROOT, ".venv-omlx/bin/omlx")
# GGUF artifacts are not resident; the aliases resolve from the cache and can download
# on the first resolve when MODEL_STORE_ONLINE=1.
GGUF = os.environ.get("LLAMA_GGUF") or "qwen38-27b-gguf-q4km"
MTP = os.environ.get("MTP_GGUF") or "qwen38-27b-mtp"
OUT = os.path.join(ROOT, "results/text/ab_mtp.jsonl")
MAXTOK = 128
LOGS = os.path.join(ROOT, "logs/text")
PROMPT_FILE = os.path.join(ROOT, "benchmarks/text/prompts/prompt_short.txt")


def open_log(path):
    """Open a server log for writing, failing loudly instead of raising a bare OSError."""
    try:
        return open(path, "w")  # noqa: SIM115 - closed by the caller in one()
    except OSError as e:
        raise RuntimeError(f"cannot open log {path}: {e}") from e


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"cannot create dir {path}: {e}") from e


def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)


PROMPT = read_text(PROMPT_FILE)


def run(args, timeout=60):
    """Run a command without a shell. Returns stdout ('' on failure)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def load1():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return -1.0


def swap_used_mb():
    """Swap currently allocated (may include stale, unpaged-out allocations)."""
    out = run(["sysctl", "-n", "vm.swapusage"])
    try:
        return float(out.split("used =")[1].split("M")[0].strip())
    except (IndexError, ValueError):
        return -1.0


def swapouts_pages():
    """Cumulative pages swapped OUT - the honest signal that paging actually happened."""
    for line in run(["vm_stat"]).splitlines():
        if "Swapouts" in line:
            try:
                return int(line.split(":")[1].strip().replace(".", ""))
            except (IndexError, ValueError):
                return -1
    return -1


def servers_running():
    return bool(run(["pgrep", "-f", "llama-server|omlx-server"]).strip())


def kill_all():
    for pat in ("llama-server", "omlx-server", "omlx serve"):
        run(["pkill", "-f", pat])
    for _ in range(30):
        if not servers_running():
            return True
        time.sleep(1)
    return False


def check_url(url):
    """Only allow loopback http - prevents file:// or remote fetches."""
    u = urllib.parse.urlparse(url)
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"refusing non-loopback URL: {url}")
    return url


def wait_http(url, timeout=300):
    check_url(url)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310 - loopback validated
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def post(url, payload, timeout=600):
    check_url(url)
    req = urllib.request.Request(  # noqa: S310 - loopback validated in check_url
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        raw = r.read()
    wall = time.perf_counter() - t0
    try:
        return json.loads(raw), wall
    except json.JSONDecodeError as e:
        raise RuntimeError(f"non-JSON response from {url}: {e}") from e


def start_engine(engine, rnd):
    """Start the engine's server. Returns (proc, url, payload, logfile_handle, base_dir)."""
    if engine == "llama":
        log = open_log(f"{LOGS}/ab_llama_r{rnd}.log")
        proc = subprocess.Popen(
            [LLAMA, "-m", GGUF, "-md", MTP, "--spec-type", "draft-mtp",
             "--spec-draft-n-max", "3", "-c", "4096", "-ngl", "99", "-fa", "on",
             "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--port", "18200",
             "--metrics", "--jinja"],
            stdout=log, stderr=subprocess.STDOUT)
        return (proc, "http://127.0.0.1:18200/v1/completions",
                {"prompt": PROMPT, "max_tokens": MAXTOK, "temperature": 0,
                 "cache_prompt": False}, log, None)

    base = f"{LOGS}/ab_omlx_r{rnd}_base"
    ensure_dir(base)
    # oMLX discovers models from subdirectories, so point it at the snapshot parent and
    # use the discovered name (the revision) in the settings and the request payload.
    omlx_model = resolve("qwen38-27b-oq4e-mtp")
    omlx_dir, omlx_name = os.path.dirname(omlx_model), os.path.basename(omlx_model)
    try:
        with open(f"{base}/model_settings.json", "w") as f:
            json.dump({"version": 1,
                       "models": {omlx_name: {"mtp_enabled": True}}}, f)
    except OSError as e:
        raise RuntimeError(f"cannot write model_settings.json: {e}") from e
    log = open_log(f"{LOGS}/ab_omlx_r{rnd}.log")
    proc = subprocess.Popen(
        [OMLX, "serve", "--model-dir", omlx_dir, "--port", "18201",
         "--no-cache", "--memory-guard", "off", "--base-path", base, "--headless"],
        stdout=log, stderr=subprocess.STDOUT)
    return (proc, "http://127.0.0.1:18201/v1/completions",
            {"model": omlx_name, "prompt": PROMPT,
             "max_tokens": MAXTOK, "temperature": 0, "cache_prompt": False}, log, base)


def one(engine, rnd):
    """Run one measurement for engine in {llama, omlx}; return a row dict."""
    if not kill_all():
        return {"round": rnd, "engine": engine, "valid": False,
                "error": "could not clear servers before start"}
    row = {"round": rnd, "engine": engine, "valid": False}
    proc = log = base = None
    try:
        proc, url, payload, log, base = start_engine(engine, rnd)
        if not wait_http(url.replace("/v1/completions", "/health" if engine == "llama" else "/v1/models")):
            row["error"] = "server did not become healthy"
            return row

        post(url, {**payload, "max_tokens": 32})  # warmup, discarded

        row["loadavg_before"] = load1()
        row["swap_used_mb_before"] = swap_used_mb()
        sw0 = swapouts_pages()

        body, wall = post(url, payload)
        row["swapouts_delta_pages"] = swapouts_pages() - sw0
        row["loadavg_after"] = load1()
        row["swap_used_mb_after"] = swap_used_mb()

        ntok = (body.get("usage") or {}).get("completion_tokens") or 0
        tim = body.get("timings") or {}
        row.update({
            "completion_tokens": ntok,
            "wall_s": round(wall, 2),
            "decode_tps": round(ntok / wall, 2) if wall and ntok else None,
            "server_dec_tps": round(tim["predicted_per_second"], 2)
            if tim.get("predicted_per_second") else None,
            "server_pp_tps": round(tim["prompt_per_second"], 2)
            if tim.get("prompt_per_second") else None,
            "text_head": (body.get("choices") or [{}])[0].get("text", "")[:120].replace("\n", " "),
        })
        # a row is only valid if it produced tokens AND did not page to disk
        row["valid"] = bool(ntok and wall and row["swapouts_delta_pages"] == 0)
        return row
    except Exception as e:  # noqa: BLE001 - record and continue the sweep
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        return row
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()
        kill_all()
        if base:
            shutil.rmtree(base, ignore_errors=True)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    global GGUF, MTP
    GGUF, MTP = resolve(GGUF), resolve(MTP)
    rows = []
    try:
        out = open(OUT, "a")  # noqa: SIM115
    except OSError as e:
        print(f"cannot open {OUT}: {e}", file=sys.stderr)
        return 1
    with out:
        for r in range(1, rounds + 1):
            for engine in ("llama", "omlx"):
                row = one(engine, r)
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"round {r} {engine:<6} decode={row.get('decode_tps') or 'FAIL'} "
                      f"wall={row.get('wall_s')} load={row.get('loadavg_before')} "
                      f"swapouts={row.get('swapouts_delta_pages')} "
                      f"err={row.get('error', '')}", flush=True)

    print("\n--- paired results (same round = same machine conditions) ---")
    ratios = []
    for r in range(1, rounds + 1):
        a = next((x for x in rows if x["round"] == r and x["engine"] == "llama"), None)
        b = next((x for x in rows if x["round"] == r and x["engine"] == "omlx"), None)
        if a and b and a.get("decode_tps") and b.get("decode_tps"):
            ratio = b["decode_tps"] / a["decode_tps"]
            ratios.append(ratio)
            print(f"  round {r}: llama {a['decode_tps']:6.2f}  oMLX {b['decode_tps']:6.2f}  "
                  f"ratio {ratio:.3f}   valid {a['valid']}/{b['valid']}")
    if ratios:
        print(f"\n  median ratio oMLX/llama = {statistics.median(ratios):.3f} "
              f"(n={len(ratios)}, spread {min(ratios):.3f}-{max(ratios):.3f})")
    print(f"\nrows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
