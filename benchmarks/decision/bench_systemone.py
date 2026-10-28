#!/usr/bin/env python3
"""Measure a running System One decision server on a frozen typed-decision suite.

One row = one measurement, one process, one model load. This harness never starts or
stops the engine; point it at a server that answers POST /v1/systemone (llama.cpp,
clef_mlx.py, kev.serve, ...), so the engine and the model stay the only variables.

    . "$(dirname "$0")/../common/env.sh"
    python3 benchmarks/decision/bench_systemone.py \
        --base-url http://127.0.0.1:8091 --engine llama.cpp --model clef-flash-q4km \
        --server-pid 12345 --label "ctx=8192 quant=Q4_K_M" \
        --parquet "$DATA/cache/typed-decisions/all/test-00000-of-00001.parquet" \
        --out results/decision/typed.jsonl

Metrics follow the suite's own scorer (LocalLLaMA/typed-decisions, bench/score_with_jobs.py):
accuracy / KL from gold / multiclass Brier / ECE with 10 equal-width confidence bins.
noul answers arrive as a single P(true); they are expanded to {true, false} here, and the
suite's yes/no -> true/false alias is applied, so the numbers are comparable with the
published leaderboard.

Latency: `latency_ms_*` is the first ask about a state (cold prefix). `cached_latency_ms_*`
re-asks the first --cached-sample cases after the run, which should hit the server's
prefix cache. Never average the two together.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any

NOUL_ALIAS = {"yes": "true", "no": "false"}
ALLOWED_SCHEMES = ("http", "https")


# --------------------------------------------------------------------------- numbers
def as_float(value: Any, default: float | None = None) -> float | None:
    """A probability or metric from a server body; None instead of a crash on junk."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- metrics
def normalize_gold(gold: dict, question: dict) -> dict:
    """Some suites store a label; the server metric needs a distribution. One-hot the label."""
    if isinstance(gold, dict) and isinstance(gold.get("probabilities"), dict):
        return gold
    qtype = question.get("type", "choice")
    if qtype == "noul":
        value = gold.get("label") if isinstance(gold, dict) else gold
        p_true = 1.0 if bool(value) else 0.0
        return {"type": "noul", "probabilities": {"true": p_true, "false": 1.0 - p_true}}
    criteria = question.get("criteria")
    if qtype == "score":
        levels = range(len(criteria)) if isinstance(criteria, list) else range(5)
        keys = [str(i) for i in levels]
    else:
        keys = list(criteria.keys()) if isinstance(criteria, dict) else []
    label = gold.get("label") if isinstance(gold, dict) else gold
    label = str(label)
    return {
        "type": qtype,
        "probabilities": {key: (1.0 if key == label else 0.0) for key in keys},
    }


def score_one(pred: dict, gold: dict) -> tuple[bool, float, float, float] | None:
    """-> (argmax_correct, kl_from_gold, brier, confidence_of_top_option), None if unusable."""
    probs_in = pred.get("probabilities")
    probs: dict[str, float] = dict(probs_in) if isinstance(probs_in, dict) else {}
    if gold.get("type") == "noul":
        # engines answer {noul: p(true)}; some answer a probabilities map
        if not probs and "noul" in pred:
            p_true = as_float(pred.get("noul"))
            if p_true is None:
                return None
            probs = {"true": p_true, "false": 1.0 - p_true}
        probs = {NOUL_ALIAS.get(k, k): v for k, v in probs.items()}

    gold_probs = gold.get("probabilities")
    if not isinstance(gold_probs, dict) or not gold_probs:
        return None
    opts = list(gold_probs.keys())
    g = [as_float(gold_probs[o], 0.0) or 0.0 for o in opts]
    p = [as_float(probs.get(o, 0.0), 0.0) or 0.0 for o in opts]
    total = sum(p)
    p = [v / total for v in p] if total > 0 else [1.0 / len(opts)] * len(opts)

    correct = p.index(max(p)) == g.index(max(g))
    kl = sum(
        gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p, strict=True) if gi > 0
    )
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g, strict=True))
    return correct, kl, brier, max(p)


def ece(oks: list[bool], confs: list[float], n_bins: int = 10) -> float:
    """Equal-width confidence bins, as in the suite's scorer."""
    total = len(oks)
    if not total:
        return 0.0
    hits = [0] * n_bins
    count = [0] * n_bins
    conf_sum = [0.0] * n_bins
    for ok, conf in zip(oks, confs, strict=True):
        b = min(int(conf * n_bins), n_bins - 1)
        count[b] += 1
        hits[b] += int(ok)
        conf_sum[b] += conf
    return sum(
        count[b] / total * abs(hits[b] / count[b] - conf_sum[b] / count[b])
        for b in range(n_bins)
        if count[b]
    )


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    i = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[i]


# --------------------------------------------------------------------------- plumbing
def checked_url(url: str) -> tuple[str, int, str]:
    """(host, port, path) after rejecting file:/custom schemes; HTTP server only."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.hostname:
        raise SystemExit(f"refusing non-HTTP URL: {url}")
    return parsed.hostname, parsed.port or 80, parsed.path or "/"


def post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    host, port, path = checked_url(url)
    body = json.dumps(payload).encode()
    start = time.perf_counter()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    elapsed = (time.perf_counter() - start) * 1000.0
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response: {raw[:200]!r}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"response is not an object: {raw[:200]!r}")
    return decoded, elapsed


def http_ok(url: str, timeout: float = 5.0) -> bool:
    host, port, path = checked_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        return conn.getresponse().status == 200
    except (http.client.HTTPException, OSError):
        return False
    finally:
        conn.close()


def wait_ready(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            if http_ok(base_url + path):
                return
        time.sleep(1.0)
    raise SystemExit(f"server at {base_url} not ready after {timeout_s}s")


def swap_used_mb() -> float:
    """macOS swap in use, MB. A run that swaps is not a valid speed measurement."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return float("nan")
    for part in out.replace("=", " ").split():
        if part.endswith("M"):
            value = as_float(part[:-1])
            if value is not None:
                return value
    return float("nan")


class RssSampler(threading.Thread):
    def __init__(self, pid: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak = pid, interval, 0.0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                rss = as_float(out)
                if rss is not None:
                    self.peak = max(self.peak, rss / 1024.0)
            except (OSError, subprocess.SubprocessError):
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


def load_cases(path: str, limit: int | None) -> list[dict]:
    """Read the suite's parquet; state/questions/gold are JSON-encoded columns."""
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("pyarrow is required: .venv/bin/python -m pip install pyarrow") from exc

    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if limit:
        rows = rows[:limit]

    cases = []
    for i, raw in enumerate(rows):
        parsed = {}
        for key in ("state", "questions", "gold"):
            value = raw.get(key)
            if isinstance(value, str):
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"row {i} ({raw.get('id')}): bad {key} JSON: {exc}") from exc
            else:
                parsed[key] = value
        cases.append({"id": raw.get("id", f"row-{i}"), **parsed})
    return cases


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--parquet", required=True, help="test split parquet of the typed-decisions suite")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N cases")
    ap.add_argument("--task", default="typed-decisions")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--model-name", default=None, help="value of the request's `model` field")
    ap.add_argument("--out", required=True, help="JSONL file to append the row to")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--cached-sample", type=int, default=20, help="cases re-asked to measure a warm prefix")
    ap.add_argument("--server-pid", type=int, default=None, help="sample this process's peak RSS")
    ap.add_argument("--ctx-tokens", type=int, default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--engine-version", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--per-case-out", default=None, help="optional JSONL of one line per decision")
    args = ap.parse_args()

    cases = load_cases(args.parquet, args.limit)
    if not cases:
        raise SystemExit("no cases loaded")
    url = args.base_url.rstrip("/") + "/v1/systemone"
    model_name = args.model_name or args.model

    wait_ready(args.base_url)
    sampler = None
    if args.server_pid:
        sampler = RssSampler(args.server_pid)
        sampler.start()

    swap_before = swap_used_mb()
    latencies: list[float] = []
    input_tokens: list[int] = []
    oks: list[bool] = []
    kls: list[float] = []
    briers: list[float] = []
    confs: list[float] = []
    by_type: dict[str, list[bool]] = {"choice": [], "noul": [], "score": []}
    failures = 0
    answered = 0
    expected = 0
    per_case: list[dict] = []

    for i, case in enumerate(cases):
        payload = {"model": model_name, "state": case["state"], "questions": case["questions"]}
        try:
            body, elapsed = post_json(url, payload, args.timeout)
        except (OSError, TimeoutError, RuntimeError) as exc:
            failures += 1
            print(f"[bench] case {case['id']} failed: {exc}", file=sys.stderr, flush=True)
            continue
        latencies.append(elapsed)
        usage = body.get("usage")
        if isinstance(usage, dict):
            tokens = as_int(usage.get("input_tokens"))
            if tokens is not None:
                input_tokens.append(tokens)
        answers = body.get("answers")
        answers = answers if isinstance(answers, dict) else {}
        gold_all = case.get("gold")
        if not isinstance(gold_all, dict):
            failures += 1
            continue
        for qid, raw_gold in gold_all.items():
            expected += 1
            question = (case.get("questions") or {}).get(qid, {})
            gold = normalize_gold(raw_gold, question)
            answer = answers.get(qid)
            scored = score_one(answer, gold) if isinstance(answer, dict) else None
            if scored is None:
                oks.append(False)
                kls.append(float("nan"))
                briers.append(float("nan"))
                confs.append(0.0)
                by_type.setdefault(str(gold.get("type")), []).append(False)
                continue
            answered += 1
            ok, kl, brier, conf = scored
            per_case.append(
                {
                    "case": case["id"],
                    "question": qid,
                    "type": gold.get("type"),
                    "correct": ok,
                    "confidence": conf,
                    "latency_ms": elapsed,
                }
            )
            oks.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
            by_type.setdefault(str(gold.get("type")), []).append(ok)
        if (i + 1) % 50 == 0:
            acc = sum(oks) / len(oks) if oks else float("nan")
            print(
                f"[bench] {i + 1}/{len(cases)} cases  acc={acc:.4f}"
                f"  p50={statistics.median(latencies):.0f} ms",
                flush=True,
            )

    # warm-prefix pass: same states again, server-side prefix cache should answer cheaper
    cached: list[float] = []
    for case in cases[: args.cached_sample]:
        payload = {"model": model_name, "state": case["state"], "questions": case["questions"]}
        try:
            _, elapsed = post_json(url, payload, args.timeout)
            cached.append(elapsed)
        except (OSError, TimeoutError, RuntimeError):
            continue

    if sampler:
        sampler.stop()
        sampler.join(timeout=2)
    swap_after = swap_used_mb()

    n = len(oks)
    valid_kls = [v for v in kls if not math.isnan(v)]
    valid_briers = [v for v in briers if not math.isnan(v)]
    metrics = {
        "accuracy": sum(oks) / n if n else float("nan"),
        "kl_from_gold": sum(valid_kls) / len(valid_kls) if valid_kls else float("nan"),
        "brier": sum(valid_briers) / len(valid_briers) if valid_briers else float("nan"),
        "ece": ece(oks, confs),
        "schema_valid_rate": answered / expected if expected else 0.0,
        "n_cases": len(cases),
        "n_decisions": n,
        "questions_per_request": round(expected / len(cases), 2) if cases else 0,
        "accuracy_choice": _mean(by_type.get("choice", [])),
        "accuracy_noul": _mean(by_type.get("noul", [])),
        "accuracy_score": _mean(by_type.get("score", [])),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else float("nan"),
        "latency_ms_p50": pct(latencies, 0.50),
        "latency_ms_p95": pct(latencies, 0.95),
        "cached_latency_ms_p50": pct(cached, 0.50),
        "cached_latency_ms_p95": pct(cached, 0.95),
        "input_tokens_mean": statistics.fmean(input_tokens) if input_tokens else float("nan"),
        "input_tokens_p50": pct([float(v) for v in input_tokens], 0.50) if input_tokens else float("nan"),
        "request_failures": failures,
        "peak_rss_mb": round(sampler.peak, 1) if sampler else None,
        "swap_mb_before": swap_before,
        "swap_mb_after": swap_after,
    }
    # A row that swapped or lost requests is evidence, but not a valid speed/quality row.
    valid = (
        failures == 0
        and answered == expected
        and expected > 0
        and not (swap_after - swap_before > 512)
    )
    row = {
        "modality": "decision",
        "task": args.task,
        "engine": args.engine,
        "model": args.model,
        "label": args.label,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "valid": valid,
        "metrics": metrics,
        "harness": {
            "dataset": "smoke" if args.task == "smoke" else "LocalLLaMA/typed-decisions",
            "split": "test",
            "parquet": os.path.basename(args.parquet),
            "protocol": "POST /v1/systemone",
            "ctx_tokens": args.ctx_tokens,
            "quant": args.quant,
            "engine_version": args.engine_version,
            "base_url": args.base_url,
            "machine": "M5 32GB",
            "notes": args.notes,
        },
    }
    if args.per_case_out:
        try:
            with open(args.per_case_out, "w", encoding="utf-8") as f:
                for record in per_case:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[bench] cannot write per-case file: {exc}", file=sys.stderr)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise SystemExit(f"cannot append to {args.out}: {exc}") from exc
    print(json.dumps(row, indent=2, ensure_ascii=False))


def _mean(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
