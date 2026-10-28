#!/usr/bin/env python3
"""Start one engine, measure it on the typed-decisions suite, stop it. One row per run.

The harness keeps the engine the only variable: it spawns the server, waits for it, hands
the port to bench_systemone.py (which owns the metrics), samples the server's peak RSS, then
terminates the process. Nothing is reused between configs — a row is one load of one model.

    python3 benchmarks/decision/run_config.py --engine llamacpp \
        --model-path clef-flash-q4km --quant Q4_K_M --ctx 8192

Engines:
  llamacpp   native decision model in GGUF, served through POST /v1/systemone
  mlx-clef   Cloudflare Clef / Clef-Flash through the MLX loader shipped with the weights
  mlx-d3     vLLM-SR d3-flash through its MLX loader (thin local /v1/systemone wrapper)
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
LLAMA_SERVER = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-server"
BENCH = ROOT / "benchmarks" / "decision" / "bench_systemone.py"
D3_SERVER = ROOT / "benchmarks" / "decision" / "serve_d3_mlx.py"
DATA = ROOT / "data" / "decision"
PARQUET = DATA / "cache" / "typed-decisions" / "all" / "test-00000-of-00001.parquet"
RESULTS = ROOT / "results" / "decision" / "typed.jsonl"
LOGS = ROOT / "logs" / "decision"

sys.path.insert(0, str(ROOT / "benchmarks" / "common"))
from model_store import resolve  # pyright: ignore[reportMissingImports]  # noqa: E402 - repo-local helper


def engine_version(engine: str) -> str:
    if engine == "llamacpp":
        try:
            out = subprocess.run(
                [str(LLAMA_SERVER), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
            ).stdout
            return next((line.strip() for line in out.splitlines() if line.startswith("version:")), "?")
        except (OSError, subprocess.SubprocessError):
            return "?"
    try:
        import importlib.metadata

        parts = [f"mlx={importlib.metadata.version('mlx')}"]
        for pkg in ("mlx-lm", "mlx-vlm"):
            with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                parts.append(f"{pkg}={importlib.metadata.version(pkg)}")
        return " ".join(parts)
    except Exception:  # noqa: BLE001 - version string is metadata, never fatal
        return "?"


def build_command(args: argparse.Namespace) -> list[str]:
    if args.engine == "llamacpp":
        cmd = [
            str(LLAMA_SERVER),
            "-m",
            args.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "-c",
            str(args.ctx),
            "-t",
            str(args.threads),
            "-ngl",
            "99",
            "-b",
            "2048",
            "-ub",
            "2048",
            "--parallel",
            "1",
            "--no-webui",
        ]
        if args.mmproj:
            cmd += ["--mmproj", args.mmproj]
        if args.extra:
            cmd += args.extra
        return cmd
    if args.engine == "mlx-clef":
        cmd = [
            str(PY),
            str(Path(args.model_path) / "clef_mlx.py"),
            "serve",
            "--model",
            args.model_path,
            "--port",
            str(args.port),
            "--max-length",
            str(args.ctx),
            "--quiet",
        ]
        return cmd + args.extra
    if args.engine == "mlx-d3":
        return [
            str(PY),
            str(D3_SERVER),
            "--model",
            args.model_path,
            "--port",
            str(args.port),
            *args.extra,
        ]
    raise SystemExit(f"unknown engine: {args.engine}")


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    import http.client

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}; see its log")
        for path in ("/health", "/v1/models"):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                try:
                    conn.request("GET", path)
                    if conn.getresponse().status == 200:
                        return
                finally:
                    conn.close()
            except (http.client.HTTPException, OSError):
                pass
        time.sleep(1.0)
    raise SystemExit(f"server on :{port} not ready after {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=["llamacpp", "mlx-clef", "mlx-d3"])
    ap.add_argument("--model-path", required=True, help="alias, org/repo[:file], or a local path")
    ap.add_argument("--model-id", default=None, help="name recorded in the row; defaults to --model-path")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mmproj", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--engine-arg", action="append", default=[], help="extra flag for the engine (repeatable)")
    ap.add_argument("extra", nargs="*", help="extra engine flags")
    args = ap.parse_args()

    if not PARQUET.exists():
        raise SystemExit(f"missing suite parquet: {PARQUET} (see data/decision/manifest.json)")
    if args.engine == "llamacpp" and not LLAMA_SERVER.exists():
        raise SystemExit(f"missing llama-server: {LLAMA_SERVER}")
    args.model_path = resolve(args.model_path)
    args.model_id = args.model_id or Path(args.model_path).name
    if not Path(args.model_path).exists():
        raise SystemExit(f"missing model path: {args.model_path}")

    args.extra = [*args.extra, *args.engine_arg]
    cmd = build_command(args)
    LOGS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.engine}_{args.model_id}".replace("/", "_")
    log_path = LOGS / f"serve_{tag}.log"
    if args.dry_run:
        print(" ".join(cmd))
        return

    print(f"[run] {' '.join(cmd)}", flush=True)
    with contextlib.ExitStack() as stack:
        try:
            log = stack.enter_context(open(log_path, "w", encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(f"cannot write server log {log_path}: {exc}") from exc
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.time()
        try:
            wait_ready(args.port, proc, args.startup_timeout)
            startup_s = time.time() - started
            print(f"[run] ready in {startup_s:.1f}s (pid {proc.pid})", flush=True)
            bench = [
                str(PY),
                str(BENCH),
                "--base-url",
                f"http://127.0.0.1:{args.port}",
                "--parquet",
                str(PARQUET),
                "--engine",
                args.engine,
                "--model",
                args.model_id,
                "--label",
                args.label or f"ctx={args.ctx} quant={args.quant or 'none'}",
                "--out",
                args.out,
                "--server-pid",
                str(proc.pid),
                "--ctx-tokens",
                str(args.ctx),
                "--engine-version",
                engine_version(args.engine),
                "--notes",
                f"server_log={log_path.name} startup_s={startup_s:.1f}",
            ]
            if args.quant:
                bench += ["--quant", args.quant]
            if args.limit:
                bench += ["--limit", str(args.limit)]
            bench += ["--per-case-out", str(LOGS / f"cases_{tag}.jsonl")]
            subprocess.run(bench, cwd=str(ROOT), check=True)
        finally:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print(f"[run] stopped; log {log_path}", flush=True)


if __name__ == "__main__":
    if not shutil.which(str(PY)):
        raise SystemExit(f"missing interpreter: {PY}")
    main()
