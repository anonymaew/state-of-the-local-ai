#!/usr/bin/env python3
"""MLX (mlx-lm) throughput bench: prefill + decode at several context depths.

Reports prompt tps, generation tps, peak MLX memory. Appends JSONL rows.
Usage: .venv/bin/python benchmarks/text/bench_mlx.py <model_ref> <label>

<model_ref> is an alias (`qwen38-27b-oq4e-mtp`), an `org/repo` Hub id, or a path;
see benchmarks/common/model_store.py.
"""
import json
import os
import sys
import time

import mlx.core as mx  # pyright: ignore[reportMissingImports]
from mlx_lm import load, stream_generate  # pyright: ignore[reportMissingImports]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "common"))
from model_store import resolve  # pyright: ignore[reportMissingImports]  # noqa: E402 - repo-local helper, path needs ROOT first

OUT = os.path.join(ROOT, "results", "text", "speed_mlx.jsonl")

PROMPT_PARA = (
    "The quick brown fox jumps over the lazy dog. In 2026, local inference on "
    "Apple Silicon became practical because unified memory bandwidth grew and "
    "quantization improved. A benchmark must control for context length, "
    "batch size, and thermal state, otherwise the numbers are meaningless. "
)


def make_prompt(tok, approx_tokens: int) -> str:
    """Repeat filler until the *tokenized* length is >= approx_tokens."""
    text = ""
    while len(tok.encode(text)) < approx_tokens:
        text += PROMPT_PARA
    return text


def run(model, tok, approx_tokens: int, max_tokens: int = 128):
    prompt = make_prompt(tok, approx_tokens)
    n_prompt = len(tok.encode(prompt))
    # warmup on a short prompt so kernels/weights are resident
    for _ in stream_generate(model, tok, "hi", max_tokens=4):
        pass
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens):
        last = r
    wall = time.perf_counter() - t0
    if last is None:
        raise RuntimeError("stream_generate produced no tokens")
    return {
        "engine": "mlx-lm",
        "prompt_tokens": n_prompt,
        "gen_tokens": last.generation_tokens,
        "prompt_tps": round(last.prompt_tps, 1),
        "gen_tps": round(last.generation_tps, 1),
        "wall_s": round(wall, 1),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def main():
    path, label = resolve(sys.argv[1]), sys.argv[2]
    depths = [int(x) for x in os.environ.get("DEPTHS", "64,4096,16384").split(",")]
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    except OSError as e:
        print(f"cannot create results dir: {e}", file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    loaded = load(path)
    model, tok = loaded[0], loaded[1]
    load_s = time.perf_counter() - t0
    rows = []
    for d in depths:
        try:
            row = run(model, tok, d)
        except Exception as e:  # OOM etc. -> record, keep going
            row = {"engine": "mlx-lm", "target_depth": d, "error": type(e).__name__ + ": " + str(e)[:200]}
        row["label"] = label
        row["target_depth"] = d
        row["load_s"] = round(load_s, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
        try:
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            print(f"cannot append results: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
