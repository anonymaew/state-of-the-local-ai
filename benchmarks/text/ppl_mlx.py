#!/usr/bin/env python3
"""Perplexity over a local text file for an MLX model, mirroring llama-perplexity.

llama-perplexity: PPL over N non-overlapping chunks of `ctx` tokens; token i in a chunk is
scored given tokens [start..i-1]. This script does exactly that so the GGUF and MLX numbers
are comparable on the same file and the same tokenizer.

Usage: .venv/bin/python benchmarks/text/ppl_mlx.py <model_ref> <text_file> [chunks] [ctx]

<model_ref> is an alias, an `org/repo` Hub id, or a path (benchmarks/common/model_store.py).
"""
import os
import sys

import mlx.core as mx  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from mlx_lm import load  # pyright: ignore[reportMissingImports]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from model_store import resolve  # pyright: ignore[reportMissingImports]  # noqa: E402 - repo-local helper


def main() -> int:
    model_dir, text_file = resolve(sys.argv[1]), sys.argv[2]
    n_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 2048

    loaded = load(model_dir)
    model, tok = loaded[0], loaded[1]
    try:
        with open(text_file, errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"cannot read {text_file}: {e}", file=sys.stderr)
        return 1
    ids = tok.encode(text)
    need = n_chunks * ctx
    if len(ids) < need:
        print(f"text too short: {len(ids)} tokens < {need}", file=sys.stderr)
        return 1

    total_nll, total_tok = 0.0, 0
    for c in range(n_chunks):
        chunk = np.array(ids[c * ctx : (c + 1) * ctx], dtype=np.int32)
        logits = model(mx.array(chunk)[None])[0]
        # predict token t+1 from position t
        logp = logits[:-1].astype(mx.float32)
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        tgt = chunk[1:]
        nll = -mx.take_along_axis(logp, mx.array(tgt)[:, None], axis=-1)
        total_nll += float(mx.sum(nll))
        total_tok += len(tgt)
        print(f"  chunk {c + 1}/{n_chunks}: ppl={np.exp(total_nll / total_tok):.4f}", flush=True)

    ppl = float(np.exp(total_nll / total_tok))
    print(f"Final estimate: PPL = {ppl:.4f} over {total_tok} tokens, n_ctx={ctx}, chunks={n_chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
