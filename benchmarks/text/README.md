# Text track

Measured. Write-ups in [`docs/`](../../docs/README.md); rows in [`results/text/`](../../results/text/).

## Resident model

`qwen38-27b-oq4e-mtp` (Hub repo `Jundot/Qwen3.8-27B-oQ4e-mtp`) — MLX, 4.887 bits/weight, 17.8 GB, MTP
head, vision+video encoder. It resolves from the Hub cache; there is no path in the repo.
Checkpoints below that need GGUF or a second model are **not resident by default**; their aliases
restore them into the cache on demand with `MODEL_STORE_ONLINE=1 model_path <alias>` (see
[`models/README.md`](../../models/README.md)).

## Harnesses

| Harness | Emits | Measures |
| --- | --- | --- |
| `bench_mlx.py <model> <label>` | `results/text/speed_mlx.jsonl` | prompt/gen t/s and peak memory at `DEPTHS` (default 64,4096,16384) |
| `ppl_mlx.py <model> <text> [chunks] [ctx]` | stdout | perplexity, chunked exactly like `llama-perplexity` |
| `quality_mlx.sh <model> <tag> <limit> <task…>` | `results/text/quality/<tag>/` | lm-eval task scores, one process per task |
| `extract_quality.py` | `results/text/quality/summary.jsonl` | scrapes the lm-eval JSON blocks from `logs/text/q_*.log` |
| `bench_speed_gguf.sh <model.gguf> <label>` | `results/text/speed_llamacpp.jsonl` | llama-bench tg/pp across KV types and depths |
| `bench_server.sh <label> <model.gguf>` | `results/text/server.jsonl` | server decode/prefill/TTFT + swap delta |
| `bench_spec.sh <label> <model.gguf>` | `results/text/spec.jsonl` | draft acceptance and spec-decode speedup |
| `ab_mtp.py [rounds]` | `results/text/ab_mtp.jsonl` | interleaved llama.cpp-vs-oMLX paired ratio |

## Commands

```bash
. benchmarks/common/env.sh

# MLX throughput (mlx-lm). If the loader rejects vision tensors, run mlx-vlm instead
# and record engine=mlx-vlm: the checkpoint is a VLM, not a text-only model.
.venv/bin/python benchmarks/text/bench_mlx.py qwen38-27b-oq4e-mtp oq4e-mtp

# Perplexity on the same file llama-perplexity used
.venv/bin/python benchmarks/text/ppl_mlx.py qwen38-27b-oq4e-mtp data/text/wiki.test.raw 12 2048

# Task quality, then collect the rows
benchmarks/text/quality_mlx.sh qwen38-27b-oq4e-mtp oq4e-mtp 200 \
  arc_challenge truthfulqa_mc2
.venv/bin/python benchmarks/text/extract_quality.py

# GGUF paths need a re-downloaded checkpoint (not resident): the aliases carry the
# org/repo and filename, MODEL_STORE_ONLINE=1 lets the first resolve download.
MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km
MODEL_STORE_ONLINE=1 model_path qwen38-27b-mtp
.venv/bin/python benchmarks/text/ab_mtp.py 3

benchmarks/text/bench_speed_gguf.sh qwen38-27b-gguf-q4km mylabel
```

## Reading a result

- `decode_tps` is bandwidth-bound and drifts up to 25 % within a session on this fanless M5.
  Compare **within a run**, use the paired ratio when engines differ.
- `swap_delta_mib` / `swapouts_delta_pages` must be 0. A row that paged is not a row.
- Depth sweeps are the point: zero-depth numbers overstate long-context usability.
- `valid` follows [`docs/text/02-benchmark-plan.md`](../../docs/text/02-benchmark-plan.md); the historical
  rows predate the field and are treated as valid only where the doc says so.

## History

Rows in `results/text/` were produced 2026-10-20…21 against checkpoints that are no longer resident
(`qwen38-27b-gguf`, `qwen38-aux` MTP head, `qwen36-35ba3b-oq4e-mtp`, `qwen38-27b-mlx4bit`). They are
kept as evidence for the write-ups and labelled by their original tags; do not re-run a sweep and
append under the same label expecting comparable numbers.
