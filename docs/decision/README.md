# Decision (System One) track

**Question:** which model + engine answers typed `choice` / `noul` / `score` questions accurately,
calibrated, and fast enough to sit in a request path on 32 GB? The model must choose rather than
write: one forward pass, a probability per option, no generation.

Status: **measured**. Rows live in [`results/decision/`](../../results/decision), harnesses in
[`benchmarks/decision/`](../../benchmarks/decision/README.md).

## Write-ups, in reading order

| File | What it is |
| --- | --- |
| [`01-plan.md`](01-plan.md) | Question, hypotheses, the engine constraint, the measurement contract |
| [`02-engines.md`](02-engines.md) | Which runtimes can execute a decision head, and the `/v1/systemone` protocol |
| [`02-engines.sources.md`](02-engines.sources.md) | Source excerpts behind every engine claim (P/D labelled) |
| [`03-models.md`](03-models.md) | 9B candidates, exact HF sizes, the adapter-fold cliff, memory math |
| [`03-models.candidates.json`](03-models.candidates.json) | Machine-readable candidate table used to build `03-models.md` |
| [`04-eval.md`](04-eval.md) | Decision Index v0.2 formula, `breadth-v1`, calibration definitions |
| [`05-results.md`](05-results.md) | **Measured:** 2,000 typed decisions × 4 configs, latency/thermal curves |
| [`06-recommendation.md`](06-recommendation.md) | The pick, memory budget, what not to run, next experiments |

## The answer in one screen

Three 9B native decision models, two engines, the `typed-decisions` test split (400 cases → 2,000
typed questions):

- **Clef-Flash 9B, MLX 4-bit (`clef-flash-4bit`) → 0.697 accuracy, 1.04 s per five-question
  request, 5.6 GB.** The same weights under llama.cpp Q4_K_M: 0.702, 2.14 s, 6.8 GB. **The engine is
  worth ~1.8× latency on this box.**
- **Kev-9B Q4_K_M (llama.cpp) → 0.680 accuracy, 2.76 s, 7.2 GB.** Its MLX adapter path does not fit:
  the adapter fold peaks at ≈36 GB against a 19.3 GB bf16 base.
- **32 GB is not the constraint at 9B** — 6–7 GB resident. The format is: a GGUF carries the decision
  head; the MLX adapter path carries a second copy of the base while folding.
- **8-bit is not worth it here:** 0.699 vs 0.697 accuracy, 1.4× slower, 10.7 GB vs 6.2 GB.
- Quality is deterministic across re-runs (0.6965 twice); latency is not — the chassis matters more.
  Sustained runs on this fanless Air lose 50–100 % of latency; first-100 and full-run p50 are both
  reported because either alone is misleading.

## Reproducing

```bash
. benchmarks/common/env.sh

# the fastest config
.venv/bin/python benchmarks/decision/run_config.py \
  --engine mlx-clef --model-path clef-flash-4bit \
  --model-id clef-flash-4bit-mlx --quant 4bit --ctx 8192

# the portable GGUF arm
.venv/bin/python benchmarks/decision/run_config.py \
  --engine llamacpp --model-path clef-flash-q4km \
  --model-id clef-flash-q4km --quant Q4_K_M --ctx 8192

benchmarks/decision/run_matrix.sh        # all four configs in one command
```

`--model-path` takes an alias, `org/repo[:file]`, or a path. One invocation = one server = one row.
