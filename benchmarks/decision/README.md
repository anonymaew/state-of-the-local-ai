# Decision (System One)

Everything where the model must **choose** rather than write: typed `choice` / `noul` / `score`
questions answered with a probability per option, in one forward pass. Plan and hypotheses:
[`docs/decision/01-plan.md`](../../docs/decision/01-plan.md). Engine matrix:
[`docs/decision/02-engines.md`](../../docs/decision/02-engines.md). Models:
[`docs/decision/03-models.md`](../../docs/decision/03-models.md). Evaluation:
[`docs/decision/04-eval.md`](../../docs/decision/04-eval.md).

## What we measure

| Metric | Definition | Why |
| --- | --- | --- |
| `accuracy` | argmax of the predicted distribution vs argmax of gold | the capability |
| `kl_from_gold` | mean KL(gold ‖ predicted) over decisions | ranks the whole distribution, not just the top |
| `brier` | mean multiclass Brier (0..2) | calibration + sharpness together |
| `ece` | expected calibration error, 10 equal-width confidence bins | overconfidence is worse than abstention |
| `schema_valid_rate` | answers parsed / decisions expected | with a native decision model this must be 1.00 |
| `latency_ms_p50/p95` | first ask about a state, wall clock incl. HTTP | interactive; prefill dominates |
| `cached_latency_ms_*` | same state re-asked (server prefix cache) | the multi-question cost that actually matters |
| `input_tokens_*` | state + question tokens reported by the engine | the battery cost |
| `peak_rss_mb` | sampled from the server process | 32 GB is a shared pool |

The metric definitions are the suite's own, so a local number is directly comparable with the
published leaderboard: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(test split, 400 cases → 2,000 decisions, five typed questions per case).

## Harnesses

| File | Role |
| --- | --- |
| `bench_systemone.py` | client + metrics for any server that answers `POST /v1/systemone`. Writes one JSONL row. |
| `run_config.py` | spawns one engine on one checkpoint, runs the bench, kills it. One row = one load. |
| `serve_d3_mlx.py` | `/v1/systemone` in front of the d3-flash MLX loader, which ships without a server. |
| `load_suite.py` | resolves `data/decision/manifest.json` into normalized items (breadth-v1, smoke). |
| `run_matrix.sh` | the 9B comparison in one command. |

Two engines are in scope because they are the only ones that execute a decision head:

- **llama.cpp** — a GGUF with decision metadata (`clef`, `kev`, `vllm-d3`, `openjev`, `laya`, …) is
  served natively at `/v1/systemone`; no Python model code involved.
- **MLX** — the checkpoint's own Python module (`clef_mlx.py`, `kev.mlx_model`) runs the backbone and
  the head; needed for head implementations that llama.cpp has not ported.

## Run

```bash
. benchmarks/common/env.sh

# full 2,000-decision score, one config
.venv/bin/python benchmarks/decision/run_config.py \
  --engine llamacpp \
  --model-path clef-flash-q4km \
  --model-id clef-flash-q4km --quant Q4_K_M --ctx 8192 --port 8092

# the same on the MLX arm
.venv/bin/python benchmarks/decision/run_config.py \
  --engine mlx-clef --model-path clef-flash-4bit \
  --model-id clef-flash-4bit-mlx --quant 4bit --ctx 8192 --port 8091

# everything with weights present
benchmarks/decision/run_matrix.sh
```

`--model-path` accepts an alias (`clef-flash-q4km`), an `org/repo[:file]` ref, or a path; the alias is
resolved from the Hub cache ([`benchmarks/common/model_store.py`](../common/model_store.py)).
`--model-id` defaults to the resolved file/dir name.

Smoke before a full run (5 cases, ~30 s):

```bash
.venv/bin/python benchmarks/decision/run_config.py --engine mlx-clef \
  --model-path clef-flash-4bit --model-id smoke --limit 5 --port 8099
```

## Context is part of the config

A `choice` question with N options puts the whole option schema in the prompt. At N=255 the schema
alone is **~8.8 k tokens**, so `ctx=8192` fails with HTTP 413 before any state is read. The smoke set
catches this; use `--ctx 16384` when option sets can be large. Long states compound it: a 6 k-token
state with 255 options needs both.

## Validity

- Same item set, same request shape, one model load per row; `results/decision/typed.jsonl` is append-only.
- A config is `valid: true` only when every decision parsed and the run did not swap by >512 MB.
- `temperature` is a checkpoint property in a native decision model, not a request parameter; it is
  recorded in `harness` via the engine's own report, never set by the harness.
- Numbers from `breadth-v1` (Kev's index) and `typed-decisions` are different suites. Do not compare
  them, and do not mix them in one table.
