# Results

Append-only JSONL, one directory per modality, one file per harness. Never edit a row after the run;
if a run was invalid, it carries `valid: false` and stays.

```text
results/<modality>/<harness>.jsonl       one measurement per line
results/<modality>/quality/<tag>/        per-task evaluator output (lm-eval, etc.)
```

## Row contract

See [`benchmarks/common/README.md`](../benchmarks/common/README.md). Required:
`modality, task, engine, model, label, ts, valid, metrics`, plus an optional `harness` object for
versions, flags, load/swap/thermal state.

## Text track (historical, 2026-10-20…21)

These files predate the contract and use harness-local schemas. They are the evidence for
[`docs/text/03-results-speed.md`](../docs/text/03-results-speed.md),
[`docs/text/04-results-quality.md`](../docs/text/04-results-quality.md) and
[`docs/text/06-omlx.md`](../docs/text/06-omlx.md); do not normalise them retroactively.

| File | Rows are | Supports |
| --- | --- | --- |
| `text/speed_llamacpp.jsonl` | llama-bench tg/pp per KV type and depth | dense-vs-MoE, depth decay |
| `text/speed_llamacpp.loaded.jsonl` | same, with weights resident | load-time costs |
| `text/speed_mlx.jsonl` | mlx-lm prompt/gen t/s, peak memory | MLX throughput vs depth |
| `text/server.jsonl` | llama-server timings + swap delta | decode/prefill/TTFT validity gates |
| `text/spec.jsonl` | llama-cli decode + draft acceptance | spec-decoding speedup |
| `text/ppl.jsonl` | perplexity per checkpoint/context | quality-first vs speed-first |
| `text/quality/` | lm-eval per-task outputs + `summary.jsonl` | ARC / TruthfulQA / GSM8K / IFEval |
| `text/ab_mtp.jsonl` | paired llama.cpp/oMLX rounds | the +23.6 % engine-config claim |
| `text/omlx*.jsonl` | oMLX engine, continuous batching, concurrency | [`docs/text/06-omlx.md`](../docs/text/06-omlx.md) |

## Rules

- One row per measured request. Medians are computed by the reader, not baked in.
- A rerun with different weights, flags, or machine state gets a new `label`, even into the same file.
- Keep `ts` in ISO-8601; the machine drifts and the date is part of the interpretation.
- Never delete a failing row. Delete nothing, in fact: annotate instead.
- Home-directory paths in engine status dumps were redacted to `.` / `~` before publication. No
  measurement, flag or verdict was changed; see `git log --follow` on the file if you need the original.
