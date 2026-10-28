# `benchmarks/common`

Shared plumbing for every modality harness. Nothing here is text-specific.

| File | Purpose |
| --- | --- |
| `env.sh` | Bash layout contract: `ROOT`, `MODALITY`, `HF_HOME`, `HF_HUB_CACHE`, `MODELS`, `DATA`, `RESULTS`, `LOGS`, `PRIMARY_TEXT_MODEL`, `model_path()`. Source it, don't copy paths. |
| `model_store.py` | Alias/repo-id → filesystem path, backed by the Hub cache. Also imports a `--local-dir` checkpoint into the cache (hard links, no copy). |
| `row.schema.json` | The JSONL row every harness writes. One row = one measurement, one process, one load of the model. |

## The layout contract

```text
benchmarks/<modality>/<harness>      code
data/<modality>/                     inputs (datasets, prompts, clips, wavs)
results/<modality>/<harness>.jsonl   append-only measurements
logs/<modality>/<run>.log            raw engine output
$HF_HUB_CACHE/models--<org>--<repo>  weights (default ~/.cache/hf/hub); no copy in models/
models/README.md                     the registry: alias -> repo id, format, size
```

Weights are addressed, not pathed. `model_path <alias|org/repo[:file]|path>` prints a snapshot path and
never downloads unless `MODEL_STORE_ONLINE=1`; a benchmark must not pull bytes inside a measured
window. `model_store.py list` shows which aliases the cache satisfies.

A harness may add columns, never rename or redefine these:

| Field | Type | Meaning |
| --- | --- | --- |
| `modality` | string | `text` \| `image` \| `video` \| `audio` \| `speech` \| `decision` |
| `task` | string | what is being measured, e.g. `decode`, `txt2img`, `asr`, `tool-call` |
| `engine` | string | `llama.cpp` \| `mlx-lm` \| `mlx-vlm` \| `mlx-whisper` \| `oMLX` \| … |
| `model` | string | weights id, including quant, e.g. `qwen38-27b-oq4e-mtp` |
| `label` | string | free-form config tag (`kv=q8_0 depth=4096`) |
| `ts` | ISO-8601 | wall clock, second precision |
| `valid` | bool | passes the validity gates in [`docs/text/02-benchmark-plan.md`](../../docs/text/02-benchmark-plan.md) |
| `metrics` | object | modality-owned numbers (t/s, s/image, WER, RTF, pass@1…). Units in the key name. |
| `harness` | object | optional: versions, flags, hardware state (load, swap, thermal) |

`valid: false` rows stay in the file. They are evidence of a rejected run, not garbage to delete.

## Adding a modality

1. `benchmarks/<modality>/README.md` — scope, what is measured, candidate engines/models for 32 GB, status.
2. `benchmarks/<modality>/<harness>` — source `common/env.sh`, call `model_path`, write `$RESULTS/<harness>.jsonl`.
3. `data/<modality>/` and `results/<modality>/` and `logs/<modality>/` — created by `env.sh`.
