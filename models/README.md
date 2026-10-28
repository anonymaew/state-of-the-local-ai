# Models

Weights are **not** stored in this repo. Every checkpoint lives in the Hugging Face Hub
cache (`$HF_HOME/hub`, default `~/.cache/hf/hub`), one `models--<org>--<repo>` directory
per checkpoint. Harnesses name a checkpoint by alias or `org/repo[:file]`
([`benchmarks/common/model_store.py`](../benchmarks/common/model_store.py)) and get a
snapshot path back, so the cache is the only copy on disk.

```bash
. benchmarks/common/env.sh          # sets HF_HOME=~/.cache/hf and HF_HUB_CACHE
model_path qwen38-27b-oq4e-mtp      # -> .../models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/<rev>
model_path clef-flash-q4km          # -> .../Clef-Flash-Q4_K_M.gguf
.venv/bin/python benchmarks/common/model_store.py list
```

`models/` is retained only for this registry file. A deleted checkpoint frees its bytes;
there are no per-model symlinks or shared blobs to keep in sync.

## Resident set

| Alias | Hub repo | Format | Size | Modalities | License |
| --- | --- | --- | --- | --- | --- |
| `qwen38-27b-oq4e-mtp` | [`Jundot/Qwen3.8-27B-oQ4e-mtp`](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) | MLX safetensors, 4 shards | 16 GiB (17.8 GB) | text, image, video → text; MTP head | Apache-2.0 |

`qwen38-27b-oq4e-mtp` is an oQ mixed-precision MLX checkpoint of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): 27B dense, Qwen3.5 architecture
(16 of 64 layers full-attention), 262k native context, vision encoder + MTP head included. It is the
reference model for the [text](../docs/text/README.md), [image](../docs/image/README.md),
[video](../docs/video/README.md), [audio](../docs/audio/README.md) and
[decision](../docs/decision/README.md) tracks.

```bash
.venv/bin/python benchmarks/text/bench_mlx.py qwen38-27b-oq4e-mtp oq4e-mtp
.venv/bin/mlx_vlm.generate --model "$(model_path qwen38-27b-oq4e-mtp)" --image shot.png --prompt "..."
```

## Decision models

Weights for the [decision track](../docs/decision/README.md). All Apache-2.0, all 9B, all
native decision models: they answer typed questions with a probability per option and never write
text. Two formats because the head has to travel with the weights:

| Alias | Hub repo | Format | Size | Head | Runs on |
| --- | --- | --- | --- | --- | --- |
| `clef-flash-4bit` | `mlx-community/clef-flash-4bit` | MLX 4-bit + `joint_head.safetensors` + `clef_mlx.py` | 6.2 GB | joint schema head | MLX (`clef_mlx.py serve`) |
| `clef-flash-8bit` | `mlx-community/clef-flash-8bit` | MLX 8-bit + head + loader | 10.7 GB | joint schema head | MLX; measured, **not recommended** |
| `clef-flash-q4km` | `ggml-org/Clef-Flash-GGUF` | GGUF Q4_K_M | 6.3 GB | baked in (`clef` metadata) | llama.cpp `-m` |
| `kev-9b-q4km` | `ggml-org/Kev-9B-GGUF` | GGUF Q4_K_M | 6.0 GB | baked in (pointer head, `kev` metadata) | llama.cpp `-m` |
| `d3-flash-q4km` | `ggml-org/vllm-d3-flash-GGUF` | GGUF Q4_K_M | 6.2 GB | baked in (`vllm-d3` metadata) | llama.cpp `-m` |

Kev-9B is deliberately **not** stored as the MLX adapter: the adapter is 173 MB on a 19.33 GB bf16
base, and the MLX loader folds the adapter into a second copy while loading (~1.9× weights ≈ 36 GB
peak), which does not fit a 32 GB machine. The GGUF carries the pointer head and needs 6 GB.

## Removed on 2026-10-21 (≈40 GB freed)

| Checkpoint | Size | Why it was here | To restore |
| --- | --- | --- | --- |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M | 16 GB | llama.cpp quality-first config (PPL 6.037) | `MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km` |
| `mtp-Qwen3.8-27B-Q4_0.gguf` + dflash heads | 2.6 GB | MTP/draft heads for llama.cpp speculative decode | `MODEL_STORE_ONLINE=1 model_path qwen38-27b-mtp` |
| `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` | 21 GB | MoE speed baseline (54.5 tok/s) | see [model survey](../docs/text/01-models.md) |

Results produced with these checkpoints remain in [`results/text/`](../results/text/) as evidence for
[`docs/text/03-results-speed.md`](../docs/text/03-results-speed.md) and
[`docs/text/05-recommendation.md`](../docs/text/05-recommendation.md). Re-running a sweep needs the
weights back and a **new label** — appending under an old label mixes incompatible runs.

## Rules

- One checkpoint = one cache directory. `hf cache rm <repo>` is the only deletion path.
- Record the exact repo id and revision in `model`; quant and shard layout are part of the result.
- `HF_TOKEN` lives in `.env` and is read by the download tooling, never committed (see `.gitignore`).
- Benchmark runs are cache-only (`MODEL_STORE_ONLINE=0`, the default). Download before measuring,
  never during: a load inside the measured window invalidates the row.
- Weights imported from a `--local-dir` download keep their original revision. `model_store.py
  import` hard-links the bytes, so adoption costs no extra disk.

`.cache/` under `data/decision/` holds the typed-decisions suite (1 MB), not weights.
