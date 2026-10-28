# Image track

**Question:** what image understanding (and, secondarily, image generation) is usable on 32 GB, and
what does each cost in seconds and peak memory?

Status: **scoped** — no rows yet, no diffusion weights resident. Harness scope and exact metrics:
[`benchmarks/image/README.md`](../../benchmarks/image/README.md).

## Two tasks, deliberately separate

- `image-qa` — image in, text out. The resident
  [`qwen38-27b-oq4e-mtp`](../../models/README.md) already carries a vision encoder, so this is
  measurable today with no download.
- `txt2img` / `img2img` — pixels out. Needs a diffusion checkpoint; quantized DiTs are the only
  plausible 1024² candidates at 32 GB.

## Metrics (fixed before measuring)

| Metric | Why |
| --- | --- |
| `prefill_tps` (vision) | fixed encoder tax per image; dominates interactive latency |
| `answer_acc` | a caption that looks right can be wrong on text-in-image |
| `s_per_image` | warm wall seconds; the only honest generation number |
| `peak_mem_gb` | diffusion activations, not weights, decide whether 1024² survives |
| `clip_t` / `image_reward` | prompt adherence |

Validity: fixed resolution, steps, sampler, seed; warmup excluded; swap-outs invalidate; never compare
across step counts or resolutions.

## Start here

```bash
. benchmarks/common/env.sh
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --image data/image/<file>.png \
  --prompt "Transcribe every field in this form as JSON." \
  --max-tokens 512
```

The VLM path needs no new weights. Generation is blocked on choosing and downloading one checkpoint;
record the resolution and step count in `label` because they change the runtime, not just the quality.
Physics and engine traps: [`../shared/`](../shared/README.md).
