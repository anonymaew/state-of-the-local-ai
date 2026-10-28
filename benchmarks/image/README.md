# Image

Two different problems share this folder. Keep them separate in `task`:

- `task: image-qa` — image **in**, text out. The resident `qwen38-27b-oq4e-mtp` already carries its
  vision encoder, so this is measurable today.
- `task: txt2img` / `task: img2img` — text/image **in**, pixels out. Needs a diffusion checkpoint that
  is not resident.

## What we measure

| Metric | Definition | Why it is the one that matters |
| --- | --- | --- |
| `prefill_tps` (vision) | image+prompt tokens / s | the encoder pass is a fixed tax per image; it dominates interactive latency |
| `answer_acc` | exact-match / rubric score on a fixed QA set | a caption that looks right can still be wrong on text-in-image |
| `s_per_image` | wall seconds per generated image, warm | the only honest generation number |
| `peak_mem_gb` | MLX peak during the denoise loop | diffusion activations, not weights, decide whether 32 GB survives 1024² |
| `clip_t` / `image_reward` | prompt adherence | catches "pretty but ignored the prompt" |

Validity: fixed resolution, steps, sampler and seed; warmup image excluded; swap-outs invalidate;
never compare across step counts or resolutions.

## Understanding (available now)

```bash
. benchmarks/common/env.sh
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --image data/image/<file>.png \
  --prompt "Transcribe every field in this form as JSON." \
  --max-tokens 512
```

Data: a small local set of screenshots/forms/charts plus public subsets (OCRBench, ChartQA, DocVQA,
MMMU) capped at a fixed `--limit` so runs stay comparable and cheap.

## Generation (needs a checkpoint)

```bash
.venv/bin/mlx_vlm.generate \
  --model <diffusion-checkpoint> \
  --output-modality image \
  --prompt "..." --size 1024x1024 --steps 4 --seed 42 --output results/image/out.png
```

Candidates for a 32 GB box: FLUX.1-schnell (4-bit, few steps), Qwen-Image / Qwen-Image-Edit,
SDXL/SD3.5 at 1024². Quantized DiTs are the only ones likely to fit at 1024²; 512² is the fallback,
and the resolution must be recorded because it changes the runtime, not just the quality.

## Status

- scoped; no image datasets or diffusion weights resident
- next: build `benchmarks/image/` harness that emits `results/image/<name>.jsonl` with the row fields
  from [`../common/README.md`](../common/README.md), starting with the VLM path (no download needed)
