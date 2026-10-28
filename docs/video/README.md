# Video track

**Question:** how many frames can this box attend to before the KV cache — not the weights — becomes
the limit, and what does video QA or a short generated clip cost?

Status: **scoped** — no rows yet, no video data or video checkpoint resident. Harness scope and exact
metrics: [`benchmarks/video/README.md`](../../benchmarks/video/README.md).

## Consumption vs generation

- `video-qa` — frames in, text out. Affordable today with the resident
  [`qwen38-27b-oq4e-mtp`](../../models/README.md); the cost curve to map is
  `frames → prefill seconds → peak memory`.
- `txt2video` / `fl2va` / `ref2va` — a budget experiment, not a workflow. One conditioned shot is
  cheaper than an open-ended clip.

## Metrics (fixed before measuring)

| Metric | Why |
| --- | --- |
| `frames_per_s` | frame sampling is the dominant prefill cost |
| `qa_acc` | easy to fake by sampling too few frames |
| `s_per_clip`, `s_per_frame` | makes clips of different lengths comparable |
| `peak_mem_gb` | latent video is quadratic-ish in frames × resolution |

Validity: fixed `--fps`, `--video-num-frames`, resolution, steps, seed, clip length. Record the
sampling policy (uniform vs scene-change) — it changes both accuracy and cost.

## Start here

```bash
. benchmarks/common/env.sh
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --video data/video/<clip>.mp4 \
  --fps 1 --video-num-frames 16 \
  --prompt "What changes between 0:02 and 0:06? Answer with timestamps." \
  --max-tokens 512
```

16 frames at 1 fps is the cheap point; 32+ frames is where a 27B VLM on this box starts to page.
Physics and engine traps: [`../shared/`](../shared/README.md).
