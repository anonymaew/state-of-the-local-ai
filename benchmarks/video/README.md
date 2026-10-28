# Video

Video is the modality where 32 GB stops being "enough". Separate consumption from generation in `task`:
`video-qa` is affordable; `txt2video` is a budget experiment, not a workflow.

## What we measure

| Metric | Definition | Why |
| --- | --- | --- |
| `frames_per_s` | frames encoded or attended per second | frame sampling is the dominant prefill cost |
| `qa_acc` | accuracy on a fixed Video-MME-style subset | answers are easy to fake by sampling too few frames |
| `s_per_clip` | wall seconds per generated clip | the number that decides if a shot is usable |
| `s_per_frame` | s_per_clip ÷ frames | lets clips of different lengths compare |
| `peak_mem_gb` | MLX peak | latent video is quadratic-ish in frames × resolution |

Validity: fixed `--fps`, `--video-num-frames`, resolution, steps, seed and clip length. Record the
sampling policy (uniform vs scene-change) — it changes both accuracy and cost.

## Understanding (available now)

```bash
. benchmarks/common/env.sh
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --video data/video/<clip>.mp4 \
  --fps 1 --video-num-frames 16 \
  --prompt "What changes between 0:02 and 0:06? Answer with timestamps." \
  --max-tokens 512
```

The cost curve to map: frames → prefill seconds → peak memory. 16 frames at 1 fps is the cheap point;
32+ frames is where a 27B VLM on this box starts to page.

## Generation (needs a checkpoint)

```bash
.venv/bin/mlx_vlm.generate \
  --model <video-checkpoint> \
  --output-modality video \
  --workflow t2va \
  --prompt "..." --num-frames 49 --steps 4 --seed 42 \
  --output results/video/out.mp4
```

`--workflow` also accepts `fl2va` (first/last frame) and `ref2va` (reference image) — those are the
realistic local uses, because a single conditioned shot is cheaper than an open-ended one.

## Status

- scoped; no video data resident
- next: a `video` harness that sweeps `frames × resolution` for the VLM path first (no download), then
  generation behind an explicit memory ceiling and a hard timeout
