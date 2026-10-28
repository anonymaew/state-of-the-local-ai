# Audio track

**Question:** can this box render music / sound effects faster than real time at a usable quality, and
how long a clip fits before audio tokens become the memory limit?

Status: **scoped** — no rows yet, no music/sfx checkpoint resident. Harness scope and exact metrics:
[`benchmarks/audio/README.md`](../../benchmarks/audio/README.md).

Sound, not speech. STT/TTS live in [`../speech/`](../speech/README.md).

## Three tasks

- `music` — caption (+ optional lyrics) → stereo track.
- `sfx` — one-shot sound effect from a prompt.
- `audio-qa` / `tagging` — audio in, text out. The resident
  [`qwen38-27b-oq4e-mtp`](../../models/README.md) takes `--audio`, so this needs no new checkpoint.

## Metrics (fixed before measuring)

| Metric | Why |
| --- | --- |
| `rtf` (wall ÷ audio seconds) | `< 1` is real-time; portable across models |
| `s_per_10s` | reads better for short clips |
| `peak_mem_gb` | 30 s of audio is a large prompt, not a context window |
| `clap_score` | did it make the sound that was asked for |
| `fad` | needs ≥ 100 clips to mean anything; do not report on 5 |
| `tag_acc` | understanding, not generation |

Validity: fixed sample rate (44.1 kHz music, 16 kHz tagging), duration, steps, seed; report duration
in the row; measure warm, excluding model load.

## Start here

```bash
. benchmarks/common/env.sh

# understanding / tagging — no new weights
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --audio data/audio/<clip>.wav \
  --prompt "Describe the scene, list every sound source, and estimate their directions." \
  --max-tokens 512
```

Generation starts with one music model and a duration sweep (5/10/20 s) to find the memory knee.
`mlx_vlm.generate --output-modality audio` is a separate `task` from `mlx_audio.music.generate`.
Physics and engine traps: [`../shared/`](../shared/README.md).
