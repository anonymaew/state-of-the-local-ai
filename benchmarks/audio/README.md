# Audio

Sound, not speech. Music, sound effects, environmental audio, audio tagging and audio-text reasoning.
Speech in and out lives in [`../speech/`](../speech/README.md).

## What we measure

| Metric | Definition | Why |
| --- | --- | --- |
| `rtf` | wall seconds ÷ audio seconds | `< 1` is real-time; the only portable number across models |
| `s_per_10s` | seconds to render 10 s of audio | reads better than RTF when comparing short clips |
| `peak_mem_gb` | MLX peak | long contexts are audio tokens; 30 s is already a large prompt |
| `clap_score` | generated audio ↔ caption match | a proxy for "did it make the sound I asked for" |
| `fad` | distribution distance vs reference clips | needs ≥ 100 clips to mean anything; do not report on 5 |
| `tag_acc` | top-k tagging accuracy | for audio understanding, not generation |

Validity: fixed sample rate (44.1 kHz for music, 16 kHz for tagging), duration, steps and seed;
report duration in the row, never compare across it; measure warm, excluding model load.

## Generation

```bash
# music: caption + structured lyrics → stereo track
.venv/bin/mlx_audio.music.generate \
  --model <music-checkpoint> \
  --caption "warm analog synth, brushed drums, 90 bpm" \
  --lyrics-file data/audio/lyrics.txt \
  --duration 20 --steps 8 --seed 42 \
  --output results/audio/sample.wav
```

Candidates: MusicGen-style and ACE-Step-style checkpoints via `mlx_audio.music`; stable-audio-style
models for one-shot sound effects. A 10–20 s clip is the realistic local unit.

## Understanding / tagging

```bash
. benchmarks/common/env.sh
.venv/bin/mlx_vlm.generate \
  --model "$(model_path qwen38-27b-oq4e-mtp)" \
  --audio data/audio/<clip>.wav \
  --prompt "Describe the scene, list every sound source, and estimate their directions." \
  --max-tokens 512
```

The resident VLM takes `--audio`; use it for audio captioning and as the cheap baseline before adding
a dedicated audio encoder. `mlx_audio` also ships language-ID and classification utilities.

## Status

- scoped; no audio data or music/sfx checkpoint resident
- next: `benchmarks/audio/` emitting `results/audio/<name>.jsonl`; start with the VLM tagging path and
  one music model, sweeping duration 5/10/20 s to find the memory knee
- "audio output" from `mlx_vlm.generate --output-modality audio` is also in scope — treat it as a
  separate `task` from music generation
