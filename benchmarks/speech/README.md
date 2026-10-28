# Speech

STT and TTS. The interesting numbers are latency-shaped, not throughput-shaped: a transcriber at
6× real time that emits nothing for 4 s is worse than one at 2× that streams.

## What we measure

### STT (`task: asr`)

| Metric | Definition | Why |
| --- | --- | --- |
| `wer` / `cer` | normalized word/char error rate | normalize case, punctuation and numerals or the number is fiction |
| `rtf` | audio seconds ÷ wall seconds (**> 1** is real-time) | the headline |
| `ttft_ms` | time to first partial token | latency users feel |
| `peak_mem_gb` | MLX peak | long files are chunked; chunk size is the knob |

### TTS (`task: tts`)

| Metric | Definition | Why |
| --- | --- | --- |
| `rtf` | audio seconds ÷ wall seconds (> 1 is real-time) | same scale, opposite direction from STT |
| `first_audio_ms` | time to first audio chunk | what makes streaming TTS usable |
| `utmos` | predicted naturalness | cheap MOS proxy, no listening panel |
| `wer_roundtrip` | WER of STT(TTS(text)) | intelligibility, not just timbre |
| `spk_sim` | speaker embedding similarity | only for voice-cloning models |

Validity: fixed audio (16 kHz mono) or a fixed transcript and voice, fixed chunking (`--chunk-duration`,
`--max-parallel-segments`), load excluded from RTF, sample rate and model version in the row.

## STT

```bash
.venv/bin/mlx_audio.stt.generate \
  --model <whisper-or-parakeet-checkpoint> \
  --audio data/speech/<utt>.wav \
  --format json --language en \
  --output-path results/speech/<utt>.json
```

Candidates: Whisper large-v3-turbo (mlx-audio / whisper.cpp / WhisperKit all support it), Parakeet
TDT 0.6B. Cross-engine comparisons are the point: Whisper Metal kernels differ a lot between
implementations, and the same weights can be 2× apart in RTF.

```bash
# harness to be added: benchmarks/speech/bench_asr.py <model> <label>
.venv/bin/python benchmarks/speech/bench_asr.py <model> <label>
```

Data: LibriSpeech `test-clean` capped at 100 utterances for sweeps, full split for the final number;
plus 10 local recordings with noise and accents, because clean-listening WER flatters every model.

## TTS

```bash
.venv/bin/mlx_audio.tts.generate \
  --model <tts-checkpoint> \
  --text "..." --voice af_heart \
  --output_path results/speech/ \
  --file_prefix sample --audio_format wav
```

Candidates: Kokoro-82M (fast, tiny, non-cloning), Chatterbox / Dia-class models (expressive, cloning,
slower). Measure first-audio latency with `--stream` separately from full-clip RTF.

## Status

- scoped; no speech data or models resident (`mlx-audio` is installed and its CLIs run)
- next: `benchmarks/speech/bench_asr.py` and `bench_tts.py`, writing `results/speech/`
