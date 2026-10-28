# Speech track

**Question:** is transcription and synthesis fast and accurate enough to sit in an interactive loop —
and is the bottleneck throughput or first-token latency?

Status: **scoped** — no rows yet, no speech model resident (`mlx-audio` is installed and its CLIs run).
Harness scope and exact metrics: [`benchmarks/speech/README.md`](../../benchmarks/speech/README.md).

## Latency-shaped, not throughput-shaped

A transcriber at 6× real time that emits nothing for 4 s is worse than one at 2× that streams.

### STT (`task: asr`)

| Metric | Why |
| --- | --- |
| `wer` / `cer` | normalize case, punctuation and numerals or the number is fiction |
| `rtf` (audio ÷ wall, **> 1** is real-time) | the headline |
| `ttft_ms` | latency users feel |
| `peak_mem_gb` | long files are chunked; chunk size is the knob |

### TTS (`task: tts`)

| Metric | Why |
| --- | --- |
| `rtf` (audio ÷ wall) | same scale, opposite direction |
| `first_audio_ms` | what makes streaming TTS usable |
| `utmos` | cheap naturalness proxy, no listening panel |
| `wer_roundtrip` | intelligibility, not just timbre |
| `spk_sim` | only for voice-cloning models |

Validity: fixed audio (16 kHz mono) or fixed transcript + voice, fixed chunking, load excluded from
RTF, sample rate and model version in the row.

## Start here

```bash
.venv/bin/mlx_audio.stt.generate \
  --model <whisper-or-parakeet-checkpoint> \
  --audio data/speech/<utt>.wav \
  --format json --language en \
  --output-path results/speech/<utt>.json
```

Candidates: Whisper large-v3-turbo, Parakeet TDT 0.6B for STT; Kokoro-82M and
Chatterbox/Dia-class for TTS. Cross-engine comparisons are the point: the same Whisper weights are
~2× apart in RTF across Metal implementations. Data: LibriSpeech `test-clean` capped at 100
utterances for sweeps, plus local noisy/accented recordings. Physics and engine traps:
[`../shared/`](../shared/README.md).
