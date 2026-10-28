# Data

One directory per modality. Inputs are small and fixed: a benchmark whose dataset changes between
runs is not a benchmark.

```text
data/text/      wiki.test.raw (fetched, not committed; see data/text/README.md)
data/image/     screenshots, forms, charts, QA subsets
data/video/     short clips (≤ 60 s) + frame-sampling manifests
data/audio/     wav/flac clips, lyrics, captions
data/speech/    utterances for ASR, transcripts and reference voices for TTS
data/decision/  manifest, smoke set, model-eval provenance; suite cached on demand
```

## Rules

- **Pin provenance.** Every dataset gets a line in the modality README: source URL, split, `--limit`,
  and the date fetched. "LibriSpeech test-clean" without a limit is not reproducible.
- **Deterministic sampling.** Sample once, store the manifest (file list or indices). Do not re-sample
  per run; the score change will be sampling noise and you will believe it.
- **Same audio settings.** 16 kHz mono for speech, 44.1 kHz stereo only where music requires it.
  Resampling inside the harness is allowed only if the harness records the source rate.
- **No generated outputs here.** Model outputs go to `results/<modality>/`; `data/` is inputs only.
- **Keep it small.** A subset that runs in minutes beats a full split that nobody runs. Full-split
  numbers are a final pass, not a sweep.

## Licensing

Only redistribute a dataset if its license allows it. For restricted corpora, store the fetch script
and the manifest, not the bytes.
