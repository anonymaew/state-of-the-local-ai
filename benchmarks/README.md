# Benchmarks

One directory per modality. Same contract everywhere:

```text
benchmarks/<modality>/       harnesses (source common/env.sh)
data/<modality>/             inputs
results/<modality>/          JSONL, append-only
logs/<modality>/             raw engine output
$HF_HUB_CACHE/               weights (default ~/.cache/hf/hub)
```

Harnesses take a model **ref** — alias (`qwen38-27b-oq4e-mtp`, `clef-flash-q4km`), `org/repo[:file]`,
or an existing path — and resolve it through [`common/model_store.py`](common/model_store.py).
Weights are never copied into the repo; the registry is [`models/README.md`](../models/README.md).
Track write-ups are organised by modality in [`docs/`](../docs/README.md).

## Modalities

| Modality | Owning runtime | First harness to build | Why it is hard on 32 GB |
| --- | --- | --- | --- |
| [Text](text/README.md) | llama.cpp, mlx-lm, oMLX | (built) | density: weights are the whole cost, decode is bandwidth-bound |
| [Image](image/README.md) | mlx-vlm | image→text accuracy + s/image at 1024² | diffusion activations, not weights, set the peak |
| [Video](video/README.md) | mlx-vlm | frames/s prefill, video QA, s/clip | KV grows with frames; clips must be capped, not downscaled blindly |
| [Audio](audio/README.md) | mlx-audio, mlx-vlm | real-time factor for music/sfx, CLAP score | duration is the knob: 30 s of audio is a generation budget, not a context |
| [Speech](speech/README.md) | mlx-audio | WER, RTF, first-token latency | streaming latency ≠ throughput; chunking dominates |
| [Decision](decision/README.md) | llama.cpp `/v1/systemone`, MLX loaders | (built) | constrained decoding costs prefill/TTFT, not decode |

## Tooling already in `.venv`

| Package | Version | Gives you |
| --- | --- | --- |
| `mlx` | 0.32.4 | Metal backend |
| `mlx-lm` | 0.32.2 | text generation, `mlx_lm.evaluate` |
| `mlx-vlm` | 0.7.9 | image/video/audio input, **text/image/video/audio output**, `mlx_vlm.decide` |
| `mlx-audio` | 0.6.1 | `stt.generate`, `tts.generate`, `music.generate`, `sts.generate` |
| `lm-eval` | 0.4.13 | task accuracy (text) |
| `llama.cpp` | `src/llama.cpp` | GGUF engine, server, grammars, MTP spec decode |

`.venv-omlx` additionally has `omlx 0.7.1` and `mlx-embeddings` (decision/retrieval work).

## Adding a harness

1. Read [`common/README.md`](common/README.md) — the row schema is not optional.
2. Put code in `benchmarks/<modality>/`, source `../common/env.sh`, write `$RESULTS/<name>.jsonl`.
3. Relative paths are resolved from `$ROOT`, never from the caller's cwd.
4. Record the engine version, quant, and the parameters that change the number in `label`.
5. Reject the run (valid: false) on swap-outs, thermal throttling, or a model load inside the
   measured window. Keep the row.
