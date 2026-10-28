# Research notes

Evidence for what runs on this box — an M5 MacBook Air with 32 GB of unified memory — organised
by **modality**. Each folder owns one track: its question, its engines, its model survey and its
measured results. Cross-cutting facts (hardware ceilings, engine behaviour) live in
[`shared/`](shared/README.md) and are referenced, not repeated.

| Modality | Question | Status | Write-ups | Harness |
| --- | --- | --- | --- | --- |
| [Text](text/README.md) | tokens/s, TTFT, PPL, task accuracy | **measured** | [`text/`](text/README.md) | [`benchmarks/text`](../benchmarks/text/README.md) |
| [Decision](decision/README.md) | typed `choice`/`noul`/`score`, accuracy + calibration | **measured** | [`decision/`](decision/README.md) | [`benchmarks/decision`](../benchmarks/decision/README.md) |
| [Image](image/README.md) | image→text accuracy; s/image, steps, peak memory | scoped | [`image/`](image/README.md) | [`benchmarks/image`](../benchmarks/image/README.md) |
| [Video](video/README.md) | frames/s, video QA; s/clip | scoped | [`video/`](video/README.md) | [`benchmarks/video`](../benchmarks/video/README.md) |
| [Audio](audio/README.md) | music/sfx RTF, CLAP/FAD, tagging | scoped | [`audio/`](audio/README.md) | [`benchmarks/audio`](../benchmarks/audio/README.md) |
| [Speech](speech/README.md) | WER/CER, RTF, first-token latency, TTS MOS | scoped | [`speech/`](speech/README.md) | [`benchmarks/speech`](../benchmarks/speech/README.md) |

`measured` means there are rows in [`results/<modality>/`](../results/README.md) and numbers below are
measurements. `scoped` means the question, metrics and validity gates are fixed, but no weights or
data are resident and no row exists yet.

## Cross-cutting

| File | What it is |
| --- | --- |
| [`shared/01-hardware.md`](shared/01-hardware.md) | Machine facts, the capability bar, and the three physics ceilings |
| [`shared/02-engines.md`](shared/02-engines.md) | Engine survey for Apple M5, with M5-specific version traps |

Read `shared/` first if you are new: decode is bandwidth-bound, 32 GB is a shared pool, and M5 needs a
recent llama.cpp. Those three facts bound every track.

## The contract every track shares

- One row = one measurement, one process, one model load. No averaging inside a harness.
- Record thermal/load/swap state. Swap-outs invalidate a row: at 32 GB, paging is the real ceiling.
- Fix the prompt, seed, steps, resolution/clip length, and report them in `label`.
- Report what failed (OOM, timeout, refusal) as a row, not as a missing row.
- A number without its engine version, quant and context length is a rumour.
- Weights are named by Hub repo/alias, never by a path that can drift
  ([`models/README.md`](../models/README.md)).

The row schema and the mechanics live in [`benchmarks/common/`](../benchmarks/common/README.md).
The long-form methodology is [`text/02-benchmark-plan.md`](text/02-benchmark-plan.md); the other
tracks reuse its validity gates and replace its datasets, metrics and thresholds.

## Current resident set

| Alias | Checkpoint | Serves |
| --- | --- | --- |
| `qwen38-27b-oq4e-mtp` | `Jundot/Qwen3.8-27B-oQ4e-mtp` | text, image, video, audio input; decision baselines |
| `clef-flash-4bit`, `clef-flash-8bit` | `mlx-community/clef-flash-*` | decision |
| `clef-flash-q4km`, `kev-9b-q4km`, `d3-flash-q4km` | `ggml-org/*-GGUF` | decision |

Everything lives in the Hub cache (`~/.cache/hf/hub`); see [`models/README.md`](../models/README.md).
