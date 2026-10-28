# Hardware and the physics ceiling

Everything measured in this repo runs on one machine, and the machine — not the engine — sets the
hard limits. This file is the shared factual base for every modality track.

## Target hardware (measured, not assumed)
| Item | Value |
|---|---|
| Machine | MacBook Air Mac17,3 (Apple M5) |
| RAM | 32 GB unified (34,359,738,368 B) |
| CPU | 10 cores (4P "Super" + 6E), **no** SMT |
| GPU | 10 cores, Metal 4, built-in |
| OS | macOS 27.0.1 (build 26A434) |
| Disk free | ~88 GB at planning time |
| Toolchain present | nix, uv, git, node, xcodebuild. **No** brew/ollama/llama.cpp |

## The capability bar
`Qwen/Qwen3.8-27B` — 27B dense, native multimodal, Apache-2.0, released 2026-08-14.
The question every track answers: **what combination of engine + quant + model gets closest to that
capability inside a 32 GB envelope (weights + KV cache + OS), and how fast is it?**

## How the evidence is organised

One folder per modality — [`../README.md`](../README.md) is the index. Cross-modality facts live in
this folder; per-track datasets, metrics and recommendations live with the track:

| Folder | Owns |
| --- | --- |
| [`../text/`](../text/README.md) | Model survey, methodology, speed/quality results, text recommendation |
| [`../decision/`](../decision/README.md) | Decision-head engines, models, evaluation, results, recommendation |
| [`../image/`](../image/README.md), [`../video/`](../video/README.md), [`../audio/`](../audio/README.md), [`../speech/`](../speech/README.md) | Scope, metrics and validity gates; no measurements yet |
| [`../shared/`](README.md) | This file and the engine survey |

Claims are labelled **measured** (a row exists), **derived** (arithmetic on measured values) or
**ceiling** (physics). A number without its engine version, quant and context length is a rumour.

## Reality checks baked into the plan
1. **Decode is bandwidth-bound.** `tok/s ≈ bandwidth / bytes_read_per_token`. On M5
   (~153 GB/s) a 16 GB Q4 27B model caps out near 9-10 tok/s no matter the engine.
   Engines differ far more on prefill/TTFT and memory behaviour than on decode.
2. **32 GB is a shared pool.** Metal's default wired limit leaves little headroom;
   KV cache at 32k context on a 64-layer model is ~1-2 GB fp16 and grows per-slot.
   Every config must be measured at realistic context, not at ctx=512.
3. **MoE vs dense matters more than parameter count.** A3B-class MoE reads ~3B
   params/token → several times the decode speed of a dense 27B.
4. Benchmark only what fits: no point ranking 60 tok/s configs that OOM at 16k ctx.
