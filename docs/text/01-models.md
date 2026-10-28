# Models

**Verdict (32 GB / M5, 153.6 GB/s):**

1. **Qwen3.8-27B, `UD-Q4_K_M` (16.46 GB) + `mmproj-F16` (0.93 GB)** — `unsloth/Qwen3.8-27B-GGUF`. ~19.4 GiB total at 32k ctx. Decode ceiling 9.3 tok/s. Nothing at this size class comes close (AA II 33.7 vs 18.2 for the next open small model). **This is the pick.**
2. **Qwen3.8-27B `MLX-4bit` (16.08 GB)** — `lmstudio-community/Qwen3.8-27B-MLX-4bit`. Same weights, 0.4 GB smaller, MLX-native KV quant; use if MLX-LM wins the engine bake-off.
3. **Qwen3.6-35B-A3B `UD-Q3_K_XL` (16.85 GB)** — the *speed* pick: 3B active params means ~1.8 GB read/token instead of 16.5 GB, so ~35-50 tok/s vs ~8. Quality drops hard (AA II 18.2 vs 33.7) — take this only for agent loops and draft generation.

**Do not buy:** Qwen3.8-27B Q6_K (21.98 GB → 23.5 GiB at 16k, over the Metal ceiling with vision) and Q8_0 (29.05 GB → does not fit at any ctx).

---

## Memory math

### Budget arithmetic

| Line item | Value | Source |
|---|---|---|
| Installed unified memory | 34,359,738,368 B = **32 GiB** | `sysctl hw.memsize` on the target box |
| macOS baseline RSS (idle desktop, chat-only) | **5–7 GiB** | community norm; not measured idle here |
| macOS baseline with a dev session running | **12.2 GiB** measured (wired 5.37 + active 6.80 GiB) | `vm_stat` at time of research, page size 16384 |
| Metal `recommendedMaxWorkingSetSize` | **~75% of RAM ≈ 24.6 GiB** | [Metal wired-limit behaviour](https://llmconfigurator.com/en/guides/troubleshooting/increase-metal-vram-limit-apple-silicon) |
| `iogpu.wired_limit_mb` | `0` (unset → default 75%) | `sysctl iogpu.wired_limit_mb` |
| **Budget for weights + KV + mmproj** | **≈ 22.5 GiB comfortable, 24.6 GiB hard** | derived |

> The Metal cap is a *wired* ceiling, not an availability ceiling. llama.cpp uses `mmap`, so a 22 GB model will "load" on a 16 GB machine — by paging. Wired-limit overrun = swap = 1-2 tok/s. Treat 24.6 GiB as the real limit.

### KV cache formula

```
KV_bytes_per_token = 2 (K and V) × n_full_attn_layers × n_kv_heads × head_dim × bytes_per_elem
```

GQA divides `n_kv_heads` by the group ratio. **Hybrid/linear-attention layers are the far bigger lever**: Qwen3.8-27B has only 16 full-attention layers out of 64 (`full_attention_interval: 4`), so it pays 1/4 the KV of a classic dense 27B. Sliding-window layers contribute a *fixed* buffer instead of a growing one.

`bytes_per_elem`: fp16 = 2, q8_0 ≈ 1.0625 (34 values + 1 fp16 scale per 32), q4_0 ≈ 0.5625 (32 values + 1 fp16 scale per 32).

### KV cost per candidate

| Model | Full-attn layers | KV heads × dim | KV/token (fp16) | MiB / 1k tok fp16 | q8_0 | q4_0 | Fixed SWA buffer |
|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B** | 16 / 64 | 4 × 256 | 64.0 KiB | **62.5** | 66.4 | 35.2 | — |
| Qwen3.6-27B | 16 / 64 | 4 × 256 | 64.0 KiB | 62.5 | 66.4 | 35.2 | — |
| Qwen3.5-27B | 16 / 64 | 4 × 256 | 64.0 KiB | 62.5 | 66.4 | 35.2 | — |
| Qwen3.5-9B | 8 / 32 | 4 × 256 | 32.0 KiB | 31.2 | 33.2 | 17.6 | — |
| Qwen3.5/3.6-35B-A3B | 10 / 40 | 2 × 256 | 20.0 KiB | 19.5 | 20.8 | 11.0 | — |
| gpt-oss-20b | 24 / 24 (SWA-128 on half) | 8 × 64 | 48.0 KiB | 46.9 | 49.8 | 26.4 | ~3 MiB |
| Gemma-4-31B | 10 / 60 | 16 × 256 | 160.0 KiB | 156.2 | 166.0 | 87.9 | **800 MiB** |
| Gemma-4-26B-A4B | 5 / 30 | 8 × 256 | 40.0 KiB | 39.1 | 41.5 | 22.0 | 200 MiB |
| Muse-Glimmer-30B | 13 / 52 | 2 × 128 | 13.0 KiB | **12.7** | 13.5 | 7.1 | 78 MiB |
| GLM-4.7-Flash (MLA) | 47 / 47 | 576 latent | 52.9 KiB | 51.6 | 54.9 | 29.0 | — |
| Nemotron-3.5-L-30B-A3B | 52 / 52 | 2 × 128 | 52.0 KiB | 50.8 | 54.0 | 28.6 | — |
| Devstral-S2-24B | 40 / 40 | 8 × 128 | 160.0 KiB | 156.2 | 166.0 | 87.9 | — |
| Kimi-Linear-48B-A3B | 0 (KDA) | — | ~0 | ~0 | ~0 | ~0 | constant state |

Per-1k-token KV is computed from each published `config.json`. Qwen3.8-27B's real value is 64 KiB/token, **not** the ~256 KiB/token a naive 64-layer/8-head assumption gives — the hybrid layout is a 4× KV discount.

Linearity check for the baseline: 32k ctx → 2.0 GiB; 64k → 4.0 GiB; 262k (native max) → 16 GiB, i.e. the headline 262k context is **not reachable at fp16 on this box**; q8_0 + 64k (2.1 GiB) is the practical ceiling.

### Weight sizes — measured, not estimated

`Qwen/Qwen3.8-27B` is 27,781,427,952 BF16 params (`safetensors` API). Actual file sizes from the HF tree API:

| Quant | Repo | File size | effective bpw | decode ceiling @153.6 GB/s |
|---|---|---|---|---|
| UD-IQ3_XXS | unsloth | 10.93 GB | 3.15 | 14.1 tok/s |
| UD-Q3_K_XL | unsloth | **13.15 GB** | 3.79 | 11.7 tok/s |
| UD-IQ4_XS | unsloth | 14.25 GB | 4.10 | 10.8 tok/s |
| **UD-Q4_K_M** | unsloth | **16.46 GB** | 4.74 | **9.3 tok/s** |
| Q4_K_M | lmstudio | 16.81 GB | 4.84 | 9.1 tok/s |
| MLX-4bit | lmstudio | **16.08 GB** | 4.63 | **9.6 tok/s** |
| UD-Q4_K_XL | unsloth | 17.56 GB | 5.06 | 8.7 tok/s |
| MLX-5bit | lmstudio | 19.44 GB | 5.60 | 7.9 tok/s |
| UD-Q5_K_M | unsloth | **19.77 GB** | 5.69 | 7.8 tok/s |
| UD-Q6_K | unsloth | 21.98 GB | 6.33 | 7.0 tok/s |
| MLX-6bit | lmstudio | 22.80 GB | 6.57 | 6.7 tok/s |
| Q8_0 | unsloth | 29.05 GB | 8.37 | 5.3 tok/s |
| mmproj-F16 (vision) | unsloth | **0.93 GB** | — | required for image input |

Why Q4_K_M is 4.74 bpw and not 4.5: the 248,320-token vocab produces **two** untied 1.271B matrices (`tie_word_embeddings: false`), i.e. 2.54B params = 9.1% of the model, and GGUFs keep embeddings at Q6/Q8. Body-only effective rate is ≈4.36 bpw.

**Rule of thumb that holds on this data:** Q4_K_M ≈ 4.7-4.9 bpw on large-vocab models, Q5_K_M ≈ 5.6-5.8, Q6_K ≈ 6.3, Q8_0 ≈ 8.4, MLX-4bit ≈ 4.6, MLX-6bit ≈ 6.6. Budget `GB ≈ params_B × bpw / 8`.

### Fit verdicts (weights + mmproj + 1.2 GiB overhead + fp16 KV, vs 24.6 GiB)

| Config | 8k | 16k | 32k | 64k | Verdict |
|---|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M | 17.9 | 18.4 | 19.4 | 21.4 | **fits comfortably** (to 64k) |
| Qwen3.8-27B MLX-4bit | 16.7 | 17.2 | 18.2 | 20.2 | **fits comfortably** |
| Qwen3.8-27B Q4_K_XL | 18.9 | 19.4 | 20.4 | 22.4 | fits comfortably |
| Qwen3.8-27B Q5_K_M | 21.0 | 21.5 | 22.5 | 24.5 | fits tight at 64k; OK to 32k |
| Qwen3.8-27B Q6_K | 23.0 | 23.5 | 24.5 | 26.5 | **fits tight at 8k only** — no context |
| Qwen3.8-27B Q8_0 | 29.6 | 30.1 | 31.1 | 33.1 | **does not fit** |
| Qwen3.5-35B-A3B Q4_K_M | 22.7 | 22.9 | 23.2 | 23.8 | fits tight at 16k |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 17.9 | 18.1 | 18.4 | 19.0 | **fits comfortably** |
| Qwen3.5-9B Q8_0 | 11.2 | 11.4 | 11.9 | 12.9 | fits comfortably |
| gpt-oss-20b MXFP4 | 12.4 | 12.8 | 13.5 | 15.0 | fits comfortably |
| Gemma-4-31B Q4_K_M | 19.5 | 20.8 | 23.3 | 28.3 | fits tight at 16k; **fails at 64k** |
| Gemma-4-26B-A4B QAT-4bit | 16.1 | 16.4 | 17.0 | 18.3 | fits comfortably |
| Muse-Glimmer-30B Q4_K_M | 16.9 | 17.0 | 17.2 | 17.6 | **fits comfortably** (cheap KV) |
| GLM-4.7-Flash Q4_K_M | 18.7 | 19.1 | 19.9 | 21.6 | fits comfortably |
| Nemotron-3.5-L-30B Q4 | 18.4 | 18.8 | 19.6 | 21.2 | fits comfortably |
| Devstral-S2-24B Q4_K_M | 15.8 | 17.0 | 19.5 | 24.5 | fits tight at 64k |

---

## Candidate table

Sizes are **actual bytes** from the HF tree API; "—" in the fit column means no published 4-bit GGUF/MLX found under a canonical repo id.

| Model | Arch | Active | Ctx | Quant | Size GB | License | MM | HF repo id (GGUF / MLX) | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B** | dense hybrid (48 DeltaNet + 16 attn) | 27.8B | 262k | Q4_K_M | 16.46 | Apache-2.0 | **y** | `unsloth/Qwen3.8-27B-GGUF` / `lmstudio-community/Qwen3.8-27B-MLX-4bit` | **comfortable** |
| Qwen3.8-27B | " | " | " | Q5_K_M | 19.77 | " | y | same | tight at 64k |
| Qwen3.8-27B | " | " | " | Q6_K | 21.98 | " | y | same | tight, ~8k only |
| Qwen3.6-27B | identical shape to 3.8 | 27.8B | 262k | Q4_K_M | 16.82 | Apache-2.0 | y | `unsloth/Qwen3.6-27B-GGUF` | comfortable |
| Qwen3.5-27B | identical shape | 27.8B | 262k | Q4_K_M | 16.74 | Apache-2.0 | y | `unsloth/Qwen3.5-27B-GGUF` / `mlx-community/Qwen3.5-27B-4bit` (16.08) | comfortable |
| Qwen3.6-35B-A3B | MoE 256 experts, 8 active | **~3B** | 262k | UD-Q3_K_XL | 16.85 | Apache-2.0 | y | `unsloth/Qwen3.6-35B-A3B-GGUF` | comfortable |
| Qwen3.6-35B-A3B | " | " | " | UD-Q4_K_M | 22.13 | " | y | same | tight at 16k |
| Qwen3.5-35B-A3B | MoE 256/8 | ~3B | 262k | Q4_K_M | 22.02 | Apache-2.0 | y | `unsloth/Qwen3.5-35B-A3B-GGUF` / `mlx-community/Qwen3.5-35B-A3B-4bit` (20.42) | tight at 16k |
| Qwen3.5-9B | dense hybrid | 9.65B | 262k | Q4_K_M | 5.68 | Apache-2.0 | y | `unsloth/Qwen3.5-9B-GGUF` / `mlx-community/Qwen3.5-9B-4bit` (5.98) | comfortable |
| Qwen3.5-9B | " | " | " | Q8_0 | 9.53 | " | y | same | comfortable |
| gpt-oss-20b | MoE 32 experts, 4 active | **3.6B** | 131k | MXFP4 (native) | 11.62 | Apache-2.0 | n (text) | `unsloth/gpt-oss-20b-GGUF` / `mlx-community/gpt-oss-20b-MXFP4-Q8` (12.10) | comfortable |
| Gemma-4-31B-it | dense, 10/60 global attn | 31.27B | 262k | Q4_K_M | 18.32 | Apache-2.0 | y | `unsloth/gemma-4-31B-it-GGUF` / `lmstudio-community/gemma-4-31B-it-MLX-4bit` (18.44) | tight at 16k |
| Gemma-4-26B-A4B-it | MoE 128 experts | **~4B** | 262k | QAT-4bit | 15.64 | Apache-2.0 | y | `unsloth/gemma-4-26B-A4B-it-GGUF` (UD-Q4_K_M 16.95) / `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` | comfortable |
| Muse-Glimmer-30B | dense, 13/52 global, SWA-2048 | 29.6B | 131k | Q4_K_M | 16.76 | Apache-2.0 | y (+1.8B ViT) | `meta-models/Muse-Glimmer-30B-GGUF` / `mlx-community/Muse-Glimmer-30B-4bit` (19.44) | **comfortable** |
| GLM-4.7-Flash | MoE 64+1 shared, 4 active, MLA | ~3B | 203k | Q4_K_M | 18.31 | **MIT** | n | `unsloth/GLM-4.7-Flash-GGUF` / `mlx-community/GLM-4.7-Flash-4bit` (16.87) | comfortable |
| Nemotron-3.5-Lightning-30B-A3B | MoE 128+1, 6 active | ~3B | 262k | Q4_K_M | ~18.0 `[unverified]` | NVIDIA Open (`other`) | n | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` | comfortable |
| Devstral-Small-2-24B-2512 | dense | 24.01B | 393k | Q4_K_M | 14.33 | Apache-2.0 | n | `unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF` / `mlx-community/...-4bit` (15.14) | comfortable |
| Kimi-Linear-48B-A3B | MoE 256, KDA linear attn | ~3B | 1M | Q4_K_M | ~28 `[unverified]` | MIT | n | `bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF` | borderline / fails |
| Llama-4-Scout-17B-16E | MoE 16 experts | 17B | 10M | Q4_K_M | ~60 `[unverified]` | Llama 4 Community | y | — | **does not fit** |
| MiniMax-M2.5 | MoE | — | — | — | 228.7B total | `other` | n | — | **does not fit** |
| GLM-5.3-Flash | MoE | — | — | — | 321.3B total | MIT | n | — | **does not fit** |
| Qwen3.7-Plus | — | — | — | — | API-only, no weights on HF | — | y | — | not downloadable |

Notes:
- **Qwen3.7-Plus has no open weights.** The `Qwen3.8-27B` card lists it as a comparison column only; HF search returns no `Qwen/Qwen3.7-*` repo. Qwen3.7 was API-only.
- **Qwen3.5-27B, Qwen3.6-27B and Qwen3.8-27B are the same 27.78B shape** (`qwen3_5`, hidden 5120, 64 layers, `full_attention_interval: 4`). Quant sizes transfer across all three within ±0.4 GB — so a 3.8 upgrade only costs a re-download, not a re-plan.
- **gpt-oss is stale.** `openai/gpt-oss-20b` was last modified 2025-08-26; the only newer OpenAI releases are `gpt-oss-safeguard-20b/120b` (safety-tuned, same base). Still relevant only as an 11.6 GB "fits anywhere" option.
- **GGUF/MLX quants cannot improve on MXFP4 for gpt-oss** — the weights are natively MXFP4, so Q4_K_M (11.62) ≈ Q8_0 (12.11). Don't bother re-quantizing.
- Qwen3.8-27B also ships MTP and DFlash draft heads (`ggml-org/Qwen3.8-27B-GGUF`: `mtp-Q4_0` 1.68 GB, `dflash-Q4_0` 1.09 GB) — speculative decoding material, relevant to `03`/`04`.

---

## Capability vs Qwen3.8-27B

Cells marked `—` were **not reported**. The harness column is the crux: Qwen's card and Google's card are different harnesses, and Qwen3.8's card **dropped MMLU-Pro/MMMU entirely**, so the flagship comparison benchmark is missing for the baseline.

| Benchmark | Qwen3.8-27B | Qwen3.6-27B | Qwen3.6-35B-A3B | Qwen3.5-27B | Gemma-4-31B | Gemma-4-26B-A4B | Muse Glimmer-30B | gpt-oss-20b | Qwen3.5-9B | same harness? |
|---|---|---|---|---|---|---|---|---|---|---|
| MMLU-Pro | **—** | 86.2 | 85.2 | 86.1 | 85.2 | 82.6 | — | 74.8 | 82.5 | **n** — 3.6 card + Google card + 9B card; 3.8 absent |
| GPQA Diamond | **89.2** | 87.8 | 86.0 | 85.5 | 84.3 | 82.3 | 83.5 (AA) | 71.5 | 81.7 | n — 3 cells share Qwen harness, Gemma/Muse/gpt-oss don't |
| HLE | **30.8** | 24.0 | 21.4 | 24.3 | 19.5 | 8.7 | — | — | — | y (Qwen 3.6 card, 3.8 card) |
| LiveCodeBench v6 | **90.3** | 83.9 | 80.4 | 80.7 | 80.0 | 77.1 | — | 74.6 | 65.6 | y for Qwen cols; Gemma self-reported |
| SWE-bench Pro | **61.7** | 53.5 | 49.5 | 51.2 | 35.7 | — | 51.2 | — | — | y (Qwen 3.6 + 3.8 cards); Muse own |
| SWE-bench Verified | — | 77.2 | 73.4 | 75.0 | 52.0 | — | **76.0** | — | — | y |
| Terminal-Bench 2.x | **73.0** (2.1) | 59.3 (2.0) | 51.5 (2.0) | 41.6 (2.0) | 42.9 (2.0) | — | 51.7 (2.1) | — | — | **n** — 3.8/Muse use TB-2.1, rest TB-2.0 |
| IFBench | **79.5** | — | — | — | — | — | 77.0 | 65.1 | 64.5 | n |
| τ²-bench (avg 3) | — | — | — | 57.4* | 76.9 | 68.2 | — | — | 79.1* | **n** — *9B card figure is τ², Google's is "average over 3" |
| MMMU | — | 82.9 | 81.7 | 82.3 | 80.4 | — | — | — | — | y (Qwen 3.6 card) |
| MMMU-Pro | — | 75.8 | 75.3 | 75.0 | 76.9 | 73.8 | 74 | 59.7 | — | n |
| AIME26 | — | 94.1 | 92.7 | 92.6 | 89.2 | 88.3 | 94.7 | — | — | n |
| **AA Intelligence Index** | **33.7** (xhigh) | **not listed** | 18.2 | **not listed** | 14.7 | 16.7 | 17.5 | 9.0 | 13.3 | **y — independent harness** |
| **Arena Elo (overall)** | **1440.5** (#75) | **not listed** | **not listed** | 1408.9 (#142) | **1442.9** (#67) | 1434.1 (#90) | 1388.9 (#167) | 1287.4 (#260) | not listed | **y — human preference** |

Harness caveats, explicitly:
- **Qwen3.8-27B's own card changed benchmark sets.** It reports Terminal Bench 2.1, SWE-bench Pro, NL2Repo, DeepSWE 1.1, QwenSWEBench, CoWorkBench, JobBench, IFBench, GPQA, HLE, LiveCodeBench v6, OSWorld, WebArena, AndroidWorld, SWE-MM, Vision2Web, CharXiv, OmniDocBench, RealWorldQA, ERQA. **MMLU-Pro, MMMU, MMMU-Pro, AIME, SWE-bench Verified are all absent.** Any MMLU-Pro or SWE-bench-Verified number quoted for Qwen3.8-27B is third-party or inferred — mark it `[unverified]`.
- The **Gemma 4 columns in the Qwen3.6-27B card are byte-identical to Google's own card** (MMLU-Pro 85.2, GPQA 84.3, LCB 80.0). Qwen reused Google's numbers rather than re-running, so those cells are *vendor-vs-vendor*, not a controlled comparison. Still "same harness y" only in the sense that both sides used their own preferred config.
- **AA Intelligence Index gaps:** Qwen3.6-27B and Qwen3.5-27B are not on artificialanalysis.ai at all (only Qwen3.8-27B is, plus the A3B MoE). So the single most important comparison — 3.8-27B vs 3.6-27B on a neutral harness — **does not exist**.
- **Arena gaps:** `qwen3.6-27b` and `qwen3.6-35b-a3b` are absent from the text leaderboard (only `qwen3.6-max/plus` appear). `qwen3.8-27b` **is** there at 1440.5.
- **Arena anomaly worth flagging:** Gemma-4-31B (1442.9, #67) edges out Qwen3.8-27B (1440.5, #75) — inside overlapping CIs, and Arena rewards chat style. On AA's index Qwen3.8-27B more than doubles Gemma-4-31B (33.7 vs 14.7). Trust AA for capability; treat Arena as a style prior.
- **Reasoning tier matters more than the model.** Qwen3.8-27B scores 33.7 / 27.6 / 26.2 / 20.2 at xhigh / medium / low / non-reasoning. A quantized Q4 model run at `reasoning_effort=low` is *worse than a 9B at full effort*. Pin effort in the config.

---

## Per-gigabyte efficiency

AA Intelligence Index ÷ weight GB, using each model's best reasoning tier:

| Model | Quant | GB | AA II | **II per GB** | decode ceiling |
|---|---|---|---|---|---|
| **Qwen3.8-27B** | Q4_K_M | 16.46 | 33.7 | **2.05** | 9.3 tok/s |
| **Qwen3.8-27B** | Q5_K_M | 19.77 | 33.7 | 1.70 | 7.8 tok/s |
| Qwen3.8-27B | MLX-4bit | 16.08 | 33.7 | 2.10 | 9.6 tok/s |
| Qwen3.5-9B | Q4_K_M | 5.68 | 13.3 | **2.34** | 27 tok/s |
| Qwen3.5-9B | Q8_0 | 9.53 | 13.3 | 1.40 | 16 tok/s |
| Muse Glimmer-30B | Q4_K_M | 16.76 | 17.5 | 1.04 | 9.2 tok/s |
| Gemma-4-26B-A4B | QAT-4bit | 15.64 | 16.7 | 1.07 | ~40 tok/s (4B active) |
| Qwen3.6-35B-A3B | UD-Q3_K_XL | 16.85 | 18.2 | 1.08 | ~45 tok/s |
| Qwen3.6-35B-A3B | Q4_K_M | 22.13 | 18.2 | 0.82 | ~40 tok/s |
| Gemma-4-31B | Q4_K_M | 18.32 | 14.7 | 0.80 | 8.4 tok/s |
| gpt-oss-20b | MXFP4 | 11.62 | 9.0 | 0.77 | ~13 tok/s (3.6B active) |
| GLM-4.7-Flash | Q4_K_M | 18.31 | — | — | ~45 tok/s |
| Nemotron-3.5-L-30B | Q4_K_M | ~18.0 | 12.9 | 0.72 | ~40 tok/s |
| Kimi-Linear-48B-A3B | Q4_K_M | ~28 | 7.3 | 0.26 | high (KDA) |

**Does Q4_K_M of a 27B dense beat Q6/Q8 of a 9B? Yes, and it is not close.**
- Qwen3.8-27B Q4_K_M: 16.46 GB, II **33.7**
- Qwen3.5-9B Q8_0: 9.53 GB, II **13.3** (non-reasoning; 11.2 reasoning — the 9B's best tier is *lower* than its non-reasoning tier, an oddity of AA's sampling)
- Gap: **2.5×**, for 1.7× the bytes. For 7 GB saved you buy a model that is worse at everything except decode speed and KV footprint.

The reverse also holds and is worth stating plainly: **Qwen3.5-9B Q4_K_M has the best quality-per-byte in the field (2.34)**. Per-GB efficiency is a small-model metric by construction — it rewards being small, not being good, because KV cache and fixed costs are excluded. Do not optimize the per-GB column; optimize absolute quality subject to the fit constraint. That constraint binds at ~20 GB weights, and at 20 GB the answer is unambiguous.

**MoE at Q4: costs more RAM, buys a lot of speed.**
- `Qwen3.6-35B-A3B UD-Q4_K_M` = 22.13 GB → 23.1 GiB at 16k ctx. It barely fits and leaves no room for a second context slot. Reads ~1.8 GB/token (3B active × 4.9 bpw) vs ~16.5 GB/token for the dense 27B — a **9× reduction in bytes per token**.
- Decode ceiling: **83 tok/s** (22 GB file, but only active params are read) vs **9.3 tok/s** for dense 27B Q4_K_M. Real M5 numbers will land near 30-50 and 7-8 respectively.
- The honest trade: **II 18.2 at 45 tok/s vs II 33.7 at 8 tok/s.** Roughly 2× the quality for 1/5 the speed. If the workload is batch/eval/one-shot, dense wins outright. If it is an interactive agent loop with retries, the MoE's speed may win on wall-clock-per-solved-task.
- **The underrated middle option:** `Qwen3.6-35B-A3B UD-Q3_K_XL` at 16.85 GB / 18.4 GiB @32k. Same quality as its Q4, 5 GB smaller, more-than-competitive speed. 3-bit is unusually safe on Qwen's UD recipes because embeddings/lm_head stay at high precision.

**Where the 27B is actually waste:** weights are read once per token regardless of context. A 16.5 GB model on a 153.6 GB/s bus sets a hard 9.3 tok/s ceiling that no engine, no `--n-cpu-moe` trick and no flash-attention kernel can lift. That is the physical fact this whole exercise is bounded by.

---

## Sources

**Target hardware / memory (measured locally on the box)**
- `sysctl hw.memsize` → 34,359,738,368 B; `hw.model` Mac17,3; macOS 27.0.1 (26A434); `iogpu.wired_limit_mb` = 0
- `vm_stat` (page 16384 B): wired 5.37 GiB, active 6.80 GiB at research time
- https://en.wikipedia.org/wiki/Apple_M5 — base M5: up to 32 GB, **153.6 GB/s** (LPDDR5X-9600), 10-core CPU / 10-core GPU
- https://llmconfigurator.com/en/guides/troubleshooting/increase-metal-vram-limit-apple-silicon — `recommendedMaxWorkingSetSize` ≈ 75% of unified RAM
- https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap — same 75% behaviour, `iogpu.wired_limit_mb` override

**Baseline model**
- https://huggingface.co/Qwen/Qwen3.8-27B — card: 27B, 5120 hidden, 64 layers, 16×(3×Gated DeltaNet + 1×Gated Attention), 24 Q / 4 KV heads, head_dim 256, 248,320 vocab, MTP, 262,144 native ctx (extensible to 1M), Apache-2.0
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json — `full_attention_interval: 4`, `tie_word_embeddings: false`, `linear_num_value_heads: 48`, `linear_value_head_dim: 128`, vision `depth: 27, hidden 1152`
- HF API `safetensors.total` = 27,781,427,952 (BF16)
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md — all Qwen3.8-27B scores in the capability table

**Quant file sizes (HF tree API, actual bytes)**
- `unsloth/Qwen3.8-27B-GGUF`, `ggml-org/Qwen3.8-27B-GGUF`, `lmstudio-community/Qwen3.8-27B-GGUF`
- `lmstudio-community/Qwen3.8-27B-MLX-4bit` (16.08 GB), `-MLX-5bit` (19.44), `-MLX-6bit` (22.80), `mlx-community/Qwen3.8-27B-4bit` (16.08)
- `unsloth/Qwen3.5-27B-GGUF`, `unsloth/Qwen3.5-35B-A3B-GGUF`, `unsloth/Qwen3.5-9B-GGUF`, `unsloth/Qwen3.6-27B-GGUF`, `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gpt-oss-20b-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`, `unsloth/GLM-4.7-Flash-GGUF`, `unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF`, `meta-models/Muse-Glimmer-30B-GGUF`, `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
- `mlx-community/Qwen3.5-27B-4bit`, `-Qwen3.5-35B-A3B-4bit`, `-Qwen3.5-9B-4bit`, `-Muse-Glimmer-30B-4bit`, `-GLM-4.7-Flash-4bit`, `-gpt-oss-20b-MXFP4-Q8`, `-Devstral-Small-2-24B-Instruct-2512-4bit`, `lmstudio-community/gemma-4-31B-it-MLX-4bit`, `-gemma-4-26B-A4B-it-QAT-MLX-4bit`

**Model cards / configs used for arch and benchmarks**
- https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md — the same-harness table (Qwen3.5-27B, Qwen3.5-397B-A17B, Gemma4-31B, Claude 4.5 Opus, Qwen3.6-35B-A3B, Qwen3.6-27B)
- https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md — 6-column table incl. **gpt-oss-20b** and gpt-oss-120b on Qwen's harness
- https://huggingface.co/google/gemma-4-31B-it/raw/main/README.md — Gemma 4 family table (31B / 26B-A4B / 12B / E4B / E2B / Gemma 3 27B)
- https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/README.md — 29.6B dense, 52 layers, 32/2 GQA, SWA 2048, 13 global layers, 131k ctx, Apache-2.0, 1.8B ViT, "Apple M4 Max 23.7 tok/s" reference data point
- https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md — 21B total / 3.6B active, MXFP4, "within 16GB"
- configs: `openai/gpt-oss-20b` (24 layers, 8 KV × 64, SWA 128), `zai-org/GLM-4.7-Flash` (MLA, kv_lora 512, 64+1 experts, MIT), `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (128+1 experts, 6 active), `mistralai/Devstral-Small-2-24B-Instruct-2512`, `moonshotai/Kimi-Linear-48B-A3B-Instruct`, `google/gemma-4-26B-A4B-it`

**Harness-independent leaderboards**
- https://artificialanalysis.ai/leaderboards/models — Intelligence Index (embedded JSON), retrieved 2026-10-20
- https://datasets-server.huggingface.co/rows?dataset=lmarena-ai%2Fleaderboard-dataset&config=text&split=latest — Arena Elo, retrieved 2026-10-20 (`lmarena.ai` itself is Cloudflare-gated)

**News corroboration**
- https://www.alibabacloud.com/blog/alibaba-unveils-qwen3-8-27b_603463
- https://datanorth.ai/news/alibaba-releases-qwen3-8-27b — confirms 14 Aug 2026, Apache-2.0, 262,144 ctx
- https://benchlm.ai/models/qwen3-8-27b — third-party aggregate (48 benchmark rows) `[unverified]`

**Marked `[unverified]` in this document**
- Nemotron-3.5-Lightning Q4_K_M file size (~18.0 GB) — extrapolated from Q8_0 33.59 GB; no canonical Q4 in the tree I read
- Kimi-Linear-48B-A3B Q4_K_M (~28 GB) — extrapolated from 49.12B params at 4.6 bpw
- Llama-4-Scout Q4_K_M (~60 GB) — extrapolated from 108.6B params; bartowski repo tree not readable
- Any MMLU-Pro / SWE-bench-Verified figure attributed to Qwen3.8-27B — not on its card
- macOS 5-7 GiB idle baseline — community norm, not measured on this box in an idle state
- All tok/s figures in this document are **bandwidth-derived ceilings** (`153.6 GB/s ÷ bytes-read-per-token`), not measurements. [`03-results-speed.md`](03-results-speed.md) owns the real numbers.
