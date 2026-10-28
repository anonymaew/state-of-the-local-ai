# Engines

**Verdict up front.** On this box — M5 base, 10-core GPU, **153 GB/s**, 32 GB, no brew, no ollama — the shortlist is **MLX (mlx-vlm for Qwen3.8-27B, mlx-lm for the MoE) as the primary runtime** and **llama.cpp built from a `b11xxx`-or-newer nightly as the co-primary** for GGUF portability, `llama-server`, KV-quantized long context, MTP/draft speculative decoding and one-command `-hf` model pulls. This is not a coin flip: on **this exact SoC (M5 10-CPU/10-GPU/32GB)** the community `mac-llm-bench` run measures MLX at **1.8-1.9× llama.cpp GGUF** for the same models — Qwen3.5-35B-A3B-4bit **58.8 tok/s (MLX) vs 31.3 (GGUF Q4_K_M)**, Gemma-3-27B **5.5 vs 3.0** ([results/m5/base](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/README.md)). The **runner-up is Ollama 0.40.6**, which now ships *both* engines under one API ([v0.30.0 notes](https://github.com/ollama/ollama/releases/tag/v0.30.0)) and is the lowest-effort way to get either — but you lose KV-quant/spec-decode knobs and its MLX engine lags. **Do not plan around vLLM Metal, Splash, oMLX, Rapid-MLX or Kapitsa as your daily driver yet** — vLLM Metal is the real deal for *serving* but is a brew-only plugin and heavy; Splash is arguably the fastest thing for exactly your two models but states **36 GB minimum (48 GB recommended)** for its 4-bit targets, i.e. one tier above your box ([LM Studio blog](https://lmstudio.ai/blog/splash-engine)). Dense-27B on 32 GB is a *fit* problem first and a speed problem second: budget ~17 GB weights + ~1 GB mmproj + KV, and expect **single-digit tok/s without speculative decoding** on 153 GB/s.

## Comparison

| Engine | Version (Oct 2026) | Backend on M5 | Install on nix macOS | KV-quant | Spec-decode | MTP | Multimodal | Server / OpenAI API | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **MLX** (mlx-lm / mlx-vlm) | mlx **0.32.4** (10-14), mlx-lm **0.32.2** (10-16), mlx-vlm **0.7.9** (10-19) | Native Metal + **M5 NAX tensor units** (TensorOps/MPP, Metal 4) | `uv pip install mlx-lm mlx-vlm` (pure wheels, no brew) | ✅ `kv_bits`, `quantized_kv_start`, `max_kv_size`, TurboQuant (mlx-vlm) | ✅ draft model (`--draft-model`); mlx-vlm: dflash/dflash2/dspark/eagle3 | ⚠️ **mlx-lm drops `mtp.*` tensors**; **mlx-vlm supports `--draft-kind mtp`** | ✅ mlx-vlm (Qwen3.8-27B vision+video) | ✅ `mlx_lm.server`, `mlx_vlm.server` (FastAPI, batching, APC) | **PRIMARY** |
| **llama.cpp** | **v0.7.0** (10-15) / nightly **b11761** (10-20) | Metal; Metal 4 tensor API on M5 since **b10734**; NAX kernels underused | `nix run nixpkgs#llama-cpp`; or prebuilt `llama-b11761-bin-macos-arm64.tar.gz` | ✅ `-ctk/-ctv` (f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1) | ✅ draft, EAGLE-3, DFlash, DSpark, ngram-mod | ✅ `--spec-type draft-mtp` (+`--mtp` download) | ✅ `mmproj` (Qwen3.8-27B mmproj exists) | ✅ `llama-server` `/v1/*`, `--jinja` tools, GBNF, **port 9931** now | **CO-PRIMARY** |
| **Ollama** | **0.40.6** (10-16) | llama.cpp **+ MLX engine** on Apple Silicon | not installed; brew/nix not officially packaged → tarball | ✅ `OLLAMA_KV_CACHE_TYPE` (default f16) | ⚠️ model-package dependent (`draft_num_predict`) | ✅ inline MTP in `qwen3.8:27b-mlx` | ✅ | ✅ `/v1` | Runner-up / easiest ops |
| **LM Studio** (+ Bionic) | **0.4.26**; Bionic **1.1.8**; llama.cpp runtime **2.53.0** | GGUF (llama.cpp) **and** MLX engine **and** Splash | .dmg app (GUI); `lms` CLI | ⚠️ via llama.cpp runtime flags only | ✅ GGUF path; MLX path limited | via Splash / GGUF | ✅ | ✅ `lms server start` → OpenAI API | Best GUI; paid-ish; not nix-friendly |
| **vLLM Metal** (plugin) | **0.29.0** (0.28.0 announced 09-22) | MLX + Metal, **paged varlen kernel**, **M5 NAX prefill** | brew tap only (or `install.sh` venv) | ⚠️ vLLM-style mem budget `--gpu-memory-utilization` | ✅ batched MTP `--speculative-config` | ✅ (`method: mtp`) | ✅ (Qwen3.8-27B served) | ✅ full vLLM V1 scheduler | Real, but overkill for 1 user |
| **vllm-mlx** (waybarrios) | **0.6.0** | MLX | `pip install vllm-mlx==0.4.1`+ | ✅ paged KV, SSD tier | ✅ `--enable-mtp` | ✅ | ✅ text/img/video/audio | ✅ OpenAI **+ Anthropic** | Good agent-server fallback |
| **oMLX** | **0.7.1** (0.7.0 stable) | MLX + custom kernels (+Splash in 0.7.1) | .dmg / brew tap | ✅ tiered KV, SSD block cache | ✅ external MTP drafter | ⚠️ external drafter only | ✅ | ✅ OpenAI+Anthropic | Great for long agent contexts |
| **Rapid-MLX** | **0.16.5** | MLX | pip | ✅ paged KV, SSD cache | ✅ | ✅ | ✅ | ✅ OpenAI+Anthropic | Fine; benchmark claims are self-reported |
| **Kapitsa** | nace-ai/kapitsa | custom Metal kernels, Qwen/Nemotron only | source / brew | ? | ✅ | ? | ❌ | ✅ OpenAI+Anthropic | Narrow; no |
| **Splash** (Inco AI) | v1.x via LM Studio 1.1.5+ | purpose-built Metal | brew tap **only** | ✔ (memory plan) | ✅ **DFlash 2** per-model drafter | ❌ (DFlash2 instead) | ✅ | ✅ OpenAI+Anthropic | Fastest, but **36 GB floor** → not this box |
| **MLC-LLM / WebLLM** | no tagged release; repo last push 10-06 | TVM→Metal (pre-Metal-4 path) | pip only, model libs prebuilt | ⚠️ | ❌ | ❌ | ⚠️ | ✅ | Skip |
| **llamafile** | **0.10.6** (09-15) | Metal (llama.cpp inside) | single binary | ✅ (inherits) | ⚠️ | ❌ | ⚠️ | ✅ | Skip (single-file novelty) |
| **KoboldCpp** | **1.123.1** (10-13) | Metal (`koboldcpp-mac-arm64`) | one binary | ✅ (inherits llama.cpp) | ✅ (inherits) | ⚠️ | ✅ | ✅ (Kobold + OpenAI-ish) | Skip unless you want its UI |
| **Jan** | **0.8.7** | llama.cpp default, MLX optional | .dmg | ⚠️ | ⚠️ | ❌ | ✅ | ✅ | Skip (GUI wrapper) |
| **GPT4All** | **3.10.0** (2025-02-25, stale) | llama.cpp | — | ❌ | ❌ | ❌ | ❌ | local API | **Dead** for this purpose |
| **exo** | **v1.0.71** (04-23) | MLX distributed | pip | ⚠️ | ❌ | ❌ | ⚠️ | ✅ | Only for multi-Mac clusters |

## llama.cpp (ggml-org)

- **Version/naming:** stable release is **v0.7.0, published 2026-10-15** (https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0). Note the shape change: the `v0.x` tag carries **only `nightly-tag.txt`** — zero binaries. Every prebuilt asset lives on the rolling `bNNNNN` prerelease tags; the newest at time of writing is **b11761 (2026-10-20T22:14Z)**. So "latest release" ≠ "download a binary"; pin a `b` number explicitly.
- **macOS arm64 asset (verified present in b11761):** `llama-b11761-bin-macos-arm64.tar.gz` (16,084,489 bytes). URL pattern: `https://github.com/ggml-org/llama.cpp/releases/download/b11761/llama-b11761-bin-macos-arm64.tar.gz`. Sibling assets include `llama-b11761-xcframework.zip` and `llama-b11761-ui.tar.gz`.
- **Metal is default-on for Apple builds:** `set(GGML_METAL_DEFAULT ON)` under `if (APPLE)` in `ggml/CMakeLists.txt`, so the macOS arm64 tarball is a Metal build — no `-DGGML_METAL=ON` needed.
- **M5-specific history (important):** the Metal **4.0 tensor API** (i.e. the M5 GPU Neural Accelerators / NAX) was silently **inert on M5/A19** for ~10 months because `MTLCompileOptions.languageVersion` was never raised above 3.2, so the `mpp::tensor_ops` probe failed and ggml logged `- the tensor API is not supported in this environment - disabling` ([issue #27473](https://github.com/ggml-org/llama.cpp/issues/27473)). Fixed by **[PR #27461](https://github.com/ggml-org/llama.cpp/pull/27461), merged 2026-09-01**, shipping in **builds from b10734 onward** ([modelfit write-up](https://modelfit.io/blog/m5-mac-metal-tensor-api-llama-cpp-fix)). Consequence for you: **any M5-relevant llama.cpp must be ≥ b10734**; older bottles (Homebrew bottles were stale at b10450 in Aug 2026) silently run without NAX.
- **Metal maturity vs. the M5 ceiling:** a detailed feature request reports that on an M5 Max the ggml Metal backend **prefills at ~60% of oMLX**, "because the M5 NAX matrix units are mostly unused by the current ggml-metal kernels", and a 6-kernel patch set measured **pp 8k: 633 → 1064 t/s (+68%)**, pp 32k 528 → 884 (+67%), decode unchanged ([issue #30287](https://github.com/ggml-org/llama.cpp/issues/30287)). That issue was **closed as `not_planned`** in Oct 2026 — i.e. this is a real, acknowledged, *unfixed* gap. Read it as: llama.cpp decode is competitive; llama.cpp **prefill on M5 lags the MLX-derived engines**.
- **Per-chip tuning does exist:** `metal : add fa-vec tunings for M3 Max, M5 and M5 Pro` merged 2026-08-28 ([PR #27863](https://github.com/ggml-org/llama.cpp/pull/27863)), and a tensor-API flash-attention kernel for F16 KV landed as [#29570](https://github.com/ggml-org/llama.cpp/pull/29570).
- **Flash attention:** `-fa/--flash-attn [on|off|auto]` (env `LLAMA_ARG_FLASH_ATTN`) — present in `common/arg.cpp`.
- **KV-cache quantization:** `-ctk/--cache-type-k` and `-ctv/--cache-type-v`, allowed values `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`; separate draft-model variants `-ctkd/-ctvd`. On a 32 GB box this is the difference between a 32K and a 262K window: the Qwen3.8-27B MTP guide reports context creation **fails past ~90K next to 17 GB of weights without `--cache-type-k/v q4_0`**, and fits 262K at 22.2 GB *with* it ([qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)).
- **MoE partial offload:** `-ncmoe/--n-cpu-moe N` (first N layers' experts on CPU), `-cmoe/--cpu-moe` (all), and new in v0.7.0: **`--moe-cache-mib N`** — a GPU LRU cache for host-resident experts, "up to ~2.2x faster decode" (PR #29887). Also `-ot/--override-tensor` patterns. On a 32 GB Mac with a 20 GB MoE this is less relevant than on 12 GB GPUs, but `--moe-cache-mib` is free upside if you offload any experts.
- **Speculative decoding:** the richest menu of any engine — `--spec-type` takes `none | draft-simple | draft-eagle3 | draft-dflash | draft-dspark | draft-mtp | ngram-cache | ngram-simple | ngram-map-k | ngram-map-k4v | ngram-mod`, comma-composable ([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). **MTP is real and generic**: `draft-mtp` uses "Multi Token Prediction (MTP) heads from the main model"; `--mtp` also *downloads* the MTP head sidecar from the `-hf` repo. The Qwen3.5/3.8 family loader (`src/models/qwen35.cpp`) allocates `LLM_TENSOR_NEXTN_{EH_PROJ,ENORM,HNORM}` blocks and marks them dense, so **Qwen3.8-27B's in-checkpoint MTP head is loadable** — but you must pass the flag, since the head is otherwise silently ignored (added in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), Jul 2026).
- **Draft-model search UI:** `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-split`, `--spec-draft-p-min` (a ~0.60 confidence gate helps bandwidth-starved machines and hurts fast ones), `--spec-draft-n-ubatch` (new), `--spec-draft-ngl`. Community rule: **`--parallel 1` when measuring**, because the spec-decode win disappears by 4 concurrent streams.
- **Server:** `llama-server` is OpenAI-compatible (`/v1/chat/completions`, `/v1/responses`-style, `/v1/systemone`, rerank, embeddings), has `--jinja` (now **enabled by default**; `--no-jinja` to disable) giving real tool-calling/chat-template fidelity, GBNF grammars plus llguidance (`docs/llguidance.md`), multimodal via `--mmproj`, prompt-cache slot save/restore (`--slot-save-path`, `LLAMA_ARG_SLOT_SAVE_PATH`), and **changed its default port to 9931** in v0.7.0 (Docker keeps 8080 inside the container).
- **Batch tuning:** `-b/--batch-size` (logical, default 2048) and `-ub/--ubatch-size` (physical). The M4-24GB MTP sweep needed `-b 512 -ub 512` because "the default `-b 2048` OOMs the spec arm mid-generation and takes the server down" — expect the same class of failure on 32 GB with a 17 GB model.
- **Memory/loading:** `--mlock` as a standalone flag is **gone**; it's now `-lm/--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`.
- **Why it still matters on M5:** it is the only engine here with (a) `mmproj` multimodal GGUF support for Qwen3.8-27B, (b) full prefill/decode quant matrix, (c) the `ngram-mod` draftless accelerator that costs ~16 MB and is genuinely useful for coding agents, (d) `nixpkgs#llama-cpp`.

**Verdict:** co-primary. Use `b11761`+ (never below b10734 on M5). Accept that prefill is ~60-70% of MLX-derived engines on M5 and that maintainers declined the NAX prefill work — so for prompt-heavy agent workloads MLX-side engines win; for decode-heavy chat + GGUF convenience + tooling, llama.cpp wins.

## MLX (mlx-lm, mlx-vlm, mlx core)

- **Versions:** `mlx 0.32.4` (2026-10-14), `mlx-lm 0.32.2` (2026-10-16, Python ≥3.11), `mlx-vlm 0.7.9` (2026-10-19, Python ≥3.10) — all from PyPI JSON. Fast-moving but wheels only, no compiler needed.
- **M5/NAX support is first-party and early:** Apple's own post documents that "The GPU Neural Accelerators introduced with the M5 chip provides dedicated matrix-multiplication operations… MLX leverages the Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4" and reports **M5 vs M4 speedups of 3.33-4.06× TTFT and 1.19-1.27× generation** across Qwen3-1.7B/8B/14B, gpt-oss-20b-MXFP4 and **Qwen3-30B-A3B-4bit (TTFT 3.52×, gen 1.25×, 17.31 GB)** on a 24 GB M5 MacBook Pro ([machinelearning.apple.com](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). That is the single best citation for "MLX already exploits M5 NAX".
- **Measured on your exact SoC (M5 10/10/32GB, `mlx_lm.benchmark`):** Qwen3.5-35B-A3B-4bit **58.8 tok/s @ 20.12 GB**; Qwen3.6-35B-A3B-4bit **20.3 @ 20.13 GB**; Qwen3.5-9B-4bit **21.5**; Gemma-3-27B-4bit **5.5 @ 16.72 GB**; Qwen3-8B-4bit **24.0** ([mac-llm-bench m5/base](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/README.md)). Note the Qwen3.5→3.6 cliff (**58.8 → 20.3**) — hybrid-attention kernels are immature and this is not a bandwidth story.
- **KV-cache quantization:** `generate()`/CLI expose `kv_bits`, `kv_group_size`, `quantized_kv_start` (default 5000) and `max_kv_size`; `maybe_quantize_kv_cache()` swaps each `KVCache` for a `to_quantized()` cache once offset ≥ start ([mlx_lm/generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)). mlx-vlm adds a server-level KV-quantization option plus a **TurboQuant KV cache** path.
- **Speculative decoding:** yes, draft-model only in mlx-lm: `--draft-model`, `--num-draft-tokens`, `speculative_generate_step()` with a tokenizer-match warning path in `server.py`. In practice the drafter must share the vocab.
- **MTP: mlx-lm explicitly discards it.** `mlx_lm/models/qwen3_5.py` loads with `weights = {k: v for k, v in weights.items() if "mtp." not in k}`. So **mlx-lm can never exploit Qwen3.8-27B's built-in MTP head** — that speedup is llama.cpp-only or mlx-vlm-only. This is the single most decision-relevant asymmetry in this document.
- **mlx-vlm is the multimodal answer and it *does* do MTP:** `mlx_vlm.models` ships `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `prism_hadamard_qwen35`; the README documents `--draft-model` + `--draft-kind {dflash (default), eagle3, mtp}` + `--draft-block-size`, plus continuous batching, Automatic Prefix Caching, KV-cache quantization and a FastAPI server.
- **Prompt cache:** `mlx_lm.generate --prompt-cache`-style reuse exists as `load_prompt_cache`/`trim_prompt_cache`/`LRUPromptCache` with `--prompt-cache-bytes` accounting in the server; the same machinery is what oMLX/Rapid-MLX/vllm-mlx bolt SSD tiers onto.
- **Known M5 gotcha:** LM Studio's MLX models failed on M5 until users manually installed and selected the **"LM Studio MLX (Apple M5)" runtime** ([lmstudio-bug-tracker #1356](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1356)) — evidence that "MLX" is not one binary; the per-chip kernel selection matters.

**Verdict:** primary. Highest tok/s per watt on this box, first-party NAX kernels, only engine where Qwen3.8-27B *vision* + *MTP* + KV-quant can all be on at once (via mlx-vlm). Cost: you juggle two packages (mlx-lm for text/MoE, mlx-vlm for the VLM) and MLX has no GGUF ecosystem.

## Ollama

- **Engine:** **both**. v0.30.0 (2026-05-13) is explicit: "improved compatibility and performance using **llama.cpp**… This **augments the MLX engine on Apple Silicon**" ([release](https://github.com/ollama/ollama/releases/tag/v0.30.0)). Model packages decide which engine you get — e.g. `qwen3.8:27b-mlx` (MLX, inline MTP) vs `qwen3.6:35b-a3b-mtp-q4_K_M` (GGUF/llama.cpp + MTP drafter).
- **Version:** **0.40.6** (2026-10-16); repo actively pushed (2026-10-20).
- **KV cache control:** `OLLAMA_KV_CACHE_TYPE` ("Quantization type for the K/V cache (default: f16)"), `OLLAMA_FLASH_ATTENTION` (default false), `OLLAMA_CONTEXT_LENGTH` ("default: 4k/32k/256k based on VRAM") — from `envconfig/config.go`. Per-request `num_ctx` still exists. So yes: KV quant ✅, memory control ✅ (coarse).
- **Real numbers on M5 Pro 64 GB** (bandwidth 307 GB/s): `qwen3.6:35b-a3b-mtp-q4_K_M` **84.9 tok/s** 3-scenario average (86.8 short / 88.0 long / 79.9 @11k), `qwen3.8:27b-mlx` **33.8**, `gemma4:26b-nvfp4` 55.5, `gemma4:31b-nvfp4` 9.9 ([m5pro-llm-bench](https://github.com/daniel29348679/m5pro-llm-bench)). The 27B number exceeds the naive 307/16.7 ≈ 18 tok/s weight-read ceiling — that only happens because of MTP, which is the useful lesson.
- **Tools/multimodal:** OpenAI-compatible `/v1`, tool calling supported per-model; multimodal via vision projectors. But: it is **not** in nixpkgs by default and you said no brew, so the install route is a GitHub release tarball you have to hand-manage.

**Verdict:** runner-up — best "one daemon, both engines, zero config" story, and its MTP-packaged Qwen tags are genuinely fast. Loses on: no per-request KV-quant type, opaque memory planning, no `mmproj`-level control, and a non-nix install path.

## LM Studio

- **Versions:** app **0.4.26**; companion agent app **Bionic 1.1.8**; bundled llama.cpp runtimes are versioned separately (**2.53.0 extension packs** as of 1.1.8) ([changelog](https://lmstudio.ai/changelog)).
- **Three engines on a Mac now:** GGUF (llama.cpp), **MLX**, and since **Bionic 1.1.5 (2026-09-23)** "[Mac only] Ultra fast Qwen3.8 inference using a new engine: **Splash (by Inco AI)**".
- **MLX vs GGUF:** LM Studio's own docs describe it as running llama.cpp everywhere *plus* MLX on Apple silicon. Independent measurement on this SoC (see MLX section) puts MLX ~1.8-1.9× ahead of llama.cpp GGUF; LM Studio exposes both, so the practical rule is **pick MLX for decode-bound chat, GGUF when you need `-ctk/-ctv`, MTP, or a quant that only exists as GGUF**.
- **Headless:** `lms` CLI is real and covers it — `lms server start|stop|status`, `lms load --gpu=max --context-length=N`, `lms ps`, `lms ls`, `lms runtime`, plus a **daemon** (`lms daemon up`) for headless operation; `lms` v0.0.47 ([docs](https://lmstudio.ai/docs/cli)). Server is OpenAI-compatible with structured output/JSON-schema support.
- **Cost:** closed-source app, GUI-first, `.dmg` install — a poor fit for a nix-managed machine, and Bionic appears to be a paid tier.
- **Known M5 issue:** needs the M5-specific MLX runtime selected manually ([#1356](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1356)).

**Verdict:** best GUI and the only single app that gives you llama.cpp + MLX + Splash side by side — worth borrowing for A/B tests, not for a nix/CLI-first workflow.

## vLLM on Apple Silicon

This is no longer "unusable" — that answer is out of date.

- **`vllm-project/vllm-metal`** (1,892 ★, pushed 2026-10-20) is an **official-ish, community-maintained hardware plugin**: "vLLM supplies the API server, scheduler, and paged block manager; **mlx_lm supplies the model layers**; vllm-metal owns the request-aware attention path — the **paged varlen kernel, M5 NAX prefill, and speculative decoding**". v**0.29.0** current; v0.28.0 announced 2026-09-22 ([blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0), [repo](https://github.com/vllm-project/vllm-metal)).
- Concrete Apple-specific wins: **Aug 2026 — "vLLM Metal now uses M5 NAX tensor units to accelerate MHA, GQA, and MQA prefill"**; **Qwen3.8-27B runs** (`mlx-community/Qwen3.8-27B-8bit`); **batched MTP** under concurrent load; `--gpu-memory-utilization` as a hard serving budget; sleep mode that returns memory to other apps.
- **Blockers for you:** install is **Homebrew tap** (`brew install vllm-project/vllm-metal/vllm-metal`) or a curl venv installer — no brew on this box, so you'd be hand-rolling the venv installer. It also pulls the full vLLM V1 stack (heavy Python) to serve one user. Metal-plugin capability is still model-matrix-limited.
- **SGLang on Apple: still unusable.** No Metal/MLX backend; the Qwen3.8-27B SGLang docs in the wild are ROCm/CUDA (`AMD-AIM/sglang-radeon`). Treat as CUDA/ROCm-only in 2026 `[unverified for any private Apple fork]`.
- **`waybarrios/vllm-mlx`** (1,629 ★, PyPI **0.6.0**) is the pragmatic middle: pure `pip install vllm-mlx`, continuous batching, paged KV, SSD-tier KV, prefix cache, `--enable-mtp`, `--moe-top-k` (+7-16% on Qwen3-30B-A3B per its README), OpenAI **and** Anthropic `/v1/messages`, JSON-schema output, validated against Claude Code/Codex/pi/OpenCode.

**Verdict:** vLLM Metal is the real Apple story for *serving many clients* and the fastest-moving thing with NAX prefill — but it is a brew-installed, heavyweight plugin. For a single-user 32 GB Air, `vllm-mlx` (pip) is the only vLLM-family option worth installing, and only if you need concurrency.

## Also-rans, one line each

- **MLC-LLM / WebLLM:** no tagged release (repo last pushed 2026-10-06, no release assets), TVM-compiled Metal path predates Metal 4 TensorOps, model libs must be prebuilt per model — **skip**; nothing on M5 NAX.
- **llamafile (mozilla-ai, 0.10.6, 2026-09-15):** single-file executable wrapping llama.cpp's Metal backend; strictly a subset of llama.cpp with no `--spec-type`/KV-quant ergonomics exposed — **skip**.
- **KoboldCpp (1.123.1, 2026-10-13):** `koboldcpp-mac-arm64` is a llama.cpp+Metal bundle with a UI and its own agentic framework; inherits llama.cpp's M5 limitations, adds nothing — **skip** unless you want the UI.
- **Jan (0.8.7):** Electron GUI over llama.cpp with optional MLX engine; its own docs say "for the broadest model support and feature set, use llama.cpp" — **skip**, use llama.cpp directly.
- **GPT4All (3.10.0, 2025-02-25):** effectively **abandoned** for this purpose (last release 20 months ago) — **dead**.
- **exo (v1.0.71):** distributed MLX across multiple Macs over Thunderbolt RDMA; irrelevant and actively harmful on a single 32 GB Air — **skip**.

## 2026 newcomers that actually target Apple Silicon

- **Splash (Inco AI)** — purpose-built Metal engine *for exactly your two models*: "optimized specifically for **Qwen3.6-35B-A3B and Qwen3.8-27B**", ships a per-model **DFlash 2** drafter, vision, tool calling, OpenAI+Anthropic APIs, auto memory planning, prefix reuse, concurrent batching. Claims **210 tok/s on Qwen3.6-35B-A3B** and 74 tok/s short-prompt / 54 @32K on Qwen3.8-27B (48 GB M5 Pro), ~2× the next-fastest engine ([inco.ai](https://inco.ai/blog/splash), [repo](https://github.com/incoai/splash)). **But**: "Apple M3 or newer, macOS 26.4+, **Homebrew**… The 4-bit examples need at least **36 GB** of unified memory (48 GB recommended); 24 GB Macs can use smaller GGUF variants." Your 32 GB sits in the gap → **Q4 works only at reduced quant or not at all, and there's no brew**. Watch it; don't build on it. (Bundled in LM Studio Bionic ≥1.1.5 and picked up by oMLX 0.7.1.)
- **oMLX (jundot/omlx, 22,966 ★, Apache-2.0, v0.7.1)** — MLX server with **continuous batching + two-tier (RAM→SSD) paged KV cache**, menu-bar app, OpenAI+Anthropic APIs, MCP optional. Measured on M5 Pro 64 GB: Qwen3.8-27B-4bit **22.99 tok/s** average with an external 250.9 MB MTP drafter (53.7% acceptance, 368 tok/s prefill, 30.2 s TTFT for an uncached 11k prompt) — i.e. **slower than Ollama's inline-MTP package** on the same box. Its real win is **prefix/TTFT for coding agents**, not raw decode. Needs full Xcode for the custom kernels (GLM-5.2/MiniMax M3/Qwen3.5 families), else silently slower.
- **Rapid-MLX (raullenchai, 4,002 ★, v0.16.5)** — MLX server, tool-calling focused, OpenAI+Anthropic, continuous batching, paged + SSD KV. Its public corpus (58 submissions, 11 machines) contains the cleanest M5 numbers anywhere: **M5 Max 36 GB — Qwen3.8-27B-4bit 25.5 tok/s decode / 562 tok/s prefill / 18.7 GB peak**; M5 Pro 64 GB — Qwen3.6-35B-4bit 101.2. Its "up to 4× faster than mlx-lm" headline is self-reported — treat as marketing, use the corpus numbers.
- **Kapitsa (nace-ai, Apache-2.0)** — Metal engine hand-tuned **only** for Qwen and Nemotron; 154 tok/s on Qwen3.6-35B-A3B on an M4 Pro 48 GB, ahead of oMLX/Ollama/LM Studio/llama.cpp in its own benchmarks. Narrow by design; no multimodal story for Qwen3.8; **skip**.
- **Apple Foundation Models framework** — `import FoundationModels` gives you Apple's **on-device AFM + Private Cloud Compute** models with guided generation and tool calling ([docs](https://developer.apple.com/documentation/foundationmodels)). It is *not* a runtime for Qwen/Llama weights: you cannot point it at a GGUF. Irrelevant to "run a 27B locally" except as the zero-install baseline to benchmark against.

## M5 hardware facts (verified)

| Fact | Value | Source |
|---|---|---|
| M5 base memory bandwidth | **153 GB/s** (Apple's own MacBook Air spec line, 10-core GPU config: "10‑core GPU, Neural Accelerators… **153GB/s memory bandwidth**") | [apple.com/macbook-air/specs](https://www.apple.com/macbook-air/specs/) |
| M5 base theoretical | **153.6 GB/s**, LPDDR5X @ 9600 MT/s, "nearly 30 percent increase over M4"; M4 = **120 GB/s** (Apple ML post) | [Wikipedia Apple M5](https://en.wikipedia.org/wiki/Apple_M5), [Apple ML](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |
| M5 Pro | up to **307 GB/s**, 64 GB max, 16 or 20 GPU cores, dual-die "Fusion" | [Wikipedia Apple M5](https://en.wikipedia.org/wiki/Apple_M5) |
| M5 Max | **460 GB/s** (32-core GPU) / **614 GB/s** (40-core GPU), 128 GB max | same |
| M5 GPU AI compute | "dedicated **Neural Accelerator per GPU core**"; Apple claims **>4× peak GPU AI compute vs M4**, >6× vs M1; 16-core Neural Engine (42 TOPS) | [Wikipedia Apple M5](https://en.wikipedia.org/wiki/Apple_M5) (citing Apple's Sept 2025 briefing) |
| Metal 4 inference features | **tensor resource + Metal Performance Primitives (MPP) TensorOps** — cooperative tensors, reductions, and **"new quantized tensor formats and support for scale factors … native support in Metal performance primitives to efficiently compress weights and take advantage of the Neural Accelerators on the M5 Pro and M5 Max"** | [Metal – What's New](https://developer.apple.com/metal/whats-new), [MPP Programming Guide (PDF)](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), [WWDC26-330](https://developer.apple.com/videos/play/wwdc2026/330) |
| Do MLX / llama.cpp exploit NAX in 2026? | **MLX: yes**, since M5 launch (MLX → TensorOps/MPP; Apple's own M5 benchmarks). **llama.cpp: partially** — tensor API enabled only from b10734 (PR #27461), but NAX matmul kernels largely unused; the +68%-prefill patch was closed as not-planned. **vLLM Metal: yes** for MHA/GQA/MQA prefill (Aug 2026). | [Apple ML](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), [PR #27461](https://github.com/ggml-org/llama.cpp/pull/27461), [issue #30287](https://github.com/ggml-org/llama.cpp/issues/30287), [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| GPU wired-memory ceiling | `sysctl iogpu.wired_limit_mb` defaults to **0 = kernel-derived**. Measured on a 32 GiB M2 Max / macOS 26.5.2: Metal reports a **26,800,603,136 B (24.96 GiB, 78%)** recommended working set, leaving **7.04 GiB** unreachable. `sudo sysctl iogpu.wired_limit_mb=28000` moves it; **session-scoped** (reverts on reboot; `=0` resets). It is a ceiling, not an allocation | [modelpiper](https://modelpiper.com/blog/iogpu-wired-limit-mb-mac) |
| Do you need it on 32 GB? | **Probably yes, at the margin, for 27B+.** ~16.7 GB weights + mmproj + 16-32K KV lands right at the ~24.9 GiB GPU cap once macOS and apps take their share. Raise to ~26,000-27,000 MB, verify with a Metal query, and **never** approach hw.memsize — the spared RAM is exactly what keeps the machine responsive under pressure. | derived from the above `[unverified for M5 specifically]` |

### Measured throughput, this SoC (M5 10-CPU/10-GPU/32 GB)

`llama-bench` tg128 (Q4_K_M) and `mlx_lm.benchmark` 4-bit, same machine — [mac-llm-bench m5/base](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/README.md):

| Model | Runtime | tok/s | Peak RSS |
|---|---|---|---|
| Qwen3.5-**35B-A3B** MoE | **MLX 4bit** | **58.8** | 20.12 GB |
| Qwen3.5-35B-A3B MoE | GGUF Q4_K_M | 31.3 | 20.73 GB |
| Qwen3-**30B-A3B** MoE | GGUF Q4_K_M | 23.1 | 17.48 GB |
| Qwen3.6-35B-A3B MoE | MLX 4bit | 20.3 | 20.13 GB |
| Qwen3.6-35B-A3B MoE | GGUF Q4_K_M | 16.7 | 20.14 GB |
| Qwen3.5-**27B dense** | GGUF Q4_K_M | **4.4** | 15.93 GB |
| Gemma-3-27B | MLX 4bit | 5.5 | 16.72 GB |
| Gemma-3-27B | GGUF Q4_K_M | 3.0 | 15.64 GB |

Cross-box anchors: M5 Max 36 GB, Qwen3.8-27B-4bit → **25.5 tok/s decode / 562 t/s prefill / 18.7 GB** ([rapid-mlx corpus](https://rapidmlx.com/benchmarks)); M5 Pro 64 GB, `qwen3.8:27b-mlx` (Ollama, inline MTP) → **33.8 tok/s** ([m5pro-llm-bench](https://github.com/daniel29348679/m5pro-llm-bench)); M4 24 GB Qwen3.8-27B Q3_K_XL with MTP → 5.8 → 5.8 (**a wash**, the flag added nothing) and M3 Ultra 96 GB Q6_K → 22.8 → 24.2 (**+6%**) ([qwen38-mtp community table](https://github.com/sudoingX/qwen38-mtp)).

**Derived expectation for your box on Qwen3.8-27B `[derived, unverified]`:** 153.6 GB/s ÷ ~16.7 GB of 4-bit weights ≈ **9.2 tok/s hard ceiling without speculative decoding**; the Qwen3.5-27B GGUF's measured 4.4 suggests llama.cpp's young hybrid GDN kernels get roughly half of that today. With a working MTP/DFlash drafter the ceiling is bypassed — that's how a 307 GB/s M5 Pro reaches 33.8 — so plan on **~8-10 tok/s now, ~15-20 once you find a drafter path that engages** on 27B dense; and **~30-60 tok/s** on a 3B-active MoE (measured 23-59 above), where the fit is ~17-21 GB and bandwidth is not the bottleneck.

### Known M5-specific regressions / issues (as of 2026-10-20)

1. **llama.cpp Metal tensor API silently inert on M5/A19** — logs `- the tensor API is not supported in this environment - disabling`, `has tensor = false`, on macOS 26.2-26.4; prefills slow with no error. Fixed by PR #27461 from **b10734**; escape hatch `GGML_METAL_TENSOR_DISABLE=1`. [issue #27473](https://github.com/ggml-org/llama.cpp/issues/27473), [modelfit](https://modelfit.io/blog/m5-mac-metal-tensor-api-llama-cpp-fix)
2. **M5 NAX units unused by ggml-metal** — M5 Max prefill ~60% of oMLX; +68% patch **closed not planned**. [issue #30287](https://github.com/ggml-org/llama.cpp/issues/30287)
3. **Failed tensor-API compile can leave the GPU in a bad state** (`ggml_metal_synchronize: command buffer 0 failed with status 5`) — one more reason to require a modern build. [modelfit](https://modelfit.io/blog/m5-mac-metal-tensor-api-llama-cpp-fix)
4. **LM Studio MLX broken on M5** until the M5-specific MLX runtime is installed and selected. [#1356](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1356)
5. **Default batch size OOMs spec-decode arms on small unified memory** — `-b 2048` killed the server on a 24 GB M4; use `-b 512 -ub 512`. [qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)
6. **Hybrid GDN throughput regression Qwen3.5→3.6** (58.8 → 20.3 MLX; 31.3 → 16.7 GGUF) on identical hardware — treat any "Qwen3.x-note" tok/s claim as runtime-version-specific. [mac-llm-bench](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/README.md)

## Install commands on this machine

All uv/pip/tarball — no brew, no nix system changes beyond profiles.

### 1. llama.cpp — newest Metal (recommended for M5)

```bash
# (a) nixpkgs attr (verified name from upstream docs/install.md: `llama-cpp`)
nix run nixpkgs#llama-cpp -- --version
nix profile install nixpkgs#llama-cpp

# (b) OR pin a nightly when you need the M5 tensor-API path / newest kernels (>= b10734, ideally b11761)
#     verified asset name in b11761: llama-b11761-bin-macos-arm64.tar.gz
curl -fL -o /tmp/llama-b11761.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11761/llama-b11761-bin-macos-arm64.tar.gz
mkdir -p ~/opt/llama-b11761 && tar -xzf /tmp/llama-b11761.tar.gz -C ~/opt/llama-b11761
export PATH="$HOME/opt/llama-b11761:$PATH"

# verify Metal + tensor API are actually live on M5 (the whole point):
llama-bench --list-devices            # expect: MTL0 Apple M5
llama-server --version 2>&1 | head
# if you ever see "- the tensor API is not supported in this environment - disabling",
# you are on an old build; GGML_METAL_TENSOR_DISABLE=1 only *hides* it.
```

### 2. llama.cpp server for the 27B, MTP + KV-quant (llama.cpp-only feature combo)

```bash
llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 131072 -fa on --jinja \
  -b 512 -ub 512 \
  --load-mode mmap+mlock \
  --host 127.0.0.1 --port 9931
# port 9931 is the v0.7.0 default; /v1/chat/completions is OpenAI-compatible.
# -hf auto-pulls mmproj for the VLM; add --mmproj <file> if loading files manually.
# Drop -c to 262144 exactly and raise KV to f16 if you prefer window over speed.
# --load-mode replaces the old --mlock standalone flag.
```

### 3. MLX — primary runtime (wheels only, nix-friendly)

```bash
uv venv ~/.venvs/mlx
uv pip install --python ~/.venvs/mlx/bin/python mlx-lm==0.32.2 mlx-vlm==0.7.9

# text / MoE, OpenAI-compatible server
~/.venvs/mlx/bin/mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080

# 27B VLM with KV quant + MTP drafter (mlx-lm CANNOT do MTP; mlx-vlm can)
~/.venvs/mlx/bin/mlx_vlm.server --model mlx-community/Qwen3.8-27B-4bit \
  --draft-kind mtp --draft-block-size 3
# mlx-vlm also accepts --draft-model z-lab/...-DFlash with --draft-kind dflash (default)
# one-shot generation / prompt-cache / kv-quant knobs:
~/.venvs/mlx/bin/mlx_lm.generate --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt "hi" --kv-bits 4 --quantized-kv-start 5000 --max-kv-size 32768
```

### 4. Optional: Ollama as the fallback daemon (tarball, since no brew)

```bash
# GitHub release tarball is the only non-brew route; pick the darwin arm64 asset
curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz \
  -o /tmp/ollama.tgz && mkdir -p ~/opt/ollama && tar -xzf /tmp/ollama.tgz -C ~/opt/ollama
OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=32768 \
  ~/opt/ollama/bin/ollama serve     # then: ollama run qwen3.8:27b-mlx
```

### 5. Optional: vllm-mlx when you need concurrency + Anthropic API

```bash
uv tool install 'vllm-mlx==0.6.0'
vllm-mlx serve mlx-community/Qwen3.8-27B-4bit --host 127.0.0.1 --port 8000 --continuous-batching --enable-mtp
```

**Raise the GPU ceiling first if the 27B won't load:**
```bash
sysctl iogpu.wired_limit_mb hw.memsize        # read: 0 = kernel default
sudo sysctl iogpu.wired_limit_mb=26000        # session only; ~26 GB of your 32
sudo sysctl iogpu.wired_limit_mb=0            # reset after the run
```

## Sources

- llama.cpp v0.7.0 release notes — https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0
- llama.cpp releases / nightly asset list (b11761) — https://github.com/ggml-org/llama.cpp/releases
- llama.cpp speculative decoding doc — https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- llama.cpp install doc (nix attr `llama-cpp`) — https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md
- llama.cpp `common/arg.cpp` (all flags quoted) — https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
- llama.cpp `src/models/qwen35.cpp` (GDN + MTP/nextn tensors) — https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- llama.cpp `ggml/CMakeLists.txt` (GGML_METAL_DEFAULT ON for APPLE) — https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt
- PR #27461 "metal: enable Metal 4.0 tensor API on M5+/A19+" (merged 2026-09-01) — https://github.com/ggml-org/llama.cpp/pull/27461
- PR #27863 "metal: add fa-vec tunings for M3 Max, M5 and M5 Pro" — https://github.com/ggml-org/llama.cpp/pull/27863
- PR #22673 (draft-mtp) referenced by community repo — https://github.com/sudoingX/qwen38-mtp
- Issue #27473 "Metal tensor API silently inert on M5+/A19+" — https://github.com/ggml-org/llama.cpp/issues/27473
- Issue #30287 "Metal prefill speedup (~68%) on M5 Max (NAX)" — closed not planned — https://github.com/ggml-org/llama.cpp/issues/30287
- modelfit: M5 Metal tensor API, fix from b10734 — https://modelfit.io/blog/m5-mac-metal-tensor-api-llama-cpp-fix
- Qwen3.8-27B MTP flag community benchmark repo — https://github.com/sudoingX/qwen38-mtp
- Real-world M5 llama.cpp Qwen3.8-27B config (512K ctx) — https://dev.to/dmitryame/inside-my-llamacpp-setup-tuning-qwen-38-27b-for-512k-context-o3c
- mlx-lm on PyPI (0.32.2) — https://pypi.org/project/mlx-lm/
- mlx-vlm on PyPI (0.7.9) + spec-decode/KV-quant docs — https://github.com/Blaizzy/mlx-vlm
- mlx-lm `qwen3_5.py` (drops `mtp.*` tensors) — https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
- mlx-lm `generate.py` (kv_bits / draft-model) — https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- Apple ML Research: "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" — https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Apple MacBook Air specs (153GB/s, Neural Accelerators) — https://www.apple.com/macbook-air/specs/
- Wikipedia: Apple M5 (bandwidth table, NAX per GPU core, 42 TOPS NPU) — https://en.wikipedia.org/wiki/Apple_M5
- Apple Developer: Metal What's New (quantized tensor formats, MPP, M5 NAX) — https://developer.apple.com/metal/whats-new
- Apple: Metal Performance Primitives Programming Guide (PDF) — https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
- WWDC26 session 330 "Optimize custom ML operations with Metal tensors" — https://developer.apple.com/videos/play/wwdc2026/330
- Apple Foundation Models framework — https://developer.apple.com/documentation/foundationmodels
- Ollama v0.30.0 (llama.cpp + MLX engines) — https://github.com/ollama/ollama/releases/tag/v0.30.0
- Ollama v0.40.6 (latest) — https://github.com/ollama/ollama/releases
- Ollama `envconfig/config.go` (OLLAMA_KV_CACHE_TYPE / FLASH_ATTENTION / CONTEXT_LENGTH) — https://github.com/ollama/ollama/blob/main/envconfig/config.go
- LM Studio changelog (Bionic 1.1.8, llama.cpp 2.53.0, Splash) — https://lmstudio.ai/changelog
- LM Studio download (0.4.26) — https://lmstudio.ai/download
- LM Studio `lms` CLI docs — https://lmstudio.ai/docs/cli
- LM Studio blog: Splash Engine (36 GB floor, M5 Pro numbers) — https://lmstudio.ai/blog/splash-engine
- Splash repo (Inco AI) — https://github.com/incoai/splash
- Inco AI Splash post (210 tok/s, 48 GB M5 Pro) — https://inco.ai/blog/splash
- LM Studio bug #1356 "MLX models not working on M5" — https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1356
- vLLM Metal repo (v0.29.0, NAX prefill, Qwen3.8) — https://github.com/vllm-project/vllm-metal
- vLLM blog: vllm-metal v0.28.0 — https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0
- vllm-mlx (waybarrios, PyPI 0.6.0) — https://github.com/waybarrios/vllm-mlx
- oMLX (jundot) — https://github.com/jundot/omlx ; https://omlx.ai
- Rapid-MLX — https://github.com/raullenchai/Rapid-MLX ; benchmark corpus — https://rapidmlx.com/benchmarks
- Kapitsa (nace-ai) — https://github.com/nace-ai/kapitsa ; https://www.nace.ai/kapitsa
- mac-llm-bench, M5 10/10/32GB results (primary same-hardware evidence) — https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/README.md
- M5 Pro 64GB Ollama+oMLX leaderboard — https://github.com/daniel29348679/m5pro-llm-bench
- iogpu.wired_limit_mb measured ceiling — https://modelpiper.com/blog/iogpu-wired-limit-mb-mac
- Qwen3.8-27B spec sheet (48 GDN + 16 GQA, 262K, MTP, Apache-2.0) — https://www.llm-releases.com/models/qwen3-8-27b
- Qwen3.8-27B model card — https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B GGUF + mmproj (bartowski) — https://huggingface.co/bartowski/Qwen3.8-27B-GGUF
- Qwen3.6-35B-A3B model card — https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- MLC-LLM repo (maintained, no tagged release) — https://github.com/mlc-ai/mlc-llm
- llamafile 0.10.6 — https://github.com/mozilla-ai/llamafile/releases
- KoboldCpp 1.123.1 — https://github.com/LostRuins/koboldcpp/releases
- Jan (llama.cpp default, MLX optional) — https://docs.jan.ai/desktop/local-engine/llama-cpp ; https://github.com/menloresearch/jan/releases
- GPT4All (last release v3.10.0, 2025-02-25) — https://github.com/nomic-ai/gpt4all/releases
- exo v1.0.71 — https://github.com/exo-explore/exo/releases/tag/v1.0.71
