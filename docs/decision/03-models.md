# Decision models — what actually runs on 32 GB

Sizes below are **bytes from the HF `tree/main` API**, not prose. Scores are the Decision Index (DI)
where one exists, and every row names its suite version because 0.2.1 and 0.3 are not comparable.
`[UNVERIFIED]` = I could not check it against a primary source in this pass; nothing is invented.

**Verdict.** Two open 9B checkpoints both *run here* and score on the same suite: **`clef-flash`**
(MLX 4-bit 6.21 GB or GGUF Q4_K_M 6.3 GB, Apache-2.0) and **`kev-9b`** (GGUF Q4_K_M 6.0 GB).
**`jaredpalmer/kev-4b`** is the best-documented on *this exact SoC* (721 ms new / 136 ms
cached, measured by upstream on an M5 32 GB) at DI 38.0. The 9B cliff is **only** the MLX adapter
fold — the adapter merge peaks at ~1.9× the weights (~37 GB) — and the GGUF conversion bypasses it:
`ggml-org/Kev-9B-GGUF` carries the pointer head in a 6.0 GB Q4_K_M file and was measured here at
**0.680 accuracy on `typed-decisions`, 4.97 s p50, 7.2 GB peak RSS** ([results](05-results.md)).

---

## 1. Candidates

`readout` = how the answer leaves the model. `run` = what can execute it on macOS arm64 today.
Disk = sum of the files you must download (`hf download` byte total), not resident memory.

### 1a. LLM backbone + trained decision head (pointer / joint head)

| id | HF repo | readout | params | format | disk GB | ctx | license | score (suite) | run here | mac-ready |
|---|---|---|---|---|---|---|---|---|---|---|
| **clef-flash-4bit** | `mlx-community/clef-flash-4bit` | joint schema head (transformer over backbone hidden states; 1 logit per option, softmax per question) | 9.65B | MLX safetensors + `clef_mlx.py` | **6.21** (5.34 + 0.60 shards + 0.244 head) | 16,384 trained | Apache-2.0 | **54.65** (DI 0.2.1, 2k sample, *third-party*); bf16 same rows 55.63; published 57.07 | `clef_mlx.py` + mlx-vlm 0.7.4 | **yes** |
| clef-flash-8bit | `mlx-community/clef-flash-8bit` | " | 9.65B | MLX 8-bit | **10.71** | 16,384 | Apache-2.0 | — (card quotes Δ≈1 pt vs bf16) `[UNVERIFIED size]` | same | **yes** |
| clef-flash (bf16 ref) | `Cloudflare/clef-flash` | " | 9.65B | torch safetensors + custom code | **19.08** (4 shards + 0.244 head) | 16,384 | Apache-2.0 | 57.07 (DI 0.2.1, full suite, *vendor*) | torch 2.11 + transformers 5.10.2, MPS untested by vendor | partial |
| Clef (27B) | `Cloudflare/clef` | " | 27.8B | torch safetensors | **54.99** | 16,384 | Apache-2.0 | higher than flash (vendor, not tabulated for 32 GB) | 4-bit MLX exists (16.3 GB) | no (peak 17.1-19.6 GB, 32 GB min) |
| **Kev-9B** | `jaredpalmer/kev-9b` | pointer head: `<decide>` hidden state scored against each `</opt>` hidden state | 9.65B + LoRA r16 | PEFT adapter 0.173 GB + `head.pt` 0.0084 GB | **0.202** (+ base `Qwen/Qwen3.5-9B-Base` **19.329**) | 65,536 accepted / 8,192 validated | Apache-2.0 | **41.0** (breadth-v1 test, chance-corrected, vendor) | `kev.mlx_model` (bf16 only) | **no** — merge peak ≈37 GB (§3) |
| **Kev-4B** | `jaredpalmer/kev-4b` | " | 4.7B + LoRA | adapter 0.130 GB + head 0.0053 GB | **0.160** (+ base `Qwen/Qwen3.5-4B-Base` ≈8.4) | 8,192 validated | Apache-2.0 | **38.0** (breadth-v1) | `kev.mlx_model`, measured on M5 32 GB | **yes** |
| **Kev-0.8B** | `jaredpalmer/kev-0.8b` | " | 0.85B + LoRA | adapter 0.0433 GB + head 0.0021 GB | **0.066** (+ base ≈1.7 `[UNVERIFIED]`) | 8,192 validated | Apache-2.0 | **23.3** (breadth-v1) | " | **yes** |
| Kev-27B (ref) | `jaredpalmer/kev-27b` | " | 27.8B | full bf16 | **51** (+head) | 65,536 | Apache-2.0 | **52.3** (breadth-v1) | needs 96-128 GB Mac | no |
| **Jeeves** | `PostHog/jeeves` | pointer head, + optional reasoning pass when p<0.9 | 9.65B + LoRA | torch safetensors 21 GB + `drafter_k4` | **21.0** | 8,192 cache default | MIT (code) / Apache-2.0 (weights) | 0.896 JevBench public, 0.905 own test (vendor) | torch MPS (bf16) or `--precision fp8` | **fp8 only** |
| Jeeves fp8 | `PostHog/jeeves-fp8` | " | " | fp8 + Metal w8a16 | **11.5** `[UNVERIFIED]` | " | " | same outputs as `--precision fp8` (vendor) | MPS | **yes, tight** |
| **Nimble-9B v4** | `bespokelabs/Bespoke-Nimble-9B` | one-token answer-code logits, softmax over codes | 9.65B | torch safetensors (LoRA merged at serve) `[UNVERIFIED]` | ≈18-19 `[UNVERIFIED]` | 8,192 | **not stated in repo** | **63.70** DI 0.3 public suite (*vendor*, own run 2026-10-12); v3 54.7 on 0.2.x | `ParallelScorer` on Metal, `ollama pull nimble` (v2) | **probably** |
| **Open-Jev-9B** | `ZefanCai/Open-Jev-9B` | LoRA + trained scalar decision head over Qwen3.5 | 9.65B + adapter | adapter + head, base not included `[UNVERIFIED size]` | ≈19.4 with base `[UNVERIFIED]` | 16,384 (JevBench run) | Apache-2.0 | 179/231 JevBench public (77.49%), hard 66/111 (vendor, audited); DI v1.5.4 #46 / 24.356 (different scale) | torch, CUDA-oriented; MPS stated in deployment matrix | partial |
| OpenJev-4B | `shenjunhao/OpenJev-4B` | Option-Set Interactor + shared head, permutation-equivariant | 4.7B | full-parameter safetensors | ≈9.4 `[UNVERIFIED]` | 8,192 `[UNVERIFIED]` | MIT | MMDM/JevBench vs Jev, Decider 4B, JevK5 (vendor) | CUDA validated (`fla-core==0.5.2`) | unlikely |
| OpenDecider-small | `manjunathshiva/opendecider-small` | logprob over option tokens read from the server | 4.7B | safetensors / GGUF | 8.4 / Q8_0 ≈5 `[UNVERIFIED]` | 8,192 `[UNVERIFIED]` | Apache-2.0 | 0.796 typed-decisions (nano; *author*), vs Jev 0.754, Laya 0.766 | `opendecider[mlx]`, LM Studio, Ollama | **yes** |
| Decider-4B | `Mapika/decider-4b` | one-pass typed head | 4.7B | safetensors 8.4 GB / GGUF Q4_K_M 2.7 GB | 8.4 | 32,768 `[UNVERIFIED]` | Apache-2.0 | 51.02 (DI 0.3 sample, vendor); JevBench hard 0.640 | CUDA→MPS→CPU; GGUF | **yes (GGUF)** |
| Tev1-4B | `togethercomputer/Tev1-4B-experimental` | **LM head**, one option letter + logprobs | 4.67B | safetensors (2 shards) | **9.343** | 32,768 `[UNVERIFIED]` | **being finalized** (base Apache-2.0) | 88.0% dev set, not a benchmark (vendor) | any logits engine | yes, but license blocks it |
| OpenThai-SystemOne | `iapp/OpenThai-SystemOne` | LM head replaced by 256-way slot softmax | 0.85B | safetensors `[UNVERIFIED]` | ≈1.7 `[UNVERIFIED]` | `[UNVERIFIED]` | Apache-2.0 | Thai/English only, no DI | CUDA, **MPS**, CPU | probably |
| Jev-Omni | `akhilaaa3/Jev-Omni` | head over Gemma-4-12B | 12B | safetensors 23.976 + `head.pt` 0.004 | **23.98** | `[UNVERIFIED]` | Apache-2.0 | 86.15% matched JevBench groups (vendor) | NVIDIA-oriented | **no** (22.3 GiB, no headroom) |

### 1b. Encoders / non-autoregressive (one pass, no LM)

| id | HF repo | readout | params | format | disk GB | ctx | license | score | run here | mac-ready |
|---|---|---|---|---|---|---|---|---|---|---|
| **Anarkali** | built from `jhu-clsp/ettin-encoder-150m`; ONNX in `release-web/` | spans/`[CLS]` decision heads | **68M per awesome-system-one, 150M per repo README — contradiction** | ONNX 273 MB | **0.273** `[UNVERIFIED]` | page-sized | Apache-2.0 | 74.0% typed-decisions vs Jev 72.7 (author) | ONNX Runtime CPU | **yes** |
| **open-jev-deberta** | `com-kotobalabs/open-jev-deberta-v3-large` | encoder + head (`head.safetensors` 12.6 MB) | 434M (fp32) | safetensors + `typed_decisions/` code | **1.760** | 512 `[UNVERIFIED]` | Apache-2.0 | public gold labels (vendor, no DI) | torch CPU/GPU | **yes** |
| **GLiNER2.5-Decide** | `fastino/GLiNER2.5-Decide` | multi-head span/label scorer, `classify_text` | stated 340M; **1.95 GB fp32 ⇒ ~487M, bf16 ⇒ ~1B** | safetensors + `SKILL.md` | **1.955** | `[UNVERIFIED]` | Apache-2.0 | no DI; not `/v1/systemone` | `gliner2` CPU/GPU | **yes** |
| **Laya** | `convaiinnovations/laya`, `convaiinnovations/laya-multilingual` | choice/score/noul heads on ModernBERT / mmBERT | 322M / 421M | safetensors `[UNVERIFIED]` | ≈1.3 / 1.7 `[UNVERIFIED]` | 1,024 default, 8,192 with `max_len` | Apache-2.0 | 0.766 typed-decisions (fine-tuned), 0.362 base | `pip install laya`, ONNX extra, MPS | **yes** |
| **Von** | `wfzyx/von` | per-option independent scoring on ModernBERT-Large | 395M | safetensors, `von-sdk` | ≈0.8 weights (**README says ~3 GB fetch — prose, likely env-inclusive**) | 8,192 `[UNVERIFIED]` | Apache-2.0 | threshold sweep only; no DI | OpenVINO / CUDA / **MPS** | yes — **archived** |

### 1c. Logprob read-out (the only 9B path in llama.cpp)

| id | HF repo | readout | params | format | disk GB | ctx | license | score | run here |
|---|---|---|---|---|---|---|---|---|---|
| **SemIf (ex-OpenJev)** | none of its own — `TheoLeeCJ/SemIf` + any base, e.g. `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | option-token / criterion logits from the frozen LM head | model-dependent | Python pkg, + GGUF for llama.cpp | 0 (code) + base | base model's | MIT (code) + Apache-2.0 (Qwen) | Qwen3.5-4B 0.813 balanced acc vs Jev 0.883 (author) | torch/MPS, **MLX**, llama.cpp |
| Jev-Style v3 | `[UNVERIFIED repo id]` — site `jevstyle.com` says v3 = 0.8B GGUF | one-token decision, logprobs | 0.8B | GGUF | ≈0.5 (Q4) `[UNVERIFIED]` | 25,600 | Apache-2.0 | 79.2% on 2,000 typed decisions (vendor) | llama.cpp / LM Studio |

### 1d. Multimodal small

| id | HF repo | readout | params | format | disk GB | ctx | license | score |
|---|---|---|---|---|---|---|---|---|
| **d1-3B** | `LiquidAI/d1-3B` | `modeling_d1.py` heads over LFM2.5-VL-3B (SigLIP2 NaFlex 400M) | 3.12B | safetensors + 5 `.py` | **6.265** | 32,768 | **LFM Open License v1.0** (`license_name: lfm1.0`) | **48.57** DI 0.2.1 (vendor, official scorer) |
| d1-omni-600M | `LiquidAI/d1-omni-600M` | " (+audio) | 587M | safetensors + code | **2.354** | `[UNVERIFIED]` | LFM Open v1.0 | not tabulated on the 3B card |

### 1e. Reference only (hosted, not candidates)

| Model | Size | Number | Source |
|---|---|---|---|
| **Jev** (TypeSafe) | undisclosed | DI 54.0 (breadth-v1), 57.07 (Clef's 0.2.1 full suite), 57.96 / 60.1 (Bespoke's views) | hosted; **four different numbers for one model — suite version is the variable** |
| **Microsoft-Decision-1** | 9.7B | DI 43.49 as "Decision 1.0 Lux" (0.2.1, third-party table); $0.042/M in | Foundry only |

---

## 2. Architecture and read-out, per family

- **Pointer head (Kev, Jeeves, Open-Jev).** The state and each question form one causal row
  `<state> … <q> instr <opt> o1 </opt> … <decide>`; the head scores `</opt>` hidden states against the
  `<decide>` hidden state, softmax over options. Qwen3.5's Gated DeltaNet layers are recurrent and ignore
  attention masks, so **each question is its own row** and the state is prefilled once into a prompt cache
  (`kev/model.py`, `kev/mlx_model.py`). Consequence: cost ≈ prefill(state) + n_questions × branch.
- **Joint schema head (Clef).** A small transformer over the backbone's final hidden states routes
  evidence from state to each question and scores **all options of all questions jointly** — so questions
  *can* see each other, unlike Kev. One logit per allowed option, softmax per question.
- **Answer-code logits (Nimble, Jev-Style, Tev1, SemIf, OpenDecider-GGUF).** No custom module: pick the
  first token of each option (or a single-letter code) and softmax its logit. **This is the only family a
  stock llama.cpp / MLX server can serve**, because it needs logits and nothing else.
- **Encoder heads (Laya, Von, Anarkali, open-jev-deberta, GLiNER).** `[CLS]`/span pooling into
  choice/score/noul heads; one pass, no cache, no KV growth. 273 MB – 1.76 GB and milliseconds.

---

## 3. Memory math for 32 GB

Budget (from the [text model survey](../text/01-models.md), unchanged): 32 GiB installed; macOS + a dev session 5-7 GiB; Metal
`recommendedMaxWorkingSetSize` ≈ **24.6 GiB** hard; **≈22.5 GiB comfortable**.

KV for Qwen3.5-9B: 8 of 32 layers full-attention × 4 KV heads × 256 dim × 2 (K,V) × 2 B = **32 KiB/token**
→ 0.06 GiB @2k, 0.25 @8k, **1.00 @32k**. DeltaNet layers add a *constant* recurrent state, not KV.
Kev-4B assumed 36 KiB/token `[inference]` (its config wasn't re-read in this pass).

Activations, **measured upstream on this SoC**: Kev-4B at 8,192 tokens costs **1.2 GB** above the weights
with `PREFILL_CHUNK=1024`, and **4.6 GB** in one pass — the same 6.5 s either way. 9B scaled ×1.6
`[inference]` → ~1.9 GB @8k.

| Config | weights GiB | KV 2k / 8k / 32k | act 2k / 8k / 32k | **total @32k** | vs 22.5 GiB |
|---|---|---|---|---|---|
| Kev-9B bf16 (pre-merged) | 18.00 | 0.06 / 0.25 / 1.00 | 0.47 / 1.77 / 1.77 | **20.8** | fits, ~1.7 GiB spare |
| **Kev-9B GGUF Q4_K_M** | 5.92 (+ host buffers) | llama.cpp-managed | — | **7.2 measured (RSS)** | **fits; measured** |
| Kev-9B bf16 **adapter path** | 18.00 → **peak 34.0 during fold** | — | — | **≈37 GB** | **does not fit** |
| Kev-4B bf16 | 7.82 (fold peak 14.8) | 0.07 / 0.28 / 1.10 | 0.33 / 1.12 / 1.12 | **10.0** | comfortable |
| Kev-0.8B bf16 | 1.58 | 0.02 / 0.06 / 0.25 | 0.1 / 0.3 / 0.3 | **2.1** | trivial |
| clef-flash **4-bit** MLX | 5.78 | vendor-measured jointly | " | **8.0 @16k (vendor peak 8.6 GB)** | comfortable; **do not exceed 16k** |
| clef-flash **8-bit** MLX | 9.10 | " | " | **12.1 @16k (vendor peak 13.0 GB)** | comfortable |
| clef-flash bf16 (torch) | 17.77 | 0.06 / 0.25 / 1.00 | ~1.0 | **19.8** | fits, no quant headroom |
| clef-flash Q4_K_M GGUF | 6.48 (+mmproj) | llama.cpp-managed | — | ~8 | comfortable |
| d1-3B bf16 | 5.83 | small (hybrid conv) | image-dependent | **~6.5** | comfortable to 32k |
| Jeeves fp8 (MPS) | 10.71 | 0.25 @8k | think cache | **~13 with `--max-rows 4 --max-len 4096`** | fits; **bf16 (19.6) does not with default caches (28 GB)** |
| Jev-Omni bf16 | 22.33 | — | — | **>23** | **no** |

### The 9B cliff is the adapter fold, not the weights — **H1 is refuted as stated**

`kev/mlx_model.py` folds the LoRA with `merge_lora()`: it materialises the adapter delta in fp32, per
tensor, on top of a model that already holds the base. Upstream measured the ratio on Kev-4B: full bf16
weights load at **8.4 GB peak for 8.4 GB of weights**, the adapter path at **15.9 GB** (1.9×). Extrapolated
`[inference]` to Kev-9B: **1.9 × 19.33 ≈ 36.7 GB**. `kev.mlx_model`'s own comment records the same failure
mode one level down ("one graph over every target peaks at ~2.9x the base weights… 25.8 GB RSS for the 4B").

So: **Kev-9B cannot be loaded on a 32 GB Mac through either shipped *MLX* path.** The fix is one merge step
off-box (`scripts/merge_lora_checkpoint.py`, or a Modal H100 for ~$1), then `load_full` — which peaks at
**exactly** the weight size (verified upstream on the 4B: 8.4 GB for 8.4 GB). After that the 20.8 GiB row
above is the budget, and it fits at 32k with ~1.7 GiB to spare. There is a cheaper escape hatch that does
not need a second machine: **GGUF**. `ggml-org/Kev-9B-GGUF` Q4_K_M (6.0 GB) bakes the pointer head into
the file and runs in llama.cpp 0.7.0's native `kev` decision type — measured here, 0.680 accuracy and
7.2 GB peak RSS, no fold.

### Quantizability: there is **no** 4/6/8-bit Kev

`MLX_DTYPES = {"bfloat16": …, "float32": …}`; `load_full` hard-fails on any tensor that is not the
config's dtype; `merge_lora` merges into fp weights. **bf16 is the only Kev dtype on Metal**, and the
pointer head (≈1.1M params, ~4 MB fp32, `PointerHead` runs on torch CPU) is far too small to be worth
quantizing — the head is not the quant risk; the *backbone* is, and it has no quant path at all.
Clef is the opposite: Cloudflare's `mlx-community` build quantizes only the backbone
(`mlx_vlm.convert -q --q-bits 4`, vision tower in bf16) and copies `joint_head.safetensors` **unchanged in
bf16** — the head is 243 MB of the 6.21 GB, i.e. 3.9%.

### Which precision is the realistic default

| tier | default | why |
|---|---|---|
| encoders (≤434M) | **fp32/bf16 as shipped** | 273 MB - 1.76 GB; quantizing buys nothing |
| 0.8-4B heads (Kev) | **bf16** | only supported dtype; 8.2-10 GiB total |
| 9B with a joint head (Clef) | **8-bit** if you want published parity, **4-bit** for headroom/latency | both fit; 4-bit costs ~1 DI point (55.63 → 54.65 on identical rows) |
| 9B pointer-head (Kev/Jeeves/Open-Jev) | **bf16** (fp8 for Jeeves only) | no quant path exists |
| 6-bit | **not a default anywhere** | no published 6-bit decision model; only a hand `mlx_lm.convert` with no parity data |

---

## 4. Ranking — runnable × score × latency × license

Score axis = DI where published; latency = published macOS number (M5 Pro/Max ⇒ M5 base is slower);
license gate = Apache-2.0/MIT pass, LFM Open v1.0 conditional, unstated = fail.

| # | Model | run here | score | latency on Apple | license | verdict |
|---|---|---|---|---|---|---|
| **1** | `mlx-community/clef-flash-4bit` | **yes, today** (`clef_mlx.py`, mlx-vlm 0.7.4-0.7.x) | **54.65** (2k sample) | 311 ms median / 0.31 s @1k / 7.0 s @16k, peak 7.0-8.6 GB (M5 **Max**) | Apache-2.0 | **the 9B that runs.** M5 base: expect 3-5× the long-prompt times. Only config with a published quant-vs-bf16 parity study |
| **2** | `LiquidAI/d1-3B` | **yes** (transformers ≥5.14, `device="mps"`) | **48.57** DI 0.2.1 | **30 ms**/question, 41 ms/3 questions, 640 ms @3.4k tokens (M5 **Pro**, vendor) | **LFM Open v1.0** — not OSI; check terms | best score-per-GB on the box; multimodal; license is the only reason it isn't #1 |
| **3** | `jaredpalmer/kev-4b` | **yes**, measured on this SoC | 38.0 (breadth-v1) | **721 ms new / 136 ms cached** (5 questions, ~270 tok, M5 32 GB, vendor) | Apache-2.0 | the reference row: only model with a published M5-32GB number *and* a server you can point a Jev SDK at |
| **4** | `PostHog/jeeves-fp8` | **yes, tight** (MPS fp8 + reduced caches) | 0.896 JevBench / 0.905 own test (vendor) | 0.3-0.5 s nothink (M4 Pro); 1.2 s median with thinking (H100) | MIT + Apache-2.0 | highest measured 9B accuracy that still fits 32 GB — but 1.2 s is not System-One latency |
| **5** | `bespokelabs/Bespoke-Nimble-9B` (v4) | **probably** (Metal `ParallelScorer`; `ollama pull nimble` for v2) | **63.70** DI 0.3 (vendor, own run) | not published for Apple `[UNVERIFIED]` | **not stated ⇒ needs a decision** | best open score in the class; blocked on license, not hardware |
| — | `jaredpalmer/kev-9b` | **no** (merge peak ≈37 GB) | 41.0 | not measured on any Mac | Apache-2.0 | only after a pre-merge; do it once, then it fits at 32k |
| — | `jaredpalmer/kev-0.8b` | **yes** | 23.3 | 149 ms / 28 ms cached (M5 32 GB) | Apache-2.0 | use as the harness smoke test, not a candidate |
| — | SemIf + `Qwen/Qwen3.5-9B` | **yes** (MLX backend) | no DI; 4B = 0.813 balanced acc | torch/MPS/MLX/llama.cpp | MIT + Apache-2.0 | the *only* way to get a 9B decision score out of llama.cpp on this box — worth one row for engine A/B |

Latency honesty: the only Apple numbers published for #1, #2 and #4 are on **M5 Max / M5 Pro**. On an
M5 base (153.6 GB/s, 10-core GPU) treat 16k-token prefills as multi-second and 2k-4k prefills as
hundreds of ms. Kev-4B's 721 ms is the one figure measured on a 32 GB base M5.

---

## 5. Minimal benchmark set

`HF_TOKEN` from `.env` (see [`models/README.md`](../../models/README.md)); downloads land in the Hub
cache (`~/.cache/hf/hub`) and are referenced by repo id. Kev's `--include` list is the same
allow-pattern set `kev/checkpoint.py` uses for `snapshot_download`.

```bash
. benchmarks/common/env.sh   # HF_HOME=~/.cache/hf, so downloads land where the resolver looks

# --- Tier A1: everything ≤3B + encoders — 18.0 GB, all runnable today -----------------
hf download mlx-community/clef-flash-4bit \
  --include "*.json" "*.safetensors" "clef_mlx.py" "*.jinja"          # 6.21 GB
hf download LiquidAI/d1-3B \
  --include "*.json" "*.safetensors" "*.py" "*.jinja"                 # 6.27 GB
hf download fastino/GLiNER2.5-Decide \
  --include "*.json" "*.safetensors" "*.md"                           # 1.96 GB
hf download com-kotobalabs/open-jev-deberta-v3-large \
  --include "*.json" "*.safetensors" "*.model" "*.py"                 # 1.76 GB
hf download jaredpalmer/kev-0.8b \
  --include "*.json" "*.safetensors" "*.pt" "*.txt" "*.jinja"         # 0.07 GB
hf download Qwen/Qwen3.5-0.8B-Base \
  --include "*.json" "*.safetensors" "*.txt"                          # ~1.7 GB [UNVERIFIED]

# --- Tier A2: add the 4B pointer-head model — +8.56 GB (running total 26.5 GB) --------
hf download jaredpalmer/kev-4b \
  --include "*.json" "*.safetensors" "*.pt" "*.txt" "*.jinja"         # 0.16 GB
hf download Qwen/Qwen3.5-4B-Base \
  --include "*.json" "*.safetensors" "*.txt"                          # ~8.4 GB (Kev's full-bf16 build measures 8.4 GB)

# --- Tier B: 9B, only if you accept the merge cost / license — +31 GB ----------------
hf download Qwen/Qwen3.5-9B-Base \
  --include "*.json" "*.safetensors" "*.txt"                          # 19.33 GB
hf download jaredpalmer/kev-9b \
  --include "*.json" "*.safetensors" "*.pt" "*.txt" "*.jinja"         # 0.20 GB
hf download PostHog/jeeves-fp8 \
  --include "*.json" "*.safetensors"                                  # 11.5 GB [UNVERIFIED]
```

| tier | GB | configs it unlocks |
|---|---|---|
| A1 | **18.0** | clef-flash-4bit, d1-3B, GLiNER, DeBERTa, Kev-0.8B (5 configs, 3 engines) |
| +A2 | **26.5** | + Kev-4B, SemIf over Qwen3.5-4B |
| +B | **57.5** | + a pre-merged Kev-9B, SemIf over Qwen3.5-9B, Jeeves fp8 |

Do **not** download `Cloudflare/clef-flash` (19.08 GB) for benchmarking — `mlx-community/clef-flash-4bit`
is a byte-for-byte conversion of it with a published parity table; fetch the bf16 only to re-derive the
quant delta. Skip `Cloudflare/clef` (54.99 GB), `akhilaaa3/Jev-Omni` (23.98 GB, doesn't fit), and
`togethercomputer/Tev1-4B-experimental` (9.34 GB, license unresolved).

---

## 6. Contradictions and gaps

**Contradictions, recorded not resolved**

1. **Kev-9B memory.** `jaredpalmer/kev` says "Kev-9B needs about 17 GB of GPU memory", but
   `Qwen/Qwen3.5-9B-Base` is 19.33 GB of bf16 safetensors. Either the 17 GB is a quantized/downcast
   measurement or it excludes buffers. Trust the file listing.
2. **Anarkali size: 68M vs 150M.** `awesome-system-one` (checked 2026-10-02) says 68M; the repo README
   now trains `jhu-clsp/ettin-encoder-150m` and keeps `release-150m/` as "the earlier general-purpose
   model". The README also says **"No web-decision results are published yet"**, so the 74.0%-vs-Jev-72.7%
   figure in the index is from the earlier general model, not the web model.
3. **Jev's Decision Index has four values**: 54.0 (Kev breadth-v1), 57.07 (Clef's 0.2.1 full suite),
   57.96 (Bespoke's DI view), 60.1 (a second Bespoke view). Same model, different suite revisions/views.
   Never put two of these in one column.
4. **Kev-9B's 41.0 and Clef's per-benchmark "Kev 9B" column are different suites.** Clef's table is DI
   0.2.1 per-benchmark; Kev's 41.0 is chance-corrected breadth-v1. Kev looks like 94.5 on BFCL and 25.0 on
   the home-appliance simulator in Clef's run — that is a *different model generation* (Kev-9B v1 vs v2).
5. **d1-3B license.** `license: other`, `license_name: lfm1.0`. Apache-2.0 it is not; treat as
   commercial-restricted until the LFM Open License text is read.
6. **Clef-Flash CLINC150+OOS = 66.8 vs Clef 97.4**, the largest single-benchmark gap in that table —
   suspect the 9B's head is weaker on many-option fine-grained classification rather than on tool routing.
7. **Clef's 16k trained limit vs its hosted "24,576-token context".** The weights' `max_length` is 16,384;
   the extra 8k on Workers AI is either a different build or truncation. On-device: 16,384.

**Missing / unverified**

- Exact byte size of `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-4B-Base` (≈8.4 GB inferred from Kev's own
  full-bf16 measurement), `mlx-community/clef-flash-8bit`, `PostHog/jeeves-fp8`,
  `bespokelabs/Bespoke-Nimble-9B`, `ZefanCai/Open-Jev-9B`, `convaiinnovations/laya`, and the Jev-Style v3
  repo id (the site names v3 but I have no HF id) — **`[UNVERIFIED]`, do not quote these as file sizes.**
- `Bespoke-Nimble-9B` license is genuinely unstated in the repo → **cannot be ranked above #5** and should
  not be downloaded into the cache until resolved.
- No third-party latency for **any** decision model on an M5 (base). Every macOS number in circulation is
  vendor-run on M5 Pro/Max, except Kev-4B/0.8B.
- No published 6-bit or 8-bit parity study for Kev (no path) or for d1-3B (no quant build found).
- `glm`-side: whether `llama.cpp` in `src/` actually carries `/v1/systemone` (PR ggml-org/llama.cpp#29831)
  is **stream A's** call; I did not verify it, and the GGUF rows above assume it lands.
- Brier/ECE per model: only Kev publishes Brier (0.289 / 0.217 for 9B on new/trained sources) and ECE
  (0.041 after temperature). Clef publishes a ForecastBench Brier column (10.6) per model. d1-3B, Nimble,
  Jeeves and Open-Jev publish accuracy, not Brier — **a calibration comparison across families is not
  currently possible from public material.**

## 7. What to run first

1. `clef-flash-4bit` vs `d1-3B` vs `kev-4b` — three read-out families, 20.8 GB total, all runnable today.
2. Only then pre-merge Kev-9B (one H100-minute, or 64 GB box) and add it as a fourth row; that row is the
   only direct test of H5 (4B vs 9B ≈ 3 index points for 4× the memory) on this hardware.
3. Add `SemIf + Qwen3.5-9B` in llama.cpp as the engine-portability control, not as a quality candidate.
