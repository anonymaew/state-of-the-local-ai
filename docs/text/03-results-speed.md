# Results — speed (measured on this machine)

All numbers measured **on the target box**, not derived:
MacBook Air Mac17,3 · Apple M5 (4P+6E, 10-core GPU, Metal 4) · 32 GB · macOS 27.0.1 (26A434)
· 153.6 GB/s memory bandwidth.

| Component | Version |
|---|---|
| llama.cpp | **v0.7.0**, commit `620d222`, built from source, `GGML_METAL=ON` |
| MLX | mlx **0.32.4**, mlx-lm **0.32.2** |
| lm-eval | 0.4.13 |

The Metal 4 tensor-API probe **passes** on this build (`MTLGPUFamilyMetal4 (5002)`,
`simdgroup matrix mul. = true`, no "tensor API is not supported" warning), so the M5 GPU neural
accelerators are active. That is the key version gate on M5: builds before `b10734` silently disable
the tensor API.

## 0. Clean-machine revision — read this first

The machine was later cleared (an unrelated VM stopped, swap went **15.4 GB → 0**, load average
**10.4 → 1.3**). The whole llama.cpp matrix was re-run. **These supersede the loaded-window numbers
in the sections below**, which are kept for comparison:

| Config (Qwen3.8-27B Q4_K_M, llama.cpp v0.7.0, Metal) | loaded window | **clean machine** | delta |
|---|---|---|---|
| pp512, f16 KV | 138.5 | **152.8** | +10 % |
| pp512, q8_0 KV | 120.5 | **132.2** | +10 % |
| pp512, q4_0 KV | 115.1 | **109.8** | −5 % |
| tg128, f16 / q8_0 / q4_0 KV | 7.1 / 6.6 / 6.0 | **7.0 / 6.2 / 5.6** | ~parity |
| pp512 @ 4k depth | 96.5 | **94.0** | ~parity |
| pp512 @ 16k depth | 78.9 | **90.9** | +15 % |
| tg128 @ 16k depth | 4.3 | **5.1** | +19 % |
| pp512 @ 32k depth | 65.1 | **71.4** | +10 % |
| tg128 @ 32k depth | 4.1 | **4.3** | +5 % |
| pp2048 | 74.2 | **96.0** | +29 % |
| **end-to-end, no spec** (llama-server, 128 gen) | 7.25 | **7.40** | +2 % |
| **end-to-end, MTP n=3** | 11.05 | **13.90** | **+26 %** |

**Two things this changes:**

1. **The headline decode number is 13.9 t/s, not 11.1.** With MTP the clean machine is **+88 % over its
own no-spec baseline** (7.40 → 13.90), not the +53 % the loaded window showed. Draft acceptance is
identical in both runs (63.1 %, mean draft length 2.86), so this is purely the machine being able to
execute the verify step at full speed. Effective bandwidth at 13.9 t/s × 16.46 GB ≈ **229 GB/s**, i.e.
1.5× the physical bus — the clearest evidence that MTP is bypassing the memory wall rather than
approaching it.
2. **The load sensitivity is asymmetric, and that is diagnostic:** decode barely moved (±10 %, and it
is the bandwidth-bound phase) while **prefill moved up to +29 %** (the compute-bound phase). Contention
steals GPU compute, not memory bandwidth. So *prefill numbers are the ones to distrust from a busy
machine*; decode numbers survive. The earlier 2.4-8.0 t/s (3.4×) spread reported by a parallel test
was therefore an outlier condition, not a general property.

The sections below retain the loaded-window values with their variance, because the *rankings* — and
the reasoning about bandwidth ceilings, KV cost and engine crossover — are the same in both datasets.

**Where this leaves the ranking.** Within llama.cpp, the clean numbers above are final. But a
then-untested config later beat them: **oMLX 0.7.1 + `oQ4e-mtp` + its own MTP implementation measured
15.22 t/s decode vs llama.cpp+MTP's 12.33 in a paired, interleaved A/B** (ratio 1.206/1.236/1.260),
for +3.3 % perplexity. The engine and the quant are both exonerated — at identical weights oMLX and
mlx-lm tie (0.98), and `oQ4e` is within 0.3 % PPL of MLX-4bit, so the entire gain is the MTP
implementation (75.2 % vs 63.1 % draft acceptance). See [`06-omlx.md`](06-omlx.md) §"Round 4" and
[`05-recommendation.md`](05-recommendation.md) for the two-way recommendation.

## Headline table (loaded-window run)

| # | Engine | Model | Weights | Prefill | Decode |
|---|---|---|---|---|---|
| A | llama.cpp | Qwen3.8-27B **Q4_K_M**, f16 KV | 16.46 GB | **138 t/s** (pp512) | **7.1 t/s** (tg128) |
| B | llama.cpp | same, q8_0 KV | 16.46 GB | 120 t/s | 6.6 t/s |
| C | llama.cpp | same, q4_0 KV | 16.46 GB | 115 t/s | 6.0 t/s |
| D | llama.cpp + **MTP n=3** | same, q8_0 KV | 16.46 + 1.68 GB | (see §3) | **11.1 t/s** |
| E | MLX | Qwen3.8-27B **4bit** | 16.08 GB | 122 t/s | 5.7 t/s |
| F | MLX | Qwen3.6-**35B-A3B** 4bit | 20.40 GB | **881 t/s** | **56.9 t/s** |

**The number that decides everything: 7.1 vs 56.9 tok/s.** An 8× gap between a dense 27B and a
3B-active MoE — and it is not an engine bug, it is the arithmetic of bytes-read-per-token.

## 1. Decode vs context depth (the number users feel)

`llama-bench -d N` prefills the KV cache to N tokens, then decodes. Same model, q8_0 KV:

| Depth | prefill (pp512) | decode (tg128) | decode vs depth-0 |
|---|---|---|---|
| 0 | 120.5 t/s | **6.6 t/s** | — |
| 4 096 | 96.5 | **4.8** | −27 % |
| 16 384 | 78.9 | **4.3** | −35 % |
| 32 768 | 65.1 | **4.1** | −38 % |

MLX, same weights (4-bit), random-token prompts via `mlx_lm.benchmark`:

| Prompt (= KV depth) | prefill | decode | peak memory |
|---|---|---|---|
| 512 | 122.3 t/s | 5.75 t/s | 16.27 GB |
| 4 096 | 125.2 | 5.81 | 18.27 GB |
| 16 384 | 132.3 | **5.70** | 19.07 GB |

**The most interesting engine finding of the exercise — the two engines cross over:**

- **short context: llama.cpp wins** (7.1 vs 5.7, +25 %) — better short-context decode kernels;
- **16k+: MLX wins** (5.7 vs 4.3, +33 %) — because llama.cpp decode *degrades 38 %* with depth while
  MLX is flat.

Qwen3.8-27B is not a vanilla transformer: 48 **Gated DeltaNet** layers + only 16 full-attention
layers (`full_attention_interval: 4`, verified in `config.json`). llama.cpp's Metal kernels handle
that recurrent state update poorly as context grows. Practical rule: **long-context/agent work → MLX;
short chat → llama.cpp.**

## 2. Speculative decoding — the only way past the bandwidth wall

Dense-27B decode is bandwidth-bound: `153.6 GB/s ÷ 16.46 GB ≈ 9.3 tok/s` theoretical ceiling.
Baseline measures 7.1 = **76 % of the physical ceiling**. Nothing pushes past that ceiling except
*not reading all the weights for every token* — which is what speculative decoding does.

`llama-server`, 128 generated tokens, `temperature 0`, `cache_prompt: false`, q8_0 KV:

| Config | Decode | vs baseline | Notes |
|---|---|---|---|
| none | 7.25 t/s | — | |
| `--spec-type draft-mtp --spec-draft-n-max 2` | 9.80 | **+35 %** | |
| `--spec-type draft-mtp --spec-draft-n-max 3` | **11.05** | **+52 %** | acceptance 0.631, mean draft len 2.86 |
| `--spec-type draft-mtp --spec-draft-n-max 4` | **11.10** | +53 % | no gain past 3 |
| `--spec-type draft-dflash` (DFlash drafter) | 9.60 | +32 % | worse than MTP |
| `--spec-type ngram-mod` (draftless) | 6.2 – 19.75 | −14 % … **+172 %** | workload-dependent, see below |

**MTP wins and costs almost nothing to enable**: the `mtp-Qwen3.8-27B-Q4_0.gguf` head is 1.68 GB and
lifts decode 7.1 → 11.1 tok/s. Effective bandwidth at 11.1 tok/s is **183 GB/s — above the memory bus
peak**, which is the proof the ceiling was routed around rather than approached.

MLX **cannot** do this: `mlx_lm/models/qwen3_5.py` explicitly drops `mtp.*` tensors on load. MTP is a
llama.cpp-only (or mlx-vlm-only) win on this model.

**`ngram-mod` is a trap and a gift.** Needs no drafter (16 MB) and hit **19.75 t/s** — but the server
log showed `draft acceptance = 1.00000 (115 accepted / 115 generated)`: the output was copying
n-grams out of the prompt. On a fresh prompt requiring novel prose it measured **6.2 t/s, 14 % slower
than baseline**. Enable for code editing / quoting / document rewrite; leave off for free-form
generation.

## 3. Prefill is a different story

| Config | Prefill |
|---|---|
| Dense 27B, llama.cpp pp512 | 120 – 138 t/s |
| Dense 27B, MLX (512) | 122 t/s |
| Dense 27B, MLX (16k prompt) | 132 t/s |
| **MoE 35B-A3B, MLX (4k prompt)** | **1 071 t/s** |

The dense model sits at ~130 t/s prefill on both engines — the 153 GB/s bus limits prefill too, since
every prompt token still needs a full weight sweep. The MoE does 8× better because only ~3B of 35B
parameters are touched per token.

(An M5 *Max* reaches ~562 t/s prefill on this model class. llama.cpp's Metal backend is reported at
~60 % of MLX-derived engines for prefill on M5 due to unused NAX matmul paths; not dramatic here
because this box is bandwidth-bound, not compute-bound.)

## 4. Memory — the real 32 GB constraint

Measured with `mx.get_peak_memory()` (MLX):

| Config | Weights | Peak @512 | Peak @4k | Peak @16k | Headroom @16k |
|---|---|---|---|---|---|
| Qwen3.8-27B MLX 4bit | 16.08 GB | 16.27 GB | 18.27 GB | **19.07 GB** | 12.9 GB |
| Qwen3.6-35B-A3B MLX 4bit | 20.40 GB | 20.20 GB | 21.30 GB | ~22 GB | ~10 GB |

KV cost, derived from `config.json` rather than assumed:
`2 × 16 full-attn layers × 4 KV heads × 256 head_dim × 2 B = 64 KiB/token` (fp16) = **1 GiB per 16k
tokens**, 2 GiB at 32k, halved with q8_0. The 48 linear-attention layers hold a *fixed* recurrent
state instead of a growing cache — that is why a 27B at 32k context fits at all.

### Wired memory, measured directly (the number that decides whether it fits)

`vm_stat` "Pages wired down", before vs after loading the recommended config:

| State | Wired |
|---|---|
| idle (this box, other apps running) | 4 GiB |
| + Qwen3.8-27B Q4_K_M + MTP head + **32k ctx**, q8_0 KV | **22 GiB** |
| **delta attributable to the config** | **+17 GiB** |

The Metal wired ceiling is ~75 % of RAM ≈ **24.6 GiB**, so the recommended config sits at ~69 % of the
ceiling with ~2.5 GiB of wired headroom. Comfortable — but the box was already carrying **14.9 GiB of
swap in use** from other applications, which is what makes a 32 GB machine feel tight long before the
model does. Close other apps before a long run.

**Do not use `/usr/bin/time -l` max RSS for GGUF fit decisions:** it reported 6.4 GB for a 16.46 GB
model (mmap + Metal buffers are not attributed to RSS). Use `mx.get_peak_memory()` for MLX and
`vm_stat` wired pages for llama.cpp.

## 5. Efficiency against the physics

| Config | Bytes read / token | Theoretical ceiling | Measured | % of ceiling |
|---|---|---|---|---|
| Dense 27B Q4_K_M | ~16.5 GB | 9.3 t/s | 7.1 | **76 %** |
| Dense 27B + MTP n=3 | ~16.5 GB (amortized) | 9.3 (moot) | 11.1 | **119 %** |
| MoE 35B-A3B 4bit | ~1.7 GB (3B active) | ~90 t/s | 56.9 | 63 % |

## 6. Fairness caveats

### Load sensitivity — measured, and worse than expected

Every number above was taken on a lightly loaded machine between 15:25 and 18:00 local. Later, with an
unrelated `Virtualization.framework` VM at ~20 % CPU, load average 3.6-7.6 on 10 cores, and swap at
**15.4 of 16 GB**, the identical anchor config re-measured as:

| Run | pp512 | tg128 | load avg |
|---|---|---|---|
| earlier (all tables above) | 120.5 t/s | 6.6 t/s | < 2 |
| re-run under load | **105.6 ± 8.5** (−12 %) | **4.80 ± 0.55 (−27 %)** | 3.6→2.6 |

Two consequences:

1. **Treat every decode number here as a best-case figure for a quiet machine.** On a box with a VM or
   a full swap file, expect ~25 % less. Note also that the ±8.5 / ±0.55 spreads are *sqrt-of-variance*
   from llama-bench on 2 reps, i.e. **variance tripled under load** as well as the mean dropping.
2. **Any engine-vs-engine comparison made on this machine must be paired and interleaved**, never two
   sequential runs — otherwise the drift is larger than the effect being measured. A parallel oMLX
   test on this box saw the *identical* config span 2.4-8.0 t/s (3.4×) during the loaded window.

The direction of the bias is reassuring for the headline claim: contention makes all engines slower
roughly together, and the 7.1 vs 5.7 (dense) and 56.9 (MoE) gaps are far larger than the ±25 % drift.

### Other caveats

- **Variance.** Repeated runs move ±8 % on prefill (e.g. `pp2048` = 74 t/s in one pass, `pp8192` = 84
  t/s in the next — not monotonic). Decode was stable to ±3 %.
- **Thermals.** This is a **fanless** MacBook Air and every test was ≤60 s. These are *burst* numbers;
  sustained multi-minute generation will clock down. No dedicated throttle run was done.
- **Swap.** The machine carried ~11-15 GB of swap in use from other work during testing. Per-run swap
  deltas were 0-184 MiB; the 184 MiB case is flagged rather than hidden.
- **`llama-bench`** excludes tokenisation and sampling. The `llama-server` end-to-end numbers
  (7.25 baseline / 11.05 MTP) agree with `llama-bench` within 3 %, so that is not a source of bias.
- Raw rows: `results/text/speed_llamacpp.jsonl`, `results/text/speed_mlx.jsonl`, `results/text/server.jsonl`,
  `results/text/spec.jsonl`. Scripts: `benchmarks/text/bench_speed_gguf.sh`, `benchmarks/text/bench_mlx.py`,
  `benchmarks/text/bench_server.sh`.
