# oMLX on M5 / 32 GB

**Verdict (5 lines).**
1. **oMLX 0.7.1 runs correctly** on this M5/32 GB machine and is OpenAI-compatible on
   `/v1/completions` with plain prompts (no chat template). It emits **no `timings` object**,
   so every number here is external wall-clock.
2. **Best observed, identical MLX-4bit weights: oMLX 8.00 t/s vs mlx-lm 5.75 t/s (+39%).**
   But that 8.00 came from one quiet window and **I could not reproduce it**: in a same-session,
   same-weights, back-to-back test oMLX scored **3.76** vs mlx-lm **3.77** — a **tie**.
   So *"is oMLX faster than mlx-lm at identical weights?" is **inconclusive**, not yes.*
3. **The decisive experiment (C, oQ4e-mtp) is not-measured.** The 4 weight files (16.97 GB)
   never downloaded — `models/.partial-oq4e-27b` (the then-flat weights directory) contains **0 of 4 `.safetensors`**. So
   *"does oQ4e-mtp beat llama.cpp+MTP's 11.05 t/s?" is **unanswered***.
4. **No recommendation change.** oMLX has not been shown to beat llama.cpp+MTP (11.05), nor to
   reliably beat mlx-lm. Nothing here justifies moving the project's primary recommendation.
5. **The blocker was the measurement environment, not oMLX.** External load (a VM, CacheDelete,
   Spotlight — no workloads of mine) put load average anywhere in **2.9–39.2** on 10 cores and
   grew swap from 11.4 to 15.7 GB during the session, which swamped the ~40% engine effect I was
   trying to resolve.

## What worked

```bash
# discovered models from subdirectories; --no-cache matches the "cache disabled" baseline.
# At the time models/ was a flat checkpoint directory; weights now live in the Hub cache, so
# point --model-dir at the snapshot parent.
. benchmarks/common/env.sh
OMLX_DIR="$(dirname "$(model_path qwen38-27b-oq4e-mtp)")"
.venv-omlx/bin/omlx serve --model-dir "$OMLX_DIR" --port 8000 \
  --no-cache --memory-guard off --base-path logs/text/omlx/a --headless
```

- `omlx serve` takes **`--model-dir`**, *not* a model path (the main `--help` example
  `omlx serve <model-path>` is misleading; `serve --help` is authoritative). Model is chosen
  per-request via the JSON `model` field.
- `/health` and `/v1/models` answered within **1–3 s** of start on every attempt. No crashes.
- `/v1/completions` accepted `{prompt, max_tokens, temperature:0, cache_prompt:false, stream}`
  and returned correct `usage`. `cache_prompt` is accepted (not rejected).
- Useful telemetry: **`/v1/models/status` → `current_model_memory`** (bytes) is the only reliable
  memory figure (see Caveats).
- Harnesses (all inside `logs/text/`, per the write constraints): `logs/text/omlx/bench.py` (non-streaming),
  `logs/text/omlx/bench_stream.py` (SSE → TTFT + inter-token), `logs/text/omlx/bench_concurrent.py` (exp E),
  `logs/text/omlx/control_mlxlm.py` (same-load mlx-lm control).

## Results

Median of 2–3 reps per config; first request after startup discarded in every case.

| config | quant | decode t/s | prefill t/s | TTFT ms | peak mem GB | valid | measurement method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (measured earlier, different session) — llama.cpp v0.7.0 | Q4_K_M | 7.25 | 138 (pp512) | not-measured | not-measured | yes | server timings |
| baseline (measured earlier, different session) — llama.cpp + `--spec-type draft-mtp` | Q4_K_M + MTP | 11.05 | 138 (pp512) | not-measured | not-measured | yes | server timings |
| baseline (measured earlier, different session) — mlx-lm | MLX 4-bit | 5.75 | 122 | not-measured | 16.27 | yes | internal timings |
| **A: oMLX, same MLX-4bit weights — best observed** | MLX 4-bit | **8.00** ¹ | n/m ² | 508 | 16.86 | yes (load not sampled) | wall-clock, streaming |
| A: oMLX, non-streaming cross-check | MLX 4-bit | 7.00 ³ | n/m | n/m | 16.86 | yes | wall-clock, `usage` field |
| **A: oMLX, matched-load, back-to-back** | MLX 4-bit | **3.76** | 22.4 ² | 1115 | 16.86 | yes (load 11.6) | wall-clock, streaming |
| **F/G: mlx-lm, same-session control, same load** | MLX 4-bit | **3.77** ⁴ | not-measured | not-measured | 15.43 | yes (load 7.5–12) | wall-clock around `generate` |
| **D: oMLX prefill at depth (4394 tok)** | MLX 4-bit | 7.19 | **131.6** | **33786** | 16.86 | yes | wall-clock, streaming |
| **E: oMLX, 4 concurrent streams (aggregate)** | MLX 4-bit | **20.27** agg (5.07/stream) | not-measured | 910–2499 | 16.86 | yes | wall-clock, concurrent SSE |
| **C: oMLX on oQ4e-mtp** | oQ4e-mtp | not-measured | not-measured | not-measured | not-measured | **NO** | weights absent (0/4 safetensors) |
| A-ablation: oMLX burst-decode **off** | MLX 4-bit | 3.9–6.7 (median 5.4) | n/m | 590–1127 | 16.86 | **no** — load-uncontrolled | wall-clock, streaming |

¹ median of 2 reps: 8.03, 7.98 (one run, quiet window, load not sampled).
² TTFT-implied prefill on a **25-token** prompt is dominated by fixed overhead — **not a real
prefill figure**; use row D. ³ median of 2: 7.04, 6.97. ⁴ includes prefill, so pure decode ≈3.88.
`n/m` = not-measured. oMLX's marketing figure (299.5/18.2) is **deliberately excluded** from this
table; it is an external, unverified self-report.

## Engine vs quant: which explains the difference?

The experiment was designed to hold weights constant (A) and then change only quant+MTP (C), so
the two causes could be separated. **Only the engine half produced data, and even that is
inconclusive.**

**Engine — measured, but not reproducible.** A and the mlx-lm baseline load *bit-identical* weights
(`mlx-community/Qwen3.8-27B-4bit`, 3 safetensors, 16.39 GB actual; since removed), so any difference is 100% engine.
The first, quiet-window measurement showed **8.00 vs 5.75 = +39%**. That is a real observation, and
because oMLX's figure is external wall-clock while mlx-lm's 5.75 is an internal timer, the true
margin would if anything be larger. **However**, when I re-ran the two engines back-to-back under a
matched, recorded system load, they **tied: 3.76 vs 3.77** (and a separate same-load mlx-lm control
gave 3.59–3.78). So the +39% is **a best-case point estimate, not a reproducible effect** at the
95%-worth-trusting level. Reported honestly: **engine effect = somewhere between 0% and +39%,
most likely small; not resolved.**

**Why the tie is plausible, and a hypothesis (not measured).** oMLX is a *server*: uvicorn + a
scheduler + per-request Python orchestration on every decode step. On a quiet machine its
batching wins; on a CPU-contended machine that per-step Python overhead is exactly what gets
starved, while mlx-lm's tight single process degrades less. This would explain a win collapsing to
a tie as load rose, and it is testable (pin CPU, re-run) — I did **not** test it.

**Burst decode — ablated, inconclusive.** oMLX batches decode steps per GPU submission
(`burst_decode_mode`, default `balanced`; 64 steps / 0.1 s). It is *not* speculative decoding and
changes no output. It is directly observable: a 128-token completion arrives in **65 SSE chunks**
with bursting on vs **128 chunks** with it off. But its *throughput* effect was **not
established** — across all replications burst-off scored 3.9–6.7 (**median 5.38, n=4**) and
burst-on 2.4–8.0 (**median 4.46, n=12**), i.e. **the noise band is wider than the effect**.
Earlier I measured 8.00 (on) vs 3.99 (off) and then 6.69 (off) vs 4.10 (on) — the sign **flipped**.
I therefore **retract** any claim that burst decoding explains the gap.

**Quant + MTP head — not-measured.** With the oQ4e-mtp weights absent I cannot attribute any
speedup to the oQ4e quantisation or to the 29-tensor built-in MTP head. I confirmed from
`model.safetensors.index.json` that those tensors exist (`language_model.mtp.*`) and that **4**
weight files totalling **16.97 GB** are required — **none are on disk**. So the quant+MTP share of
any difference is **unknown**, and the 11.05 t/s MTP bar is **untested** by oMLX.

**Net attribution:** engine = 0 to +39% (inconclusive); quant+MTP = unknown. Neither is
established, so neither can be credited.

## Caveats

- **Environment was not controlled — this is the dominant error source, not oMLX.**
  Load average was sampled between **2.9 and 39.2** on 10 cores (8.1 at session end); culprit
  processes were **not mine**:
  `Virtualization.framework` (a VM), `CacheDelete` (disk pressure), `Storage.appex`, Spotlight
  (`mdworker`/`mds`), `opendirectoryd`, `AssistantServices`. Swap rose **11.4 → 15.7 GB** (of
  17.4 GB) and free disk fell **38 → 24 GB** over the session. Repeats of the *identical*
  configuration spanned **2.4–8.0 t/s (3.4×)**. Absolute numbers are therefore unreliable and
  **must not be compared** to the earlier baselines, which the brief already labels
  "different session".
- **Swap rule.** Per the >500 MiB abort rule: individual runs moved swap by +405, +358, and
  −16 MB (no abort); the multi-GB growth was cumulative and driven by external workloads. Rows
  taken once load exceeded ~10 are marked invalid/qualified above.
- **Thermal.** This is a **fanless** MacBook Air; `ThermalThrottlingSupported = Yes`. Two long
  prefill runs degraded 145.6 → 117.5 t/s. `pmset -g therm` recorded **no** warning level, and
  after a 7-minute idle cooldown throughput got *worse* (2.38 t/s), so the drift is attributed to
  **system load, not throttling** — but throttling is **unconfirmed-excluded**, not excluded.
- **Timing method.** oMLX gives no `timings`, so decode is external wall-clock. Streaming numbers
  are inter-token (`(gen−1)/(t_last−t_first)`), which strips per-token HTTP overhead but makes
  burst-mode chunking a small bias (first/last chunk arrive up to 0.1 s late). Non-streaming
  7.00 t/s is the pessimistic view. Baselines used *internal* timers → cross-engine comparison is
  **slightly unfair to oMLX**, which does not rescue the tie.
- **Memory.** `ps -o rss` is **useless for MLX**: it read 883 MB (and 190 MB) because Metal
  allocations are not in RSS. The 16.86 GB is oMLX's own `/v1/models/status →
  current_model_memory` (actual_size 16.39 GB). `vm_stat` wired pages moved 5.65 → 22.43 GB at
  load then drifted back to 20.69 GB, so wired is noisy. mlx-lm's 15.43 GB here / 16.27 GB earlier
  came from `mx.get_peak_memory()`. **Different instruments** — treat the ~16–17 GB as agreement
  in magnitude only.
- **Prompt/params.** `benchmarks/text/prompts/prompt_short.txt` verbatim → **25 prompt tokens** (not 40 as the brief
  assumed). Plain completions, **no chat template**, temperature 0, `max_tokens` 128.
  Prefill-at-depth used a 4394-token repeated prompt and only **8** output tokens.
- **Single stream** for all decode rows except E. Continuous batching **does** work (4 streams →
  **20.27 t/s aggregate = 2.53×** a single stream, 5.07 t/s each) but that was one uncontested
  sample; llama.cpp with 4 slots would also batch, so **no cross-engine throughput claim** is made.
- **`--memory-guard off`** was used to match the unguarded mlx-lm baseline; the default `balanced`
  tier sets an 18.9 GB process ceiling that 16.86 GB peak would likely clear — **not-measured**.
  I never raised the Metal wired cap.
- No weights were downloaded by me; no disk-pressure abort was needed (never below 8 GB).

## Raw data

`results/text/omlx.jsonl` — 27 lines, one JSON object per rep. Labels: `exp-A-*` (engine, incl.
`burstoff`/`reverify`/`clean`/`final`/`pair` replications), `exp-B-mem-*` (memory record),
`exp-D-prefill4096`, `exp-E-concurrent4`, `exp-F/G-control/pair-mlxlm`. Each row carries
`label`, `rep`, `ttft_ms`, `n_chunks`, `prompt_tokens`, `measurement_method`, and wall-clock
derived rates. Full server logs in `logs/text/omlx/a/`.

---

## Paired re-test under load (round 2)

Session 2026-10-20 20:44–21:06 on the same M5 / 32 GB machine. The previous section's
weakness was that engine A and engine B were measured at *different* machine conditions while
external load swung 2.9–39.2. This round used the only design that survives that: **every
measurement of oMLX is bracketed by an mlx-lm measurement of the *same* weights, minutes apart,
with `uptime` and `vm.swapusage` recorded immediately before and after each individual rep.**
Both engines are timed the same way (**external wall clock, `time.perf_counter()` around the
`POST /v1/completions` call**) — so there is no timings-object asymmetry. `mlx_lm.server`
0.32.2 returns **no `timings` object at all** (verified: `usage` has only token counts), so
wall clock is in fact the only common instrument. 5 paired rounds were run (the brief asked
for 3; 2 were added because the first 3 stayed noisy).

### Verdict sentence

**No — at identical weights oMLX is not faster than mlx-lm: the paired median ratio is
`oMLX/mlx-lm = 0.895` (oMLX ≈10% slower), with 4 of 5 paired rounds at or below 1.01 and only
one round favouring oMLX.** The paired design does resolve the *direction* (it is not a tie in
oMLX's favour and the earlier +39% "best observed" does not replicate), but the *magnitude* is
still not resolved to better than ±20%: the individual ratios span 0.507–1.210 (2.4×), and the
round that produced 0.507 did so because mlx-lm got one anomalously fast rep (6.14 t/s vs its
own 2.86–3.98 in every other round), not because oMLX slowed down. Excluding the round whose
measurement straddled a load spike (round 5, load1 9.84→40.92), the median is 0.937 — still
"oMLX not faster", still short of a clean win for either engine.

### Paired rounds (identical weights: `mlx-community/Qwen3.8-27B-4bit`, prompt `benchmarks/text/prompts/prompt_short.txt`, `max_tokens` 128, temperature 0, 1 discard warmup + 1 measured rep per arm)

| round | mlx-lm t/s | oMLX t/s | ratio oMLX/mlx-lm | load1 mlx-lm (before→after) | load1 oMLX (before→after) | swap used MB (mlx-lm before→after / oMLX before→after) | valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.139 | 3.110 | **0.507** | 3.87 → 4.49 | 4.09 → 3.48 | 14915→14883 / 15949→15821 | yes |
| 2 | 3.308 | 4.004 | **1.210** | 2.70 → 3.06 | 3.28 → 2.70 | 15757→15669 / 16323→16219 | yes |
| 3 | 3.148 | 3.183 | **1.011** | 8.04 → 10.02 | 7.07 → 5.47 | 16107→16051 / 16659→16555 | yes |
| 4 | 3.979 | 3.433 | **0.863** | 3.15 → 3.27 | 9.33 → 6.61 | 14864→14792 / 16556→16396 | yes |
| 5 | 2.860 | 2.561 | **0.895** | 9.84 → 13.93 | 36.98 → 40.92 | 16292→16228 / 16395→16339 | **no** — oMLX rep straddled load1 >25 (abandon threshold) |
| **median (n=5)** | 3.308 | 3.183 | **0.895** | — | — | swap used 14,864–16,659 MiB throughout | |
| median (n=4, excl. round 5) | 3.644 | 3.308 | 0.937 | — | — | — | |

(Per-round ratios are computed within a round; the "median" rows report the *median of the
per-round ratios*, which is not the same thing as the ratio of the two median t/s columns.)

Spread of the ratios: **0.507 – 1.210 (2.39×)**. Per-round `decode_tps = completion_tokens /
elapsed`, `completion_tokens = 128` in every row (so no truncation asymmetry). Elapsed times:
mlx-lm 20.85/38.70/40.66/32.17/44.76 s, oMLX 41.16/31.97/40.21/37.29/49.99 s.

**Cross-check with oMLX's own timer (not compared against mlx-lm).** oMLX 0.7.1 *does* return
per-request timings, just not in a llama.cpp-style `timings` object — they are fields inside
`usage`: `generation_tokens_per_second`, `generation_duration`, `time_to_first_token`,
`total_time`. Those agree with my wall clock to within **~3%** (e.g. round 2: self-reported 4.14
t/s vs wall 4.00 t/s; round 5: 2.62 vs 2.56), which is exactly what you want from a sanity check
and confirms the wall-clock numbers are not being inflated by HTTP/queueing. Self-reported values:
3.19 / 4.14 / 3.28 / 3.55 / 2.62 t/s. (This corrects the earlier "no timings" statement — that
referred to a `timings` object; the fields exist in `usage`.)

### MTP mechanism

**The flag is not a CLI flag.** `omlx serve --help` exposes **no** `--mtp`, `--draft`,
`--num-draft*` or `--speculative*` option (full option list checked). MTP is a **per-model
setting** named **`mtp_enabled`** (bool), stored in `<base-path>/model_settings.json` and also
settable through the admin API (`PUT /api/models/{model_id}/settings`, field `mtp_enabled`).
Related knobs, same object: `mtp_adaptive_max_depth` (max draft depth; controller picks 1..max),
`mtp_fixed_depth` (fixed depth, overrides the controller), clamped by the in-package ceiling
`MAX_LIGHTNING_MTP_DRAFT_TOKENS = 8`. Sources grepped in
`.venv-omlx/lib/python3.12/site-packages/`: `omlx/model_settings.py`,
`omlx/utils/model_loading.py`, `omlx/engine_pool.py`, `omlx/patches/mlx_lm_mtp/batch_generator.py`,
`omlx/patches/mlx_vlm_mtp/gemma4_vlm_runtime.py`.

**How it was toggled for the A/B** (this is the command that worked; `--model-dir` is a
*directory of models*, not a model path — the CLI's own `--help` example is wrong):

```bash
# OFF: <base-path>/model_settings.json = {"version":1,"models":{"gemma4-e4b-oq4e-mtp":{"mtp_enabled":false}}}
# At the time models/ was flat; today point --model-dir at the snapshot parent:
#   . benchmarks/common/env.sh
#   OMLX_DIR="$(dirname "$(model_path Jundot/gemma-4-E4B-it-oQ4e-mtp)")"
.venv-omlx/bin/omlx serve --model-dir "$OMLX_DIR" --port 8104 \
  --no-cache --memory-guard off --base-path logs/text/omlx_paired/base-off2 --headless
# ON: same, but base-path logs/text/omlx_paired/base-on2 with "mtp_enabled":true
# one request: POST /v1/completions {"model":"gemma4-e4b-oq4e-mtp","prompt":<benchmarks/text/prompts/prompt_short.txt>,"max_tokens":128,"temperature":0}
```

A **fresh server process per arm** is required: `mtp_enabled` is read at model load and is fixed
for the life of the loaded model.

**Gotcha that cost one run (documented because it is a real product behaviour):** oMLX
**auto-enables MTP on model discovery** for any checkpoint that ships MTP weights —
`engine_pool.apply_settings_overrides` writes `mtp_enabled: true` as an *initial* setting, log
line: *"A new checkpoint with Lightning MTP weights also starts with MTP on."* My first
"off/on" attempt used an empty base-path for "off" and was therefore silently **ON in both arms**;
those 6 rows (`exp2-pair*`, 31.2–47.4 t/s wall) are kept in the JSONL but are **6 MTP-on reps,
not a paired test**. The numbers below come from the re-run (`exp2b-*`) with an explicit
`"mtp_enabled": false`. Gate confirmation from the server logs: OFF arms contain **zero** lines
matching `Speculative backend selected` / `MTP path activated` / `MTP[...] finish=`; ON arms
contain all three, and the sanitize/runtime patches are installed either way (they are needed
for weight-correctness even when MTP is idle).

| pair | MTP off t/s | MTP on t/s | speedup (on/off) | load1 off (before→after) | load1 on (before→after) |
| --- | --- | --- | --- | --- | --- |
| 1 | 20.759 | 40.885 | **1.970×** | 6.04 → 5.65 | 5.08 → 4.76 |
| 2 | 17.390 | 35.133 | **2.020×** | 4.81 → 4.82 | 4.62 → 4.62 |
| 3 | 15.387 | 34.596 | **2.248×** | 3.83 → 3.68 | 3.57 → 3.68 |
| **median** | 17.390 | 35.133 | **2.020×** | — | — |

Same prompt / `max_tokens` 128 / temperature 0 / wall clock, interleaved **off, on, off, on, off,
on** on `Jundot/gemma-4-E4B-it-oQ4e-mtp` (then a flat `models/` directory; oQ4e-mtp, 5.54 GB). Note both arms *drift down* over the
session (off 20.8→15.4, on 40.9→34.6 — thermal/load drift), which is exactly why the paired
ratio, not the absolute t/s, is the result.

**Draft acceptance rate — logged, not returned.** oMLX does not put acceptance in the HTTP
response; it writes a one-line summary to the server log at sequence finish:

```text
MTP[1] finish=length tokens=128 cycles=39 tok/cycle=3.28 accept=17/19 (89.5%)
  depth[d1=3/4,d2=2/3,d3=2/2,d4=2/2,d5=2/2,d6=2/2,d7=2/2,d8=2/2] copy=70/70 in 35
  emits[init=2,draft=87,bonus=37,verify=2]
  timing[backbone=2807.4ms mtp=78.1ms sample=3.7ms cache=0.5ms]
```

- **acceptance = 89.5%** (17 of 19 chained draft tokens accepted) — **identical in all 3 "on"
  pairs and in all 6 earlier MTP-on reps**; the run is deterministic at temperature 0.
- **3.28 tokens per verify cycle** (128 tokens / 39 cycles), with per-depth acceptance
  `d1=3/4, d2=2/3, d3..d8=2/2` — the drafted chain almost always lands 2 tokens once past the first
  two depths.
- The head itself is cheap: **~40–135 ms of MTP head time against 2.6–3.5 s of backbone time**
  per 128-token completion — the speedup comes from emitting ~3.3 tokens per backbone forward,
  not from a fast drafter.
- A second mechanism also ran: `copy=70/70 in 35` — a context-copy proposer that accepted 70 of
  70 proposed tokens, i.e. part of the win is repetition detection, not the neural head.

**Is oMLX's MTP a real, usable path on this machine? Yes.** It binds, it runs the
draft+verify cycle, it reports honest per-cycle telemetry, and it is worth a **~2.0× median
(1.97–2.25×) decode speedup measured paired and interleaved**. Two honest qualifications:

1. The completion text on this prompt is `***\n\n***…` — raw `/v1/completions` (no chat template)
   on an instruct VLM checkpoint degenerates into a highly predictable repetition. A
   **89.5% acceptance rate on repetitive output is a best case**; the `copy=70/70` proposer is
   *specialised* for exactly that. The MTP multiplier on real prose/chat is **not-measured** here
   and will be lower (the adaptive controller is documented to settle at depth 1 for prose).
2. This says nothing about the 27B oQ4e checkpoint: that download is still blocked (0 of 4
   shards present; oMLX itself logs *"Skipping incomplete model 'qwen38-27b-oq4e-mtp'"*), so
   "does oQ4e+MTP beat llama.cpp+MTP's 11.05 t/s" remains **not-measured**.

### Contamination note

- **Load average** (10 cores) during this session, sampled per rep: **2.64 → 40.92**, with the
  1-min average above 10 in **4 of the 20 before/after samples** (10.02, 36.98, 13.93, 40.92 — none
  of them at a round's *opening* sample except round 5's oMLX arm). Non-mine processes included
  `Virtualization.framework` / `VirtualMachine` (~20% CPU), `CacheDelete`, Spotlight
  (`mdworker`/`mds`) and `Storage.appex`. A
  guard ran before every measured request with the brief's thresholds (pause 60 s and retry once
  above load1 15; abandon above 25): **no rep hit >15 at guard time** (guard-time load1 was
  2.64–13.53), so no pause/retry was taken. The spike that did occur happened *inside* a
  measurement — round 5's oMLX arm entered at load1 13.53 and the load reached **36.98 → 40.92**
  during its warmup, so that rep is marked invalid above and is additionally reported via the n=4
  median.
- **Swap** was already saturated before I started and stayed saturated. Raw `sysctl -n
  vm.swapusage` `used` went **15,211 MiB (start) → 16,307 MiB (end)**, peaking at **16,659 MiB**,
  against a total that macOS itself grew from 16,384 to 17,408 MiB — i.e. **≈14.9–16.3 GiB used of
  ~16–17 GiB**, free never above 1.5 GB. Per-rep deltas were small (≤ +840 MB, most ±100 MB), so
  no per-rep abort was triggered, but the machine was in sustained swap pressure for 100% of the
  measurements.
- **Disk:** 25 GB → 21 GB free (never below the 8 GB stop line). Weight files untouched.
- **Why unpaired numbers from this session are worthless:** an earlier run on this *identical*
  machine and config spread **2.4–8.0 t/s (3.4×)** unpaired. This session reproduces that: the
  same mlx-lm config measured **6.139 t/s in round 1 and 2.860 t/s in round 5** — a 2.1× swing
  with nothing changed but the clock. **Any oMLX (or mlx-lm) number from this session quoted
  without its paired counterpart must be treated as unreliable.** The paired ratios above are the
  only defensible numbers here, and even they carry a 2.4× internal spread.
- **Quality caveat:** the output text was degenerate (`***`) on both engines with this prompt, so
  these throughput numbers describe a repetitive-output workload. They are valid as an
  *engine-vs-engine* comparison (same degenerate output both sides) and as an *MTP-on/off*
  comparison, but not as a prediction for chat workloads.
- **One server at a time** was enforced mechanically: each arm starts only after `pgrep` for
  `omlx serve` / `omlx-server` / `mlx_lm.server` returned empty, and each kill is verified empty
  before the next start (`logs/text/omlx_paired/run_exp1.sh`, `run_exp2b.sh`). Note that oMLX renames
  its process to `omlx-server`, so `pkill -f "omlx serve"` alone leaves it alive — the scripts kill
  by exact name as well.

### Raw data

- `results/text/omlx_paired.jsonl` — **22 rows**, one JSON object per request. Every row carries
  `label`, `engine`, `phase`, `round`/`pair`, `prompt_tokens`, `completion_tokens`, `elapsed_s`,
  `decode_tps`, `warmup_elapsed_s`, `loadavg_before` / `loadavg_after` (full 1/5/15 triple),
  `swap_used_mb_before` / `swap_used_mb_after`, `measurement_method`, `server_timings` (always
  `null` — neither engine returns llama.cpp-style timings) and the raw `usage` (which for oMLX
  includes its own `generation_tokens_per_second`). Labels: `exp1-round{1..5}-{mlxlm,omlx}`,
  `exp2b-pair{1..3}-mtp{off,on}` (the valid MTP A/B), `exp2-pair{1..3}-mtp{off,on}` (**invalid as
  a pair** — both arms had MTP on, see above).
- Harnesses: `logs/text/omlx_paired/req.py` (wall-clock request + load/swap sampler),
  `logs/text/omlx_paired/run_exp1.sh`, `run_exp1b.sh` (rounds 4–5), `run_exp2.sh` (the invalid run),
  `run_exp2b.sh` (valid MTP A/B); progress logs `run_exp1.progress`, `run_exp1b.progress`,
  `run_exp2.progress`, `run_exp2b.progress`.
- Server logs: `logs/text/omlx_paired/mlxlm_r{1..5}.log`, `omlx_r{1..5}.log`,
  `mtp_pairb{1..3}_{off,on}.log` plus extracted gate/telemetry in
  `mtp_pairb{1..3}_{off,on}.mtp.txt` (and `mtp_pair{pair}_{arm}.*` for the invalid run).
- Toggle state: `logs/text/omlx_paired/base-off2/model_settings.json` (`mtp_enabled:false`),
  `base-on2/model_settings.json` (`mtp_enabled:true`); `base-off/` shows the auto-written
  `mtp_enabled:true` default that invalidated the first attempt.

---

## Round 3: clean-machine re-test with oQ4e-MTP

**Verdict: YES — oMLX on `oQ4e-mtp` with MTP enabled beats the llama.cpp bar of 13.90 t/s:
15.65 t/s decode (median of 3), i.e. +12.6% on wall clock and +20.2% on the
like-for-like inter-token metric (16.71 vs 13.90).** The win is attributable almost
entirely to the **MTP speculative head**, not to the quantisation format and not to
oMLX's raw decode speed: with MTP turned off, the same oQ4e weights on oMLX decode at
**7.32 t/s — statistically identical to llama.cpp's 7.40 t/s no-spec number (0.99×)**.
oMLX's MTP multiplies decode by **2.14×**, versus **1.88×** for llama.cpp's
`--spec-type draft-mtp`. The oQ4e quant itself buys nothing on decode; the spec head
buys everything, and oMLX's head is the better of the two.

Measured on the clean machine (load 1.9–2.7, no VM, **zero paging during every rep** —
see Caveats for the swap accounting). Full method: `logs/text/omlx_clean/`, one JSONL row per
request in `results/text/omlx_clean.jsonl`.

### Decisive result

Paired, interleaved arms (`on, off, on, off, on` — 3 pairs, fresh oMLX process per arm,
since `mtp_enabled` is bound at model-load time), 128 gen tokens, `temperature 0`,
`benchmarks/text/prompts/prompt_short.txt` (24 prompt tokens), one discarded warmup per server start.

| config | decode (wall, med of 3) | decode (inter-token, med) | TTFT | peak mem | MTP acceptance | load1 / swap used | valid |
|---|---|---|---|---|---|---|---|
| **oMLX `oQ4e-mtp`, MTP ON** | **15.65** (15.67 / 15.65 / 13.37) | **16.71** | 0.58 s | 17.82 GB (16.60 GiB) | 74.8–75.5%, median **75.2%**; 2.98 tok/cycle, adaptive depth d1–d4 | 1.92–2.68 / 2991–3055 MB (Δ 0, 0 swapouts) | yes |
| oMLX `oQ4e-mtp`, MTP OFF | 7.32 (7.32 / 7.50 / 6.56) | 7.50 | 0.55 s | 17.82 GB | n/a (no MTP path) | 2.06–2.34 / 2967–3015 MB (Δ 0, 0 swapouts) | yes |
| *llama.cpp v0.7.0, Q4_K_M, `--spec-type draft-mtp`, n-max 3* | *13.90* | *not comparable (engine timings)* | *not-measured* | *not-measured* | *63.1%, 2.86 mean draft len* | *clean, per brief* | *reference* |
| *llama.cpp v0.7.0, Q4_K_M, no spec* | *7.40* | *not comparable* | *not-measured* | *not-measured* | *n/a* | *clean, per brief* | *reference* |

- Decode is quoted two ways because they are not the same quantity. **wall** = tokens ÷
  the whole streaming HTTP call (the brief's rule, includes the 24-token prefill).
  **inter-token** = `(completion_tokens − 1) / (t_last_delta − t_first_delta)`, which is
  what llama.cpp's `timings.predicted_per_second` actually measures. **The 13.90 bar is
  a decode-only number, so the inter-token column is the fair comparison.** Both columns
  beat 13.90.
- Spread across the 3 pairs is tight: MTP ON 13.37–15.67 (1.17×), MTP OFF 6.56–7.50
  (1.14×). Compare the earlier contaminated session, where an identical config spread
  2.4–8.0 t/s (3.4×). The clean-machine re-run is reproducible to ~15%.
- Prefill: **not-measured**. oMLX exposes no `timings` object and no prompt-processing
  rate; TTFT (0.55–0.58 s on a 24-token prompt) is the only timing available and it is
  not comparable to llama-bench `pp512` = 152.8.
- Peak memory is `current_model_memory` from `/v1/models/status`: 17,820,265,635 B,
  identical in both arms — the MTP head is folded into the checkpoint, not an extra
  allocation worth reporting separately.
- MTP acceptance comes from oMLX's own log line (`MTP[n] finish=… accept=…`), not from
  HTTP: 83/110 = 75.5%, 82/109 = 75.2%, 83/111 = 74.8% across the three measured
  requests — self-consistent to within 0.7 points.

### Engine vs quant vs MTP

| decomposition step | measurement | ratio | interpretation |
|---|---|---|---|
| **MTP contribution** | (oQ4e MTP ON) − (oQ4e MTP OFF) = 15.65 − 7.32 | **2.14×** (inter-token: 2.23×) | The spec head is the whole story. |
| llama.cpp's MTP contribution | 13.90 − 7.40 | 1.88× | oMLX's MTP converts acceptance into more speed than llama.cpp's. |
| **Quant / raw-engine parity (no MTP)** | oQ4e MTP OFF 7.32 vs llama.cpp Q4_K_M 7.40 | **0.99×** | oQ4e against Q4_K_M, and oMLX against llama.cpp, are indistinguishable on plain decode. |
| oQ4e-MTP ON vs llama.cpp+MTP | 15.65 vs 13.90 | **1.126×** (inter-token 1.202×) | The oMLX win. |

**Which component dominates: the MTP head.** Strip MTP out and the two stacks land within
1% of each other on decode (7.32 vs 7.40), despite different engines *and* different 4-bit
formats — so neither the engine nor the quant format is responsible for the gap. oMLX wins
because its speculative head is more effective: **75.5% acceptance at ~2.98 tok/cycle
against llama.cpp's 63.1% at 2.86 draft length**, which compounds 2.14× instead of 1.88×.
Two caveats on that acceptance comparison: the two numbers come from different tokenisers
and prompts (identical prompt text, but llama.cpp was measured separately), and
acceptance is prompt-dependent — see the sanity section below before trusting 75.5%
as a general figure.

Note the engine-vs-quant question the earlier rounds could not answer is now answered
*for decode*: with the spec head disabled, oQ4e+oMLX and Q4_K_M+llama.cpp are a tie, so
there is no evidence that the oQ4e quant format is intrinsically faster on this hardware.

### Generated-text sanity

First ~200 characters of the decisive run (pair 1, MTP ON), verbatim:

```text
\n\n thinking\nThe user wants a detailed technical explanation of how unified memory bandwidth (not compute) limits token generation speed in local LLMs. They want concrete numbers and a worked example. Le
```

This is **not** the degenerate `***` repetition that inflated acceptance in the earlier
contaminated session, and there is independent evidence that acceptance is not
copy-inflated: oMLX's telemetry for the earlier `***` run carried `copy=70/70` (the copy
proposer produced every draft from copied context), whereas **none of the six telemetry
lines from the three measured reps here contains a `copy=` field at all** — every draft
came from the real MTP head. The acceptance curve by depth decays the way genuine
acceptance does
(d1 = 33/40 = 83%, d2 = 23/32 = 72%, d3 = 16/22 = 73%, d4 = 11/16 = 69%) instead of
sitting flat near 100%.

Two honest caveats. (1) The model answered in **thinking mode** — the 128-token budget was
consumed by a ` thinking` block, so this measures reasoning-trace decoding, not polished
prose; the visible answer never starts. (2) 75.5% is therefore an acceptance rate *for a
reasoning trace on this one prompt*, and should not be extrapolated to chat output or to
other prompts.

### Engine A/B on identical weights (substituted for the mlx4bit A/B)

**The planned experiment could not run as specified:** the MLX 4-bit checkpoint
(`mlx-community/Qwen3.8-27B-4bit`) **was not on disk** during this round. The model store then held
only `qwen38-27b-oq4e-mtp`, `qwen36-35ba3b-oq4e-mtp`, `qwen38-27b-gguf` and `qwen38-aux`, and
downloading was forbidden, so the "same weights, two engines" A/B was re-pointed at the **oQ4e
weights**, which exp5 proved stock mlx-lm can load. This is a *stronger* test than the original: it
removes the quantisation variable instead of holding it constant by construction.
(Weights now live in the Hub cache; see [`models/README.md`](../../models/README.md).)

mlx-lm has no MTP head, so the like-for-like engine comparison is **oMLX with MTP OFF**.
Arms alternate `mlxlm, omlx-off` within each round; wall clock on both sides; one server
in memory at a time.

| round | mlx-lm (`mlx_lm.server`) | oMLX (MTP off) | ratio oMLX/mlx-lm |
|---|---|---|---|
| 1 | 7.26 | 7.11 | 0.98 |
| 2 | 5.57 | 6.74 | 1.21 |
| 3 | 6.99 | 6.36 | 0.91 |
| **median** | **6.99** | **6.74** | **0.98** |

**The engines are equivalent on plain decode: median ratio 0.98, round-to-round spread
0.91–1.21.** oMLX is not intrinsically faster than mlx-lm; it is the same mlx stack with a
speculative head bolted on. Independently confirmed by exp5, where stock mlx-lm loaded the
oQ4e checkpoint directly and reported `generation_tps = 7.196` — essentially the same as
oMLX-with-MTP-off (7.11/6.74/6.36 in this A/B, 7.32/7.50/6.56 earlier).

Note the round-to-round drift, which is the honest weakness of this table: values fall
7.26 → 5.57 while nothing else changes, and the two oMLX-off series measured in different
phases of the session (7.11 → 6.74 → 6.36, and earlier 7.32 → 7.50 → 6.56) both trend
downward. `pmset -g therm` recorded **no thermal warning level and no performance warning
level**, and load stayed 1.9–2.7 throughout, so this is not OS-reported throttling and not
contamination; it is unexplained drift. Interleaving is what makes the *ratios* survive it —
both arms of a round sit minutes apart — but the absolute level of any single row should be
treated as ±20%.

### Experiment 5: does stock mlx-lm load the oQ4e quant?

**Yes.** `.venv/bin/mlx_lm.benchmark --model "$(model_path qwen38-27b-oq4e-mtp)" -p 512 -g 128 -n 2`
(the historical run used `models/qwen38-27b-oq4e-mtp`):

```text
Trial 1:  prompt_tps=191.511, generation_tps=7.433, peak_memory=16.863, total_time=19.914
Trial 2:  prompt_tps=157.697, generation_tps=6.958, peak_memory=16.863, total_time=21.667
Averages: prompt_tps=174.604, generation_tps=7.196, peak_memory=16.863
```

This settles the engine/quant question definitively: **the same oQ4e weights run in stock
mlx-lm at 7.20 t/s decode.** The oQ4e format is not an mlx-lm-incompatible black box, and
mlx-lm gets no decode speed from it that oMLX does not.

It also gives the only *prefill* number this round produced. mlx-lm prompt processing on
512 tokens = **174.6 t/s**, from the engine's internal timer (rows 185/158, 20% spread).
Against llama-bench `pp512` on the same machine that is **+14% over f16-KV (152.8) and
+32% over q8_0-KV (132.2)**. Treat as indicative only: different KV-cache configuration,
different internal timing scopes, and n=2. oMLX's own prefill remains **not-measured** —
it exposes no timings, so no oMLX number appears here.

### MoE result

`Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` (since removed), paired `on, off, on, off` (2 pairs), same prompt/params.
Peak memory 22.69 GB.

| config | decode (wall, median) | decode (inter-token) | TTFT | peak mem | MTP acceptance | load1 / swap | valid |
|---|---|---|---|---|---|---|---|
| MoE oQ4e, MTP ON | **54.53** (51.66 / 57.40) | 58.61 | 0.18 s | 22.69 GB | 75.3% / 80.0%, 2.29–2.56 tok/cycle, depth d1–d3 | 2.54–2.62 / 4570–4746 MB (Δ 0, 0 swapouts) | yes |
| MoE oQ4e, MTP OFF | 48.17 (48.32 / 48.01) | 50.67 | 0.15 s | 22.69 GB | n/a | 2.57–2.74 / 5006–5243 MB (Δ 0, 0 swapouts) | yes |
| *standard MLX-4bit MoE (earlier, under load)* | *56.9* | *not comparable* | *not-measured* | *not-measured* | *n/a* | *contaminated — underestimate* | *reference* |

Pair ratios: **1.07** (51.66/48.32) and **1.20** (57.40/48.01), median **~1.13**.

**MTP does *not* compound usefully with MoE sparsity — it contributes only ~1.13×, against
2.14× on the dense 27B.** The reason is visible in the telemetry: on the MoE the head
reaches depth d1–d3 only (never d4), tok/cycle falls to 2.29–2.56 from 2.98, and the
absolute gain is capped because MoE decode is already cheap per token. With 3B active
parameters the dense-memory-bandwidth bottleneck that MTP attacks is mostly absent, so
there is far less to win. Prefill speedup is also large in absolute terms (TTFT 0.15–0.18 s).

Against the stale 56.9 t/s reference for standard MLX-4bit MoE, oQ4e+MTP's 54.53 does **not
clearly win** — but that reference was measured under the contaminated load and is an
underestimate of unknown size, so this is **not-measured**, not a loss.

### Caveats

- **Swap, the one rule this session could not satisfy literally.** Every row has
  `"valid": false` with reason `swap_used ≈2.9–5.2 GB > 2048 MB`. That is the brief's
  threshold applied honestly, and it fires on **all 19 rows**. It is also, I am confident, a
  false alarm, for evidence rather than assertion:
  - `memory_pressure` reported **83–86% of system memory free** at every sample; 32 GB
    against a 17.8 GB model.
  - `Swapouts` moved **exactly 0 pages during every measured rep of experiments 1, 2, 3 and
    4** — `swapouts_delta_mb = 0.0` in **all 19 rows, without exception**. A rep that is
    actually paging cannot show zero swapouts.
  - Swap `used` **fell** across the session (3062 → 2870 MB) while measuring — measuring
    freed swap, it did not consume it.
  - The machine did carry ~3.0 GB of *stale allocated* swap at session start (macOS does
    not reclaim allocated swap until reboot), left over from the earlier contaminated run.
    The brief's "swap 0 MB" observation was not reproducible from here, and no non-`sudo`
    command clears that allocation (I did not use `sudo`).
  - Exception worth flagging: during the **MoE** experiment macOS *grew* the swap file from
    4096 → 6144 MB and free memory fell to 20–22%, so the 21.6 GB MoE does create real
    allocation pressure at model load. Per-rep swapouts stayed 0, so no measured generation
    was paged, but the MoE rows have less headroom than the dense ones.
  - Rows therefore carry both signals: `swap_used_mb_*` (fails the literal rule) and
    `swapouts_delta_mb` (the live signal). A reader who insists on the literal rule should
    read the whole round as invalid; I judged the reps valid on active-paging grounds and
    am stating that judgement openly rather than hiding it.
- **Load average** was clean and stayed clean: 2.28–2.74 across all 19 rows, never
  approaching the 6.0 abort threshold. Before the session I killed three user-owned
  CPU hogs that were pushing load1 to 4.2 (`ApplicationsStorageExtension` 96%,
  `StorageManagementService` 89–103%, `duetexpertd` 54%); StorageManagement respawned once
  and was killed again. No VM, no `Virtualization.framework`, no Spotlight storm during
  measurement.
- **Wall clock, not engine timings.** Every number is `perf_counter` around the HTTP call;
  neither oMLX nor mlx-lm returns a llama.cpp `timings` object, so none is compared to one.
  Decode is reported two ways (see the decisive table) because llama.cpp's 13.90/7.40 bars
  are decode-only.
- **Unexplained drift.** Absolute decode fell ~25% over the session with load, swap and
  thermal state flat (see Engine A/B). Paired/interleaved ratios are the defensible output;
  absolute single rows are ±20%.
- **The 4-bit mlx4bit model is gone**, so the originally specified experiment 3 (and the
  mlx-lm 5.75/122 stale figures) could not be re-run. No weight was downloaded.
- **Thinking mode.** Every arm generated a ` thinking` trace rather than an answer — 128
  tokens was always consumed by reasoning. Throughput therefore describes reasoning-trace
  decoding. Acceptance (74.8–75.5%) is for this workload and prompt only.
- Disk 18 GB → 16 GB free (swap file growth), never below the 8 GB stop line. No file in
  `models/`, the Hub cache, either venv, `benchmarks/text/`, `src/` or the earlier `.md` files was modified.

### Raw data

- `results/text/omlx_clean.jsonl` — **19 rows**, one JSON object per measured request. Fields:
  `label`, `exp`, `engine`, `arm`, `pair`, `model`, `prompt_tokens`, `completion_tokens`,
  `ttft_s`, `decode_window_s`, `elapsed_s`, `decode_tps_wallclock`, `decode_tps_window`,
  `warmup_elapsed_s` (17–28 s on the dense model — the model load is absorbed by the
  discarded warmup, which is why measured reps hit a warm model), `loadavg_before/after`
  (full triple), `swap_used_mb_before/after`, `swapouts_pages_before/after`,
  `swapouts_delta_mb`, `mem_free_pct_before/after`, `current_model_memory` and full
  `models_status`, `valid` + `invalid_reasons` + the `validity_rule` used (thresholds are
  in the JSON so anyone can re-apply the literal rule), `measurement_method`,
  `server_timings` (always `null`), `text_len`, `text_head` (first 200 chars), `raw_usage`.
  - `exp1+2`: `exp12-pair{1..3}-mtp{on,off}` — the decisive paired MTP A/B (6 rows).
  - `exp3` (aborted attempt): 3 rows, oMLX arms only — `mlx_lm.server` was handed a bare
    model name, resolved it as a HuggingFace repo id and 404'd. Kept as drift evidence.
  - `exp3b`: `exp3b-round{1..3}-{mlxlm,omlx-mtpoff}` — the valid engine A/B (6 rows).
  - `exp4`: `exp4-pair{1,2}-mtp{on,off}` — the MoE paired test (4 rows).
- Harnesses in `logs/text/omlx_clean/`: `req.py` (streaming request + TTFT + load/swap/paging
  sampler), `run_exp12.sh` (exp 1+2, 3 interleaved pairs), `run_exp3.sh` (engine A/B),
  `run_exp4.sh` (MoE), progress logs `run_exp12.progress`, `run_exp3b.progress`,
  `run_exp4.progress` (note `run_exp4.sh` still writes the `exp12` prefix inside its
  progress lines — cosmetic, the JSONL labels are correct).
- Server logs `logs/text/omlx_clean/logs/exp12_p{pair}_{arm}.log`, `exp3b_r{r}_{engine}.log`,
  `exp4_p{pair}_{arm}.log`; MTP acceptance extracted to `*.mtp.txt`, effective MTP state at
  load time to `*.gate.txt`; `exp5_mlxlm_bench_oq4e.log`.
- Toggle state: `logs/text/omlx_clean/base-on/model_settings.json` (`mtp_enabled: true`),
  `base-off/`, `base-ab-off/`, `base-moe-on/`, `base-moe-off/`. Effective state was
  verified by reading the file back after startup **and** by the server log
  (`Speculative backend selected … Lightning MTP … active` present in ON arms, absent in
  OFF arms, with no `MTP[n]` telemetry in OFF arms).

## Round 4: paired head-to-head (orchestrator), and the quant's quality cost

Round 3 compared oMLX (15.65) against llama.cpp (13.90) measured in **different sessions**, on a
machine that drifts ~25 % within a session. That gap was therefore not trustworthy. Re-run as a
strict A/B: both engines alternate within the same round, one model resident at a time, identical
prompt (`benchmarks/text/prompts/prompt_short.txt`), 128 tokens, `temperature 0`, `cache_prompt false`, warmup
discarded, wall-clock timing for both.

| round | llama.cpp Q4_K_M + MTP | oMLX oQ4e-MTP | ratio | swapouts (llama/oMLX) |
|---|---|---|---|---|
| 1 | 12.62 | **15.22** | **1.206** | 0 / 0 |
| 2 | 12.33 | **15.24** | **1.236** | 0 / 0 |
| 3 | 11.92 | **15.02** | **1.260** | 0 / 0 |
| **median** | **12.33** | **15.22** | **1.236** | valid: 6/6 |

**Verdict: oMLX on `oQ4e-mtp` is +23.6 % faster than llama.cpp + MTP, paired, with zero paging.**
The ratio is stable across all three rounds (1.206-1.260) even though both engines drifted downward
within the session — which is exactly why the pairing was necessary. Raw rows: `results/text/ab_mtp.jsonl`.

### Causal decomposition (all four measurements exist now)

| Comparison | Result | Attributable to |
|---|---|---|
| oQ4e weights, MTP **off**: oMLX 7.32 vs mlx-lm 6.99 vs llama.cpp 7.40 | within ±5 % | **engine: no effect** |
| oQ4e weights, MTP **on**: oMLX 15.22 vs 7.32 off | **2.08×** | **MTP head** |
| Q4_K_M weights, MTP on: llama.cpp 13.90 vs 7.40 off | 1.88× | MTP head (weaker impl.) |
| oQ4e vs Q4_K_M perplexity (wikitext-2, 12×2048) | 6.245 vs 6.037 | quant: **+3.3 %** |

So the entire +23.6 % is **not** the MLX engine and **not** the quantisation recipe — it is oMLX's
MTP implementation converting 75.2 % acceptance at 2.98 tok/verify-cycle into a 2.08× multiplier,
where llama.cpp converts 63.1 % at 2.86 into 1.88×.

### What that costs in quality

Perplexity on the same corpus and tokeniser (12 chunks x 2048, `benchmarks/text/ppl_mlx.py`):

| Weights | PPL | delta vs Q4_K_M |
|---|---|---|
| Q4_K_M (llama.cpp) | **6.037** | — |
| oQ4e-mtp (oMLX / mlx-lm) | 6.245 | **+3.3 %** |
| MLX 4bit (mlx-lm) | 6.265 | +3.8 % |

oQ4e is indistinguishable from standard MLX-4bit (+0.3 % PPL) and 3.3 % behind Q4_K_M. So the
recommendation becomes a genuine two-way choice: **+23.6 % speed for +3.3 % perplexity.**

### Caveats specific to this round

- Output was **thinking-mode** (the 128-token budget is consumed by the ` thinking` trace), so these
  ratios describe reasoning decoding, not short chat replies.
- Load average rose 1.8 → 2.8 across the three rounds and both arms fell ~5 %; the *ratio* is the
  robust quantity, the absolute t/s are ~5-10 % pessimistic versus an idle machine.
- oMLX `mtp_enabled` was forced on via `model_settings.json` in a fresh base-path per round
  (`benchmarks/text/ab_mtp.py`); it auto-enables on discovery, so a naive "off" arm is silently invalid.
