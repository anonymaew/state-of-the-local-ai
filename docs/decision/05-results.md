# Decision results — three 9B models, two engines, one M5 / 32 GB

Suite: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `test`
split — 400 cases × 5 typed questions = **2,000 decisions**, scored with the suite's own metric
definitions (accuracy, KL from gold, multiclass Brier, ECE over 10 equal-width confidence bins).
Rows: `results/decision/typed.jsonl`; raw engine logs: `logs/decision/`.

Published reference points on the same split (zero-shot): clef-flash 0.710, d3-flash 0.717,
Amos-1-medium 0.765, Jev 0.727, d1-3B 0.656, prior 0.470, uniform 0.308.

## Quality

| model | engine | quant | n | accuracy | KL ↓ | Brier ↓ | ECE ↓ | choice | noul | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **clef-flash** | llama.cpp 0.7.0 | Q4_K_M | 2000 | **0.702** | **0.208** | **0.109** | **0.011** | 0.685 | 0.797 | 0.651 |
| **clef-flash** | MLX 0.32.4 | 4-bit | 2000 | 0.697 | 0.213 | 0.111 | 0.024 | 0.673 | 0.783 | 0.649 |
| clef-flash | MLX 0.32.4 | 8-bit | 2000 | 0.699 | 0.211 | 0.111 | **0.009** | 0.673 | 0.787 | 0.651 |
| **kev-9b** | llama.cpp 0.7.0 | Q4_K_M | 2000 | 0.680 | 0.216 | 0.117 | 0.067 | 0.652 | 0.738 | 0.658 |
| **d3-flash** | llama.cpp 0.7.0 | Q4_K_M | 500 (100 cases) | 0.584 | 0.258 | 0.135 | 0.034 | 0.650 | 0.680 | 0.470 |
| d3-flash (partial, stopped) | llama.cpp 0.7.0 | Q4_K_M | 1475 (295 cases) | 0.702 | — | — | — | — | — | — |
| clef-flash (published, bf16) | — | bf16 | 2000 | 0.710 | 0.209 | 0.111 | 0.016 | — | — | — |
| d3-flash (published) | — | — | 2000 | 0.717 | 0.322 | 0.131 | 0.031 | — | — | — |

Reading it:

- **Quantization is nearly free on this suite, and 8-bit buys nothing measurable.** Clef-Flash:
  bf16 0.710 (published), 8-bit 0.699, Q4_K_M 0.702, 4-bit 0.697. The four are within the suite's
  spread; 4-bit keeps the accuracy of 8-bit at 58 % of the disk and ~1.4× the speed. The one metric
  that does move is ECE (8-bit 0.009 vs 4-bit 0.024), i.e. 8-bit is *sharper*, not more accurate.
- **A lower ECE is not a better model.** Both quantized Clef runs beat the published bf16 ECE
  (0.011 / 0.024 vs 0.016) while being slightly less accurate: flatter distributions pass through
  fewer confidence bins. ECE ranks calibration, accuracy ranks decisions; report both.
- **Kev-9B is the weakest of the three on this suite** (0.680) although its own index (41.0 on
  `breadth-v1`) is respectable. It was trained on the breadth datasets, not on these five
  workflows; the numbers are not transferable across suites, which is exactly why this track uses
  one suite for every candidate.
- Clef's joint head is the best-calibrated by a wide margin (ECE 0.011–0.024 vs Kev 0.067): it
  scores all questions of a request together, so the answers share evidence instead of five
  independent passes.
- **d3-flash's full 400-case run did not complete.** The first attempt was stopped at 295 cases after
  its p50 climbed to 9.6 s; a second 100-case prefix was then run on a cooler machine for a valid,
  labelled row (0.584 at 100 cases, marked `limit=100`). Its `score` questions are the weak spot
  (0.470 vs 0.65–0.68 for the others at the same prefix). The two partial rows stay in the JSONL as
  `valid: false`; only the 100-case row is a usable measurement, and it is comparable with the other
  models at their 100-case checkpoint (Clef MLX 0.584, Clef GGUF 0.632, Kev 0.512), not with the
  full runs.
- **Same accuracy does not mean same distribution.** On the stopped d3 run, accuracy reached 0.702 at
  295 cases (level with Clef's 0.702) but its KL was 0.303 and Brier 0.127 versus 0.208 / 0.109 for
  Clef — it picks the same top answer while spreading probability worse. If the downstream use is a
  threshold, that difference is the one that matters.

## Speed (first ask, one request = one case with five questions)

| model | engine | quant | p50 ms | p95 ms | cached p50 ms | input tokens | peak RSS | valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **clef-flash** | MLX | 4-bit | **1,579** | 2,065 | 1,384 | 879 | 5.6 GB | no (swap) |
| clef-flash | llama.cpp | Q4_K_M | 2,819 | 3,852 | 2,208 | 879 | 6.8 GB | yes |
| clef-flash | MLX | 8-bit | 2,776 | 4,690 | 2,599 | 879 | 10.7 GB (weights) | no (swap) |
| kev-9b | llama.cpp | Q4_K_M | 4,975 | 6,881 | 3,273 | 1,491 | 7.2 GB | yes |
| d3-flash | llama.cpp | Q4_K_M | 3,135 | 16,703 | 16,257 | 1,164 | 6.9 GB | yes (n=100) |

- **Same model, same accuracy, 1.8× latency:** MLX 4-bit beats llama.cpp Q4_K_M on Clef-Flash. The
  MLX loader prefills in 1,024-token chunks; llama.cpp evaluates the whole Clef prompt in one
  `--ubatch-size` batch, which on this part is the slower shape.
- **Kev is 3.2× slower than MLX Clef because of its read-out, not its size.** The pointer head needs
  one causal row per question (5 passes, 1,491 prompt tokens vs 879). Prompt caching does not rescue
  it: in a paired 100-case A/B, `--cache-ram 4096` moved the cached re-ask from 3,590 ms
  (`--no-cache-prompt`) to 3,276 ms — **−6.7 %**, not the ~10× a text model would see. A decision
  model reads hidden states at option positions; the KV cache does not store those.
- **Memory is not the constraint: 5.6–7.2 GB peak RSS.** All three 9B checkpoints fit with >20 GB to
  spare. The only 9B configuration that does *not* fit is Kev's MLX **adapter** path, whose fold
  needs ≈1.9× the 19.3 GB base (~36 GB peak). The GGUF sidesteps it.
- **Sustained vs burst.** p50 by checkpoint over the 400-case run (cumulative):

| config | 50 | 100 | 200 | 300 | 400 |
| --- | --- | --- | --- | --- | --- |
| clef-flash MLX 4-bit | 1,024 | 1,040 | 1,190 | 1,504 | 1,579 |
| clef-flash Q4_K_M | 1,653 | 2,143 | 2,679 | 2,775 | 2,819 |
| kev-9b Q4_K_M | 2,421 | 2,758 | 3,449 | 5,001 | 4,975 |

  A fanless M5 Air loses 50–100 % of its latency over a sustained run. The first-100 p50 is the
  number for interactive use; the full-run p50 is the number for batch work. Quoting one without
  the other is how benchmarks lie.
- **The suite's case order is a workflow order, not a difficulty-neutral shuffle.** Accuracy climbs
  because the early `agent_trace_observability` cases are harder: on the d3 run's per-decision
  records, agent traces scored **0.584**, customer service **0.762**, invoice processing **0.764**.
  Both the cumulative accuracy and the cumulative p50 curves are therefore warm-up curves as well
  as performance curves; compare models at the same case count, never a full run against a smoke.
  The harness now writes per-decision records to `logs/decision/cases_*.jsonl` for this analysis.

## Reproducibility (re-runs on an idle machine)

The uncertain rows were re-run the next pass, one process at a time:

| Run | First | Re-run | Delta |
| --- | --- | --- | --- |
| clef-flash MLX 4-bit, accuracy | 0.6965 | **0.6965** | **0.0000** |
| clef-flash MLX 4-bit, p50 | 1,579 ms | 1,999 ms | +27 % |
| clef-flash MLX 4-bit, first-100 p50 | 1,040 ms | 1,125 ms | +8 % |
| Kev-9B 100-case checkpoint, accuracy | 0.512 | **0.512** | **0.0000** |
| Kev-9B 100-case checkpoint, p50 | 2,758 ms | 3,690 ms | +34 % |

Quality is deterministic — the same checkpoint, suite and harness reproduce accuracy to four decimals.
Latency is *not* a property of the config alone on this machine: the re-run started on a chassis still
warm from the 8-bit download and the previous run, and paid 27–34 %. Report the thermal state or the
first-100 number, never a bare p50.

The re-run also reproduced `valid: false` for the MLX 4-bit config (swap grew again, with no other
workload running). That is a property of MLX + this 32 GB Air under `clef_mlx.py`, not background noise.

**RSS caveat:** `peak_rss_mb` comes from `ps`, and MLX/Metal allocations are not fully reflected in
it — the 8-bit run reports 1.6 GB for 10.7 GB of weights, which is impossible. Treat RSS as a lower
bound for MLX configs; the llama.cpp numbers (6.8–7.2 GB) are the trustworthy ones.

## Validity

- `clef-flash-4bit-mlx` is marked `valid: false`: macOS swap grew by 14 GB during that run (the
  baseline was already 4 GB). Quality is unaffected; the latency row is a *pressured* number and is
  kept rather than deleted.
- All runs: `schema_valid_rate = 1.00`, zero request failures. These are native decision models —
  there is no JSON to parse and no malformed-output failure mode.
- `temperature` is baked into each checkpoint (Kev 2.19, Clef/ d3 per their conversion); the harness
  never sets it.
- Dataset order is by workflow (agent traces → customer service → invoices → security), and both
  accuracy and latency drift over the run, so the per-checkpoint columns are also a *warm-up* curve.
  Comparing models on the same suite/order is fair; comparing a full run with a 50-case smoke is not.

## Plumbing check (the smoke set, not a score)

`data/decision/smoke.jsonl` → `results/decision/smoke.jsonl`, run through the same harness on
Clef-Flash MLX 4-bit. Purpose: catch harness bugs before they contaminate a 2,000-decision run.

| Pass | ctx | items | failures | choice | noul | score | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | 8,192 | 20 | **3** | — | — | — | 377 ms | — |
| second | 16,384 | 20 | **0** | 0.727 | 1.00 | 0.833 | 424 ms | 12.5 s |

The three first-pass failures were HTTP 413, not model errors:

```text
maximum context length exceeded: schema requires 8803 tokens before state; maximum is 8192
```

The 255-option items need **~8.8 k tokens of schema before a single token of state**, so an 8 k
context is silently too small for large option sets. `ctx=16384` clears it. The 6 k-token needle
items then take 12–15 s on this machine — the p95 in the second row is entirely those two items,
not the ordinary short states (p50 424 ms).

This is the reason the suite exists: a full run at `ctx=8192` would have failed those items and
reported a *lower accuracy* instead of a configuration error.
