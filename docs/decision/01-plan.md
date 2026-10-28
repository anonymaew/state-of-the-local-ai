# Decision models (System One) on M5 / 32 GB — plan

**Question:** what actually runs a *decision model* on this box — at usable latency, usable
calibration, and inside 32 GB — with a preference for the 9B tier?

This track is different from `text/`. A System One model does not generate prose. It reads a state
(string / JSON) and answers **typed questions**: `choice` (pick one of N), `noul` (yes/no), `score`
(rating), returning **probabilities**, not tokens. TypeSafe's Jev named the category on 2026-09-15;
open weights followed within days. That changes the whole measurement problem:

- there is no decode loop to count tok/s on; the unit is **one forward pass per state**,
- the answer is read from a **head or from option-token logits**, so the engine must expose hidden
  states / logprobs, or the head has to be baked into the checkpoint,
- the quality metric is **accuracy + calibration (Brier / ECE)**, not PPL,
- the cost that matters is **prefill of the state** (and cache reuse when the same state is asked
  many questions), not generation.

## Two families (this is the central engine constraint)

| Family | How the answer is read | What can run it |
| --- | --- | --- |
| **Encoder / non-autoregressive** (Von, Laya, Anarkali, GLiNER2.5-Decide, open-jev-deberta) | one pass, decision heads on `[CLS]`/option spans | PyTorch, ONNX Runtime, CoreML, MLX only if ported |
| **LLM backbone + decision head** (Kev, Clef, Open-Jev, Jeeves) | hidden state at `<decide>` scored against each `</opt>` hidden state (pointer head) | framework that can load custom module code: PyTorch, MLX, vLLM (no Metal) |
| **LLM backbone, logprob read-out** (SemIf, Jev-Style, Tev1, Rizzo Flow) | option letter / yes-no token probability from the LM head | **any** engine that exposes logits: llama.cpp, MLX, transformers |

The third row is the bridge: it is the only way a 9B model runs in **llama.cpp** on this machine.
The second row is the only way to run the published 9B checkpoints (Kev-9B) without retraining.

## Hypotheses to test

- **H1 — the 9B cliff is the adapter fold, not the weights.** Kev-9B is a 173 MB LoRA adapter on a
  19.33 GB bf16 base, and the MLX adapter path merges it into a second copy while loading
  (≈1.9× weights ≈ 36 GB peak) — so the *MLX adapter* path cannot load on 32 GB even though the
  bf16 weights alone would. The GGUF conversion (`ggml-org/Kev-9B-GGUF`, Q4_K_M 6.36 GB) carries
  the pointer head and sidesteps the fold entirely. Corollary: for 9B, llama.cpp is not just an
  engine choice, it is the memory escape hatch.
- **H2 — MLX is the only no-port engine for pointer-head models.** `kev.mlx_model` already exists;
  torch MPS is the fallback and is expected to be slower and heavier.
- **H3 — llama.cpp wins state-cache latency at short states, loses at long ones** (same crossover
  seen in `text/`, ~8k). Only reachable via logprob read-out models.
- **H4 — chunked-prefill dominates first-ask latency.** Kev reads long states 1024 tokens at a time;
  re-asking about the same state should be ~10× cheaper than the first ask (Kev-4B: 721 → 136 ms
  on M5). Measure first-ask vs cached-ask separately; never average them.
- **H5 — the 4B is the real answer; 9B buys ~3 index points for ~4× the memory.** Kev's own
  numbers: 0.8B 23.3, 4B 38.0, 9B 41.0, 27B 52.3 (chance-corrected Decision Index, `breadth-v1`
  test split). Confirm or refute on this machine.

## Candidates (to be confirmed by research)

| Model | Size | Family | License | Mac story |
| --- | --- | --- | --- | --- |
| Kev-9B | 9B | LLM + pointer head | Apache-2.0 | GGUF + llama.cpp **measured**; MLX adapter does not fit |
| Kev-4B / Kev-0.8B | 4B / 0.8B | LLM + pointer head | Apache-2.0 | measured by upstream on M5 32 GB (speed only) |
| Cloudflare Clef-Flash | 9B | LLM + joint decision head | Apache-2.0 | MLX `clef_mlx.py` and llama.cpp **measured** |
| Jev-Style v3 | 0.8B | LLM, one-token GGUF | Apache-2.0 | llama.cpp / LM Studio; 25.6k ctx |
| SemIf over Qwen3.5-9B | 9B | LLM, logprob read-out | MIT + Apache-2.0 | llama.cpp — the 9B-on-llama.cpp path |
| Liquid d1-3B | 3.1B | LFM2.5-VL + heads | LFM Open v1.0 | 30 ms/decision on M5 Pro (vendor) |
| Anarkali / open-jev-deberta | 68M / 434M | encoder, ONNX | Apache-2.0 | latency floor, no GPU |

## Measurement contract (adds to `benchmarks/common/`)

One row = one measurement, one process, one model load. Extra fields for this modality:

| Key | Meaning |
| --- | --- |
| `metrics.accuracy` | top-1 option accuracy over the fixed item set |
| `metrics.brier` | mean Brier over the full option distribution (choice/noul) |
| `metrics.ece` | expected calibration error, 10 equal-width confidence bins (suite definition) |
| `metrics.latency_ms_p50` / `p95` | first-ask, wall clock, state unseen |
| `metrics.cached_latency_ms_p50` | same state, second question |
| `metrics.state_tokens` | tokenized length of the state |
| `metrics.questions_per_request` | batch size of the question set |
| `metrics.peak_rss_mb` / `peak_mlx_mb` | weight + activation high-water mark |
| `harness.engine_version`, `harness.quant`, `harness.device` | required, as in `text/` |

Validity gates: same item set and same `temperature` for every config; probabilities from the same
read-out path (`/v1/systemone` or logits); swap-outs invalidate; a config that changes the item set
is a new label, not a new row in the same series.

## Work split

| Stream | Owner | Deliverable |
| --- | --- | --- |
| Engines | subagent A | [`02-engines.md`](02-engines.md) — capability matrix, install story, M5 flags |
| Models | subagent B | [`03-models.md`](03-models.md) — exact repos, size/quant math, runtime path |
| Eval | subagent C | [`04-eval.md`](04-eval.md) + `data/decision/` — item sets, Decision Index spec |
| Harness | orchestrator | `benchmarks/decision/bench_systemone.py`, servers, rows |
| Runs | orchestrator | `results/decision/*.jsonl`, [`05-results.md`](05-results.md) |
| Recommendation | orchestrator | [`06-recommendation.md`](06-recommendation.md) |

## Non-goals

- Training or fine-tuning a decision model (Kev publishes the recipe; out of scope for this pass).
- Hosted APIs (Jev, OpenAI Decisions, Microsoft Decision-1): reference numbers only, not candidates.
- Multimodal decision models: `d1-omni-600M` and Clef-Omni are noted, not benchmarked here.

## Status (2026-10-27)

Measured. Four configurations ran on this machine; three completed the full 400-case / 2,000-decision
suite, one was stopped on latency and finished as a labelled 100-case prefix:

| Config | Accuracy | p50 | Peak RSS | Valid |
| --- | --- | --- | --- | --- |
| Clef-Flash 9B, MLX 4-bit | 0.697 | 1,579 ms | 5.6 GB | no (swap) |
| Clef-Flash 9B, llama.cpp Q4_K_M | 0.702 | 2,819 ms | 6.8 GB | yes |
| Kev-9B, llama.cpp Q4_K_M | 0.680 | 4,975 ms | 7.2 GB | yes |
| d3-flash 9B, llama.cpp Q4_K_M | 0.584 @100 | 3,135 ms | 6.9 GB | yes (prefix) |

Re-runs on an idle machine confirmed the quality is deterministic (0.6965 twice, 0.512 twice) and
added the quant axis: MLX 8-bit 0.699/2,776 ms vs 4-bit 0.697/1,579 ms — **8-bit is not worth it**.
A paired cache A/B on Kev-9B showed `--cache-ram 4096` improves the cached re-ask by only 6.7 %,
because a decision head needs hidden states that the KV cache does not store.

Results: [`05-results.md`](05-results.md). Recommendation:
[`06-recommendation.md`](06-recommendation.md). Hypotheses: H1 half-refuted (the
fold is the cliff, but GGUF routes around it), H2 confirmed, H3 untested (no logprob-read-out 9B
config was built), H4 confirmed (Clef's joint head amortises one prompt; Kev's 5 passes do not),
H5 rejected on this suite (9B buy ~2 points over nothing here — no 4B was run — but the 9B fits
comfortably, so there is no reason to drop to 4B).
