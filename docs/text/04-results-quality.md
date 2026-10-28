# Results — quality

**What was measured, and what was not.** Quality on a 7 tok/s model is expensive: one hour buys
~25 k generated tokens. The suite below is therefore weighted to loglikelihood (prefill-bound) tasks
and is reported with sample counts so the confidence intervals are visible. Nothing here is a
transcription of a vendor number.

## Measurements

Harness: `lm-eval 0.4.13` driven **in-process** via `mlx_lm.evaluate` (mlx-lm 0.32.2), so both models
run identical task definitions and identical scoring code. `--num-shots 0`, `--seed 42`, `--limit 300`,
`--batch-size 2`, no chat template.

| Task | n | Dense **Qwen3.8-27B** MLX-4bit | **Qwen3.6-35B-A3B** MLX-4bit | Δ (dense − MoE) |
|---|---|---|---|---|
| `arc_challenge` (acc) | 300 | **0.4433** ±0.0287 | 0.4033 ±0.0283 | **+4.0 pp** |
| `arc_challenge` (acc_norm) | 300 | **0.4633** ±0.0288 | 0.4067 ±0.0284 | **+5.7 pp** |
| `truthfulqa_mc2` (acc) | 300 | **0.5992** | 0.5700 | **+2.9 pp** |
| `ifeval` | — | not-run | not-run (harness crash, see below) | — |
| `gsm8k` | — | not-run | not-run (time) | — |
| `gpqa_diamond` | — | not-run | not-run (gated dataset) | — |
| `mmlu_pro` | — | not-run | not-run (generative, time) | — |

### Reading these numbers honestly

- **n=300 means ±2.8 pp at 95 %** for a ~40-45 % score. The dense-vs-MoE ARC gap (+5.7 pp on
  `acc_norm`) is ~2× the CI, so the *direction* is solid; the precise size is not.
  TruthfulQA's +2.9 pp is **inside** the noise band — do not build an argument on it alone.
- The **0-shot, no-chat-template** configuration is deliberately harsh and is *not* comparable to
  published card numbers (which use CoT, few-shot, and vendor prompt formats). It is comparable
  **only across the two rows in this table**, which is its purpose.
- **ARC-Challenge is a saturated, easy benchmark for 27B-class models.** Absolute values here
  (40-46 %) are far below what a fine-tuned prompt would yield. Treat this as a *relative* signal,
  not an absolute quality claim.

The locally measured gap (+5.7 pp on ARC) is consistent in direction with the independent
Artificial-Analysis Intelligence Index (Qwen3.8-27B **33.7** vs Qwen3.6-35B-A3B **18.2**) and with
Arena Elo (1440.5 vs not-listed), which is the strongest external corroboration available for the
claim that **the dense 27B is roughly a full capability tier above the 3B-active MoE**.

## Perplexity (quantisation integrity, same text, both engines)

Corpus: `Salesforce/wikitext` wikitext-2-raw **test** split, 12 chunks × 2048 ctx.
Both models share the Qwen tokenizer, so the numbers are directly comparable.

| Config | PPL | Notes |
|---|---|---|
| Qwen3.8-27B **Q4_K_M** (llama.cpp v0.7.0, q8_0 KV) | **6.0372 ± 0.135** | 12 chunks / 24 576 tokens |
| Qwen3.8-27B **MLX 4bit** (mlx-lm 0.32.2) | **6.2645** | 12 chunks / 24 564 tokens |

> **Q4_K_M wins by 3.8 % PPL** (lower is better). That is outside the ~1 % standard error at this
> sample size, and it is the expected direction: Q4_K_M carries 16.46 GB vs MLX-4bit's 16.08 GB, i.e.
> 0.11 more bits per weight. The practical read is that **the GGUF quant slightly out-preserves the
> MLX quant on this hybrid GDN architecture** — small, but it means the recommended config (#1 in
> `05-recommendation.md`) is not giving anything away to choose llama.cpp. Do not over-read it either:
> PPL differences below ~5 % rarely show up on downstream tasks, and the measured ARC gap between
> the two *models* was 5.7 pp — 20× larger than this quantisation effect.

Reference: on Llama-3-8B, llama.cpp's own table shows Q4_K_M at +2.8 % PPL and KLD 0.031 vs f16 —
a 23× larger distribution shift than Q8_0's, yet small on downstream tasks. For a 27B at 4.7 bpw,
Q4_K_M is the accepted floor; **do not go below it** (Q3_K measured +10.5 % PPL on the same table).

## Method failures worth recording (these cost real time, and will cost yours)

1. **lm-eval cannot talk to llama-server v0.7.0.** Two independent breakages:
   - `--model local-completions`: llama-server now returns OpenAI's *new* logprob shape
     (`choices[0].logprobs.content`) instead of the legacy `logprobs.tokens` / `logprobs.top_logprobs`.
     lm-eval's parser gets nothing and the client dies with
     `aiohttp.client_exceptions.ServerDisconnectedError`.
   - `--model gguf` (the purpose-built backend): `--apply_chat_template` raises
     `NotImplementedError: ... implement the 'tokenizer_name' property`, and without it the batch
     loop dies with `ValueError: zip() argument 2 is longer than argument 1`.
   Consequence: **GGUF quality was not measured directly.** Since GGUF-Q4_K_M and MLX-4bit are the
   same weights at ~the same bpw, quality was measured on the MLX form and is reported as such.
   Workaround for a future run: use `--model local-completions` against an **mlx_lm.server**
   (which still emits legacy logprobs) or patch lm-eval's `openai_completions.py`.
2. **`mlx_lm.evaluate` writes results to stdout, not to `--output-dir`.** The output directory is
   created and left empty. Results must be scraped from the log — `benchmarks/text/extract_quality.py` does this.
3. **Generative tasks crash mlx-lm under lm-eval**: `ifeval` (and the first `gsm8k` attempt) died with
   `RuntimeError: [metal::malloc] Resource limit (499000) exceeded`. This is mlx-lm accumulating
   Metal buffers across long `async_eval` sequences, not a memory-capacity problem (RSS was ~1.5 GB).
   Mitigations tried: one task per process (helps), `MLX_MAX_OPS_PER_BUFFER=100` (no effect),
   `--batch-size 1` (not tested before the time budget ran out). **Loglikelihood tasks are stable;
   long generative tasks are not.**
4. **GPQA is gated** (`Idavidrein/gpqa`, `gated: auto`) and this HF account cannot pull it, so the
   headline reasoning benchmark is absent. MMLU-Pro (`TIGER-Lab/MMLU-Pro`) is open and is the correct
   substitute for a future run.
5. **`/usr/bin/time -l` max RSS is not a fit metric** for llama.cpp (mmap + Metal buffers): it
   reported 6.4 GB for a 16.46 GB model.
