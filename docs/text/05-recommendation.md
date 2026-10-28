# Recommendation

> **Weights status (2026-11).** Checkpoints live in the Hub cache (`~/.cache/hf/hub`) and are named
> by repo/alias, not by a path in the repo. **A** (`qwen38-27b-oq4e-mtp`) is resident. **B**
> (`qwen38-27b-gguf-q4km` + `qwen38-27b-mtp`) and **C** (`mlx-community/Qwen3.8-27B-4bit`) are
> restorable on demand; **D**'s oQ4e MoE was removed to reclaim 21 GB. The numbers below remain valid
> for the checkpoints that produced them — re-running B/C/D needs the weights back and a **new
> label**. See [`models/README.md`](../../models/README.md).

## The answer: a two-way choice, not one config

All numbers below are **measured on this machine**, paired where a comparison is claimed.

| | Config | Decode | Prefill | PPL | Memory |
|---|---|---|---|---|---|
| **A — speed-first (default)** | **oMLX 0.7.1 + Qwen3.8-27B `oQ4e-mtp`**, MTP on | **15.2 t/s** | ~130 t/s | 6.245 | 17.8 GB |
| **B — quality/tooling-first** | **llama.cpp v0.7.0 + `UD-Q4_K_M` + MTP head** | **12.3 t/s** (13.9 solo) | **153 t/s** | **6.037** | 18.1 GB |
| C — long context / vision | MLX 4bit via mlx-lm (`mlx-vlm` for images) | 5.7 t/s (flat to 16k) | 132 t/s | 6.265 | 19.1 GB @16k |
| D — raw speed, half the capability | Qwen3.6-35B-A3B `oQ4e-mtp`, MTP on | 54.5 t/s | 881 t/s | not-measured | 22.7 GB |

**The honest trade: A is +23.6 % faster than B for +3.3 % worse perplexity.** That is a real but
small quality step — smaller than the Q4_K_M→Q8_0 step measured on the same kind of model — so
**A is the better default** if you are waiting on the output. **B remains the pick** if you want
scriptable telemetry (oMLX returns no `timings`), GGUF portability, or the last 3 % of quality.

## Why A beats B — and what it is *not*

The +23.6 % is **entirely oMLX's MTP implementation**. All four measurements needed to prove that
now exist:

| Comparison | Numbers | Conclusion |
|---|---|---|
| Same `oQ4e` weights, MTP **off** | oMLX 7.32 / mlx-lm 6.99 / llama.cpp 7.40 | **the engine is worth ~0 %** |
| Same engine, MTP on vs off | 15.22 vs 7.32 = **2.08×** | **the MTP head is the whole win** |
| llama.cpp MTP, on vs off | 13.90 vs 7.40 = **1.88×** | weaker MTP implementation |
| `oQ4e` vs `Q4_K_M` perplexity | 6.245 vs 6.037 | **+3.3 %** quality cost |

Draft acceptance explains it: **oMLX 75.2 % at 2.98 tok/verify-cycle vs llama.cpp 63.1 % at 2.86.**
Neither the MLX engine nor the `oQ4e` quantisation is responsible — both are within noise of the
alternatives. If a future oMLX release regresses MTP, config A loses its entire advantage.

## The exact commands

```bash
. benchmarks/common/env.sh

# ---------- A: speed-first (oMLX + oQ4e-MTP) ----------
uv venv .venv-omlx --python 3.12
uv pip install --python .venv-omlx/bin/python omlx==0.7.1     # wheel exists; no brew needed
hf download Jundot/Qwen3.8-27B-oQ4e-mtp                        # -> ~/.cache/hf/hub (16 GiB)
model_path qwen38-27b-oq4e-mtp                                 # confirm it resolved

# A is already cached; oMLX discovers models from subdirectories, so hand it the snapshot
# parent and use the discovered name (the revision) in model_settings.json.
OMLX_MODEL="$(model_path qwen38-27b-oq4e-mtp)"
# MTP is a per-model setting, NOT a CLI flag. It auto-enables on discovery, but be explicit:
mkdir -p ~/.omlx/base && cat > ~/.omlx/base/model_settings.json <<JSON
{"version": 1, "models": {"$(basename "$OMLX_MODEL")": {"mtp_enabled": true}}}
JSON

.venv-omlx/bin/omlx serve --model-dir "$(dirname "$OMLX_MODEL")" --port 8000 \
  --no-cache --memory-guard off --base-path ~/.omlx/base --headless
# NOTE: `omlx serve` takes --model-dir, not a model path (its own --help example is wrong)

# ---------- B: quality/tooling-first (llama.cpp + Q4_K_M + MTP) ----------
# llama.cpp must be >= v0.7.0 (MTP) and >= b10734 (M5 Metal-4 tensor API).
# nixpkgs ships 0.6.0 which has neither -> build from source: logs/text/build_llamacpp.sh (~2 min)
# Fetch into the cache once (the alias is uncached by default), then resolve offline:
MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km
MODEL_STORE_ONLINE=1 model_path qwen38-27b-mtp

llama-server \
  -m  "$(model_path qwen38-27b-gguf-q4km)" \
  -md "$(model_path qwen38-27b-mtp)" \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -c 32768 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 99 -b 512 -ub 512 -np 1 --jinja \
  --host 127.0.0.1 --port 8080
```

Settings that matter in practice:

- **`-b 512 -ub 512`**, not the default 2048 — default batch plus a spec-decode arm is what OOMs
  servers on unified-memory Macs (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, hit twice here).
- **`-np 1`** — the MTP gain evaporates with concurrency. If you need concurrency, use oMLX's
  continuous batching instead (20.3 t/s aggregate over 4 streams, measured).
- **`-ctk q8_0 -ctv q8_0`** halves KV (0.5 GiB/16k) for ~13 % prefill. Use `f16` for max quality,
  `q4_0` only at 32k+ where the context is worth more than the tok/s.
- **Keep the MTP head file.** It is separate (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.68 GB) and easy to lose
  in a model cleanup — without it decode silently falls from 13.9 to 7.4 t/s.
- **Do not raise `iogpu.wired_limit_mb`.** Every config here fits the default ~75 % ceiling
  (config B measured **+17 GiB wired** against a ~24.6 GiB cap).

## What I would *not* do

| Tempting choice | Why not, on this machine |
|---|---|
| **Blaming or crediting the engine** | At identical weights oMLX and mlx-lm tie (median ratio 0.98). Vendor engine comparisons that don't hold weights constant are measuring the quant and the MTP head. |
| **Qwen3.8-27B Q6_K / Q8_0** | 22 GB / 29 GB. No context fits, or it doesn't load. |
| **`ngram-mod` always on** | 2.7× on repetitive output, **−14 % on novel prose**. Workload-dependent, measured both ways. |
| **Ollama / LM Studio** | Not in nixpkgs, no brew; they hide the KV-quant and spec-decode knobs that are worth 88 % here. |
| **vLLM-metal / vllm-mlx** | Built for serving many clients; heavy install, no benefit at `-np 1`. |
| **Trusting `/usr/bin/time -l` RSS** | Reported 6.4 GB for a 16.46 GB model. Use `vm_stat` wired pages (llama.cpp) or `/v1/models/status` (oMLX). |
| **Benchmarking on a busy machine** | Swap/VM contention cut prefill by up to 29 % and produced a 3.4× spread on one config. Check `sysctl vm.swapusage` and `uptime` first, and **pair your A/B**. |

## Caveats you should hold me to

1. **Config A's quality is judged by perplexity only** (6.245 vs 6.037 — a 3.3 % gap). Downstream task
   deltas at that size are usually below the noise floor of a 300-sample benchmark, so I did not
   claim one; but I also did not run ARC/TruthfulQA on `oQ4e`, so "equal quality" is **not** proven.
2. **Config A was measured in thinking-mode.** The 128-token budget is consumed by a `-thinking`
   trace, so 15.2 t/s describes reasoning decoding, not short chat replies.
3. **Quality coverage is partial** by necessity: at 7-15 tok/s, an hour buys ~25-50k generated tokens.
   `ifeval`/`gsm8k` crash mlx-lm's Metal allocator, `gpqa` is a gated dataset, and llama-server
   v0.7.0's logprob format is unparseable by lm-eval 0.4.13. Gaps are marked `not-run`.
4. **Absolute tok/s are ±10 % session-dependent**; the *ratios* here are paired and stable
   (oMLX/llama.cpp = 1.206 / 1.236 / 1.260 across three rounds).
5. **The oMLX quality path is unproven** — no `timings` telemetry, MTP controlled by a JSON file
   rather than a flag, and its headline 18.2 t/s marketing figure was **not** reproduced (we measured
   15.2). Prefer config B when you need observability.
6. **Config D (MoE) is a capability downgrade**, not a tuning choice: AI-Index 18.2 vs 33.7, and
   locally ARC-Challenge acc_norm 40.7 vs 46.3.
