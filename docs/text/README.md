# Text track

**Question:** which engine + quant gets closest to `Qwen3.8-27B` capability on 32 GB, at what
decode/prefill speed and at what quality cost? Start with [`../shared/01-hardware.md`](../shared/01-hardware.md)
and [`../shared/02-engines.md`](../shared/02-engines.md); this track only adds what is text-specific.

Status: **measured**. Rows live in [`results/text/`](../../results/text), harnesses in
[`benchmarks/text/`](../../benchmarks/text/README.md).

## Write-ups, in reading order

| File | What it is |
| --- | --- |
| [`01-models.md`](01-models.md) | Models that fit 32 GB *including context*: KV math, weight sizes, capability tiers |
| [`02-benchmark-plan.md`](02-benchmark-plan.md) | Measurement methodology and the validity gates every row must pass |
| [`03-results-speed.md`](03-results-speed.md) | Measured prefill/decode vs depth, spec decoding, wired memory |
| [`04-results-quality.md`](04-results-quality.md) | Measured ARC / TruthfulQA / perplexity, and the harness failures behind them |
| [`05-recommendation.md`](05-recommendation.md) | The pick, exact commands, what not to do, caveats |
| [`06-omlx.md`](06-omlx.md) | oMLX 0.7.1: paired engine test, MTP mechanism, continuous batching |

## The answer in one screen

Two configs, and the choice is genuinely yours:

- **Speed-first: oMLX 0.7.1 + `qwen38-27b-oq4e-mtp` (MTP on) → 15.2 tok/s**, PPL 6.245, 17.8 GB.
- **Quality/tooling-first: llama.cpp ≥ v0.7.0 + `qwen38-27b-gguf-q4km` + `qwen38-27b-mtp` → 12.3 tok/s**
  (13.9 solo), **153 tok/s prefill**, PPL 6.037, 18.1 GB. *Not resident; see
  [`models/README.md`](../../models/README.md).*

Paired, interleaved, same machine: **oMLX is +23.6 % faster for +3.3 % worse perplexity** (ratio
1.206 / 1.236 / 1.260 across three rounds). A Qwen3.6-35B-A3B MoE gave 54.5 tok/s at roughly half the
intelligence; that checkpoint has been removed.

1. **Dense vs MoE is an 8× speed decision.** 27B dense reads ~16.5 GB/token → 9.3 tok/s ceiling on a
   153.6 GB/s bus. A 3B-active MoE reads ~1.7 GB/token → 56.9 tok/s measured.
2. **MTP speculative decoding is the whole speed story.** 7.4 → 15.2 tok/s (oMLX, 75.2 % acceptance)
   or 7.4 → 13.9 (llama.cpp, 63.1 %). Identical weights tie across engines (ratio 0.98), so the
   +23.6 % is the draft/verify implementation, not the runtime.
3. **The engines cross over at ~8k context.** llama.cpp wins short (7.1 vs 5.7); MLX wins at 16k
   (5.7 vs 4.3) because llama.cpp's hybrid-GDN decode degrades 38 % with depth while MLX stays flat.
4. **32 GB is enough — barely.** Qwen3.8-27B keeps 16 of 64 layers full-attention → KV is
   64 KiB/token (1 GiB per 16k). The recommended config wires 17 GiB against a ~24.6 GiB Metal ceiling.
5. **M5 needs a recent llama.cpp.** The Metal-4 tensor API is silently disabled before build
   `b10734`; nixpkgs ships 0.6.0, which predates MTP support entirely.

## Reproducing

```bash
. benchmarks/common/env.sh
.venv/bin/python benchmarks/text/bench_mlx.py qwen38-27b-oq4e-mtp oq4e-mtp
.venv/bin/python benchmarks/text/ppl_mlx.py qwen38-27b-oq4e-mtp data/text/wiki.test.raw 12 2048
benchmarks/text/quality_mlx.sh qwen38-27b-oq4e-mtp oq4e-mtp 200 arc_challenge truthfulqa_mc2
```

Weights resolve by alias from the Hub cache; no path in the repo points at a checkpoint. The GGUF
config is restorable with `MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km`.
