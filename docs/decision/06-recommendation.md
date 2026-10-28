# Decision recommendation — M5 / 32 GB

Answer first: **`mlx-community/clef-flash-4bit` + the bundled `clef_mlx.py` loader.** It is the
fastest 9B decision configuration measured here, ties the best accuracy on the suite, fits in 5.6 GB
of RSS, and is Apache-2.0. When an HTTP/GGUF process is required, **`ggml-org/Clef-Flash-GGUF`
Q4_K_M on llama.cpp 0.7.0** gives the same accuracy for ~1.8× the latency.

## The pick, by what you need

| Need | Config | Accuracy | p50 (first 100 cases) | Peak RSS |
| --- | --- | --- | --- | --- |
| **Lowest latency, best accuracy** | Clef-Flash **MLX 4-bit** (`clef_mlx.py serve`) | 0.697 | **1.04 s** | 5.6 GB |
| Portability / one binary / GGUF tooling | Clef-Flash **Q4_K_M**, llama.cpp 0.7.0 | **0.702** | 2.14 s | 6.8 GB |
| Kev training recipe / `breadth-v1` lineage | Kev-9B **Q4_K_M**, llama.cpp 0.7.0 | 0.680 | 2.76 s | 7.2 GB |
| Do **not** use (this box) | Kev-9B MLX **adapter** | — | — | ≈36 GB peak, OOM |
| Do **not** use (this box) | d3-flash **Q4_K_M**, llama.cpp | 0.584 @100 cases | 3.1 s p50, 16.7 s p95 | 6.9 GB |

## Why

1. **9B is comfortably a 32 GB workload — 6–7 GB resident at 4-bit.** The 32 GB question is not
   "does a 9B fit" (it does, with 20 GB spare) but "does the *engine* fit": Kev's MLX adapter fold
   peaks at ≈1.9× the 19.3 GB bf16 base. GGUF carries the pointer head without the fold, so the
   format, not the parameter count, decides whether a 9B runs here.
2. **Engine choice moves latency more than a bit-width step.** Same Clef-Flash, same accuracy:
   MLX 4-bit is 1.8–2.1× faster than llama.cpp Q4_K_M. Both are ~6 GB. On this machine the MLX
   prefill shape (1,024-token chunks) beats llama.cpp's single-batch decision path.
3. **4-bit costs ~1 point of accuracy on this suite; 8-bit does not buy it back.** Clef bf16 0.710
   (published) → LLAMA.CPP Q4_K_M 0.702 → MLX 8-bit 0.699 → MLX 4-bit 0.697. That is inside the
   suite's spread. 8-bit is 1.4× slower and 4.5 GB larger for +0.25 points (and a sharper ECE); on
   this machine **4-bit is the right quant**. Quality is deterministic: re-running the same config
   reproduced 0.6965 to four decimals.
4. **Calibration does not follow accuracy.** Clef's joint head is far better calibrated (ECE 0.011
   GGUF / 0.024 MLX) than Kev's pointer head (0.067), though Kev is only 2 points less accurate.
   If the application thresholds on confidence (routing, abstention), this matters more than 0.02
   accuracy.
5. **Sustained ≠ burst on a fanless Air.** p50 grows 50–100 % over a 400-case run. Budget for the
   first-100 number interactively and for the full-run number in batch; never mix them.

## Memory budget (measured RSS, not weights)

| Config | Weights on disk | Peak RSS | Fits 32 GB |
| --- | --- | --- | --- |
| Clef-Flash MLX 4-bit | 6.2 GB | 5.6 GB | yes, 26 GB spare |
| Clef-Flash GGUF Q4_K_M | 6.3 GB | 6.8 GB | yes |
| Kev-9B GGUF Q4_K_M | 6.0 GB | 7.2 GB | yes |
| d3-flash GGUF Q4_K_M | 6.2 GB | ~7 GB | yes |
| Clef-Flash MLX 8-bit | 10.7 GB | ~11 GB (vendor peak 13.0 @16k) | yes |
| Clef-Flash bf16 (torch) | 19.1 GB | ~20 GB + KV | yes, tight |
| Kev-9B MLX adapter | 0.2 GB + 19.3 GB base | ≈36 GB during fold | **no** |
| Clef 27B MLX 4-bit (vendor) | 16.3 GB | 19.6 GB @16k | on paper (32 GB min), **untested here** |

## Caveats that belong next to the numbers

- The fastest row (`clef-flash-4bit-mlx`) is marked `valid: false`: macOS swap grew during the run,
  and a clean re-run reproduced that with no other workload (quality identical, p50 1,579 → 1,999 ms).
  Its **first-100 p50 (1,040–1,125 ms)** is the unthrottled number and the one quoted in the table;
  quality is unaffected. `peak_rss_mb` is a lower bound for MLX configs (Metal allocations are not
  fully counted); the llama.cpp numbers are the trustworthy memory figures.
- The dataset's case order is by workflow, so the cumulative p50/accuracy curves are also a warm-up
  curve. Cross-model comparisons at equal case counts are fair; a full run vs a smoke is not.
- d3-flash's llama.cpp number is a *port* number: the vendor's 0.717 was measured with vLLM and a
  fused kernel. The port is correct (`vllm-d3` decision type, schema-valid 1.00, 0.702 at 295 cases
  on the stopped run) but slow and heavy-tailed: p50 3.1 s, p95 16.7 s on a 100-case prefix. Its
  score questions are the weak spot (0.470 at that prefix). The full 400-case run was stopped when
  p50 reached 9.6 s; treat it as a research artifact, not a deployable config.
- **Do not plan on prompt caching for repeated states.** Even with `--cache-ram 4096`, Kev's cached
  re-ask improves by ~7 %, because the head needs hidden states the KV cache does not keep.
- All three models are text-only here. Clef-Flash and d3-flash are multimodal in their own runtimes;
  image/video input was not exercised, and the GGUFs need `--mmproj` for it.

## Next experiments, in value order

1. **Clef 27B MLX 4-bit** (`mlx-community/clef-4bit`, 16.3 GB, vendor peak 19.6 GB): does the quality
   step survive on 32 GB, and at what latency? This is the only "bigger model" question left open.
2. ~~Clef-Flash MLX 8-bit vs 4-bit~~ — **done**: 8-bit 0.699 vs 4-bit 0.697, ECE 0.009 vs 0.024,
   1.4× slower. 4-bit wins on this box; no further reason to spend 10.7 GB.
3. ~~Clean cool re-run of MLX 4-bit~~ — **done**: accuracy identical (0.6965), latency +27 % because
   the chassis was still warm, swap growth reproduced. The 1.8× engine gap stands on matched first-100
   numbers (1,040–1,125 ms MLX vs 2,143 ms llama.cpp).
4. **Per-workflow accuracy** (the harness now writes `logs/decision/cases_*.jsonl`): Kev's early-case
   weakness suggests the agent-trace workflow is where the spread lives.
5. **Kev-9B via MLX with a pre-merged full-weight checkpoint**: convert the adapter to bf16 and, if a
   streaming merge can stay under 32 GB, test whether MLX's engine advantage extends to the pointer
   head. Upstream has not measured Kev-9B on a Mac at all.
