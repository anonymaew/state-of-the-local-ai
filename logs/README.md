# Logs

Raw engine output behind the numbers in [`results/`](../results/README.md). Unedited on purpose: the
stdout is what proves a run loaded the weights it claims, at the context it claims.

```text
logs/<modality>/<run>.log
logs/<modality>/<experiment>/          per-run settings, caches, sqlite state
```

## Text track

| Pattern | Produced by |
| --- | --- |
| `srv_*.log` | `llama-server` / `omlx serve` startup + request handling |
| `ab_llama_r*.log`, `ab_omlx_r*.log` | paired A/B rounds in `benchmarks/text/ab_mtp.py` |
| `q_<tag>_<task>.log` | `quality_mlx.sh` (lm-eval per task) |
| `bench_*.log`, `ppl_gguf.log` | speed/perplexity sweeps |
| `omlx/`, `omlx_paired/`, `omlx-base/` | oMLX experiments, including the ad-hoc harnesses (`bench.py`, `bench_stream.py`, `bench_concurrent.py`) that produced [`docs/text/06-omlx.md`](../docs/text/06-omlx.md) |
| `build_llamacpp.sh`, `dl*.sh` | engine build and model download scripts kept as provenance |

## Rules

- Logs are evidence, not scratch space. Do not truncate a log that a results row points to.
- A log directory may hold its own scripts (`logs/text/omlx/`). They ran in place and are kept there
  rather than rewritten into `benchmarks/` after the fact — moving them would break the record.
- New runs write through `common/env.sh` (`$LOGS`), so nothing new lands at `logs/` root.
- Disk is finite: compress (`gzip`) old logs rather than deleting them, and never delete the log behind
  a published number.
