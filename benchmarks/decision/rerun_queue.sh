#!/bin/bash
# Re-run the three uncertain measurements, sequentially, on an idle machine.
# One process at a time: a second engine would heat the SoC and invalidate the latency.
# Weights resolve from the Hub cache by alias (benchmarks/common/model_store.py).
set -uo pipefail
. "$(dirname "$0")/../common/env.sh"
B="$ROOT/benchmarks/decision"
PY="$ROOT/.venv/bin/python"

run() { echo "### $(date -u +%H:%M:%S) $*"; "$@"; echo "### rc=$? $(date -u +%H:%M:%S)"; }

# R1 — clean full run of the fastest config. Resolves valid=false caused by swap growth.
if R1="$(model_path clef-flash-4bit)"; then
  run "$PY" "$B/run_config.py" --engine mlx-clef --model-path "$R1" \
    --model-id clef-flash-4bit-mlx --quant 4bit --ctx 8192 --port 8091 \
    --label "clef-flash-4bit-mlx ctx=8192 split=test rerun2"
else echo "SKIP r1: clef-flash-4bit not in cache"; fi

# R3/R4 — does --cache-ram fix the cached re-ask? Same 100 cases, one flag apart.
if KEV="$(model_path kev-9b-q4km)"; then
  run "$PY" "$B/run_config.py" --engine llamacpp \
    --model-path "$KEV" --model-id kev-9b-q4km --quant Q4_K_M --ctx 8192 --port 8093 --limit 100 \
    --label "kev-9b-q4km ctx=8192 limit=100 no-cacheram"
  run "$PY" "$B/run_config.py" --engine llamacpp \
    --model-path "$KEV" --model-id kev-9b-q4km --quant Q4_K_M --ctx 8192 --port 8093 --limit 100 \
    --label "kev-9b-q4km ctx=8192 limit=100 cache-ram=4096" \
    --engine-arg=--cache-ram --engine-arg=4096
else echo "SKIP r3/r4: kev-9b-q4km not in cache"; fi

# R2 — 8-bit vs 4-bit on the same suite. Only if the download finished.
if R2="$(model_path clef-flash-8bit)"; then
  run "$PY" "$B/run_config.py" --engine mlx-clef --model-path "$R2" \
    --model-id clef-flash-8bit-mlx --quant 8bit --ctx 8192 --port 8095 \
    --label "clef-flash-8bit-mlx ctx=8192 split=test"
else echo "SKIP r2: clef-flash-8bit not in cache"; fi

echo "### queue done $(date -u +%H:%M:%S)"
