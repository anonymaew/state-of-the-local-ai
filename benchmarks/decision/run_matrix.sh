#!/bin/bash
# One shot per 9B candidate on the typed-decisions test split.
#
#   benchmarks/decision/run_matrix.sh 2>&1 | tee logs/decision/matrix.log
#
# Each invocation is a fresh server process and a fresh model load, so a row is one
# measurement under ../common/README.md. Weights resolve from the Hub cache by alias
# (benchmarks/common/model_store.py); a checkpoint that is not cached is a setup fact,
# not a benchmark result, so it is skipped loudly.
set -euo pipefail
. "$(dirname "$0")/../common/env.sh"
B="$ROOT/benchmarks/decision"
PY="$ROOT/.venv/bin/python"
CTX="${CTX:-8192}"

run() { echo "### $(date -u +%H:%M:%S) $*"; "$@"; }

# one <engine> <alias> <model-id> <quant> <port> <label>
one() {
  local engine="$1" ref="$2" id="$3" quant="$4" port="$5" label="$6" path
  if ! path="$(model_path "$ref")"; then
    echo "SKIP $id: $ref is not in the cache (MODEL_STORE_ONLINE=1 to download)"
    return 0
  fi
  run "$PY" "$B/run_config.py" --engine "$engine" --model-path "$path" \
    --model-id "$id" --quant "$quant" --ctx "$CTX" --port "$port" --label "$label"
}

# 1. Clef-Flash 9B, MLX 4-bit (custom joint head, bundled clef_mlx.py loader)
one mlx-clef clef-flash-4bit clef-flash-4bit-mlx 4bit 8091 "clef-flash-4bit-mlx ctx=$CTX split=test"

# 2. Clef-Flash 9B, llama.cpp Q4_K_M (native CLEF decision arch)
one llamacpp clef-flash-q4km clef-flash-q4km Q4_K_M 8092 "clef-flash-q4km ctx=$CTX split=test"

# 3. Kev-9B, llama.cpp Q4_K_M (pointer head in GGUF; the MLX adapter path needs a 1.9x merge peak)
one llamacpp kev-9b-q4km kev-9b-q4km Q4_K_M 8093 "kev-9b-q4km ctx=$CTX split=test"

# 4. d3-flash 9B, llama.cpp Q4_K_M (vLLM-SR Decision 3.0; non-causal readout)
one llamacpp d3-flash-q4km d3-flash-q4km Q4_K_M 8094 "d3-flash-q4km ctx=$CTX split=test"

echo "### done; rows in $RESULTS/typed.jsonl"
