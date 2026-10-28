#!/bin/bash
# Speed benchmark: llama.cpp (GGUF) via llama-bench.
# Emits JSONL to results/text/speed_llamacpp.jsonl with one row per test.
# Usage: benchmarks/text/bench_speed_gguf.sh <model.gguf> <label>
set -uo pipefail
# shellcheck source=../common/env.sh
. "$(dirname "$0")/../common/env.sh"
cd "$ROOT"
BENCH=./src/llama.cpp/build/bin/llama-bench
MODEL="$(model_path "$1")" || exit 1; LABEL="$2"
OUT="$RESULTS/speed_llamacpp.jsonl"

# One invocation per config so a failure (OOM) cannot kill the whole sweep.
run() { # $1=desc  $2..=extra args
  local desc="$1"; shift
  echo ">>> $LABEL | $desc" >&2
  "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -r "${REPS:-2}" -o jsonl "$@" 2>/dev/null \
    | python3 -c "
import json,sys
desc=sys.argv[1]; label=sys.argv[2]
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    d=json.loads(line)
    d['label']=label; d['config']=desc
    print(json.dumps(d))
" "$desc" "$LABEL" >> "$OUT"
}

# --- decode (tg) + prefill (pp) at zero depth, across KV cache types ---
for KV in f16 q8_0 q4_0; do
  run "kv=$KV depth=0"        -p 512 -n 128 -d 0 -ctk $KV -ctv $KV
done

# --- depth sweep (attention cost grows with KV length) on best KV type ---
# REP1 to keep runtime sane: each rep must prefill the full depth first.
REPS=1 run "kv=q8_0 depth=4096"  -p 512 -n 128 -d 4096  -ctk q8_0 -ctv q8_0
REPS=1 run "kv=q8_0 depth=16384" -p 512 -n 128 -d 16384 -ctk q8_0 -ctv q8_0
REPS=1 run "kv=q8_0 depth=32768" -p 512 -n 128 -d 32768 -ctk q8_0 -ctv q8_0

# --- prefill scaling (prompt processing throughput vs prompt size) ---
run "prefill=2048"  -p 2048  -n 0 -d 0 -ctk q8_0 -ctv q8_0
run "prefill=8192"  -p 8192  -n 0 -d 0 -ctk q8_0 -ctv q8_0

echo "DONE $LABEL" >&2
