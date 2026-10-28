#!/bin/bash
# Measure decode throughput (+ speculative-decoding acceptance) through llama-server.
# Usage: benchmarks/text/bench_server.sh <label> <model.gguf> [extra server args...]
# Emits one JSON row per repeat to results/text/server.jsonl
set -uo pipefail
. "$(dirname "$0")/../common/env.sh"
cd "$ROOT"
BIN=./src/llama.cpp/build/bin/llama-server
LABEL="$1"; MODEL="$(model_path "$2")" || exit 1; shift 2
PORT=${PORT:-18100}
NT=${NT:-128}
REPS=${REPS:-2}
CTX=${CTX:-8192}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/benchmarks/text/prompts/prompt_short.txt}
OUT="$RESULTS/server.jsonl"

# a fixed, non-chat prompt so timings are comparable across engines
[ -f "$PROMPT_FILE" ] || printf 'Explain how unified memory bandwidth limits token generation speed in local LLMs. Give concrete numbers and a worked example.\n' > "$PROMPT_FILE"

"$BIN" -m "$MODEL" -c "$CTX" -ngl 99 -fa on -np 1 --jinja --metrics --port "$PORT" "$@" \
  > "$LOGS/srv_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 240); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "SERVER FAILED $LABEL" >&2; tail -5 "$LOGS/srv_$LABEL.log" >&2; exit 1; }

sw0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
for i in $(seq 1 "$REPS"); do
  curl -sS "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' "$PROMPT_FILE"), \
         \"max_tokens\": $NT, \"temperature\": 0, \"cache_prompt\": false, \"n_predict\": $NT}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('timings') or {}
print(json.dumps({'label':sys.argv[1],'rep':int(sys.argv[2]),
  'prefill_tps':round(t.get('prompt_per_second') or 0,1),
  'decode_tps':round(t.get('predicted_per_second') or 0,1),
  'ttft_ms':round(t.get('prompt_ms') or 0,1),
  'prompt_n':t.get('prompt_n'),'predicted_n':t.get('predicted_n'),
  'swap_delta_mib':0}))" "$LABEL" "$i" >> "$OUT"
done

sw1=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
# patch swap delta into the rows just written
python3 - "$LABEL" "$((sw1-sw0))" "$OUT" <<'PY'
import json,sys
lab,swd,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
rows=[json.loads(l) for l in open(out)]
for r in rows:
    if r['label']==lab: r['swap_delta_mib']=swd
open(out,'w').write(''.join(json.dumps(r)+"\n" for r in rows))
PY

python3 - "$LABEL" "$sw0" "$sw1" "$OUT" <<'PY'
import json,statistics,sys
lab,out=sys.argv[1],sys.argv[4]
rows=[json.loads(l) for l in open(out) if json.loads(l)['label']==lab]
d=[r['decode_tps'] for r in rows if r['decode_tps']]
p=[r['prefill_tps'] for r in rows if r['prefill_tps']]
print(f"{lab}: decode {statistics.median(d):.2f} t/s | prefill {statistics.median(p):.0f} t/s | swap {sys.argv[3]}->{sys.argv[2]} MiB")
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
