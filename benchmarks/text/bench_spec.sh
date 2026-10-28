#!/bin/bash
# Speculative-decoding / plain decode comparison via llama-cli.
# Usage: benchmarks/text/bench_spec.sh <label> <model.gguf> [extra args...]
# Emits one JSON row per repeat: decode t/s, prefill t/s, draft acceptance if present.
set -uo pipefail
. "$(dirname "$0")/../common/env.sh"
cd "$ROOT"
CLI=./src/llama.cpp/build/bin/llama-cli
LABEL="$1"; MODEL="$(model_path "$2")" || exit 1; shift 2
PROMPT="${PROMPT:-Write a detailed technical explanation of how unified memory bandwidth limits token generation speed in local LLMs. Include concrete numbers.}"
NT="${NT:-128}"
REPS="${REPS:-3}"
OUT="$RESULTS/spec.jsonl"

for i in $(seq 1 $REPS); do
  "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NT" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
        --no-warmup -no-cnv --seed 42 "$@" 2>&1 >/dev/null \
  | python3 -c "
import re,sys,json
txt=sys.stdin.read()
def g(pat):
    m=re.search(pat,txt)
    return float(m.group(1)) if m else None
tg=g(r'eval time =\s*[0-9.]+ ms /\s*[0-9]+ (?:runs|tokens).*?([0-9.]+) tokens per second')
pp=g(r'prompt eval time =\s*[0-9.]+ ms /\s*[0-9]+ tokens.*?([0-9.]+) tokens per second')
draft=g(r'draft acceptance rate\s*=?\s*([0-9.]+)')
acc=g(r'([0-9]+) of ([0-9]+) accepted')
row={'label':sys.argv[1],'rep':int(sys.argv[2]),'decode_tps':tg,'prefill_tps':pp,
     'accepted':(int(acc.group(1)) if acc else None),'drafted':(int(acc.group(2)) if acc else None),
     'raw_ok': bool(tg)}
print(json.dumps(row))
" "$LABEL" "$i" >> "$OUT"
done
python3 -c "
import json,statistics,sys
rows=[json.loads(l) for l in open(sys.argv[2]) if json.loads(l)['label']==sys.argv[1]]
v=[r['decode_tps'] for r in rows if r['decode_tps']]
if v: print(f'{sys.argv[1]}: median decode {statistics.median(v):.2f} t/s over {len(v)} reps  {v}')
else: print(sys.argv[1]+': no data')" "$LABEL" "$OUT"
