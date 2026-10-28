#!/bin/bash
# Run each lm-eval task in its own process: avoids mlx "Resource limit (499000)".
# Usage: benchmarks/text/quality_mlx.sh <model_dir> <out_tag> <limit> <task...>
set -uo pipefail
# shellcheck source=../common/env.sh
. "$(dirname "$0")/../common/env.sh"
cd "$ROOT"
set -a; . ./.env; set +a
M="$(model_path "$1")" || exit 1; TAG="$2"; LIM="$3"; shift 3
mkdir -p "$RESULTS/quality/$TAG"
for T in "$@"; do
  echo ">>> $TAG $T"
  MLX_MAX_OPS_PER_BUFFER=100 .venv/bin/mlx_lm.evaluate --model "$M" --tasks "$T" \
    --num-shots 0 --limit "$LIM" --batch-size 2 --seed 42 \
    --output-dir "$RESULTS/quality/$TAG" > "$LOGS/q_${TAG}_${T}.log" 2>&1
  echo "    exit=$? $T"
done
echo DONE_ALL
