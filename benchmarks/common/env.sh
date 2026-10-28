#!/bin/bash
# Shared layout for every modality harness.
#
# Source it from a bench script:
#   . "$(dirname "$0")/../common/env.sh"
#
# Exports (all overridable from the outside):
#   ROOT      repo root (from this file's location, not from the caller's cwd)
#   MODALITY  the harness folder name (text, image, video, audio, speech, decision)
#   HF_HOME   Hugging Face cache home            default ~/.cache/hf
#   HF_HUB_CACHE  model snapshots                default $HF_HOME/hub
#   MODELS    the Hub cache (weights live there; no copy in the repo)
#   DATA      modality datasets                   default $ROOT/data/$MODALITY
#   RESULTS   modality results (JSONL)            default $ROOT/results/$MODALITY
#   LOGS      modality raw logs                   default $ROOT/logs/$MODALITY
#   PRIMARY_TEXT_MODEL  alias of the resident vision-language model
#
# Model refs: harnesses take an alias (`qwen38-27b-oq4e-mtp`, `clef-flash-q4km`),
# a `org/repo[:file]` ref, or a filesystem path. `model_path <ref>` prints the path.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_src="${BASH_SOURCE[1]:-}"
if [ -n "$_src" ] && [ -f "$_src" ]; then
  MODALITY="$(basename "$(cd "$(dirname "$_src")" && pwd)")"
  export DATA="${DATA:-$ROOT/data/$MODALITY}"
  export RESULTS="${RESULTS:-$ROOT/results/$MODALITY}"
  export LOGS="${LOGS:-$ROOT/logs/$MODALITY}"
  mkdir -p "$RESULTS" "$LOGS"
else
  # Sourced from an interactive shell (e.g. to call model_path): there is no track,
  # so do not invent data|results|logs/common directories.
  MODALITY=""
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MODELS="$HF_HUB_CACHE"

export PRIMARY_TEXT_MODEL="qwen38-27b-oq4e-mtp"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"

# model_path <alias|org/repo[:file]|path> -> absolute path, cache-only by default.
# Set MODEL_STORE_ONLINE=1 to let it download a missing checkpoint.
model_path() {
  "$PYTHON_BIN" "$ROOT/benchmarks/common/model_store.py" resolve "$@"
}

# Fail early with an actionable message instead of a shell error deep in a run.
require_file() { # <path> <what-to-do>
  [ -f "$1" ] || { echo "missing $2: $1" >&2; return 1; }
}
