# `data/decision`

Inputs for the decision track. No prompts are synthesized here: the scored suite is public and
frozen, so a number can be compared with the published leaderboard.

| Path | What it is |
| --- | --- |
| `manifest.json` | Suite registry: `typed-decisions` (primary for the 9B comparison), `breadth-v1-dev` / `breadth-v1-test` (Kev's native panel), `smoke` |
| `smoke.jsonl` | 20 hand-written plumbing checks — choice with 2/8/255 options, noul, score, option-order invariance, a 6k state. **Never report scores from it.** |
| `cache/typed-decisions/` | The `LocalLLaMA/typed-decisions` parquet files (not committed; fetch below). `all/test-*.parquet` is the 400-case / 2,000-decision suite the harness scores. |
| `model_eval/` | Upstream leaderboard rows shipped with the Clef-Flash checkpoint (not our measurements); kept as provenance for the published reference points. |

Restore the cache after a clone (this also pins `HF_HOME` to the project cache):

```bash
. benchmarks/common/env.sh
.venv/bin/hf download LocalLLaMA/typed-decisions --repo-type dataset \
  --local-dir data/decision/cache/typed-decisions
```

The `breadth-v1` partitions live in the Kev clone (`src/kev/evals/breadth-v1/`); their public
mirror is private for licensing reasons, which is why the loader resolves the git copy first.
Metrics from `breadth-v1` and `typed-decisions` are **not comparable** — different suites,
different scoring.
