# Decision evaluation — item sets, Decision Index, calibration

What a *System One* decision model is scored on, which item sets are frozen, and how the numbers are computed. Read from
code and manifests on disk in `src/kev` (a clone of `github.com/jaredpalmer/kev`), not from the web: the Hub was
unreachable for this run, and what could not be checked locally is marked **UNVERIFIED** in §9. §2 is the formula, §3 the
item-set contract, §7 what "done" means for the harness.

## 1. What is being measured

A decision model answers typed questions about a state and returns a **distribution**, not text:

| type | criteria | returned | scored as |
| --- | --- | --- | --- |
| `choice` | 1..255 named options, `desc` or `null` | `probabilities` by option name, `choice` = argmax | accuracy, Brier, ECE, NLL |
| `noul` | optional true/false descriptions | `noul` = p(true) | as a 2-option choice |
| `score` | 1..255 ordered levels | mean level, `probabilities` by level index | the above + MAE in levels, RPS |

Every question is a categorical distribution over K options (K = 2 for `noul`), and every metric below is defined on it.
`confidence` in the response is **not** accuracy and must not be scored as such: for `choice` it is
`(p_max − 1/K)/(1 − 1/K)`, for `score` `max(0, 1 − E|level − mode|/D)` — TypeSafe adapter 0.2.1 formulas.

## 2. The community Decision Index

### 2.1 Where it is defined

`evals/breadth-v1/manifest.json` holds the upstream pointer and the only copy of the formula in this tree:

```text
space      multimodalart/jev-decision-index @ 28bab1369865c3ca0aa2354564f3ae72e103ef2a
file       data/methodology.json
edition    id "v0.2", label "Decision Index 0.2", date 2026-09-24
panel      decision-index-0.2, jev-1.13.0, edition "Full release v1"
```

The local pin is **0.2**, not 0.2.1 — 0.2.1 is the version of TypeSafe's adapter (`system-one-adapter`), where the
`confidence` formulas come from. Whether the Index has published a 0.2.1 edition is UNVERIFIED.

### 2.2 The formula, verbatim

`decision_index.formula`:

```text
per benchmark k = clip((score - chance) / (1 - chance)) on the coverage-adjusted score (unanswered = wrong);
area = mean of its benchmarks;
index = 100 x mean of the five areas
```

`scoring.chance` and `scoring.skill` (same file, notes for the local panel):

```text
chance = expected score of uniform random answers:
         mean of 1/K over questions            (accuracy)
         mean over records of the product 1/K  (case_exact)
skill  = clip((score - chance) / (1 - chance)) with unanswered questions scored wrong;
         area = mean of its datasets;  index = 100 x mean of the five areas
```

`scripts/breadth_report.py` is the executable form and fixes what the prose leaves open:

```python
def clip(x):  return min(1.0, max(0.0, x))
def skill(score, chance):  return clip((score - chance) / (1 - chance)) if chance < 1 else 0.0
def option_count(q):  return 2 if q["type"] == "noul" else len(q["criteria"])
```

Together:

```text
K_j     = 2 for noul, len(criteria) otherwise                  # per question j
score   = (answered & correct) / N_units                       # coverage-adjusted: a missing row counts wrong
chance  = mean over units of 1/K_j                             # metric "accuracy"
          mean over records of prod_j 1/K_j                    # metric "case_exact"
skill_k = clip((score - chance) / (1 - chance), 0, 1)          # [0,1]: below chance scores 0, not negative
area_a  = mean over datasets d in a of skill_d                 # equal weight per dataset inside an area
INDEX   = 100 * mean over the 5 areas of area_a                # equal weight per area => 20 points per area
```

Five properties a reimplementation gets wrong:

1. **Chance is per dataset**, computed from the option counts of the questions in that partition, not from a constant
   table.
2. **Units, not rows.** `accuracy` counts *questions*; `case_exact` counts a *record* as one unit, correct only if every
   question in it is right, with `chance = prod_j 1/K_j`. SATA-Bench (974 questions / 150 records) and BFCL (305 / 150)
   are worth the same as any other dataset, not 6x.
3. **Coverage adjustment.** A refused, over-long or failed record scores wrong against the *full* partition, so a model
   that answers 20% of a dataset cannot win it. `answered` is reported next to `score` for this reason.
4. **Clipping is one-sided in practice** — above chance the value is already in [0,1); the clip only rescues below-chance
   performance into 0.
5. **Aggregation is a mean of means**: a 3-dataset area weights each dataset 1/15 of the index, and `contractnli`
   (40 records) counts 1/10 as much as `hellaswag` (150). That is the design, not an accident.

Bootstrap: `bootstrap_index` resamples **records within each dataset** (the same draw for every system compared),
recomputing skills, area means and the index — 2,000 resamples by default, paired. Report the index with that CI, never a
bare point estimate.

### 2.3 `breadth-v1` is not the Index's own data

`scripts/build_breadth_v1.py`: breadth-v1 "mirrors that structure with the fifteen held-out datasets it names… built from
the original datasets, not from the Index's prepared requests, and maps each one onto Kev's typed questions… That makes
these questions easier than the Index's versions, so scores are not comparable to the Index's numbers, only to other
systems read on this panel." The per-dataset metric differs too: the Index scores `contractnli`/`clinc150`/`sgd` with
macro-F1, `bright`/`toolret` with nDCG@10, `routerbench` with "selected quality", while breadth-v1 makes all of them
accuracy-or-case-exact over typed questions. A local index is a *chance-corrected breadth index in the Decision Index's
shape*, and must be labelled so.

## 3. `breadth-v1`: the frozen panel

`src/kev/evals/breadth-v1/` — `manifest.json` plus `development.jsonl` and `test.jsonl` (`in_git: true`; this clone lacks
the partitions, §6).

| | records | questions | bytes | sha256 (12) |
| --- | --- | --- | --- | --- |
| `development.jsonl` | 1,990 | 3,075 | 5,032,194 | `9aad8f4a7374` |
| `test.jsonl` | 1,990 | 3,089 | 5,127,662 | `91a64f0a9448` |
| total | 3,980 | 6,164 | 10,159,856 (9.7 MiB) | |

`test` is `locked`. Protocol behind the published numbers: select on `development`, read `test` **once** per candidate with
`--allow-test`, never for search. A local quant/engine sweep runs on `development`.

### 3.1 The 14 datasets

All 14 are `eval_only` for Kev — none is a training source. Counts are per partition (dev and test are the same size by
construction). `chance` is the manifest's freeze-time value for the test split; the authoritative value is recomputed
from the questions at scoring time (§2.2), so treat the column as a checksum on your implementation.

| # | dataset | area | metric | records | questions | chance | shape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `musr` | knowledge | accuracy | 150 | 150 | 0.385222 | choice, published options |
| 2 | `sata_bench` | knowledge | case_exact | 150 | 974 | 0.017917 | one noul per candidate, ≤10 |
| 3 | `chessbench` | knowledge | accuracy | 150 | 150 | 0.127579 | choice over ≤10 legal moves |
| 4 | `contractnli` | language | accuracy | 40 | 160 | 0.333333 | choice 3-way, full NDA state |
| 5 | `hellaswag` | language | accuracy | 150 | 150 | 0.25 | choice 4 endings |
| 6 | `clinc150` | retrieval | accuracy | 150 | 150 | 0.1 | choice 10 (9 intents + `out_of_scope`) |
| 7 | `sgd` | retrieval | accuracy | 150 | 150 | 0.352222 | choice over service intents + NONE |
| 8 | `bright` | retrieval | accuracy | 150 | 150 | 0.2 | choice 5 document chunks |
| 9 | `bfcl` | tools | case_exact | 150 | 305 | 0.336094 | one noul per function, ≤10 |
| 10 | `toolret` | tools | accuracy | 150 | 150 | 0.2 | choice 5 tools |
| 11 | `apibank` | tools | accuracy | 150 | 150 | 0.125119 | choice ≤8 APIs |
| 12 | `routerbench` | tools | accuracy | 150 | 150 | 0.206918 | choice over models |
| 13 | `humicroedit` | arts | accuracy | 150 | 150 | 0.5 | choice between two edits |
| 14 | `cfcolor` | arts | accuracy | 150 | 150 | 0.5 | choice between two palettes |

Areas: knowledge 3, language 2, retrieval 3, tools 4, arts 2. ContractNLI is 40 documents × up to 4 hypotheses
(`CONTRACTNLI_DOCS=40`, `CONTRACTNLI_PER_DOC=4`). **No dataset uses `score`**: the panel is `choice` + `noul` only, so
score-shaped evaluation must come from `decision-v1`/`decision-v2` (sst5, yelp 5-level legends) or your own set.

Splits: native dev/test kept for `contractnli` (dev.json/test.json), `clinc150` (plus validation/test), `sgd` (dev/test
dialogues), `humicroedit` (subtask-2 dev.csv/test.csv); everything else is split into halves by group (story, video,
dialogue, query, position, user, …) in a seeded order so no group spans both. Admission: every record fits 7,680 tokens
with the Qwen3.5 tokenizer, so it also fits an 8,192-token server; `contractnli` states are the long ones (~7k).

### 3.2 Record schema

One JSON object per line. `_meta.variant` is `clean` for every record here; other suites carry `permuted` /
`none_present` / `none_absent` variants that must be filtered out.

```json
{"state": {"subject": "…"} | "text" | [{"role": "…", "content": "…"}],
 "questions": {"<qid>": {"type": "choice", "instructions": "…",
                         "criteria": {"<option key>": "<description>" | null},
                         "label": "<option key> | true|false | <int level>", "src": "<dataset>"}},
 "_meta": {"source": "musr", "id": "musr/test/12", "group_id": "musr/test/12", "variant": "clean",
           "split": "test", "repo": "TAUR-Lab/MuSR", "revision": "7c365b43…", "row_sha256": "…", "row": 13292}}
```

`score` labels are ints, `noul` bools, `choice` option keys. `group_id` is the bootstrap cluster. `api.render()` turns a
dict/list state into labelled text — the harness must not pre-flatten it.

## 4. Calibration

From `kev/metrics.py` (pure numpy over rows `{p, label, type, keys, task, source, variant, logits?}`).

### 4.1 Brier

```python
brier_row = ((p - onehot(label)) ** 2).sum()     # multiclass Brier over the full option set
brier     = mean over rows                       # range 0..2, not the binary 0..1
```

`score` questions also get `score_mae` (levels, from `p @ arange(L)`) and `ranked_probability_score`
(`mean((cumsum(p)[:-1] - cumsum(onehot)[:-1])**2)`). Never compare against a binary Brier from another tool.

### 4.2 ECE

```python
def ece(conf, correct, bins=10):
    edges = linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)   # last bin closed
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e
```

Ten **equal-width** bins over top-1 confidence, mass-weighted, on `p.max()` and top-1 correctness.
[`01-plan.md`](01-plan.md) says "10 equal-mass bins" — that contradicts the code, and the code produced every published
number. Follow the code; fix the plan doc. `breadth_report.py` computes ECE **on answered rows only**, per dataset, per
area (pooled) and overall, plus `mean_dataset_ece` (mean of the 14 dataset ECEs): report both, since pooled rewards the
easy datasets and the mean matches the index's equal-weight-per-dataset logic.

### 4.3 Everything else already implemented, to carry in the row

`nll` (exact from logits when recorded, else floored probabilities, `EPSILON=1e-9`), `mean_conf`,
`confidence_bias = mean(p_max) − accuracy`, `confident_error_rate` (p_max ≥ 0.9 and wrong, over all questions),
`coverage_at_0_9`, `coverage_at_5pct_error` / `coverage_at_1pct_error` (largest accepted share at a fixed error budget, in
descending confidence — the "automate X% of decisions" number), `aurc`, `accuracy_at_0_9`. Policy version 2: ties
handled by whole confidence groups, and coverage is an "in-sample maximum over confidence thresholds; not a deployed
error guarantee".

### 4.4 Temperature

```python
TEMPERATURE_FIT = {"aggregation": "micro", "points": 121}   # shipped checkpoints
# "min micro mean NLL over a 121-point log grid 0.25..4"     (research trials: 81 points, macro default)
candidates = exp(linspace(log(0.25), log(4), points))
T_hat      = argmin over candidates of mean NLL over clean rows
```

Rules the code enforces and the harness must too:

- fit on **raw** logits only (`inference_temperature == 1.0`; `raw_row()` restores them), never on calibrated outputs;
  `metrics.served(fit_rows, eval_rows)` is the served-vs-served path;
- accuracy is temperature-invariant, Brier/ECE/NLL are not — every row records the temperature it was scored at;
- report the **shipped** value separately from any local refit: 0.8B 2.35, 4B 2.41, 9B 2.19, 27B 1.32 (the last two fitted
  on held-out data, the first two in-distribution);
- report the out-of-fold estimate (`cross_validated_temperature`: 5 folds disjoint in `(source, group)`, per-fold
  temperature applied to held-out rows, paired bootstrap CI on ΔECE, `separated` flag). 121 points on 100 items
  overfits; OOF is the honest number and it is cheap here. `KEV_TEMPERATURE=1.0` serves raw probabilities — the way to
  get unimpeachable numbers across engines.

Sanity target: on new sources the shipped temperature takes Kev-9B's ECE 0.103 → 0.041 and confident errors 8.2% → 2.4%.

## 5. `typed-decisions` — verifiable, and now the primary suite

**JevBench remains UNVERIFIED.** Nothing about its items, tiers or scoring exists in this tree; the only trace is a
`kev/metrics.py` docstring crediting "jev-benchmarks' 'coverage at a fixed error budget'" for one *metric*, and
`scripts/screen_overlap.py` references a build-time path `/tmp/jevbench/datasets/public` that is not shipped. Do not
cite it.

**`typed-decisions` was verified on 2026-10-27 and is the suite this track scores.** It is public and small:

| Property | Value |
| --- | --- |
| Repo | `LocalLLaMA/typed-decisions` (dataset, Apache-2.0) |
| Config / split | `all`, `test` — 400 cases, 5 typed questions each = **2,000 decisions** |
| Workflows | `agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents` (100 cases each, in that order) |
| Files | `all/test-00000-of-00001.parquet` (222 KB) + four per-workflow parquets (1 MB total) |
| Columns | `id, workflow, state, questions, gold, factors, label_agreement, n_questions`; `state/questions/gold` are JSON strings |
| Gold | a full probability distribution per question, plus `label`/`confidence`; `noul` uses `true`/`false` keys |
| Official scorer | `bench/score_with_jobs.py` — accuracy, KL(gold‖pred), multiclass Brier, ECE over 10 **equal-width** bins, with `yes/no → true/false` aliasing |
| Reference rows | `Prior` (ignores input) 0.470 acc; `Uniform` 0.308 |

Why it is the primary: every 9B candidate publishes a zero-shot number on this exact split (clef-flash 0.710,
d3-flash 0.717, Amos-1-medium 0.765, Jev 0.727), it fits in 1 MB, and the scorer fixes the four metric definitions
instead of leaving them to us. Its weaknesses are recorded too: cases are ordered by workflow (difficulty drift), and
`score` questions have only 3–4 levels, so ECE has few bins.

`breadth-v1` stays as the secondary, Kev-native panel — but see §6: the partitions are absent from this clone and its
mirror is private, so it cannot be scored here without a full clone or access.

Available and adjacent: `evals/external/typesafe-v1` (20 cases / 102 questions) was **removed** as a gate on 2026-09-30 —
gold is the argmax of two frontier models averaged, 13 of 102 references below 0.75, split-half r ≈ −0.01 across 20
checkpoints. `wanli-v1/v2` (27% annotator-disagreement labels) and `scienthoon-v1` (saturated queue, unlearnable priority)
likewise. `REMOVED_SUITES` is the best evidence of which public decision sets are not worth scoring.

## 6. What is on disk here, and the substitutes

`evals/breadth-v1/` has `manifest.json` only in this clone — both partitions are absent
(`evals/decision-v1/development.jsonl` *is* present, so this is breadth-v1-specific, not a general blob filter). Access:

1. **the git copy** — `in_git: true`, so a full clone has them; `git checkout -- evals/breadth-v1`.
   On 2026-10-27 the clone in `src/kev` did **not** have them (`load_suite.py` fails with
   `missing development.jsonl`), and its HF mirror is private. Treat `breadth-v1` as unavailable until a
   fresh clone proves otherwise; `typed-decisions` is what actually ran.
2. **the pinned mirror** — `jaredpalmer/kev-private-evals@aeaaed0f7532…`, **private**: SATA-Bench is CC-BY-NC-4.0, SGD
   CC-BY-SA-4.0, Humicroedit has no licence, cfcolor a code-only notice. So "download only the test split of breadth-v1
   from `jaredpalmer/kev-suites`" is **not possible**: `kev-suites@cc4bac80…` is the general mirror
   (`kev.suite.SUITES_DATASET`), but breadth-v1's manifest names its own private mirror and `fetch_partition` prefers
   that. Expect the `kev-suites` path to 404; the loader treats that as a loud error, never an alternative file.

| substitute | access (pinned) | usable as | caveat |
| --- | --- | --- | --- |
| `decision-v1` | local `evals/decision-v1/`: 588 dev / 588 test records, 852 questions; banking77, boolq, ag_news, mnli, sst5, yelp | **trained sources** only | all six are in Kev's SFT corpus (`decision-v2.trainable_sources`) → memorisation-flavoured, not held-out |
| `transfer-v4` / `v9` | local `evals/v4/…`, `evals/v9/…`: 764 records; QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval | held-out, small | one axis only: no tools, retrieval or arts |
| BFCL raw | `gorilla-llm/Berkeley-Function-Calling-Leaderboard@61fc0608`, `BFCL_v3_*.json` + `possible_answer/` | tools axis | that pin is **BFCL v3**, not v4 |
| CLINC150 | `clinc/clinc_oos@155b9c71`, `plus/{validation,test}-*.parquet` + `clinc/oos-eval` `domains.json` | retrieval axis | 151 intents = a 151-option question, not the panel's 10 |
| CUAD | not pinned anywhere here; model cards measure long-contract accuracy on it | long-state behaviour | no local pin → no sha256, no reproduction |
| τ-bench / τ²-bench | appear nowhere in this tree; MMLU-Pro is inside `transfer-v9` | multi-turn | breadth-v1 is single-pass by design: a real gap |

## 7. The local harness input

Resolved by `benchmarks/decision/load_suite.py`: `data/decision/manifest.json` (suites, caps, budgets, provenance,
validity) and `data/decision/smoke.jsonl` (20 items, plumbing only).

### 7.1 Primary scored suite

`breadth-v1` **development**, 14 datasets, fixed cap per dataset, ≤30 min here.

```text
cap_records   = 6 per dataset (contractnli 3: its states are ~5-7k tokens)
records       = 13*6 + 3 = 81
questions     ≈ 81 + 5 (bfcl) + 5*5.4 (sata) + 3*3 (contractnli) ≈ 128
state budget  = 150,000 tokens (manifest.budget.max_total_state_tokens)
time          = 150,000 / 90 tok/s ≈ 28 min prefill-only (question passes and weight load extra)
```

The 90 tok/s is an **assumption, not a measurement**: the repo has no Kev prefill rate for this machine (README: Kev-4B
721 ms for five questions, 136 ms cached, on an M5, state length unstated). So the cap is a knob: run `--dry-run`,
compare against a stopwatch, adjust `cap_records` **once, before recording a result**. A changed cap is a changed item set,
hence a new label.

What it is and is not: a plumbing-and-direction instrument. 6 items per dataset gives per-dataset skill a ~±20 pp standard
error, so report the **index with its record-clustered bootstrap CI**, the five area skills and per-dataset
answered-share/ECE — never a single dataset's number as a result. Selection is deterministic:
`sha256("<seed>:<source>:<id>")` ascending, first `cap` — same manifest, same items, two systems paired.
`breadth-v1-test` is declared with the same caps behind `--allow-test` for the one headline read per candidate.

### 7.2 The 20-item smoke set

Plumbing only: these items never contribute to a reported score, and are not representative of anything (they were written
by hand to be unambiguous, which real items are not).

| items | what it pins down |
| --- | --- |
| `smoke-01`, `02` | `choice` K=2; 02 embeds `<state>`, `</opt>`, `<decide>` in the state — unescaped delimiters must fail loudly, not silently shift the answer |
| `smoke-03`, `04` | `choice` K=8, same item with options reversed (`pair: "order8"`); argmax must agree and the p-vector permute — the order-shuffling invariance check |
| `smoke-05`, `06`, `07` | `choice` K=255 (API ceiling), gold at index 137, 255, 1 — the two ends are positional-bias probes. Options materialized from `criteria_builder`, so the file stays readable |
| `smoke-08`, `09` | `noul` true / false, unambiguous |
| `smoke-10` | `noul` with the deciding evidence removed; only confidence is meaningful (`expect.max_prob_below: 0.9`) |
| `smoke-11` | `score` L=3 (`Calm / Frustrated / Very angry`) |
| `smoke-12`, `13`, `14` | `score` L=5, gold at 2, 0, 4 — the edges check expected-level MAE and the extreme levels |
| `smoke-15`, `16`, `17` | ~6,000-token state (`target_chars: 24000`, filler + one needle) with the needle at 50%, 98%, 2% — long prefill, recency, primacy |
| `smoke-18`, `19` | dict state and list-of-messages state: `api.render()` on JSON, not a string |
| `smoke-20` | three questions in one request with a reference number in a **sibling** question: the probe must answer `none` (question isolation), the `score` question must be unaffected |

Checks before any real run: 20/20 items load and validate; every `gold` indexes into its `criteria`; `smoke-03/04` agree
on argmax; `smoke-05/06/07` return 255 probabilities summing to 1 ± 0.02; `smoke-15/16/17` report *measured*
`state_tokens` ≈ 6,000; `smoke-20.probe == "none"`. Any failure is a harness bug, not a model result.

### 7.3 What the harness writes

Rows follow `benchmarks/common/row.schema.json` with `modality: "decision"`, `task: "typed-decision"` and metrics
`accuracy`, `brier`, `ece`, `nll`, `mean_conf`, `coverage_at_5pct_error`, `latency_ms_p50`, `cached_latency_ms_p50`,
`state_tokens`, `questions_per_request`, plus `harness.{engine,quant,device,temperature,temperature_source}`, with
`label` carrying `suite/split/cap/seed/sha256`. Aggregates go to a `breadth_report.py`-shaped `report.json` (`datasets →
{score, chance, skill, answered, calibration}`, `areas → {skill, score, calibration}`, `overall → {index, ...}`) so local
numbers diff against upstream's directly.

## 8. Contradictions and gaps recorded

1. **Index edition**: local pin `v0.2` (2026-09-24) vs the brief's 0.2.1 (which is the adapter's version).
2. **Panel size**: `build_breadth_v1.py` docstring says the Index "groups 40 benchmarks"; the manifest's
   `decision_index.datasets` lists 15 (14 + GPQA Diamond, skipped as Hub-gated and not publishable in plain text);
   breadth-v1 implements 14.
3. **ECE bins**: plan doc says equal-mass, `kev/metrics.py` does equal-width.
4. **Two `chance` columns**: the Index block's chances (musr 0.371, sata 0.0131, bfcl 0.2592) describe the Index's
   prepared subsamples; breadth-v1's `datasets[*].chance` (0.391667, 0.01901, 0.336094) describe its own. Never mix.
5. **Published numbers to reproduce** (sanity target, not a gate): index on test — 0.8B 23.3, 4B 38.0, 9B 41.0, 27B 52.3,
   Jev 54.0; new-source accuracy 0.648 / 0.817 / 0.820 / 0.851 (Jev 0.857); Brier 0.481 / 0.269 / 0.289 / 0.225 (Jev 0.211).

## 9. TODO(verify) — needs the network

- [ ] verbatim `data/methodology.json` at `multimodalart/jev-decision-index@28bab136…`: whether `clip` is [0,1] upstream,
      whether a 0.2.1 edition exists;
- [ ] whether `jaredpalmer/kev-suites@cc4bac80…` contains `breadth-v1/{development,test}.jsonl` (expected: no), plus its
      tree size, per-file sizes and licence (not in this tree);
- [ ] JevBench: public items, tiers, scoring (still unverified).
- [x] `typed-decisions`: schema, access, scorer — verified 2026-10-27 (§5).
- [ ] CUAD access path and licence, if long-state calibration is reported.

Files read for every number above: `evals/breadth-v1/manifest.json`, `scripts/breadth_report.py`,
`scripts/build_breadth_v1.py`, `kev/metrics.py`, `kev/benchmark.py`, `kev/suite.py`, `kev/api.py`, `README.md`. There is
no `kev/decision_index.py`: the formula lives in the manifest and `breadth_report.py`.
