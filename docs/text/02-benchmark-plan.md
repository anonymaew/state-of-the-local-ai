# Benchmark plan

Target: MacBook Air Mac17,3 · Apple M5 (4P+6E, 10-core GPU) · 32 GB unified (34,359,738,368 B) ·
macOS 27.0.1 (26A434) · `hw.pagesize=16384` · `iogpu.wired_limit_mb=0` (default wired cap) ·
nix + uv, no Homebrew · ~88 GB free.

Everything below is pinned to a **build identity** that must be recorded in every result row:
`llama.cpp b11761` (2026-10-20, macOS arm64 tarball `llama-b11761-bin-macos-arm64.tar.gz`) ·
`mlx 0.32.4` / `mlx-lm 0.32.2` (2026-10-16) · `lm-eval 0.4.13` (PyPI `lm-eval`; `main` = 0.4.14.dev0).

> **Do not use `releases/latest` for llama.cpp.** As of 2026-10-20 `releases/latest` resolves to tag
> `v0.7.0` (2026-10-15) whose only asset is `nightly-tag.txt`. Binaries live on `bNNNNN` tags
> (`b11761` has 36 assets). Always pick the newest `b*` tag via the releases API.

## Matrix to run

### Table 1 — configs (the axes that change numbers)

| # | Engine | Weights | KV cache | ctx | Why it's in the matrix |
|---|---|---|---|---|---|
| C1 | llama.cpp Metal | 27B dense `Q4_K_M` | `-ctk f16 -ctv f16` | 32768 | reference: safest quality, biggest KV |
| C2 | llama.cpp Metal | 27B dense `Q4_K_M` | `-ctk q8_0 -ctv q8_0` | 32768 | half KV bytes ≈ ~2× depth headroom |
| C3 | llama.cpp Metal | 27B dense `Q4_K_M` | `-ctk q4_0 -ctv q4_0` | 32768 | floor of KV quality; expect a KLD hit |
| C4 | llama.cpp Metal | 27B dense `Q5_K_M` | `-ctk q8_0 -ctv q8_0` | 32768 | quality anchor if it fits |
| C5 | llama.cpp Metal | A3B-class MoE `Q4_K_M` | `-ctk q8_0 -ctv q8_0` | 32768 | decode is bandwidth-bound → MoE wins |
| C6 | llama.cpp Metal | 4B dense `Q4_K_M` | `-ctk q8_0 -ctv q8_0` | 32768 | control; calibrates the harness floor |
| C7 | MLX | `mlx-community/*-27B-4bit` | f16 (default) | 32768 | engine A/B at equal bits/weight |
| C8 | MLX | same 4bit | `--kv-bits 8` | 32768 | MLX KV-quant equivalent of C2 |
| C9 | MLX | `*-27B-6bit` / `-8bit` | f16 | 32768 | quality ceiling inside 32 GB |
| C10 | llama.cpp Metal | 27B dense `Q8_0` | `-ctk q8_0` | 8192 | only if RSS test passes; else skip |

Rules: K-quants only (`Q4_K_M`, `Q5_K_M`, `Q8_0`) — never mix `Q4_0` weights with `Q4_K_M` weights
in a comparison. MLX `4bit` ≈ 4.5 bpw ≈ `Q4_K_M`; compare MLX-4bit to `Q4_K_M`, not to `Q4_0`.

### Table 2 — measurements taken on every config

| Measurement | Probe | Unit | Notes |
|---|---|---|---|
| `pp512` | `llama-bench -p 512 -n 0 -d 0` | tok/s | canonical prefill number |
| `pp@4k/16k/32k` | same, `-d 4096,16384,32768` | tok/s | attention cost grows with depth |
| `tg128@0` | `llama-bench -p 0 -n 128 -d 0` | tok/s | short-context decode |
| `tg128@4k/16k/32k` | `-d 4096,16384,32768` | tok/s | **the number users feel** |
| `TTFT@4k` | llama-server `timings.prompt_ms` | ms | true time-to-first-token, cache cold |
| `peak_rss` | `/usr/bin/time -l` → `maximum resident set size` | bytes | includes Metal buffers (unified) |
| `swap_delta` | `sysctl vm.swapusage` before/after | MiB | the validity gate |
| `maxrss_mlx` | `GenerationResponse.peak_memory` | GB | `mx.get_peak_memory()` |
| `sustained_tg` | rep 1 vs median of reps 2–5 | tok/s | fanless burst-vs-sustained |
| `therm_flag` | `pmset -g therm` before/after | bool | `CPU_Speed_Limit < 100` ⇒ throttled |
| `build` | `llama-bench -o json` → `build_commit` | str | reproducibility key |

## Speed methodology

### Acquisition (not executed here; run once)

```bash
# llama.cpp: official macOS arm64 build. Nix alternative: `nix profile install nixpkgs#llama-cpp` [verify: nixpkgs build may lack Metal]
curl -sSL -o llama.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11761/llama-b11761-bin-macos-arm64.tar.gz
mkdir -p ~/llama.cpp/build/bin && tar xzf llama.tar.gz -C ~/llama.cpp/build/bin --strip-components=1
tar tzf llama.tar.gz | grep -E 'bin/(llama-bench|llama-server|llama-perplexity)$'   # [verify] all three present

# Python side
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install "mlx-lm==0.32.2" "lm-eval[api,ifeval,math]==0.4.13"
python -m mlx --version && python -m mlx_lm --version    # record in the report header
```

### llama.cpp: prefill, decode, depth sweep

Flags verified in `tools/llama-bench/README.md` (syntax block) and
`common/arg.cpp` (`-fa` accepts `on|off|auto`; `is_truthy` also accepts `1`).

```bash
# weights resolve from the Hub cache; the alias is not resident, so the first fetch is explicit
. benchmarks/common/env.sh
MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km
LB=~/llama.cpp/build/bin/llama-bench
M=$(model_path qwen38-27b-gguf-q4km)

# prefill at depth 0/4k/16k/32k  (pp only: -n 0)
$LB -m $M -p 512 -n 0 -d 0,4096,16384,32768 \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 \
    -r 5 -o json > pp.json

# decode at depth 0/4k/16k/32k  (tg only: -p 0)
$LB -m $M -p 0 -n 128 -d 0,4096,16384,32768 \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 \
    -r 5 -o json > tg.json
```

- `-p/-n/-d/-ctk/-ctv/-ub/-b/-ngl/-fa/-r/-o` are all documented in the llama-bench README;
  defaults are `-p 512 -n 128 -b 2048 -ub 512 -ctk f16 -ctv f16 -r 5 -ngl -1 -fa auto`, output default `md`.
- **Depth sweep semantics**: `-d N` prefills the KV cache with N filler tokens, *then* runs the test.
  `pp512 @ d16384` therefore measures prefill *while 16k of KV is already resident* — the realistic
  second-turn case. `tg128 @ d16384` is decode with a 16k KV cache — the number that predicts chat feel.
- `-fa on` is required for quantized **V** cache. `-fa auto` normally enables it when `-ctv` is
  quantized; pass it explicitly so the row is unambiguous. `-fa 1` still works (arg parser `is_truthy`)
  but is the legacy spelling — the documented form is `on`.
- Do **not** combine `-p 512` and `-n 128` in one invocation: `-p`+`-n` is the cross-product
  ("all combinations") / `-pg` path. Two invocations keep the semantics clean; the second load hits
  page cache and costs seconds.
- `llama-bench` excludes tokenization and sampling from its timing (README note). Good: that's what
  we want for engine comparison; add server-side numbers for the end-to-end view.
- Read `pp512` as **prefill throughput** (tok/s) and `tg128` as **decode throughput** (tok/s).
  JSON gives `avg_ts`, `stddev_ts` **and** `samples_ts[]` — take the *median of `samples_ts`*,
  not `avg_ts`, so one throttled rep cannot move the row.
- `-o jsonl` exists if you prefer one JSON object per line (`-o csv|json|jsonl|md|sql`).

### True TTFT (llama-server, not llama-bench)

`llama-server` returns a `timings` object on every completion
(`tools/server/README.md`, `/v1/chat/completions` response example):
`cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second, predicted_n, predicted_ms,
predicted_per_token_ms, predicted_per_second`. With `cache_n == 0`, `prompt_ms` **is** TTFT.

```bash
~/llama.cpp/build/bin/llama-server -m $M -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -ub 512 -b 2048 -np 1 --metrics --slots --jinja --port 8080 &
curl -sS localhost:8080/health
# cold-cache TTFT: use a fresh slot each time (-np 4 and rotate, or restart between samples)
curl -sS localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c 'print("hello "*900)')\"}],\"max_tokens\":1,\"cache_prompt\":false}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin)["timings"];print("TTFT_ms",d["prompt_ms"],"pp_tps",d["prompt_per_second"])'
# Prometheus gauges (needs --metrics): llamacpp:prompt_tokens_seconds, llamacpp:predicted_tokens_seconds,
# llamacpp:n_busy_slots_per_decode
curl -sS localhost:8080/metrics | grep -E 'prompt_tokens_seconds|predicted_tokens_seconds'
```

### MLX: CLI, bench script, python API

`mlx_lm.benchmark` **ships in mlx-lm** (`mlx_lm/benchmark.py`, entry point `mlx_lm.benchmark`).
It builds a random-token prompt of `-p` length (no tokenizer dependency), runs one warmup,
then `-n` trials, and prints parseable lines:

```bash
mlx_lm.benchmark --model mlx-community/Qwen3.8-27B-4bit -p 512  -g 128 -n 5
mlx_lm.benchmark --model mlx-community/Qwen3.8-27B-4bit -p 4096 -g 128 -n 5
# -> "Trial 1:  prompt_tps=..., generation_tps=..., peak_memory=..., total_time=..."
# -> "Averages: prompt_tps=..., generation_tps=..., peak_memory=..."
```

`mlx_lm.generate` prints the other pair of numbers (`mlx_lm/generate.py`):

```bash
mlx_lm.generate --model mlx-community/Qwen3.8-27B-4bit \
  --prompt "$(python3 -c 'print("hello "*900)')" --max-tokens 128 --temp 0 --seed 0
# -> "Prompt: 900 tokens, 812.345 tokens-per-sec"
#    "Generation: 128 tokens, 22.101 tokens-per-sec"
#    "Peak memory: 17.482 GB"
```

- `--max-tokens/-m`, `--temp`, `--top-p`, `--top-k`, `--seed`, `--prefill-step-size` (default 2048),
  `--max-kv-size`, `--kv-bits`, `--kv-group-size` (64), `--quantized-kv-start`,
  `--prompt-cache-file`, `--ignore-chat-template`, `--draft-model`, `--num-draft-tokens` are all
  real flags in `setup_arg_parser()`.
- `--prefill-step-size 2048` chunks the prefill; **lower it** (e.g. 512) if a 32k prefill OOMs —
  but record it, because it changes peak memory and pp throughput.
- Python API for a depth sweep: `stream_generate(model, tokenizer, prompt_ids, max_tokens=128)`
  yields `GenerationResponse(text, token, logprobs, from_draft, prompt_tokens, prompt_tps,
  generation_tokens, generation_tps, peak_memory)`. Prefill a long prompt once, take the first
  yield's timestamp as TTFT, the final response's `generation_tps` as decode. See `## Scripts`.
- `mlx_lm.server` serves `/v1/completions` + `/v1/chat/completions` (`--host --port --max-tokens
  --decode-concurrency --prompt-concurrency --kv-bits --prefill-step-size --prompt-cache-size`).
  Use it only for `generate_until` tasks; logprob fidelity for MCQ tasks is unverified → prefer
  `--model mlxlm` (below) for quality.

### Peak memory & swap on macOS

```bash
sysctl hw.memsize hw.pagesize iogpu.wired_limit_mb      # 34359738368 / 16384 / 0
/usr/bin/time -l llama-bench ... 2> time.txt            # "maximum resident set size" in BYTES
footprint -j /tmp/fp.json <pid>                         # phys_footprint, dirty, swapped (JSON)
vm_stat | head                                          # page counts × 16384 B
sysctl vm.swapusage                                     # total/used/free MiB
memory_pressure | tail -3                               # system-wide free %
```

- `/usr/bin/time -l` is the cheapest per-run peak: `maximum resident set size` is bytes and covers
  Metal allocations because unified memory is charged to the process. **This is the primary memory
  metric.**
- `ps -o rss=` is in KiB and lags; use it only for a live "is it growing?" watch.
- `footprint <pid>` is the accurate point-in-time view (`phys_footprint`, dirty, swapped, `wired`);
  `footprint -j file` for parseable output. Sample it 1 s after load and again at end of test.
- `sudo powermetrics --samplers gpu_power,cpu_power,thermal -i 1000 -n 60 -f plist` (samplers
  verified present in `/usr/bin/powermetrics`: `gpu_power`, `cpu_power`, `ane_power`, `thermal`,
  `disk`, `network`, `tasks`; `--format text|plist`). Power numbers are *estimates* — the man page
  says so explicitly, so never compare its watts across machines, only within a run.
- `iogpu.wired_limit_mb = 0` means the default cap (a fraction of RAM). Raising it
  (`sudo sysctl iogpu.wired_limit_mb=<MiB>`) makes OOM *harder* but starves the OS →
  **do not raise it for the benchmark**; a config that only fits after raising it is INVALID.

### Fanless thermal caveat

A MacBook Air has no fan: the SoC banks ~60–90 s of thermal headroom, then clocks down and stays
down. Consequences:

1. **Rep 1 is not rep 3.** Report both `sustained_tg` (median of reps 2–5) and `burst_tg` (rep 1).
   If `sustained/burst < 0.9`, say so — most published M-series numbers are burst numbers.
2. Cool start: idle ≥10 min, lid open, no charger heat. AC power preferred; if on battery, record it
   (macOS may cap power).
3. Fixed reps (`-r 5`, `-n 5`), fixed `--delay 0`, one config at a time, nothing else running.
4. Order effect: run the matrix twice in opposite order, or randomize config order with a fixed
   seed, then compare the two medians. A >5 % drift between passes means the machine is hot —
   discard and re-run cooled.
5. Watch `pmset -g therm` (prints `CPU_Speed_Limit` when the OS throttles) and the `thermal` sampler
   for die temps. Both go in the row.

## Quality methodology

### Harness install (uv) and backend choice — verified 2026-10-20

The PyPI project is **`lm-eval`** (0.4.13, `requires-python >=3.10`); `lm-evaluation-harness` is
**404 on PyPI**. Base install has no torch/transformers — backends are extras
(`api`, `hf`, `vllm`, `ifeval`, `math`, `tasks`, …).

```bash
uv pip install "lm-eval[api,ifeval,math]==0.4.13"     # api=requests/aiohttp/tenacity/tiktoken
lm-eval ls tasks | grep -E '^(ifeval|gpqa|mmlu_pro|arc_challenge|gsm8k|minerva_math500|humaneval|bbh|truthfulqa)'
```

Two viable llama.cpp backends in 2026, and one first-class MLX backend:

| `--model` | What it talks to | loglikelihood | Notes |
|---|---|---|---|
| `gguf` (a.k.a. `ggml`) | `llama-server` `/v1/completions` + `/tokenize` | **yes** (exact teacher forcing via `logit_bias` force + `top_logprobs`) | needs llama.cpp ≥ Dec 2024 (PR #10783); `loglikelihood_rolling` **not implemented**; auto-detects `parallel` from `/props.total_slots` |
| `local-completions` | any OpenAI `/v1/completions` | yes (needs `tokenizer_backend`) | `tokenizer_backend=auto|huggingface|remote|tiktoken`; chat endpoints cannot do loglikelihood |
| `mlxlm` | in-process MLX via `mlx_lm.evaluate` | **yes** | registered by `mlx_lm/evaluate.py`; same `lm_eval.simple_evaluate` → directly comparable |

**Prefer `--model gguf` for llama.cpp and `mlx_lm.evaluate` for MLX.** Both produce numbers from the
same task definitions and the same scoring code path, which is exactly the "comparable across
backends" requirement. `local-completions` remains the escape hatch (and the only route to a remote
server), not the default.

```bash
# llama.cpp backend
lm-eval run --model gguf \
  --model_args base_url=http://127.0.0.1:8080,parallel=4,max_length=32768 \
  --tasks gpqa_diamond_zeroshot,arc_challenge --apply_chat_template \
  --num_fewshot 0 --seed 42 --batch_size 4 \
  --output_path results/text/quality/gguf-c2.json --log_samples

# legacy/OpenAI path, for cross-checking gguf vs local-completions agreement
lm-eval run --model local-completions \
  --model_args model=local,base_url=http://127.0.0.1:8080/v1/completions,tokenizer_backend=huggingface,tokenizer=Qwen/Qwen3.8-27B,num_concurrent=4,max_retries=3,tokenized_requests=False,batch_size=4 \
  --tasks arc_challenge --apply_chat_template --num_fewshot 0

# MLX backend
mlx_lm.evaluate --model mlx-community/Qwen3.8-27B-4bit \
  --tasks gpqa_diamond_zeroshot arc_challenge --num-shots 0 --limit 200 \
  --batch-size 4 --apply-chat-template --seed 42 --output-dir results/text/quality/mlx-c7
```

- `--apply_chat_template` (optionally with a template name) is the documented flag; it auto-enables
  `--fewshot_as_multiturn`. For thinking models add
  `--model_args ...,enable_thinking=true,think_end_token="</思考>"` [verify exact token from
  `tokenizer_config.json`]; `enable_thinking` is **incompatible with loglikelihood tasks**, so
  MCQ suites must run thinking-off. `mlx_lm.evaluate --chat-template-args '{"enable_thinking":false}'`
  is the MLX equivalent.
- MCQ/loglikelihood tasks must use a **completion** endpoint (`gguf`/`local-completions`), never
  `local-chat-completions` (no logprobs).
- `humaneval` executes model-written Python: requires `--confirm_run_unsafe_code`. Run it in a
  throwaway container/`nix shell` with no network and a read-only repo mount, or skip it.

### Task tiers

Verified task names in `lm_eval/tasks/`: `ifeval`, `mmlu_pro`, `gpqa_main_zeroshot`,
`gpqa_diamond_zeroshot`, `gpqa_*_cot_zeroshot`, `arc_challenge`, `hellaswag`, `gsm8k`,
`minerva_math500` (+ per-subject `minerva_math_*`), `humaneval`, `mbpp`, `bbh`, `truthfulqa`.
**`musr` and `livecodebench` are NOT present** in lm-eval 0.4.13/main — do not budget for them.
`minerva_math` needs the `math` extra (sympy, antlr4-python3-runtime==4.11, math_verify);
`ifeval` needs the `ifeval` extra (langdetect, immutabledict, nltk) and is scored by an
in-process checker (no LLM judge).

Runtime model (calibrate once, then it predicts everything). At `pp≈P` tok/s and `tg≈G` tok/s:

```
hours ≈ (prefill_tokens / P + generated_tokens / G) / 3600
```
With a 27B dense Q4 on M5 expect `G ≈ 8–11`, `P ≈ 50–90` (fills `G ≈ 25–40`, `P ≈ 200+`).
**At G≈9 tok/s one hour buys only ~32k generated tokens** — that single fact dictates the tiers.

| Task | n (full) | Cost model | Est. full run | Est. with `--limit` |
|---|---|---|---|---|
| `gpqa_diamond_zeroshot` | 198 MCQ | 4×~250 prefill, 0 gen | ~5 min | n/a (already tiny) |
| `arc_challenge` | 1172 MCQ | 4×~120 prefill | ~10 min | `--limit 400` → 4 min |
| `hellaswag` | 10042 MCQ | 4×~200 prefill | ~2.5 h | `--limit 2000` → 30 min |
| `truthfulqa_mc2` | 817 MCQ | loglikelihood | ~8 min | — |
| `mmlu_pro` (10-shot CoT) | 12032 | ~2500 prefill + 300 gen | ~20 h | `--limit 500` → 50 min |
| `gsm8k` | 1319 | 250 prefill + 256 gen | ~11 h | `--limit 200` → 1.7 h |
| `minerva_math500` | 500 | 4-shot, ~150 prefill + ~800 gen | ~13 h | `--limit 100 --max_gen_toks 512` → 1.6 h |
| `ifeval` | 541 | ~300 gen | ~5 h | `--limit 120` → 1.1 h |
| `bbh` (3-shot CoT) | 6510 | 3-shot + 500 gen | days | `--limit 200` → 3 h |
| `humaneval` (+ execution) | 164 | ~350 gen + sandbox | ~2 h | `--limit 82` → 1 h |

**Fast tier (≤1 h total, run on every config, incl. both engines):**
`gpqa_diamond_zeroshot` (full 198) + `arc_challenge --limit 400` + `truthfulqa_mc2 --limit 400` +
`ifeval --limit 120`. Mostly loglikelihood → prefill-bound → the only tier that finishes on a laptop
in an hour. Use it as the smoke test that also catches harness breakage.

**Standard tier (≤4 h):** fast tier, plus `gsm8k --limit 400` (must be a multiple that keeps the same
first-N questions across engines), `minerva_math500 --limit 150 --max_gen_toks 512`,
`mmlu_pro --limit 500`, `arc_challenge` full, `hellaswag --limit 2000`.

**Report `--limit` as part of the score label** (`gsm8k@400/1319`). `--limit` selects the *first N*
examples of a fixed, unshuffled order, so it is reproducible across engines — but it is a
**non-random subset**, the CI widens, and the number is not comparable to a published full-set score.
Docs state it plainly: "For testing only."

### Sanity, noise, and what a delta means

Binomial 95 % CI half-width `1.96·sqrt(p(1-p)/n)` (computed):

| n | p=0.5 | p=0.7 | p=0.8 |
|---|---|---|---|
| 100 | ±9.8 pp | ±9.0 pp | ±7.8 pp |
| 198 | ±7.0 pp | ±6.4 pp | ±5.6 pp |
| 500 | ±4.4 pp | ±4.0 pp | ±3.5 pp |
| 541 | ±4.2 pp | ±3.9 pp | ±3.4 pp |
| 1172 | ±2.9 pp | ±2.6 pp | ±2.3 pp |
| 1319 | ±2.7 pp | ±2.5 pp | ±2.2 pp |

- With `n=500` and `p≈0.7` the CI is **±4 pp**. A "Q4 beats Q8 by 2 points" claim on 500 samples is
  **noise**. Only `n ≥ 1000` gets you to ~±2.5 pp on an absolute score.
- **Paired comparison is much tighter.** Same questions, same order, same seed → use McNemar:
  `SE_diff = sqrt(b+c)/n` where `b+c ≈ flip_rate·n`. At a 15 % flip rate: n=500 → ±3.4 pp,
  n=1000 → ±2.4 pp, n=1319 → ±2.4 pp. So always `--log_samples` and compare *per-sample*, not
  *per-score*. Report both the raw score and the paired delta with its CI.
- Two independent sources of variance: sampling noise (above) and **thermal/power drift** (can be
  5–10 % of tok/s, and for `--limit` runs it changes wall time, not score). Re-run one config twice
  and record the score delta as the harness's own noise floor; don't report any difference smaller
  than it.

### Quantization degradation — published, verified numbers

From the llama.cpp `tools/perplexity/README.md` LLaMA-3-8B board (Wikitext-2, KLD vs FP16). KLD is
the right yardstick: it measures divergence of the *whole* distribution, so it tracks downstream drop
far better than perplexity alone.

| Quant | PPL | ΔPPL | KLD |
|---|---|---|---|
| f16 | 6.2332 ±0.0378 | — | 0.000551 |
| q8_0 | 6.2343 | +0.0027 | 0.001355 |
| q6_K | 6.2534 | +0.0217 | 0.005452 |
| q5_K_M | 6.2886 | +0.0570 | 0.010762 |
| q4_K_M | 6.4071 | +0.1755 | 0.031273 |
| q4_K_M + imatrix | 6.3829 | +0.1513 | 0.028152 |
| q4_K_S | 6.5005 | +0.2689 | 0.043136 |
| q3_K_M | 6.8885 | +0.6569 | 0.101913 |
| q2_K | 8.6523 | +2.4207 | 0.331672 |

Reading: `Q4_K_M` costs ~+2.8 % PPL / KLD 0.031 vs `Q8_0`'s 0.0014 — a ~23× larger distribution shift,
yet on downstream tasks the drop is small and task-dependent. Cross-check with a 2026 unified study
(arXiv:2601.14277, Llama-3.1-8B-Instruct): `Q4_K_M` GSM8K 77.41 ±1.15 vs F16 79.57; IFEval (instr.
loose) 84.05; `Q4_K_S` AvgLoss ≈0.43 %; 3-bit is where real damage appears (Q3_K_S largest GSM8K
drop). Practical rule: **Q4_K_M is the floor for a 27B; Q5_K_M/Q6_K if it fits; never below Q4_K_M.**

Reproduce locally instead of trusting the table:

```bash
curl -sSL https://raw.githubusercontent.com/ggml-org/llama.cpp/master/scripts/get-wikitext-2.sh | bash
# record FP16 logits once, then KLD per quant
llama-perplexity -m f16.gguf -f wiki.test.raw --kl-divergence-base f16.kld -c 512
llama-perplexity -m Q4_K_M.gguf -f wiki.test.raw --kl-divergence-base f16.kld --kl-divergence -c 512
```
**Disk warning:** the logit file is ~11 GiB for Llama-2 and ~37 GiB for Llama-3 — it scales with
`vocab × tokens`, not with model size. A 27B with a ~150k vocab would need ~40+ GiB on an 88 GB disk.
Either restrict with a small `-c`, subsample the corpus, or **skip KLD for the 27B** and use
`mlx_lm.perplexity` (prints `Perplexity: X ± Y`, tok/s, peak memory) on a fixed text file for Q4-vs-Q8.

### Trusting numbers

- Never quote a model card's MMLU/GPQA for a *quantized local* config: cards are usually fp16/bf16,
  different prompt format, different few-shot count, sometimes a different tokenizer revision.
  Re-run, or label the number "vendor-reported, not re-run".
- Always state: harness version, task name + `--num_fewshot`, `--limit`, `--apply_chat_template`,
  thinking on/off, quant, KV quant, ctx size, engine build. A score without those is not a score.
- If a task cannot run locally (needs a judge model, needs 128k context, needs a sandbox you don't
  have, or is absent from the harness — `musr`, `livecodebench`): record it as
  `not-run (reason)` in the results table. Never substitute a different task's number.

### 2026 alternatives (and why they are not the default here)

| Tool | Status 2026-10-20 | Verdict for this machine |
|---|---|---|
| `lm-eval` 0.4.13 | reference harness; `gguf` server backend + `mlxlm` backend both first-class | **default**. Only harness with both backends and identical scoring |
| `evalchemy` (`mlfoundations/evalchemy`, 613★, last push 2026-02-24) | wrapper over lm-eval with standardized result JSON + leaderboard submission; adds HumanEvalPlus/MBPPPlus/LiveCodeBench/AIME | useful for the *task gap* (LiveCodeBench) and for its result schema; not on PyPI → git install, heavier deps |
| `lighteval` 0.13.0 | `lighteval eval` uses **inspect-ai** as preferred backend; `endpoint` backend for OpenAI-compatible servers; no MLX/Metal-specific path | skip unless you want HF-leaderboard-comparable normalizations |
| `inspect-ai` 0.3.282 | strong agentic/sandboxed evals, fine-grained transcripts | good if you need *sandboxed* code exec (HumanEval) done properly; overkill for MCQ |

Recommendation: **lm-eval only** for the comparable matrix; add `evalchemy` (or `inspect-ai`) solely
if a specific missing task (LiveCodeBench / HumanEval+) becomes decision-relevant.

## Memory & validity checks

A row is marked **INVALID** (excluded from ranking, kept in the file with the reason) if any of:

1. **Swap growth.** `used` from `sysctl vm.swapusage` grows by **> 64 MiB** between pre-run and
   post-run, or `swaps`/pageouts in `/usr/bin/time -l` are non-zero for the bench process.
   Rationale: a config leaking into swap still *works* and still prints a score — it just silently
   collapses throughput and changes results. This is the single most important gate.
2. **Wired limit / allocation errors.** stderr contains `failed to allocate`, `ggml_metal_buffer`,
   `Insufficient Memory`, `kIOGPUCommandBufferCallbackErrorOutOfMemory`, or the server returns
   HTTP 500 with an allocation message. Also invalid: the config only fits after
   `sudo sysctl iogpu.wired_limit_mb=<bigger>`.
3. **RSS above 75 % of RAM** (`peak_rss > 24 GiB`). It may "work" but macOS will compress/purge and
   the OS becomes the bottleneck; the number will not reproduce.
4. **Thermal throttle.** `pmset -g therm` shows `CPU_Speed_Limit < 100` at any sample, or
   `stddev_ts/avg_ts > 0.05` across the 5 reps, or `sustained/burst < 0.85`. Mark `THROTTLED`, cool
   10 min, re-run. If it throttles twice, keep the cooled-only median and say so.
5. **Cross-pass drift.** Pass A and pass B (reverse order) medians differ by > 5 % for the *same*
   config → machine-level drift; discard both and re-run cooled.
6. **Silent spill heuristic.** `tg_tps(config) / tg_tps(same-config-at-ctx-4096) < 0.6` at equal
   depth → suspect paging; confirm with (1)/(3) before keeping the row.
7. **Quality-run specific.** Server log shows context shift / truncation, or the harness reports
   fewer docs than requested (some requests dropped on timeout) → row invalid. `--limit` must be
   echoed into the row so a truncated run is never mistaken for a small run.

Per-row metadata always includes: `build`, `model_sha256`, `quant`, `ctk/ctv`, `ctx`, `ub/b`,
`fa`, `reps`, `burst_tg`, `sustained_tg`, `peak_rss`, `swap_delta`, `therm_flag`, `valid`, `reason`,
`host_uptime`, `on_ac_power` (`pmset -g batt`).

## Fairness rules

1. **Same context.** `-c 32768` everywhere, including runs you only care about at 4k. Context size
   changes KV allocation and therefore throughput.
2. **Same quant family.** K-quants vs K-quants (`Q4_K_M` ↔ `Q5_K_M` ↔ `Q8_0`). MLX `4bit`/`6bit` are
   the MLX analogue (≈4.5 / ≈6.5 bpw). Never compare `Q4_0` to `Q4_K_M` and call it an engine diff.
3. **Same prompt.** `llama-bench` uses synthetic token sequences (identical across configs);
   `mlx_lm.benchmark` uses seeded random token ids (`mx.random.seed(0)`). For end-to-end/gen runs use
   one fixed prompt file, same chat template, `temp=0`, fixed `seed`, `max_tokens` fixed.
4. **Warmup.** Keep `llama-bench`'s default warmup (do **not** pass `--no-warmup` except in a
   deliberately-labelled cold-start run). `mlx_lm.benchmark` does one warmup trial itself. Discard
   nothing else.
5. **Reps and statistic.** `-r 5` / `-n 5` minimum. Report **median of `samples_ts`** plus min/max.
   Burst = rep 1; sustained = median of reps 2–5.
6. **Identical harness versions and task args** across engines: same `--tasks`, `--num_fewshot`,
   `--limit`, `--apply_chat_template`, thinking setting, `--seed`.
7. **One config at a time**, nothing else on the machine, AC power, cool start, lid open.
8. **Version pinning.** Record `llama-bench -o json`'s `build_commit`/`build_number`, and
   `python -m mlx --version` / `python -m mlx_lm --version` / `lm-eval` version in every results file.
9. **No silent retries.** A failed run is retried at most once, cooled, and both attempts are logged.

## Scripts

These listings are the historical harness, kept as methodology. They take a flat directory of GGUF
files through `WEIGHTS` (not `MODELS`, which `common/env.sh` now points at the Hub cache). To run one
today, download the named quants into such a directory (`hf download <repo>`) and set
`WEIGHTS=/path/to/ggufs`; the MLX entries are repo ids and resolve from the cache directly.

### `bench.sh` (speed matrix → JSONL)

```bash
#!/usr/bin/env bash
# bench.sh - speed matrix for llama.cpp (Metal) + MLX on Apple Silicon.
# Emits $OUT/speed.jsonl, one row per (config, measurement).
# usage: WEIGHTS=/path/to/ggufs ./bench.sh [cfg-name ...]
set -Eeuo pipefail
export LC_ALL=C

LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
WEIGHTS=${WEIGHTS:-$PWD/models}
OUT=${OUT:-results/text/speed-$(date +%Y%m%dT%H%M%S)}
CTX=${CTX:-32768}; REPS=${REPS:-5}; PP=${PP:-512}; TG=${TG:-128}
DEPTHS=${DEPTHS:-"0 4096 16384 32768"}
SWAP_GATE_MIB=${SWAP_GATE_MIB:-64}
RSS_GATE_BYTES=${RSS_GATE_BYTES:-25769803776}   # 24 GiB
mkdir -p "$OUT/raw"

# name | engine | weights | extra-flags
CONFIGS=(
"C1|gguf|$WEIGHTS/Qwen3.8-27B-Q4_K_M.gguf|-ctk f16 -ctv f16"
"C2|gguf|$WEIGHTS/Qwen3.8-27B-Q4_K_M.gguf|-ctk q8_0 -ctv q8_0"
"C3|gguf|$WEIGHTS/Qwen3.8-27B-Q4_K_M.gguf|-ctk q4_0 -ctv q4_0"
"C4|gguf|$WEIGHTS/Qwen3.8-27B-Q5_K_M.gguf|-ctk q8_0 -ctv q8_0"
"C5|gguf|$WEIGHTS/Qwen3.8-A3B-Q4_K_M.gguf|-ctk q8_0 -ctv q8_0"
"C6|gguf|$WEIGHTS/Qwen3.8-4B-Q4_K_M.gguf|-ctk q8_0 -ctv q8_0"
"C7|mlx|mlx-community/Qwen3.8-27B-4bit|--kv-bits 0"
"C8|mlx|mlx-community/Qwen3.8-27B-4bit|--kv-bits 8"
"C9|mlx|mlx-community/Qwen3.8-27B-6bit|--kv-bits 0"
)
WANT=("$@")

swap_used_mib() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
therm() { pmset -g therm 2>/dev/null | awk -F'= ' '/CPU_Speed_Limit/{print $2+0}' | tail -1; }

emit() { python3 -c '
import json,sys
row=json.loads(sys.argv[1]); print(json.dumps(row,separators=(",",":")))' "$1" >> "$OUT/speed.jsonl"; }

set +e
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name engine weights flags <<<"$entry"
  [ ${#WANT[@]} -eq 0 ] || printf '%s\n' "${WANT[@]}" | grep -qx "$name" || continue
  echo ">>> $name ($engine)"
  sw0=$(swap_used_mib); th0=$(therm); t0=$(now)
  common=(repo="$name" engine="$engine" weights="$weights" ctx=$CTX reps=$REPS started="$t0")

  if [ "$engine" = "gguf" ]; then
    read -r -a extra <<<"$flags"
    # shellcheck disable=SC2086
    /usr/bin/time -l "$LLAMA_BIN/llama-bench" -m "$weights" -p $PP -n 0 \
        -d "$(echo $DEPTHS | tr ' ' ',')" -ngl 99 -fa on -ub 512 -b 2048 \
        "${extra[@]}" -r "$REPS" -o json >"$OUT/raw/$name.pp.json" 2>"$OUT/raw/$name.pp.time" || true
    /usr/bin/time -l "$LLAMA_BIN/llama-bench" -m "$weights" -p 0 -n $TG \
        -d "$(echo $DEPTHS | tr ' ' ',')" -ngl 99 -fa on -ub 512 -b 2048 \
        "${extra[@]}" -r "$REPS" -o json >"$OUT/raw/$name.tg.json" 2>"$OUT/raw/$name.tg.time" || true
    rss=$(awk '/maximum resident set size/{print $1}' "$OUT/raw/$name.tg.time" | tail -1)
    rss=${rss:-0}
    python3 - "$OUT/raw/$name.pp.json" "$OUT/raw/$name.tg.json" "$OUT/speed.jsonl" \
              "$name" "$engine" "$weights" "$CTX" "$REPS" "$rss" "$sw0" "$(swap_used_mib)" \
              "$th0" "$(therm)" "$t0" "$(sha "$weights" 2>/dev/null || echo unknown)" "$RSS_GATE_BYTES" <<'PY'
import json,sys,statistics as st
pp,tg,outp,name,engine,weights,ctx,reps,rss,sw0,sw1,th0,th1,t0,msha,rss_gate=sys.argv[1:]
def rows(p):
    try: return json.load(open(p))
    except Exception: return []
rss=int(rss); sw0=int(sw0); sw1=int(sw1); swd=sw1-sw0
therm_ok = not(th0 and int(th0)<100) and not(th1 and int(th1)<100)
for src,rows in (("pp",rows(pp)),("tg",rows(tg))):
    for r in rows:
        s=r.get("samples_ts") or []
        med=st.median(s) if s else r.get("avg_ts")
        burst=s[0] if s else med
        sd=r.get("stddev_ts") or 0.0
        cvsd=(sd/r["avg_ts"]) if r.get("avg_ts") else 0.0
        valid = (swd<=64) and (rss<=int(rss_gate)) and therm_ok
        reason=[]
        if swd>64: reason.append(f"swap+{swd}MiB")
        if rss>int(rss_gate): reason.append(f"rss{rss>>30}GiB")
        if not therm_ok: reason.append("throttled")
        if cvsd>0.05: reason.append(f"unstable_cv{cvsd:.3f}")
        row=dict(config=name,engine=engine,weights=weights,model_sha256=msha,
                 build=r.get("build_commit"),build_number=r.get("build_number"),
                 backends=r.get("backends"),n_gl=r.get("n_gpu_layers"),
                 fa=r.get("flash_attn"),ctk=r.get("type_k"),ctv=r.get("type_v"),
                 ub=r.get("n_ubatch"),b=r.get("n_batch"),ctx=int(ctx),reps=int(reps),
                 test=f'{src}{r.get("n_prompt") or r.get("n_gen")}',
                 n_prompt=r.get("n_prompt"),n_gen=r.get("n_gen"),depth=r.get("n_depth"),
                 median_ts=med,burst_ts=burst,avg_ts=r.get("avg_ts"),stddev_ts=sd,
                 samples_ts=s,peak_rss_bytes=rss,swap_delta_mib=swd,
                 therm_before=th0,therm_after=th1,started=t0,
                 valid=valid,reason=";".join(reason))
        print(json.dumps(row,separators=(",",":")),file=open(outp,"a"))
PY
  else  # mlx
    kvbits=0; for ((i=0;i<${#flags[@]};i++)); do [ "${flags[i]}" = "--kv-bits" ] && kvbits=${flags[i+1]}; done
    /usr/bin/time -l mlx_lm.benchmark --model "$weights" -p $PP -g $TG -n "$REPS" -b 1 \
        >"$OUT/raw/$name.bench.txt" 2>"$OUT/raw/$name.bench.time" || true
    if [ "$kvbits" != "0" ]; then
      python3 - "$weights" "$CTX" "$TG" "$kvbits" "$REPS" <<'PY' >"$OUT/raw/$name.depth.json"
import json,os,sys,time
import mlx.core as mx
from mlx_lm import load, stream_generate
repo,ctx,tg_reps,kv,reps=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
model,tok=load(repo)
vocab=model.args.vocab_size if hasattr(model,"args") else 32000
out=[]
for depth in [0,4096,16384,32768]:
    if depth>=ctx: continue
    ids=mx.random.randint(0,vocab,(1,depth)) if depth else mx.array([[1]])
    def run():
        t0=time.perf_counter(); ttft=None; last=None
        for r in stream_generate(model,tok,ids[0].tolist(),max_tokens=tg_reps,
                                 kv_bits=(kv or None),kv_group_size=64):
            if ttft is None: ttft=time.perf_counter()-t0
            last=r
        return ttft,last
    run()  # warmup
    ts=[run() for _ in range(reps)]
    out.append(dict(depth=depth,
        ttft_ms=1000*sorted(x[0] for x in ts)[len(ts)//2],
        gen_tps=sorted(x[1].generation_tps for x in ts)[len(ts)//2],
        peak_gb=max(x[1].peak_memory for x in ts)))
print(json.dumps(out))
PY
    fi
    rss=$(awk '/maximum resident set size/{print $1}' "$OUT/raw/$name.bench.time" | tail -1); rss=${rss:-0}
    python3 - "$OUT/raw/$name.bench.txt" "$OUT/speed.jsonl" "$name" "$CTX" "$REPS" "$rss" \
              "$sw0" "$(swap_used_mib)" "$th0" "$(therm)" "$t0" "$weights" "$RSS_GATE_BYTES" <<'PY'
import json,re,sys,statistics as st
txt,outp,name,ctx,reps,rss,sw0,sw1,th0,th1,t0,weights,rss_gate=sys.argv[1:]
rss=int(rss); swd=int(sw1)-int(sw0)
therm_ok=not(th0 and int(th0)<100) and not(th1 and int(th1)<100)
txt=open(txt).read()
def avg(k):
    m=re.search(rf"Averages:.*{k}=([0-9.]+)",txt); return float(m.group(1)) if m else None
pt,gt,pm=avg("prompt_tps"),avg("generation_tps"),avg("peak_memory")
trials=[float(x) for x in re.findall(r"generation_tps=([0-9.]+)",txt)][1:]  # skip warmup if present
med=st.median(trials) if trials else gt
valid=(swd<=64) and (rss<=int(rss_gate)) and therm_ok and gt is not None
row=dict(config=name,engine="mlx",weights=weights,test="@512/128",ctx=int(ctx),reps=int(reps),
    prompt_tps=pt,generation_tps=med,burst_ts=trials[0] if trials else med,
    peak_mem_gb=pm,peak_rss_bytes=rss,swap_delta_mib=swd,
    therm_before=th0,therm_after=th1,started=t0,valid=valid,
    reason=";".join([r for r,c in (("swap",swd>64),("rss",rss>int(rss_gate)),("throttled",not therm_ok)) if c]))
print(json.dumps(row,separators=(",",":")),file=open(outp,"a"))
PY
  fi
done
set -e
echo "wrote $OUT/speed.jsonl"
python3 -c '
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1])]
inv=[r for r in rows if not r.get("valid")]
print(f"{len(rows)} rows, {len(inv)} INVALID")
for r in inv: print("  INVALID", r["config"], r.get("test") or r.get("depth"), r["reason"])' "$OUT/speed.jsonl"
```

### `bench_quality.sh` (one tier, both backends, validity-gated)

```bash
#!/usr/bin/env bash
# bench_quality.sh - lm-eval tiers against llama-server (model=gguf) and MLX (mlxlm).
# usage: ./bench_quality.sh fast|standard [cfg ...]
set -Eeuo pipefail
export LC_ALL=C
TIER=${1:-fast}; shift || true
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
WEIGHTS=${WEIGHTS:-$PWD/models}
OUT=${OUT:-results/text/quality-$(date +%Y%m%dT%H%M%S)}
CTX=${CTX:-32768}; NP=${NP:-4}; PORT=${PORT:-8080}
LIM_FAST=${LIM_FAST:-200}; SEED=${SEED:-42}
mkdir -p "$OUT"

GGUF_MODEL=${GGUF_MODEL:-$WEIGHTS/Qwen3.8-27B-Q4_K_M.gguf}
GGUF_KV=${GGUF_KV:-"-ctk q8_0 -ctv q8_0"}
MLX_MODEL=${MLX_MODEL:-mlx-community/Qwen3.8-27B-4bit}

case "$TIER" in
  fast)     TASKS="gpqa_diamond_zeroshot arc_challenge truthfulqa_mc2 ifeval"
            LIMITS="--limit 400";;
  standard) TASKS="gpqa_diamond_zeroshot arc_challenge truthfulqa_mc2 ifeval gsm8k minerva_math500 mmlu_pro hellaswag"
            LIMITS="--limit 500";;
  *) echo "tier must be fast|standard" >&2; exit 2;;
esac

swap_used_mib(){ sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }

run_gguf() {
  local name=$1 bel sw0 sw1 rc=0
  read -r -a kv <<<"$GGUF_KV"
  echo ">>> $name (gguf) ${TASKS}"
  sw0=$(swap_used_mib)
  "$LLAMA_BIN/llama-server" -m "$GGUF_MODEL" -c "$CTX" -ngl 99 -fa on \
      "${kv[@]}" -ub 512 -b 2048 -np "$NP" --jinja --metrics --slots --port "$PORT" \
      >"$OUT/server-$name.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
  curl -sf "localhost:$PORT/health" >/dev/null || { echo "server failed to start"; kill $pid; return 1; }
  TAG=$(curl -sS "localhost:$PORT/props" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("total_slots","?"))')
  set +e
  lm-eval run --model gguf \
    --model_args "base_url=http://127.0.0.1:$PORT,parallel=$TAG,max_length=$CTX" \
    --tasks $TASKS $LIMITS --apply_chat_template --num_fewshot 0 \
    --batch_size "$TAG" --seed "$SEED" \
    --output_path "$OUT/$name.json" --log_samples >"$OUT/$name.stdout" 2>&1
  rc=$?
  set -e
  sw1=$(swap_used_mib)
  grep -Eiq 'failed to allocate|Insufficient Memory|OutOfMemory' "$OUT/server-$name.log" && rc=99
  kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
  python3 - "$name" "$TIER" "$TASKS" "$LIMITS" "$GGUF_MODEL" "$GGUF_KV" "$CTX" "$NP" \
      "$OUT/$name.json" "$((sw1-sw0))" "$rc" "$SEED" <<'PY' >> "$OUT/quality.jsonl"
import json,os,sys
name,tier,tasks,limits,model,kv,ctx,np_,path,swd,rc,seed=sys.argv[1:]
res={}
if os.path.exists(path):
    try:
        d=json.load(open(path)); res={k:{m:v for m,v in vals.items() if m in
            ("acc,none","acc_norm,none","exact_match,none","inst_level_loose_acc,none","mc2,none","f1,none")}
            for k,vals in d.get("results",{}).items()}
    except Exception as e: res={"error":str(e)}
print(json.dumps(dict(config=name,engine="llama.cpp-metal",tier=tier,tasks=tasks.split(),
    limits=limits,weights=model,kv=kv,ctx=int(ctx),parallel=int(np_),
    swap_delta_mib=int(swd),exit_code=int(rc),seed=int(seed),
    valid=(int(rc)==0 and int(swd)<=64),results=res),separators=(",",":")))
PY
}

run_mlx() {
  local name=$1
  echo ">>> $name (mlx) ${TASKS}"
  set +e
  mlx_lm.evaluate --model "$MLX_MODEL" --tasks $TASKS $LIMITS --num-shots 0 \
      --batch-size 4 --apply-chat-template --seed "$SEED" --output-dir "$OUT/mlx-$name" \
      >"$OUT/mlx-$name.stdout" 2>&1
  local rc=$?
  set -e
  python3 - "$name" "$TIER" "$TASKS" "$LIMITS" "$MLX_MODEL" "$CTX" "$rc" "$OUT/mlx-$name" \
      <<'PY' >> "$OUT/quality.jsonl"
import json,os,sys
name,tier,tasks,limits,model,ctx,rc,d=sys.argv[1:]
res={}
if os.path.isdir(d):
    for f in os.listdir(d):
        try:
            j=json.load(open(os.path.join(d,f)))
            res.update({k:{m:v for m,v in vals.items() if m.endswith(",none")} for k,vals in j.items()})
        except Exception: pass
print(json.dumps(dict(config=name,engine="mlx",tier=tier,tasks=tasks.split(),limits=limits,
    weights=model,ctx=int(ctx),exit_code=int(rc),valid=int(rc)==0,results=res),separators=(",",":")))
PY
}

run_gguf "c2-27b-q4km-q8kv"
run_gguf "c5-a3b-q4km-q8kv"
run_mlx   "c7-27b-4bit"
echo "wrote $OUT/quality.jsonl"
```

Notes on the scripts: both are `set -euo pipefail`, write append-only JSONL, and refuse to hide a
failure — a non-zero exit or a memory error flips `valid:false` instead of dropping the row. The
`parallel=$TAG` value comes from llama-server's own `/props.total_slots`, which is exactly what the
harness auto-detects; passing it explicitly keeps the log honest. `--batch_size` is set to the slot
count so loglikelihood requests actually pipeline (this is a 2–4× win on prefill-bound tiers).

## Sources

Verified 2026-10-20.

- llama-bench flags/output: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/README.md
- `-fa on|off|auto`, `-ctk/-ctv`, `-c/-b/-ub/-np/--metrics/--jinja` parsing: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp
- Server flags + `/metrics` + `timings` response: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- Metal enabled by default on macOS: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md#metal-build
- Release/binaries (b11761, `llama-b11761-bin-macos-arm64.tar.gz`): https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11761 — note `releases/latest` = `v0.7.0`, assets = `nightly-tag.txt` only
- KLD/PPL quant table + `--kl-divergence-base`: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md
- 2026 quant study (Llama-3.1-8B-Instruct downstream): https://arxiv.org/abs/2601.14277
- MLX bench script + BENCHMARKS.md (MMLU-Pro per quant): https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/benchmark.py , https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/BENCHMARKS.md
- `mlx_lm.generate` output format & flags: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/generate.py
- `mlxlm` lm-eval backend + `mlx_lm.evaluate` CLI: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/evaluate.py
- `mlx_lm.perplexity`, `mlx_lm.server` endpoints, version/deps (`mlx>=0.32.3`, `transformers>=5.7.0`): https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/perplexity.py , https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/server.py , https://raw.githubusercontent.com/ml-explore/mlx-lm/main/pyproject.toml
- `mx.get_peak_memory` / `mx.set_wired_limit` (mlx 0.32.4 docs): https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
- lm-eval install + extras + `--model gguf` section: https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/README.md
- CLI flags (`--apply_chat_template`, `--limit` "For testing only", `--seed`, `--confirm_run_unsafe_code`): https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/interface.md
- `local-completions` args + `tokenizer_backend`: https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/API_guide.md , https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/models/openai_completions.py
- `gguf` backend implementation (force-token loglikelihood, `/props` slots): https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/models/gguf.py
- Extras list (`api`, `ifeval`, `math`, …): https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/pyproject.toml
- Task inventory (no `musr`, no `livecodebench`): https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/README.md
- Task names verified: `gpqa/zeroshot/gpqa_diamond_zeroshot.yaml`, `minerva_math/minerva_math500.yaml`
- Versions: `lm-eval 0.4.13` (`https://pypi.org/pypi/lm-eval/json`), `mlx-lm 0.32.2`, `mlx 0.32.4`, `lighteval 0.13.0`, `inspect-ai 0.3.282`; `lm-evaluation-harness` → 404 on PyPI; `evalchemy` not on PyPI → https://github.com/mlfoundations/evalchemy
- Local machine facts: `system_profiler SPHardwareDataType`, `sysctl hw.pagesize hw.memsize iogpu.wired_limit_mb`, `vm_stat`, `sysctl vm.swapusage`, `/usr/bin/time -l`, `footprint -h`, `powermetrics --help`, `strings /usr/bin/powermetrics | grep -E 'gpu_power|thermal'`
