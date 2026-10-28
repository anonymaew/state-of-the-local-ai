# Local inference research on an M5 / 32 GB MacBook Air

What actually runs on this box — at usable speed and usable quality — and what it costs in memory.
Two tracks have measurements (**text** and **decision**), four are scoped with the same harness
contract (**image**, **video**, **audio**, **speech**). The interesting answer is rarely "the biggest
model": it is which engine + quant + task decomposition fits a 32 GB unified-memory envelope.

The resident general model is `qwen38-27b-oq4e-mtp` — 27B dense, **vision + video + text in, text
out**, MTP head, Apache-2.0. The decision track adds five 9B checkpoints (Clef-Flash in 4/8-bit MLX
and Q4_K_M GGUF, plus Kev-9B and d3-flash GGUFs). All weights live in the Hugging Face cache and are
named by alias or repo id — see [`models/README.md`](models/README.md).

## Quickstart

Python 3.12, ~25 GB of free disk for the default model, and Apple Silicon for MLX.

```bash
git clone <this-repo> && cd local-llm
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
# without uv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

. benchmarks/common/env.sh                    # HF_HOME=~/.cache/hf, model_path(), DATA/RESULTS/LOGS
.venv/bin/python benchmarks/common/model_store.py list

# first use fetches 16 GiB into the Hub cache; afterwards resolution is cache-only
MODEL_STORE_ONLINE=1 model_path qwen38-27b-oq4e-mtp
model_path qwen38-27b-oq4e-mtp                # -> .../snapshots/<rev>

# a short text measurement (180 s on an idle machine)
DEPTHS=64 .venv/bin/python benchmarks/text/bench_mlx.py qwen38-27b-oq4e-mtp smoke

# the decision track needs its suite (1 MB) and one 6 GB checkpoint
.venv/bin/hf download LocalLLaMA/typed-decisions --repo-type dataset \
  --local-dir data/decision/cache/typed-decisions
.venv/bin/python benchmarks/decision/run_config.py --engine mlx-clef \
  --model-path clef-flash-4bit --model-id smoke --limit 5 --port 8099
```

`model_path` never downloads unless `MODEL_STORE_ONLINE=1`: a benchmark must not pull bytes inside a
measured window. `hf download` respects the same `HF_HOME` once `env.sh` is sourced.

## What is, and is not, in this repo

| Included | Not included (fetched or built) |
| --- | --- |
| Harnesses in `benchmarks/`, row schema, model store | Model weights — resolved from the Hub cache |
| Research write-ups in `docs/`, one folder per modality | Dataset bytes — provenance + fetch commands in `data/*/README.md` |
| Append-only measurement rows in `results/` | Raw engine logs in `logs/` (kept locally; the README explains the layout) |
| Registry of aliases/repos/sizes in `models/README.md` | llama.cpp source (clone/build pinned in [`src/README.md`](src/README.md)) |

## Layout

```text
docs/                     research notes, one folder per modality
  shared/                 hardware + engine facts that bound every track
  text/  decision/        measured tracks (models, method, results, recommendation)
  image/  video/ audio/ speech/
                          scoped tracks: metrics + validity gates, no measurements yet
benchmarks/
  common/                 layout contract (env.sh), model store, row schema
  text/                   measured harnesses: llama.cpp, mlx-lm, oMLX
  image/  video/ audio/ speech/ decision/
                          scope + candidate engines, harnesses land here next
models/                   registry only; weights live in the Hub cache
~/.cache/hf/hub           weights: $HF_HOME, one models--<org>--<repo> per checkpoint
data/<modality>/          inputs; dataset bytes fetched on demand
results/<modality>/       append-only JSONL, one row per measurement
logs/<modality>/          raw engine output (not committed)
src/llama.cpp             engine source, cloned and built (not committed)
```

Every harness discovers the same places from its own location, and every model is resolved by name:

```bash
. "$(dirname "$0")/../common/env.sh"   # ROOT, HF_HOME, HF_HUB_CACHE, DATA, RESULTS, LOGS
python3 benchmarks/text/bench_mlx.py qwen38-27b-oq4e-mtp oq4e-mtp
model_path clef-flash-q4km               # -> .../snapshots/<rev>/Clef-Flash-Q4_K_M.gguf
```

A model ref is an alias, `org/repo[:file]`, or an existing path. A row is a JSON object with
`modality, task, engine, model, label, ts, valid, metrics` — see
[`benchmarks/common/README.md`](benchmarks/common/README.md). Invalid runs stay in the file; deleting
them is how benchmarks start lying.

## Tracks

| Modality | What is measured | Harness | Status |
| --- | --- | --- | --- |
| [Text](docs/text/README.md) | prefill/decode t/s vs depth, TTFT, PPL, ARC/TruthfulQA | llama.cpp, mlx-lm, oMLX | **measured** |
| [Image](docs/image/README.md) | image→text accuracy; s/image, steps, peak memory | mlx-vlm (`--image`), `--output-modality image` | scoped, not yet run |
| [Video](docs/video/README.md) | frames/s throughput, video QA accuracy; s/clip | mlx-vlm (`--video`, video-output workflows) | scoped, not yet run |
| [Audio](docs/audio/README.md) | music/sfx real-time factor, CLAP/FAD, audio tagging | mlx-audio (`music.generate`), mlx-vlm (`--audio`) | scoped, not yet run |
| [Speech](docs/speech/README.md) | WER/CER, RTF, first-token latency, TTS MOS | mlx-audio (`stt.generate`, `tts.generate`) | scoped, not yet run |
| [Decision](docs/decision/README.md) | accuracy, KL/Brier/ECE, typed-decision latency | llama.cpp `/v1/systemone`, MLX loaders | **measured** |

The environment carries the tooling for all six: `mlx-lm 0.32.2`, `mlx-vlm 0.7.9` (text/image/video/audio
output, plus typed `decide`), `mlx-audio 0.6.1` (stt/tts/music/sts), `lm_eval`, and a llama.cpp build
in `src/`. `omlx 0.7.1` needs its own venv — see [`requirements-omlx.txt`](requirements-omlx.txt).

## Text: what has been measured

Two configs, and the choice is genuinely yours:

- **Speed-first: oMLX 0.7.1 + Qwen3.8-27B `oQ4e-mtp` (MTP on) → 15.2 tok/s**, PPL 6.245, 17.8 GB.
- **Quality/tooling-first: llama.cpp v0.7.0 + `UD-Q4_K_M` + MTP head → 12.3 tok/s** (13.9 solo),
  **153 tok/s prefill**, PPL 6.037, 18.1 GB. *Not resident; restore with
  `MODEL_STORE_ONLINE=1 model_path qwen38-27b-gguf-q4km` and `... qwen38-27b-mtp`.*

Paired, interleaved, same machine: **the oMLX config is +23.6 % faster for +3.3 % worse perplexity**
(ratio 1.206 / 1.236 / 1.260 across three rounds). A Qwen3.6-35B-A3B MoE gave 54.5 tok/s at roughly
half the intelligence — that checkpoint has been removed.

1. **Dense vs MoE is an 8× speed decision, not a tuning detail.** 27B dense reads ~16.5 GB per token
   → hard ceiling 9.3 tok/s on a 153.6 GB/s bus. A 3B-active MoE reads ~1.7 GB/token → 56.9 tok/s
   measured.
2. **MTP speculative decoding is the whole speed story — and the implementation matters more than
   the engine.** 7.4 → 15.2 tok/s (oMLX, 75.2 % acceptance) or 7.4 → 13.9 (llama.cpp, 63.1 %).
   Identical weights tie across engines (ratio 0.98), so the +23.6 % is the draft/verify
   implementation, not the runtime.
3. **The engines cross over at ~8k context.** llama.cpp wins short (7.1 vs 5.7); MLX wins at 16k
   (5.7 vs 4.3) because llama.cpp's hybrid-GDN decode degrades 38 % with depth while MLX stays flat.
4. **32 GB is enough — barely, and only because of the architecture.** Qwen3.8-27B keeps 16 of 64
   layers full-attention → KV is 64 KiB/token (1 GiB per 16k). The recommended config wires 17 GiB
   against a ~24.6 GiB Metal ceiling.
5. **M5 needs a recent llama.cpp.** The Metal-4 tensor API is silently disabled before build
   `b10734`; nixpkgs ships 0.6.0, which predates MTP support entirely.

Full evidence, methodology and validity gates: [`docs/text/`](docs/text/README.md).

## Decision: what has been measured

Three 9B native decision models, two engines, 2,000 typed decisions
([typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test split):

- **Clef-Flash 9B, MLX 4-bit → 0.697 accuracy, 1.04 s per five-question request, 5.6 GB.**
  llama.cpp Q4_K_M on the same model: 0.702 accuracy, 2.14 s, 6.8 GB. Same weights, same score:
  **the engine is worth ~1.8× latency on this box.**
- **Kev-9B Q4_K_M (llama.cpp) → 0.680 accuracy, 2.76 s, 7.2 GB.** Its MLX adapter path does not
  fit: the adapter fold peaks at ≈36 GB against a 19.3 GB base.
- **32 GB is not the constraint at 9B** — 6–7 GB resident. The format is: a GGUF carries the
  decision head; the MLX adapter path carries a second copy of the base while folding.
- **8-bit is not worth it here:** 0.699 vs 0.697 accuracy, 1.4× slower, 10.7 GB vs 6.2 GB.
  Quality is deterministic across re-runs (0.6965 twice); latency is not — the chassis matters more.
- Sustained runs on this fanless Air lose 50–100 % of latency; first-100 and full-run p50 are both
  reported because either alone is misleading.

Full evidence, per-question breakdown and validity gates:
[`docs/decision/05-results.md`](docs/decision/05-results.md), recommendation:
[`docs/decision/06-recommendation.md`](docs/decision/06-recommendation.md).

## Rules that apply to every track

- One row = one measurement, one process, one model load. No averaging inside the harness.
- Record thermal/load/swap state. Swap-outs invalidate a row: at 32 GB, paging is the real ceiling.
- Fix the prompt, seed, steps, resolution/clip length, and report them in `label`.
- Report what failed (OOM, timeout, refusal) as a row, not as a missing row.
- A number without its engine version, quant and context length is a rumour.
- Results are append-only. Historical home-directory paths were redacted to `.` / `~`; no measurement
  was changed.

## License

Code and prose: [MIT](LICENSE). Model weights, datasets and engine source keep their upstream licenses
(Apache-2.0 for the Qwen and Clef checkpoints; each dataset's own terms). No weights or dataset bytes
are redistributed here.
