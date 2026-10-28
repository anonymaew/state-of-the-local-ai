# Decision engines (System One) — M5 / 32 GB

Which runtimes can execute a *decision model* on this box: no decode loop, one forward pass per state,
answer read from a head or from option-token logits. Plan: [`01-plan.md`](01-plan.md).
Raw quotes: [`02-engines.sources.md`](02-engines.sources.md).

| Tier | Meaning |
|---|---|
| **P** | Primary, local: read out of code/packaging/HF metadata on this machine — `src/llama.cpp` (v0.7.0-dev), `src/kev`, `.venv`, `.venv-omlx`, prefetched model cards and HF file listings (`/tmp/decision-sources/`). |
| **D** | Repo prior: [`shared/02-engines.md`](../shared/02-engines.md), sourced to upstream releases/issues as of 2026-10-20. Not re-verified here. |
| **TODO(verify)** | Not established in this pass. Do not build on it. |

## Verdict

1. **Two engines cover the whole space, and they are not interchangeable.** MLX runs *code-defined*
   heads (Kev `PointerHead`, Clef `joint_head`) through Python; llama.cpp runs *GGUF-defined* heads
   (`CLEF`, `VLLM_DECISION2` archs) through `/v1/systemone` (**P**). Nothing else does either.
2. **The protocol is `POST /v1/systemone`, not logprobs.** Three independent implementations agree on
   `state` + `questions{type: choice|score|noul}` → `answers{probabilities, confidence|noul}`
   (**P**: llama.cpp server docs, Cloudflare's `joint_schema_model.systemone()`, Kev's `kev/serve.py`).
3. **Everything 9B is already quantised for both paths**: `mlx-community/clef-flash-4bit` 6.21 GB and
   `ggml-org/Clef-Flash-GGUF` Q4_K_M 6.73 GB + mmproj 0.62 GB (**P**, HF file listings). Fit is not
   the problem on 32 GB; prefill latency and calibration are.
4. **PyTorch MPS is the trap.** Kev's own docstring: "MPS has no Gated DeltaNet kernels, so the
   PyTorch path runs reference code there (**Kev-4B ~0.8 s per request**)" (**P**). Use MPS as the
   correctness oracle, never as the benchmark arm.
5. **llama.cpp is the only engine whose serving knobs are all present at once** — `/v1/systemone`,
   GBNF/JSON-schema, prefix cache (`--cache-prompt`, slot save/restore), continuous batching
   (**P**). It is the harness reference; MLX is the accuracy/latency reference.

## Capability matrix

`E` encoder+head · `H` LLM+decision/pointer head (code) · `L` logprob read-out.

| Runtime (as installed / latest known) | E | H | L | Metal / M5 | Install + size on this box | Verdict |
|---|---|---|---|---|---|---|
| **MLX `mlx-lm` 0.32.2 + `mlx-vlm` 0.7.9** | ⚠️ port only | ✅ Python module (`model.predict`, Kev `PointerHead`) | ✅ logits in-process | native Metal, M5 NAX first-party (**D**) | **installed**; weights = the only download (Clef-Flash 4-bit 6.21 GB) | **primary** |
| **llama.cpp v0.7.0-dev (repo)** | ✅ BERT/ModernBERT/Nomic/Jina/EuroBERT/T5-enc | ⚠️ only `CLEF`, `VLLM_DECISION2` archs | ✅ `n_probs` | Metal; needs ≥ b10734 for M5 tensor API (**D**) | **built** in `src/llama.cpp`; GGUF 6.73 GB + 0.62 mmproj | **co-primary / protocol ref** |
| **Kev `kev.serve`** (sidecar over MLX) | ❌ | ✅ `PointerHead` fp32 + checkpoint temperature | ⚠️ via `to_answers` only | MLX on Apple Silicon by default (**P**) | `uv pip install -e src/kev[serve]` (needs torch) | **the Kev harness** |
| **PyTorch MPS + transformers** | ✅ | ✅ (reference path, slow) | ✅ full logits | MPS works; **no GDN kernels** → ~0.8 s/req 4B (**P**) | **not installed**; `torch>=2.6,<2.9` + transformers 5.x (**P** pin) | oracle only |
| **ONNX Runtime 1.31.0** | ✅ | ❌ | ⚠️ if exported as output | CoreML EP = ANE/GPU/CPU partition, silent fallback | **in `.venv-omlx`** only | encoder fast path |
| **CoreML / coremltools** | ✅ static shapes | ❌ | ❌ | ANE, lowest floor | not installed, TODO(verify) version | encoder deploy |
| **oMLX 0.7.1** | ❌ | ❌ | ⚠️ TODO(verify) | MLX + custom kernels (**D**) | `.venv-omlx` present | text server only |
| **Ollama** | ❌ | ❌ | ⚠️ TODO(verify) | llama.cpp + MLX engines (**D**) | not installed | skip |
| **LM Studio** | ❌ | ❌ | ⚠️ TODO(verify) | llama.cpp + MLX + Splash (**D**) | .dmg only | skip (GUI) |
| **vLLM** | ❌ | ⚠️ via mlx_lm layers | ✅ | core = no Metal; `vllm-metal` plugin = MLX + NAX prefill (**D**) | brew/curl venv — no brew here | serving only |
| **MLC-LLM** | ❌ | ❌ | ⚠️ | TVM→Metal, pre-Metal-4 (**D**) | pip + prebuilt libs | skip |

Why the blanks are blanks:

- **`H` needs a Python module.** Kev ships `kev/mlx_model.py` — "mlx-lm's Metal implementation of the
  hybrid backbone under Kev's own encoder and pointer head … the readout is the very same
  `PointerHead` (fp32, with the checkpoint's temperature)" (**P**). Clef-Flash ships
  `joint_schema_model.py` / `clef_mlx.py` with `encode_record`, `collate_records`,
  `load_release_model`, `systemone` (**P**). An engine without Python model code cannot run either.
- **`E` in llama.cpp is real**: `LLM_ARCH_BERT`, `MODERN_BERT`, `NOMIC_BERT(_MOE)`, `NEO_BERT`,
  `JINA_BERT_V2`, `JINA_BERT_V3`, `EUROBERT`, `T5ENCODER` are all in the local `llm_arch` enum
  (**P**) — but they surface as *pooled embeddings* (`--embedding --pooling cls`), not as a decision
  head, unless the head is inside the graph.
- **`E` in ONNX/CoreML is the natural fit**: `com-kotobalabs/open-jev-deberta-v3-large` is 1.74 GB +
  `head.safetensors` 12.6 MB + `typed_decisions/open_jev.py` (**P**); `fastino/GLiNER2.5-Decide`
  1.95 GB (**P**). Export-once, no Python at inference.

## `/v1/systemone` — the wire format (P)

Three servers implement it, and they agree: llama.cpp `llama-server`, Kev `kev.serve`, Cloudflare's
`joint_schema_model.systemone()`. llama.cpp's docs call it "TypeSafe-compatible"; Kev's package
description says "TypeSafe-compatible `/v1/systemone` API"; Clef's card says "The Clef-Flash API is
fully compatible with Jev and SystemOne."

**Request**

```json
{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "images": ["data:image/jpeg;base64,..."],   // optional; also "videos" (Clef), "files" (llama.cpp alias)
  "questions": {
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency":    {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage":     {"type": "noul",  "instructions": "Is a service down?"}
  }
}
```

- `type` ∈ `choice` (named options) · `score` (ordered levels, 2-10, lowest first) · `noul`
  ("number or logic" — the boolean primitive; `bool` in mlx-vlm).
- `criteria`: `choice` → map option-id → description (may be `null`); `score` → **ordered list**;
  `noul` → optional `{true, false}` descriptions. `instructions` is optional — the question ID is
  used when omitted (**P**, Clef card).
- `state` may be a string or any JSON value; non-strings are rendered as JSON text. Clef/llama.cpp
  allow `state: null` when media carries the whole state.
- Option ceilings per arch: **52** `openjev`; **255** `laya`, `clef`, `pplx-decider`,
  `vllm-decision2`, `vllm-d3`, `lfm2-d1(-omni)` (**P**).

**Response**

```json
{"model": "openjev",
 "answers": {
   "department": {"type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.9998, "technical": 0.0001},
                  "confidence": 0.9997},
   "outage":     {"type": "noul", "noul": 0.6328},
   "urgency":    {"type": "score", "score": 2.0858,
                  "legend": {"0": "can wait", "1": "this week", "2": "today"},
                  "probabilities": {"0": 0.0023, "1": 0.116, "2": 0.6753},
                  "confidence": 0.673}},
 "usage": {"input_tokens": 239, "output_tokens": 0}}
```

Exact semantics, quoted rather than paraphrased (llama.cpp server docs, **P**):

- `confidence` — "a value from 0 to 1, where 0 means all options are equally likely". Normalised
  entropy, **not** calibrated probability.
- `noul` — "the probability that the answer is true"; no `confidence`, no `legend`.
- `score` — "the expected level index, weighted by probability. It can be between two levels."
- `probabilities` — "sum to 1"; they are "scaled with the temperatures stored in the model file.
  They are not guaranteed to be calibrated for your data."
- `usage.output_tokens` — "always 0"; `input_tokens` is the sum over all questions, so batching
  questions multiplies prefill.
- Errors: `400` invalid request; `501` not a decision model, or media without `--mmproj`.
- Detection: `GET /v1/models` → `architecture.output_modalities` contains `"decisions"` only for a
  **native** decision model; prompt-classifying LLMs deliberately do not get it.
- Joint vs independent: answers are independent "the exception is clef: it reads all the questions
  in one prompt and decides them jointly" (**P**).
- **Temperature**: no per-request `temperature` field is documented; the checkpoint owns it
  (Kev: `head.pt` carries `temperature`; Clef: `joint_head_config.json`). The harness contract's
  "same temperature for every config" is therefore a *checkpoint* property. TODO(verify): whether a
  request-level `temperature` is accepted and ignored.
- **ubatch trap**: "for laya, clef, vllm-d3 and lfm2-d1-omni, the whole prompt is evaluated in one
  batch: it must fit in `--ubatch-size`" (**P**). For `vllm-decision2`, options + prompt tail.

Kev's server adds demo routes — `POST /v1/systemone/permute` (one `choice` under several option
orders) and `POST /v1/systemone/separate` (each question in its own pass) — plus the
`x-typesafe-request-id` response header and bearer auth when `KEV_API_KEY` is set (**P**). Default
model names are `kev-latest` and `jev-latest`, the latter being the TypeSafe SDK default, so an
unconfigured SDK client talks to it (**P**).

## Per-runtime detail

### MLX — `mlx-lm` 0.32.2 / `mlx-vlm` 0.7.9 (P)

- **H**: `mlx_vlm.decision.predict(model, processor, state, questions, **kwargs)` validates and
  delegates to `model.predict(...)`. Types `choice | score | bool | multi_label`; support gated by
  `model.decision_types`; a model without it raises `ValueError("This model does not support decision
  prediction")`. `multi_label` takes `threshold` (default 0.5); `score` criteria must be a sequence
  (a Mapping raises); labels must be unique non-empty strings. Media gated by `model.decision_media`.
  **Inference:** this module is a validating shim — head logic lives in the checkpoint, so "does MLX
  support head X" == "did the checkpoint ship `predict`".
- **L**: read `model(ids)[:, -1]`, softmax yourself. In-process, no sampler, no server.
- **Custom code**: `mlx_lm/models/__init__.py` in 0.32.2 contains only a copyright header — no
  registry to append to; arch is resolved from the checkpoint's config/`model_type` (**inference**).
  Kev sidesteps this by importing `mlx_lm.models.qwen3_5` and building `qwen3_5.Model` directly
  (**P**), and it strips `lm_head` before materialising weights ("untied bases (Qwen3.8-27B): 2.5 GB
  never read") — the same trick any hidden-state reader should use.
- **KV / prefix**: `mlx_lm.models.cache.make_prompt_cache`; Kev's `PREFILL_CHUNK = 1024` with the
  measured note "on an M5 1,024 / 2,048 / 4,096 / one pass take the same time, the peak grows with
  the chunk" (**P**) → chunk size is a peak-memory knob, not a latency knob.
- **Gotcha**: MLX's buffer cache grows per request shape; Kev caps it — `mx.set_cache_limit(1 GiB)`,
  measured "Kev-4B on an M5, 50 requests: 1.0 GB cached vs 3.7 GB unbounded, same latency" (**P**).
  A harness that loops over shapes will leak GBs without this.

### llama.cpp — v0.7.0-dev in `src/llama.cpp` (P)

- **Version**: `CMakeLists.txt` sets `LLAMA_VERSION 0/7/0` + `LLAMA_BUILD_IS_DEV=ON`. Post-v0.7.0
  tree, therefore past the M5 tensor-API fix (**P** version, **D** the b10734 link).
- **Decision support is compiled in**: `LLM_ARCH_CLEF`, `LLM_ARCH_VLLM_DECISION2` in the local
  `llm_arch` enum, and the server documents the full `/v1/systemone` contract (**P**). The ggml-org
  `Clef-Flash-GGUF` repo exists (BF16 18.16 GB, Q8_0 9.66 GB, Q4_K_M 6.73 GB, mmproj 0.62 GB) —
  upstream-converted, which is the strongest available signal that this path is intended (**P**
  listing; the card's own claim that `/v1/systemone` needs the systemone PR (#29831) is
  **TODO(verify)** — our tree is newer than v0.7.0 so it should contain it).
- **Flags**: `--grammar/--grammar-file` (GBNF), `-j/--json-schema`, `response_format
  {"type":"json_schema"}`; `n_probs` on `/completion` and `post_sampling_probs`; `--cache-prompt`
  (default **on**), `--cache-reuse N`, `--slot-save-path` + `/slots/{id}?action=save|restore|erase`,
  `-ctxcp` (32 SWA checkpoints), `-b` 2048 / `-ub` **512** defaults, `-np`, `-cb` (on),
  `-ctk/-ctv` (9 KV types), `-fa auto`, `--embedding --pooling {none,mean,cls,last,rank}`,
  `--metrics` (exposes `llamacpp:prompt_tokens_seconds` etc. — free harness instrumentation) (**P**).
- **Server-side `--logits-all`: TODO(verify)** (documented read-out is `n_probs` + `post_sampling_probs`).
- **Cache accounting**: `/v1/chat/completions` returns `timings.cache_n` / `prompt_n` /
  `predicted_n` and `usage.prompt_tokens_details.cached_tokens` (**P**) — this is how you separate
  first-ask from cached-ask (plan H3/H4) without guessing.

### Kev (`src/kev`) — the Kev-family reference server (P)

- `uv run --extra serve python -m kev.serve --run runs/kev --port 8008`; FastAPI, one checkpoint,
  prefill-only decisions (**P**).
- Backend: `LoadOptions(KEV_BACKEND, KEV_DTYPE, ...)`; "on Apple Silicon a hybrid checkpoint runs on
  **MLX by default, without a torch MPS probe**, elsewhere on torch in bf16" (**P**). Torch path is
  the reported-number path; MLX has parity tests (`tests/test_mlx.py`, max |Δp| and argmax flips).
- Prefix cache: `KEV_PREFIX_CACHE` (default **4 states**), `KEV_PREFIX_MIN_TOKENS`,
  `KEV_PREFIX_MAX_TOKENS` (default **65536**); holds "KV + DeltaNet states; attention-only backbones
  also the state's hidden states". Measured: one 64k state ≈ **1.3 GB** of keys/values/DeltaNet
  states for Kev-27B; Kev-4B on MLX a 64k state cache is **2.2 GB** (**P**).
- Batching: `MAX_BATCH = 64` requests per model thread; `kev.cuda_graphs` splits to fit buffers (**P**).
- Admission: a state over `kev.model.SERVE_MAX_STATE` is refused **422**, or truncated with
  `KEV_TRUNCATE_STATES=1` and the response then says so via `usage.state_tokens` /
  `state_tokens_used` (**P**). Wire a harness to read those two fields, not to assume.
- **Packaging conflict (act on this)**: `pyproject.toml` pins `mlx-lm>=0.31.3,<0.32` for the
  `serve`/`mlx` extras, but `.venv` has **0.32.2** (**P** both). Installing `src/kev[serve]` naively
  will pull mlx-lm down a minor version and can break `mlx_vlm 0.7.9`. Install with
  `--no-deps`/`--no-build-isolation` for the mlx extra, or install the extras minus mlx-lm, then
  run parity tests. TODO(verify): whether `kev.mlx_model` is correct against 0.32.2.
- Dependencies otherwise: `torch>=2.6,<2.9`, `transformers>=5.17,<6`, `peft`, `accelerate` (**P**).

### PyTorch MPS + transformers (P for the pins, D for behaviour)

- All three families, trivially; this is the oracle that makes other engines trustworthy.
- **The GDN problem is the headline**: MPS has no Gated DeltaNet kernels, so Kev's hybrid backbone
  falls back to reference code (~0.8 s/request for 4B) (**P**). Any Qwen3.5/3.8-derived decision
  checkpoint has the same problem. Expect the same on Clef-Flash (same base family).
- Install: not present; `torch>=2.6,<2.9` + `transformers>=5.17,<6` (**P**). Download size
  TODO(verify) — expect multiple GB of wheels, the most expensive install in this document.
- Expect to need `PYTORCH_MPS_HIGH_WATERMARK_RATIO` tuning against the ~24.9 GiB wired ceiling and a
  warm-up call before timing (**D** for the ceiling, TODO(verify) for the ratio behaviour).

### ONNX Runtime 1.31.0 / CoreML (P presence, TODO(verify) behaviour)

- Installed **only** in `.venv-omlx` (`onnxruntime 1.31.0`, MIT) (**P**). Its wheel classifiers
  advertise Linux/Windows, so verify `get_available_providers()` at runtime before trusting CoreML.
- Encoder family only, realistically: one exported graph = one option-set shape. `score` with `k`
  levels and `choice` with `n` options are different input shapes unless you pad + mask.
- `head.safetensors` next to a DeBERTa encoder (open-jev-deberta) is the canonical merge target: fold
  the head into the graph so the output is `logits[1, max_options]`.

### oMLX / Ollama / LM Studio / vLLM / MLC-LLM (D, TODO(verify) for decision surfaces)

- **oMLX 0.7.1**: MLX server, continuous batching, tiered KV, MTP drafter (**D**). No decision
  endpoint found. Its value here is TTFT for repeated states, not semantics.
- **Ollama / LM Studio**: chat servers. No `/v1/systemone`. Logprob passthrough TODO(verify) for both.
  LM Studio additionally needs the M5-specific MLX runtime selected manually (**D**) — same trap class
  as everything MLX-adjacent on this SoC.
- **vLLM**: core has no Metal; `vllm-metal` is "vLLM scheduler + mlx_lm layers + paged varlen kernel +
  M5 NAX prefill" (**D**), so its decision coverage equals mlx_lm's. Install is brew/curl-venv and
  there is no brew here. The existence of `VLLM_DECISION2` / `vllm-d3` arch names in llama.cpp (**P**)
  says the reference implementations were written vLLM-side first — useful for finding reference code,
  not for running on this box.
- **MLC-LLM**: TVM→Metal, pre-Metal-4, per-model prebuilt libs, no head story (**D**). Skip.

## Recommended engine per model family

| Family | Checkpoints (artifact size, **P**) | Engine | Why |
|---|---|---|---|
| Encoder + head | `open-jev-deberta-v3-large` 1.74 GB + head 12.6 MB; `GLiNER2.5-Decide` 1.95 GB | **ONNX Runtime CPU→CoreML EP**; PyTorch CPU as oracle | sub-100 ms class; one static graph; no GPU contention with the resident 27B |
| LLM + native decision arch | `ggml-org/Clef-Flash-GGUF` Q4_K_M **6.73 GB** + mmproj **0.62 GB**; `vllm-decision2/d3`, Laya | **llama.cpp** `/v1/systemone` | head is in the GGUF; you get protocol + ubatch control + slot cache; verify Q4 accuracy |
| LLM + pointer head | Kev-9B (LoRA 173 MB + `head.pt` 8.4 MB) / 4B (130 + 5.3) / 0.8B (43 + 2.1) **on top of a Qwen3.5 base (19.33 GB bf16)** | **MLX via `kev.serve`** | only engine that runs `PointerHead` fast here; MPS is reference-only (~0.8 s/req @4B) |
| LLM + joint schema head, MLX | `mlx-community/clef-flash-4bit` **6.21 GB** (weights 5.95 + head 0.24, head **not** quantised), 8-bit 9.78 GB | **mlx-vlm / `clef_mlx.py`** | bundled `clef_mlx.py` + `systemone()`; 4-bit fits with room for KV |
| LLM, logprob read-out | SemIf over Qwen3.5-9B, Jev-Style v3 (0.8B), Tev1-4B 9.34 GB | **llama.cpp** (`n_probs`) for portability; **MLX** if the same weights already serve text | read-out is the only way a 9B coexists with llama.cpp on 32 GB |

Rule: **the read-out picks the engine, not the parameter count.** Every recommendation above is
decided by "is there a Python module, or is the head in the file?".

## Install / commands on this box

```bash
# ── already present, nothing to install ───────────────────────────────────────
src/llama.cpp/build/bin/llama-server --version        # 0.7.0-dev, Metal
.venv/bin/python -c "import mlx_vlm; print(mlx_vlm.__version__)"         # 0.7.9
.venv-omlx/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"

# 1. llama.cpp + native decision model (Clef-Flash GGUF)
src/llama.cpp/build/bin/llama-server \
  -m Clef-Flash-Q4_K_M.gguf --mmproj mmproj-Clef-Flash-Q8_0.gguf \
  -ub 2048 -b 2048 -c 8192 -fa on --host 127.0.0.1 --port 9931

curl -s localhost:9931/v1/models | jq '.data[0].architecture.output_modalities'   # must contain "decisions"

curl -s localhost:9931/v1/systemone -H 'Content-Type: application/json' -d '{
 "state":"Our checkout started returning errors and orders are blocked.",
 "questions":{
   "department":{"type":"choice","instructions":"Which team?","criteria":{"billing":null,"technical":null}},
   "outage":{"type":"noul","instructions":"Is a service down?"},
   "urgency":{"type":"score","criteria":["can wait","this week","today"]}}}' | jq

# 2. llama.cpp + logprob read-out (plain LLM), top-k option-token logprobs
src/llama.cpp/build/bin/llama-server -m <instruct>.gguf -c 16384 -fa on \
  -ctk q8_0 -ctv q8_0 --cache-prompt --slot-save-path ./slotcache
curl -s localhost:9931/completion -d '{"prompt":"Q: ...\nA: (","n_probs":5,"temperature":0,
  "max_tokens":1,"cache_prompt":true}'

# 3. Kev on MLX — install WITHOUT letting it downgrade mlx-lm below the 0.32.2 in .venv
uv pip install --python .venv/bin/python -e ./src/kev            # core + torch deps
# mlx-lm is pinned <0.32 by kev's extras but .venv has 0.32.2: skip the extra, keep 0.32.2,
# then run kev's parity suite before trusting speed numbers
.venv/bin/python -m kev.serve --run jaredpalmer/kev-4b --port 8008
#   KEV_PREFIX_CACHE=4 KEV_PREFIX_MAX_TOKENS=65536 KEV_DATE_FACTS=0

# 4. MLX decision checkpoint with its own predict/ (clef_mlx.py, kev.mlx_model)
.venv/bin/mlx_vlm.decide --model <mlx-decision-model> --state "..." \
  --questions '{"route":{"type":"choice","criteria":{"refund":null,"cancel":null}}}'

# 5. MLX logprob read-out, in process
.venv/bin/python - <<'PY'
import mlx.core as mx
from mlx_lm import load
model, tok = load("<mlx-model>"); ids = mx.array([tok.encode("Q: ...\nA: (")])
p = mx.softmax(model(ids)[0, -1], axis=-1)
print({c: float(p[tok.encode(c)[-1]]) for c in "ABC"})
PY

# 6. PyTorch MPS oracle — most expensive install here
uv venv .venv-torch && uv pip install --python .venv-torch/bin/python "torch>=2.6,<2.9" "transformers>=5.17,<6"
```

GPU memory guard (session-scoped, reverts on reboot):

```bash
sysctl iogpu.wired_limit_mb              # 0 = kernel default (~24.9 GiB recommended on 32 GB)
sudo sysctl iogpu.wired_limit_mb=26000   # reset with =0
```

## M5 / Metal-4 gotchas for this track

1. **Wrong llama.cpp = no decision endpoint at all.** `/v1/systemone` + `CLEF`/`VLLM_DECISION2` are
   newer than nixpkgs' 0.6.0; and pre-`b10734` builds silently disable the Metal-4 tensor API
   (**D**). Pin the in-repo build. TODO(verify) exact PR (#29831) that landed systemone.
2. **`--ubatch-size` is correctness, not tuning** for `clef`/`laya`/`vllm-d3`/`lfm2-d1`: whole prompt
   in one batch (**P**). Default is 512 in this build.
3. **`state` length can be silently truncated.** Kev: 422 over `SERVE_MAX_STATE`, or truncate-and-tell
   with `KEV_TRUNCATE_STATES=1` — read `usage.state_tokens_used` (**P**). Clef: `encode_record`
   takes `max_length` (default 16,384) and `max_state_tokens` (**P**). A harness that ignores this
   compares different inputs and calls it a quality difference.
4. **Prefill is the metric; decode is irrelevant.** llama.cpp prefill on M5 ≈ 60-70 % of MLX-derived
   engines and the NAX prefill patch was closed `not_planned` (**D**). For per-ask latency, MLX wins
   before any quality argument starts.
5. **Measure first-ask vs cached-ask separately.** llama.cpp reports `timings.cache_n`; Kev's prefix
   cache holds KV **and** DeltaNet states and is capped at 65,536 tokens (**P**). Cached-ask latency
   is a different number, not a better one.
6. **MLX buffer-cache growth** — `mx.set_cache_limit` or the RSS climbs per request shape (**P**).
7. **Probabilities are the checkpoint's temperature.** No documented per-request `temperature` on
   `/v1/systemone`; Kev applies `head.pt`'s temperature inside `PointerHead` fp32 (**P**).
   Cross-checkpoint calibration comparisons are valid; "same temperature across engines" is not a
   knob you have.
8. **`confidence` ≠ probability** — entropy-shaped, use `probabilities` for Brier/ECE.
9. **Heads are usually not quantised** — `mlx-community/clef-flash-4bit` keeps `joint_head` at
   243.54 MB while the backbone drops to ~5.95 GB (**P**). Quantising the head is an open accuracy
   question, not a default.

## TODO(verify)

- llama.cpp PR #29831 (systemone) — confirm it is in `src/llama.cpp`'s history, and that
  `--logits-all`/`n_probs` cover every read-out a logprob model needs on the server.
- `kev.mlx_model` against **mlx-lm 0.32.2** (declared `<0.32`): parity, not just import.
- Whether oMLX 0.7.1 / Ollama / LM Studio expose any decision or logprob surface.
- CoreML/coremltools version and macOS-26 status; `coremltools` M5 support.
- Exact torch/transformers wheel size on macOS arm64, and MPS watermark behaviour on M5 32 GB.
- TypeSafe hosted API details beyond the primitives (rate limits, auth, model names) — the docs
  pages were HTML-only in this pass; the primitives above are confirmed via three local
  implementations instead.
- Q4 vs Q8 accuracy on Clef-Flash (head sensitivity), and whether `score` expectations survive 4-bit.

---

*Evidence: `src/llama.cpp` (v0.7.0-dev: `tools/server/README.md`, `src/llama-arch.h`, `CMakeLists.txt`);
`src/kev` (`kev/serve.py`, `kev/mlx_model.py`, `kev/checkpoint.py`, `pyproject.toml`);
`.venv` (`mlx_vlm 0.7.9` `decision.py`, `mlx-lm 0.32.2`); `.venv-omlx` (`onnxruntime 1.31.0`);
HF cards/listings for Clef-Flash, Kev, open-jev-deberta, GLiNER2.5-Decide, d1-3B; repo prior
`shared/02-engines.md` (2026-10-20). Quotes: [`02-engines.sources.md`](02-engines.sources.md).*

## Verified on this machine (2026-10-27)

Measured, not surveyed. The build in `src/llama.cpp` is `version: 0.7.0-dev (build 1, commit
620d222)`, and it already contains the whole decision surface:

- `common/common.h` defines decision types `openjev`, `lev`, **`kev`**, `nimble`, `laya`, **`clef`**,
  `pplx_decider`, `vllm_decision2`, **`vllm_d3`**, `lfm2_d1`, `lfm2_d1_omni`. `POST /v1/systemone`
  is routed in `tools/server/server.cpp`; the engine logs `decision model type: <name>` at load.
- A decision model answers with `probabilities` per option; `noul` returns only `P(true)` and the
  harness expands it; `confidence` is normalised entropy, **not** a probability. ECE must be computed
  from `probabilities`.
- **Prompt caching is off by default for decision models, and turning it on barely helps.**
  The server logs `decision model: prompt cache disabled by default, use --cache-ram N to enable it`.
  Paired 100-case run on Kev-9B (same states, same thermal state):

  | Arm | cold p50 | cached re-ask p50 | change |
  | --- | --- | --- | --- |
  | `--no-cache-prompt` | 3,537 ms | 3,590 ms | +1.5 % |
  | `--cache-ram 4096` | 3,510 ms | 3,276 ms | **−6.7 %** |

  The cache is genuinely used (the log stores and evicts ~55 MB entries), but a decision model reads
  **hidden states/embeddings at option positions**, and llama.cpp's prompt cache stores K/V — not
  those hidden states. The forward pass is therefore not skipped. Do not budget a repeated-state
  workload on prompt caching for this model class.
- For `clef`, the whole prompt is evaluated in one `--ubatch-size` batch, so `-ub 2048` is required
  for five-question requests; the MLX loader instead prefills in 1,024-token chunks and is 1.8×
  faster end to end on this part (see [`05-results.md`](05-results.md)).
- MLX side: `mlx-community/clef-flash-4bit` ships `clef_mlx.py` and runs with `mlx 0.32.4`,
  `mlx-lm 0.32.2`, `mlx-vlm 0.7.9` (it warns that it was tested on 0.32.3 / 0.32.0 / 0.7.4; no
  failure observed on 2,000 decisions).
- `d3_mlx.py` ships no HTTP server; `benchmarks/decision/serve_d3_mlx.py` adds the `/v1/systemone`
  wrapper. `kev.serve` needs `torch` at import time even when the MLX backend runs.
