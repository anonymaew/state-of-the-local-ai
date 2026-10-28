# Sources — decision engines (stream A)

Raw quotes, one block per claim, with the exact file read. Tier letters match
`02-engines.md`: **P** = primary/local, **D** = repo prior, **U/TODO** = unverified.

## 1. `/v1/systemone` protocol

### P — `src/llama.cpp/tools/server/README.md`, "TypeSafe-compatible API Endpoints"

> ### POST `/v1/systemone`: TypeSafe-compatible System One API
> Answers typed questions about a `state` with a decision model.
> Follows the [TypeSafe API](https://docs.typesafe.ai/api), streaming is not supported.

> `questions`: An object that maps a question id to a question. Each question has these fields:
> - `type`: One of `choice`, `score`, `noul`.

> `criteria`: The possible answers, the shape depends on `type`:
> - `choice`: An object that maps each option to its description. The description can be `null`.
> - `score`: An array of 2 to 10 level descriptions, lowest level first.
> - `noul`: Optional. An object with the descriptions of `true` and `false`.

> The questions of a request are answered independently, an answer does not depend on the other
> questions. The exception is clef: it reads all the questions in one prompt and decides them jointly.

> The number of options of a `choice` question is limited by the model, for example: 52 for openjev,
> 255 for laya, clef, pplx-decider, vllm-decision2, vllm-d3, lfm2-d1 and lfm2-d1-omni.

> For laya, clef, vllm-d3 and lfm2-d1-omni, the whole prompt is evaluated in one batch: it must fit in
> `--ubatch-size`. An lfm2-d1-omni prompt is cut to 16384 tokens. For vllm-decision2, the options of a
> question and the end of the prompt must fit in one batch.

> - `choice`: `choice` (highest probability), `probabilities` (sum to 1), `confidence`: "A value from
>   0 to 1, where 0 means all options are equally likely."
> - `score`: `score` — "The expected level index, weighted by probability. It can be between two
>   levels." + `legend` + `probabilities` + `confidence`.
> - `noul`: `noul` — "The probability that the answer is true."

> `usage`: `input_tokens` is the number of prompt tokens of all questions. `output_tokens` is always 0.

> The probabilities are scaled with the temperatures stored in the model file. They are not guaranteed
> to be calibrated for your data.

> An invalid request returns the error `400`. A model that is not a decision model returns the error
> `501`.

### P — same file, `GET /v1/models` capability probe

> One output value is special: **`decisions`** — The model is a native decision model. Serve it with
> [`/v1/systemone`](#post-v1systemone-typesafe-compatible-system-one-api).
> A language model that classifies with prompts does not get `decisions`. Only native decision models do.

```js
const useSystemOne = model.architecture?.output_modalities?.includes("decisions") === true;
```

### P — same file, `POST /completion` logprobs and prefix cache

`/slots` example payload contains `"n_probs": 0`, `"post_sampling_probs": false`.
Server params table:

> `--cache-prompt, --no-cache-prompt` — whether to enable prompt caching (default: enabled)
> `--cache-reuse N` — min chunk size to attempt reusing from the cache via KV shifting, requires prompt
> caching to be enabled (default: 0)
> `--slot-save-path PATH` — path to save slot kv cache (default: disabled)
> `-ub, --ubatch-size N` — physical maximum batch size (default: 512)
> `-b, --batch-size N` — logical maximum batch size (default: 2048)
> `-cram, --cache-ram N` — set the maximum cache size in MiB (default: 8192, 0 for decision models, …)
> `--pooling {none,mean,cls,last,rank}` — pooling type for embeddings
> `--port PORT` — port to listen (default: 9931)

`/v1/chat/completions` response timings:

> `"cache_n": 236, // number of prompt tokens reused from cache`

### P — `src/llama.cpp/src/llama-arch.h` (decision + encoder archs compiled in)

```
LLM_ARCH_BERT, LLM_ARCH_MODERN_BERT, LLM_ARCH_NOMIC_BERT, LLM_ARCH_NOMIC_BERT_MOE,
LLM_ARCH_NEO_BERT, LLM_ARCH_JINA_BERT_V2, LLM_ARCH_JINA_BERT_V3, LLM_ARCH_EUROBERT,
LLM_ARCH_QWEN35, LLM_ARCH_QWEN35MOE, LLM_ARCH_CLEF, LLM_ARCH_VLLM_DECISION2,
LLM_ARCH_LFM2, LLM_ARCH_LFM2MOE, LLM_ARCH_T5ENCODER
```

### P — `src/llama.cpp/CMakeLists.txt`

```
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_MINOR 7)
set(LLAMA_VERSION_PATCH 0)
option(LLAMA_BUILD_IS_DEV "llama: dev build" ON)
```

### P — `Cloudflare/clef-flash` model card (`/tmp/decision-sources/clef-flash-card.md`)

> Clef-Flash is a 9B multimodal model that turns a state and a schema of typed questions into
> decisions. It reads the state as text, JSON, images, or video, and returns a probability for every
> allowed option of every question in a single forward pass. There is no free-form text generation and
> no output parsing.
> The Clef-Flash API is fully compatible with Jev and SystemOne.

> **Joint schema head:** a small transformer head that reads the backbone's final hidden states, routes
> evidence from the state to each question, and scores all options of all questions jointly.
> **Output:** one logit per allowed option for each question. Apply a softmax per question to get
> probabilities.

> `joint_schema_model.py` | Record encoding, batching, the model, `load_release_model`, and `systemone`

> Tested with `torch` 2.11 and `transformers` 5.10.2 on a single H200.

> The server answers Jev/SystemOne requests at `/v1/systemone` …
> `systemone` takes a Jev/SystemOne `POST /v1/systemone` request body and returns the same response
> body: `model`, `answers` keyed by question ID, and `usage`. A `choice` answer has `choice`,
> `confidence`, and `probabilities`; a `score` answer has the expected `score`, `confidence`, `legend`,
> and `probabilities`; a `noul` answer has the probability of true. `instructions` is optional, and
> `images` and `videos` may be added to the request.

> `encode_record` accepts `max_length` (default 16,384 tokens) and `max_state_tokens` to bound the input.

### P — `src/kev/kev/serve.py` module docstring

> TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header,
> and bearer auth when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST
> /v1/systemone/permute (one Choice under several option orders) and POST /v1/systemone/separate (each
> question in its own pass, for the packed-vs-separate comparison).

> A state over kev.model.SERVE_MAX_STATE tokens is refused with a 422 (kev.model.admit);
> KEV_TRUNCATE_STATES=1 reads its first SERVE_MAX_STATE tokens instead, and then every response says
> whether it did.

> Backend and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple Silicon a hybrid
> checkpoint runs on MLX by default, without a torch MPS probe, elsewhere on torch in bf16.

> MODEL_NAMES = ("kev-latest", "jev-latest")  # both names serve this checkpoint; jev-latest is the
> TypeSafe SDK default model, so an unconfigured client works

### P — same file, cache/batch constants

```
PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))       # states kept (KV + DeltaNet states; attention-only backbones also the state's hidden states); 0 disables
PREFIX_MAX_TOKENS = int(os.environ.get("KEV_PREFIX_MAX_TOKENS", "65536"))
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
MAX_BATCH = 64          # requests the model thread takes at once
```

> One 64k state (Kev-27B: ~1.3 GB of keys, values and DeltaNet states), or four 16k ones, not four 64k ones

> Kev-4B on MLX: a 64k state's cache is 2.2 GB

## 2. MLX

### P — `.venv/lib/python3.12/site-packages/mlx_vlm/version.py`

```
__version__ = "0.7.9"
```

### P — `.venv/.../mlx_vlm/__init__.py`

```
from .decision import predict
from .generate import (AudioGenerationResult, BatchResponse, BatchStats, GenerationResult,
    PromptCacheState, batch_generate, generate, generate_audio, save_audio, stream_generate)
```

### P — `.venv/.../mlx_vlm/decision.py`

> """Predict named decisions using a model's native scoring and calibration.
> Questions contain a type, instructions, and criteria. Types are ``choice``, ``score``, ``bool``, and
> ``multi_label``; support is model-specific. Probabilities and independent scores remain distinct, and
> model-specific metrics stay in ``metadata``."""

```
supported = getattr(model, "decision_types", ())
if not supported:
    raise ValueError("This model does not support decision prediction")
...
media = getattr(model, "decision_media", ())
...
threshold = question.get("threshold", 0.5)
if kind == "score" and isinstance(criteria, Mapping):
    raise ValueError("score criteria must be an ordered list")
...
return model.predict(processor, state, normalized, **kwargs)
```

### P — `.venv/.../mlx_lm/models/__init__.py`

Whole file: `# Copyright © 2024 Apple Inc.` — no registry.

### P — `.venv/.../mlx_lm/generate.py` (defaults)

```
DEFAULT_TEMP = 0.0
DEFAULT_QUANTIZED_KV_START = 5000
DEFAULT_PREFILL_STEP_SIZE = 2048
from .models.cache import (QuantizedKVCache, TokenBuffer, can_trim_prompt_cache,
    load_prompt_cache, make_prompt_cache, trim_prompt_cache)
```

### P — `src/kev/kev/mlx_model.py` (docstring + constants)

> Apple Silicon backend for the Qwen3.5 checkpoints: mlx-lm's Metal implementation of the hybrid
> backbone under Kev's own encoder and pointer head.

> MPS has no Gated DeltaNet kernels, so the PyTorch path runs reference code there (Kev-4B ~0.8 s per
> request).

> the readout is the very same `PointerHead` (fp32, with the checkpoint's temperature) applied to the
> branch hidden states.

> Selected by `LoadOptions(backend="mlx")` (or "auto" on Apple Silicon) in `kev.checkpoint`; never by
> the benchmark, whose reported numbers stay on the fp32 torch path. Parity against that path is
> measured in tests/test_mlx.py.

```
CACHE_LIMIT = 1 << 30   # MLX's buffer cache keeps a buffer per new request shape; Kev-4B on an M5, 50 requests: 1.0 GB cached vs 3.7 GB unbounded, same latency
PREFILL_CHUNK = 1024    # state tokens per prefix pass (MLXDecisionModel.prefix); on an M5 1,024 / 2,048 / 4,096 / one pass take the same time, the peak grows with the chunk
```

> The LM head is removed before any weight is materialized: Kev reads hidden states, and the
> checkpoint carries none (vision tower and MTP were never loaded either).

## 3. Kev packaging / checkpoint layout

### P — `src/kev/pyproject.toml`

```
name = "kev"
version = "0.1.0"
description = "Laptop-scale reconstruction of a Jev-style decision model with a TypeSafe-compatible /v1/systemone API"
requires-python = ">=3.12,<3.14"
dependencies = ["accelerate>=1.15.0", "datasets>=3.0", "numpy>=2.5.3", "peft>=0.21",
    "pydantic>=2.9", "scikit-learn>=1.9.1", "torch>=2.6,<2.9", "transformers>=5.17,<6"]
[project.optional-dependencies]
serve = ["fastapi>=0.115", "typesafe-sdk>=0.6.0", "uvicorn>=0.30",
    "mlx-lm>=0.31.3,<0.32; sys_platform == 'darwin' and platform_machine == 'arm64'"]
```

⚠️ Conflict: `mlx-lm>=0.31.3,<0.32` declared vs **0.32.2** installed in `.venv` (see §2).

### P — `src/kev/kev/checkpoint.py` (docstring + `Meta`)

> Loader rule: `adapter_config.json` present -> a LoRA adapter on `meta.base` at `meta.base_revision`;
> no adapter and `config.json` + `model*.safetensors` -> the backbone is loaded from the checkpoint
> directory itself, nothing is merged.

```
ck = Checkpoint("jaredpalmer/kev-4b")          # or a local run directory; `@tag` pins a Hub revision
tok, model = ck.load("mps", LoadOptions.from_env())
ck.meta.temperature                             # the calibration the checkpoint carries
```

```
base: str; head: dict | None; base_revision; lora: int = 0; head_dim: int = 256;
option_isolation: bool = False; special_embeddings: bool = False; weights_dtype: str = "fp32";
temperature: float = 1.0; weights: str = "lora"   # "full": whole backbone in the checkpoint
```

## 4. Artifact sizes (HF file listings, `/tmp/decision-sources/hf-summary.txt`)

| Repo | Total | Notable files |
|---|---|---|
| `jaredpalmer/kev-9b` | **0.202 GB** | adapter_model 173.18 MB, head.pt 8.40 MB — **adapter only, needs a base** |
| `jaredpalmer/kev-4b` | 0.160 GB | adapter 129.92 MB, head.pt 5.25 MB |
| `jaredpalmer/kev-0.8b` | 0.066 GB | adapter 43.34 MB, head.pt 2.10 MB |
| `Cloudflare/clef-flash` | **19.083 GB** | 4 shards + joint_head 243.54 MB + `joint_schema_model.py` |
| `mlx-community/clef-flash-4bit` | **6.214 GB** | weights 5349.77 + 600.45 MB, **joint_head 243.54 MB (unquantised)**, `clef_mlx.py` |
| `TrevorJS/clef-flash-mlx-8bit` | 9.778 GB | same layout |
| `ggml-org/Clef-Flash-GGUF` | 36.101 GB | BF16 18164.49 MB, Q8_0 9657.26 MB, **Q4_K_M 6732.78 MB**, mmproj BF16 921.70 MB / Q8_0 624.23 MB |
| `bartowski/Cloudflare_clef-flash-GGUF` | 124.884 GB (ladder) | Q4_K_M 6040.54 MB … Q2_K 4194.03 MB |
| `Qwen/Qwen3.5-9B-Base` | 19.329 GB | Kev's base |
| `com-kotobalabs/open-jev-deberta-v3-large` | 1.760 GB | model 1736.09 MB + **head.safetensors 12.59 MB** + `typed_decisions/open_jev.py` |
| `fastino/GLiNER2.5-Decide` | 1.955 GB | single safetensors |
| `LiquidAI/d1-3B` | 6.265 GB | + `hybrid.py`, `runner.py`, `prompt.py` |
| `LiquidAI/d1-omni-600M` | 2.354 GB | + `audio.py`, `vision.py`, `encoder.py` |
| `togethercomputer/Tev1-4B-experimental` | 9.343 GB | 2 shards |
| `akhilaaa3/Jev-Omni` | 23.976 GB | model 23919.55 MB + head.pt 3.97 MB + `jev_omni.py` |

## 5. Repo prior (D) — [`../shared/02-engines.md`](../shared/02-engines.md), 2026-10-20

Used for engine-wide facts not re-verified this pass:

> Metal 4.0 tensor API … silently **inert on M5/A19** … Fixed by PR #27461, merged 2026-09-01,
> shipping in builds from **b10734** onward.

> M5 Max prefill ~60% of oMLX … a 6-kernel patch set measured pp 8k: 633 → 1064 t/s (+68%) … closed
> as `not_planned`.

> `mlx_lm/models/qwen3_5.py` loads with `weights = {k: v for k, v in weights.items() if "mtp." not in k}`

> vLLM Metal: "vLLM supplies the API server, scheduler, and paged block manager; **mlx_lm supplies the
> model layers**; vllm-metal owns the request-aware attention path".

> LM Studio's MLX models failed on M5 until users manually installed and selected the "LM Studio MLX
> (Apple M5)" runtime (bug #1356).

## 6. Local environment facts (P)

```
.venv/pyvenv.cfg     → CPython 3.12, uv 0.13.0
.venv-omlx/pyvenv.cfg→ CPython 3.12
.venv-omlx/.../onnxruntime-1.31.0.dist-info/METADATA → Name: onnxruntime, Version: 1.31.0, License: MIT
src/llama.cpp/build/CMakeCache.txt → generated by nix cmake-4.4.3; BUILD_SHARED_LIBS=ON
```

## 7. Not verified in this pass

- TypeSafe hosted API pages (`docs.typesafe.ai/api/system-one`, `/concepts/system-one`) were
  prefetched as **HTML**; the rendered text was not extracted. Wire primitives above are confirmed
  instead from three local implementations (llama.cpp server, Clef card, Kev serve) that all claim
  compatibility.
- llama.cpp PR **#29831** (systemone landing) — referenced by the ggml-org Clef-Flash card per the
  orchestrator; not read directly. The local tree documents the endpoint and compiles `CLEF`, so the
  feature is present, but the PR number/date is unconfirmed.
- `--logits-all` on `llama-server` (vs `n_probs`): not found in the auto-generated server help.
- oMLX 0.7.1 / Ollama / LM Studio decision + logprob surfaces.
- CoreML/coremltools version, macOS 26 status.
- torch/transformers wheel download sizes on macOS arm64.
- M5 wired-limit numbers are carried over from [`../shared/02-engines.md`](../shared/02-engines.md) (D), measured there on a 32 GiB M2 Max.
