# Shared — facts that apply to every modality

These files are not a track. They are the constraints a track has to respect, and they are the only
place a cross-modality claim is allowed to live. If an engine fact changes, it changes here.

| File | What it is |
| --- | --- |
| [`01-hardware.md`](01-hardware.md) | The machine, the capability bar, and the physics that bounds results |
| [`02-engines.md`](02-engines.md) | Engine survey for Apple M5: comparison, install paths, M5-specific traps |

## What belongs here

- Hardware facts: cores, bandwidth, Metal limit, disk, swap.
- Engine facts: Metal tensor API gates, MTP support, quantization formats, API surfaces.
- Derived ceilings (`tok/s ≈ bandwidth / bytes-per-token`) and the arithmetic behind them.
- Traps that would invalidate a whole run regardless of modality (thermal throttling, wiring limit,
  silent CPU fallback).

## What does not

- Datasets, metrics, thresholds, or recommendations for one modality — those live in
  [`../text/`](../text/README.md), [`../decision/`](../decision/README.md), and the scoped
  [`../image/`](../image/README.md), [`../video/`](../video/README.md),
  [`../audio/`](../audio/README.md), [`../speech/`](../speech/README.md).
- Raw results. Results belong in [`results/<modality>/`](../../results/README.md).

Engine numbers measured in the text track are tagged with the modality that measured them. Do not
assume a text-track engine gap transfers unchanged to a different prefill shape: the decision track
found the MLX/llama.cpp ordering held but the ratio moved from 1.24× to 1.8×.
