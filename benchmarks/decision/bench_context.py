#!/usr/bin/env python3
"""Prefill cost of a decision request as the state grows: cold ask vs cached re-ask.

The typed-decisions suite asks about short states (~200-400 tokens), so it cannot answer
"what does a 16k-token contract cost?". This harness measures that separately, on synthetic
states, with the same question schema at every depth. The text is filler, the questions are
fixed: the number is a *prefill* measurement, not an accuracy one.

    python3 benchmarks/decision/bench_context.py --base-url http://127.0.0.1:8091 \
        --engine mlx-clef --model clef-flash-4bit-mlx --depths 500,2000,8000,16000

Per depth, one row: `cold` = distinct states (a nonce forces a real prefill), `cached` = the
last state asked again, which should hit the server's prefix cache.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_systemone import (  # noqa: E402
    checked_url,
    pct,
    post_json,
    wait_ready,
)

FILLER = (
    "The supplier shall deliver the equipment described in Schedule A no later than thirty days "
    "after the effective date, and shall notify the customer in writing of any delay together with "
    "a revised delivery schedule and the reasons for it. "
)

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this document?",
        "criteria": {"legal": "contract terms", "billing": "payments", "support": "operations"},
    },
    "overdue": {"type": "noul", "instructions": "Does the document describe a delay?"},
    "risk": {"type": "score", "instructions": "How risky is this clause?", "criteria": ["low", "medium", "high"]},
    "party": {
        "type": "choice",
        "instructions": "Who bears the delivery obligation?",
        "criteria": {"supplier": None, "customer": None, "neither": None},
    },
    "actionable": {"type": "noul", "instructions": "Does this require immediate action?"},
}


def build_state(approx_tokens: int, nonce: int) -> str:
    # ~3.6 chars/token for this filler; the engine's usage field gives the real count.
    target = max(1, int(approx_tokens * 3.6))
    body = (FILLER * (target // len(FILLER) + 1))[:target]
    return f"[ref {nonce}] {body}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/decision/context.jsonl")
    ap.add_argument("--depths", default="500,2000,8000,16000")
    ap.add_argument("--reps", type=int, default=5, help="distinct states per depth (cold)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    checked_url(args.base_url)
    wait_ready(args.base_url)
    url = args.base_url.rstrip("/") + "/v1/systemone"
    rows = []
    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        cold: list[float] = []
        tokens: list[int] = []
        last_state = ""
        for rep in range(args.reps):
            state = build_state(depth, rep + 1)
            last_state = state
            payload = {"model": args.model, "state": state, "questions": QUESTIONS}
            body, elapsed = post_json(url, payload, args.timeout)
            cold.append(elapsed)
            usage = body.get("usage")
            value = usage.get("input_tokens") if isinstance(usage, dict) else None
            if isinstance(value, (int, float)):
                tokens.append(int(value))
        cached: list[float] = []
        for _ in range(max(1, args.reps // 2)):
            payload = {"model": args.model, "state": last_state, "questions": QUESTIONS}
            _, elapsed = post_json(url, payload, args.timeout)
            cached.append(elapsed)
        measured = statistics.fmean(tokens) if tokens else float("nan")
        row = {
            "modality": "decision",
            "task": "prefill-depth",
            "engine": args.engine,
            "model": args.model,
            "label": args.label or f"synthetic-state depth={depth}",
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "valid": len(cold) == args.reps,
            "metrics": {
                "target_tokens": depth,
                "input_tokens_mean": measured,
                "cold_ms_p50": pct(cold, 0.50),
                "cold_ms_p95": pct(cold, 0.95),
                "cached_ms_p50": pct(cached, 0.50),
                "cached_ms_p95": pct(cached, 0.95),
                "prefill_tokens_per_s": (
                    measured / (pct(cold, 0.50) / 1000.0)
                    if tokens and pct(cold, 0.50) > 0 and not math.isnan(measured)
                    else float("nan")
                ),
                "reps": args.reps,
            },
            "harness": {"protocol": "POST /v1/systemone", "state": "synthetic filler", "machine": "M5 32GB"},
        }
        rows.append(row)
        print(json.dumps(row["metrics"], ensure_ascii=False), flush=True)

    try:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise SystemExit(f"cannot append to {args.out}: {exc}") from exc


if __name__ == "__main__":
    main()
