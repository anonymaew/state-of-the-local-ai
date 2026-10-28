#!/usr/bin/env python3
"""Turn results/decision/*.jsonl into markdown tables for the docs.

    python3 benchmarks/decision/summarize.py                 # all suites
    python3 benchmarks/decision/summarize.py --task typed-decisions

Grouping is by (task, model, engine): rows are separate runs, so repeated labels are shown
individually instead of being averaged. Invalid rows are printed with `valid=false` rather
than dropped — a rejected run is evidence.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "decision"

ORDER = [
    "model",
    "engine",
    "quant",
    "ctx",
    "n",
    "accuracy",
    "kl",
    "brier",
    "ece",
    "p50_ms",
    "p95_ms",
    "cached_ms",
    "tokens",
    "rss_mb",
    "valid",
]


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"# skipping unparsable line in {path.name}")
    return rows


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[dict]) -> str:
    head = "| " + " | ".join(ORDER) + " |\n|" + "|".join(["---"] * len(ORDER)) + "|\n"
    body = []
    for row in rows:
        m = row.get("metrics") or {}
        h = row.get("harness") or {}
        body.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model", "—")),
                    str(row.get("engine", "—")),
                    str(h.get("quant") or "—"),
                    str(h.get("ctx_tokens") or "—"),
                    str(m.get("n_decisions") or m.get("reps") or "—"),
                    fmt(m.get("accuracy")),
                    fmt(m.get("kl_from_gold")),
                    fmt(m.get("brier")),
                    fmt(m.get("ece")),
                    fmt(m.get("latency_ms_p50"), 0),
                    fmt(m.get("latency_ms_p95"), 0),
                    fmt(m.get("cached_latency_ms_p50"), 0),
                    fmt(m.get("input_tokens_p50") or m.get("input_tokens_mean"), 0),
                    fmt(m.get("peak_rss_mb"), 0),
                    fmt(bool(row.get("valid"))),
                ]
            )
            + " |"
        )
    return head + "\n".join(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, help="only rows with this task")
    ap.add_argument("--paths", nargs="*", default=None)
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else sorted(RESULTS.glob("*.jsonl"))
    rows = load_rows(paths)
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
    if not rows:
        raise SystemExit(f"no rows in {', '.join(p.name for p in paths)}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task"))].append(row)
    for task, task_rows in groups.items():
        print(f"\n## {task}\n")
        print(table(task_rows))


if __name__ == "__main__":
    main()
