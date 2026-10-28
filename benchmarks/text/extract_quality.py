#!/usr/bin/env python3
"""Extract lm-eval result blocks from mlx_lm.evaluate logs into results/text/quality/summary.jsonl.

mlx_lm.evaluate prints its results dict to stdout instead of reliably writing the
output-dir, so the log is the source of truth. Parses the trailing JSON blocks.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "logs", "text")
OUT = os.path.join(ROOT, "results", "text", "quality", "summary.jsonl")

rows = []
for p in sorted(glob.glob(os.path.join(LOGS, "q_*.log"))):
    tag = os.path.basename(p)[2:-4]
    try:
        with open(p, errors="replace") as fh:
            txt = fh.read()
    except OSError as e:
        print(f"skip {p}: {e}", file=sys.stderr)
        continue
    # strip tqdm carriage-return spam
    txt = txt.replace("\r", "\n")
    for b in re.findall(r'\{\s*"name":\s*"[^"]+".*?\n\}', txt, re.S):
        try:
            d = json.loads(b)
        except json.JSONDecodeError:
            continue
        row = {"tag": tag, "task": d.get("name"), "n": d.get("sample_len")}
        for k, v in d.items():
            if k.endswith(",none") and "stderr" not in k:
                row[k.split(",")[0]] = round(v, 4)
            if k.endswith("_stderr,none"):
                row[k.split("_stderr")[0] + "_stderr"] = round(v, 4)
        rows.append(row)

out = OUT
try:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
except OSError as e:
    print(f"cannot write {out}: {e}", file=sys.stderr)
    sys.exit(1)

for r in rows:
    metrics = {k: v for k, v in r.items() if k not in ("tag", "task", "n") and not k.endswith("_stderr")}
    print(f"{r['tag']:<32} {r['task']:<26} n={r['n']:<5} {metrics}")
print(f"\n{len(rows)} rows -> {out}", file=sys.stderr)
