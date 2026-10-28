#!/usr/bin/env python3
"""Resolve data/decision/manifest.json into normalized decision items, one JSON object per line.

    python benchmarks/decision/load_suite.py --suite smoke --out /tmp/smoke.items.jsonl
    python benchmarks/decision/load_suite.py --suite breadth-v1-dev --dry-run
    python benchmarks/decision/load_suite.py --suite breadth-v1-dev --out "$DATA/breadth-v1-dev.items.jsonl"

Item shape (what the harness consumes; nothing else in the pipeline needs to know about suites):

    {"id": "musr/development/12",
     "state": <str | dict | list>,                      # passed through api.render() by the engine
     "questions": {"<qid>": {"type": "choice"|"noul"|"score",
                             "instructions": <str|dict|list|null>,
                             "criteria": {...}|[...]}},
     "gold": {"<qid>": <option key | bool | level index>},
     "meta": {"suite", "split", "source", "area", "metric", "variant", "partition_sha256"}}

Resolution order for a frozen partition (first hit wins, every hit is sha256-checked against the manifest):
  1. $DATA/cache/<suite>/<partition>            already fetched
  2. <repo>/src/kev/evals/<suite>/<partition>   the git copy the published numbers come from
  3. the manifest's Hub mirror, at its pinned revision  (huggingface_hub, imported only on this path)

Rule 3 is a fallback, never the default: breadth-v1's partitions are in a PRIVATE mirror because several sources do not
allow public redistribution (see docs/decision/04-eval.md). A 404/gated answer there is reported as an actionable error
instead of silently loading a different file.

Budgets: records per dataset and total state tokens come from the manifest. Selection is deterministic:
records are ordered by sha256("<seed>:<source>:<record id>") and the first `cap_records` of each dataset are kept, so
the same manifest always yields the same items, and changing a cap is visibly a different item set (a new label).

Dependencies: stdlib only, plus huggingface_hub on the download path. No `datasets`, no tokenizer: state length is
estimated as chars / chars_per_token_estimate for budgeting and measured exactly by the harness at run time.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "decision" / "manifest.json"
DATA = Path(os.environ.get("DATA") or (ROOT / "data" / "decision"))


class SuiteError(RuntimeError):
    """A suite that cannot be resolved as described by the manifest."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    # split on "\n" only: frozen partitions are UTF-8 with LF endings and may contain U+2028
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteError(f"cannot read partition {path}: {exc}") from exc
    try:
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    except json.JSONDecodeError as exc:
        raise SuiteError(f"invalid JSONL in {path}: {exc}") from exc


def cache_dir(spec):
    return DATA / "cache" / _name(spec["suite"])


def _name(value, what="name"):
    """A single path segment: suite and partition names come from the manifest, never from a URL."""
    text = str(value)
    if not text or text in (".", "..") or "/" in text or "\\" in text:
        raise SuiteError(f"manifest {what} {text!r} is not a plain path segment")
    return text


def repo_path(*parts):
    """<ROOT>/<parts...>, refusing a manifest path that escapes the repository."""
    path = (ROOT.joinpath(*map(str, parts))).resolve()
    if ROOT.resolve() not in path.parents:
        raise SuiteError(f"manifest path {Path(*map(str, parts))} escapes {ROOT}")
    return path


def verify(path, spec):
    want = (spec.get("provenance") or {}).get("sha256")
    got = sha256_file(path)
    if want and got != want:
        raise SuiteError(f"checksum mismatch for {path}: manifest says {want}, file is {got}. "
                         f"Refusing to score a different item set than the manifest describes.")
    return got


def _hub():
    """huggingface_hub, imported only when a download is actually needed (stdlib-only path otherwise)."""
    try:
        module = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise SuiteError("huggingface_hub is required to fetch a missing partition (pip install huggingface_hub)") from error
    errors = importlib.import_module("huggingface_hub.errors")
    return module, errors


def fetch_from_hub(spec):
    """Download only this partition into the cache. The mirror is the manifest's own if it names one."""
    prov = spec.get("provenance") or {}
    dataset = prov.get("mirror_dataset") or prov.get("fallback_dataset")
    revision = prov.get("mirror_revision") or prov.get("fallback_revision")
    if not dataset:
        raise SuiteError(f"{spec['suite']}/{spec['partition']} is missing locally and the manifest names no mirror")
    hub, errors = _hub()
    partition = _name(spec["partition"], "partition")
    target = cache_dir(spec) / partition
    target.parent.mkdir(parents=True, exist_ok=True)
    relative = f"{_name(spec['suite'])}/{partition}"        # path inside the Hub dataset repo
    try:
        got = hub.hf_hub_download(dataset, relative, repo_type="dataset", revision=revision, local_dir=str(target.parent))
    except (errors.RepositoryNotFoundError, errors.GatedRepoError) as error:
        raise SuiteError(
            f"{relative} is only in {dataset}@{str(revision)[:10]}, which is missing or private to this account. "
            f"Either clone jaredpalmer/kev (the partition is in git at {spec['repo_path']}/{spec['partition']}) or "
            f"`hf auth login` with access. Do not substitute a different file.") from error
    if Path(got).resolve() != target.resolve():
        target.write_bytes(Path(got).read_bytes())
    return target


def resolve_partition(spec, allow_download=True, dry_run=False):
    local = repo_path(spec["repo_path"], spec["partition"])
    cached = cache_dir(spec) / _name(spec["partition"], "partition")
    for candidate, origin in ((cached, "cache"), (local, "repo")):
        if candidate.is_file():
            return candidate, origin
    if dry_run or not allow_download:
        raise SuiteError(f"missing {spec['partition']}: looked in {cached} and {local}"
                         + ("" if dry_run else " (downloads disabled)"))
    return fetch_from_hub(spec), "hub"


def numbered_options(builder):
    """255-option ceiling item: keys <prefix>_001.. and a one-line description each."""
    n, prefix = int(builder["n"]), builder["prefix"]
    width = max(3, len(str(n)))
    keys = [f"{prefix}_{i:0{width}d}" for i in range(1, n + 1)]
    criteria = {k: (f"Intent class {i}: a distinct request type in this taxonomy" if builder.get("describe") else None)
                for i, k in enumerate(keys, start=1)}
    return criteria, builder["label"]


FILLER = (
    "The operations team reviewed the weekly queue and confirmed that the backlog was cleared before the shift change. ",
    "Support agents log every inbound message in the ticketing system and tag it with the product area it belongs to. ",
    "The service level target for first response is one business hour; escalations are reviewed twice per day. ",
    "Historical records from the previous system were migrated in March and reconciled against the finance export. ",
    "Agents are asked to restate the customer's request in one sentence before choosing a category for it. ",
    "Quality review samples ten percent of resolved tickets per week and scores tone, accuracy and completeness. ",
    "Refund requests over the threshold require a second approval from the team lead on duty. ",
    "The knowledge base is refreshed monthly; stale articles are flagged automatically after ninety days. ",
)


def needle_state(builder):
    """~target_chars of neutral filler with one deciding sentence inserted at a fractional position."""
    target, at = int(builder["target_chars"]), float(builder.get("at", 0.5))
    body, i = [], 0
    while sum(map(len, body)) < target:
        body.append(FILLER[i % len(FILLER)])
        i += 1
    text = "".join(body)[:target]
    cut = int(len(text) * at)
    cut = text.rfind(". ", 0, cut) + 2 if 0 < cut < len(text) else cut
    return f"{text[:cut].rstrip()} {builder['needle']} {text[cut:].lstrip()}"


def materialize_smoke(item):
    """Expand the smoke set's builders (255 options, 6k-token states) so the frozen file stays reviewable by eye."""
    state = item.get("state")
    if state is None and "state_builder" in item:
        state = needle_state(item["state_builder"])
    if isinstance(state, dict) and "kind" in state and "needle" in state:
        state = needle_state(state)
    gold = dict(item["gold"])
    questions = {}
    for qid, q in item["questions"].items():
        q = dict(q)
        if "criteria_builder" in q:
            q["criteria"], gold[qid] = numbered_options(q.pop("criteria_builder"))
        questions[qid] = q
    return {**item, "state": state, "questions": questions, "gold": gold}


def check_gold(item):
    """A normalized item must be scorable: the gold answers index into the criteria the harness will render."""
    for qid, q in item["questions"].items():
        gold = item["gold"][qid]
        if q["type"] == "choice":
            if gold not in q["criteria"]:
                raise SuiteError(f"{item['id']}/{qid}: gold {gold!r} is not one of the options")
        elif q["type"] == "noul":
            if not isinstance(gold, bool):
                raise SuiteError(f"{item['id']}/{qid}: noul gold must be a bool, got {gold!r}")
        elif q["type"] == "score":
            if not isinstance(gold, int) or not 0 <= gold < len(q["criteria"]):
                raise SuiteError(f"{item['id']}/{qid}: score gold {gold!r} is not a level of {len(q['criteria'])}")
        else:
            raise SuiteError(f"{item['id']}/{qid}: unknown question type {q['type']!r}")


def normalize(record, spec, dataset, area, metric):
    meta = record.get("_meta") or {}
    if meta.get("variant", "clean") != "clean":       # permuted / none_present / none_absent are analysis variants
        return None
    questions, gold = {}, {}
    for qid, q in record["questions"].items():
        questions[qid] = {"type": q["type"], "instructions": q.get("instructions"), "criteria": q.get("criteria")}
        gold[qid] = q["label"] if q["type"] != "score" else int(q["label"])
    item = {"id": meta.get("id") or record["id"], "state": record["state"], "questions": questions, "gold": gold,
            "meta": {"suite": spec["suite"], "split": spec["split"], "source": meta.get("source") or dataset,
                     "area": area, "metric": metric, "variant": meta.get("variant", "clean"),
                     "partition_sha256": (spec.get("provenance") or {}).get("sha256")}}
    check_gold(item)
    return item


def select(records, cap, seed, dataset):
    """Deterministic subset: sha256("<seed>:<source>:<id>") order, first `cap`."""
    def key(record):
        rid = (record.get("_meta") or {}).get("id") or str(record.get("id"))
        return hashlib.sha256(f"{seed}:{dataset}:{rid}".encode()).hexdigest()
    return sorted(records, key=key)[:cap]


def estimate_tokens(state, chars_per_token):
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return int(len(text) / chars_per_token) + 1


def build(spec, manifest, cap_scale=1.0, only=None, allow_test=False, dry_run=False, allow_download=True):
    if spec.get("locked") and not allow_test:
        raise SuiteError(f"{spec['suite']} is locked (read once per candidate, never for search); pass --allow-test")
    budget = manifest.get("budget", {})
    cpt = float(budget.get("chars_per_token_estimate", 4.0))
    max_tokens = int(budget.get("max_total_state_tokens", 10 ** 9) * cap_scale)

    if spec["kind"] == "local-jsonl":
        path = repo_path(spec["repo_path"])
        if not path.is_file():
            raise SuiteError(f"missing local item file {path}")
        raw = read_jsonl(path)
        provenance = verify(path, spec)
        items = []
        for entry in raw:
            if only and entry["id"] not in only:
                continue
            item = materialize_smoke(entry)
            item = {"id": item["id"], "state": item["state"], "questions": item["questions"], "gold": item["gold"],
                    "meta": {"suite": spec.get("suite", "smoke"), "split": spec["split"], "source": "smoke", "area": "plumbing",
                             "metric": "accuracy", "variant": "clean", "partition_sha256": provenance,
                             "checks": item.get("checks", []), "pair": item.get("pair"),
                             "note": item.get("note"), "expect": item.get("expect")}}
            check_gold(item)
            items.append(item)
        stats = {"suite": spec.get("suite", "smoke"), "split": spec["split"], "sha256": provenance, "partition": str(path),
                 "origin": "repo", "records_available": len(raw), "records": len(items),
                 "questions": sum(len(i["questions"]) for i in items),
                 "est_state_tokens": sum(estimate_tokens(i["state"], cpt) for i in items),
                 "budget_tokens": max_tokens, "dropped": []}
        return items, stats

    partition, origin = resolve_partition(spec, allow_download=allow_download, dry_run=dry_run)
    provenance = verify(partition, spec)
    records = read_jsonl(partition)
    by_source = {}
    for record in records:
        by_source.setdefault((record.get("_meta") or {}).get("source"), []).append(record)

    items, used, dropped = [], 0, []
    for entry in spec["datasets"]:
        name = entry["name"]
        if only and name not in only:
            continue
        pool = by_source.get(name) or []
        if not pool:
            raise SuiteError(f"{spec['suite']}/{spec['split']}: no records for dataset {name!r}")
        cap = max(1, int(round(int(entry["cap_records"]) * cap_scale)))
        for record in select(pool, cap, manifest["seed"], name):
            item = normalize(record, spec, name, entry["area"], entry["metric"])
            if item is None:
                continue
            cost = estimate_tokens(item["state"], cpt)
            if used + cost > max_tokens:          # budget stop: never silently truncate a dataset's item count
                dropped.append({"source": name, "id": item["id"], "reason": "state-token budget"})
                continue
            used += cost
            items.append(item)
    if not items:
        raise SuiteError(f"{spec['suite']}/{spec['split']}: selection produced no items")
    stats = {"suite": spec["suite"], "split": spec["split"], "sha256": provenance, "partition": str(partition),
             "origin": origin, "records_available": len(records), "records": len(items),
             "questions": sum(len(i["questions"]) for i in items), "est_state_tokens": used,
             "budget_tokens": max_tokens, "dropped": dropped,
             "by_source": {entry["name"]: sum(1 for i in items if i["meta"]["source"] == entry["name"])
                           for entry in spec["datasets"]}}
    return items, stats


def report(stats, assumed_tps, out=None):
    """The budget arithmetic, on stderr, so a cap is never accepted without a visible time estimate."""
    seconds = stats["est_state_tokens"] / max(assumed_tps, 1e-9)
    print(f"[load_suite] suite={stats['suite']} split={stats['split']} sha256={str(stats['sha256'])[:12]}... "
          f"records={stats['records']}/{stats['records_available']} questions={stats['questions']} "
          f"est_state_tokens={stats['est_state_tokens']}", file=sys.stderr)
    print(f"[load_suite] prefill estimate at {assumed_tps} tok/s: {seconds / 60:.1f} min "
          f"(question passes and weight load are extra)", file=sys.stderr)
    if stats["dropped"]:
        print(f"[load_suite] WARNING {len(stats['dropped'])} record(s) dropped by the token budget, e.g. "
              f"{stats['dropped'][0]['source']}/{stats['dropped'][0]['id']}: lower cap_records or raise the budget",
              file=sys.stderr)
    if out:
        print(f"[load_suite] wrote {stats['records']} items to {out}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--suite", default=None, help="suite key in the manifest (default: the manifest's `primary`)")
    ap.add_argument("--datasets", default="", help="comma-separated dataset names to keep (default: all)")
    ap.add_argument("--out", default="-", help="JSONL output path, - for stdout")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="multiply every per-dataset record cap (budget knob)")
    ap.add_argument("--allow-test", action="store_true", help="read a locked split (one read per candidate)")
    ap.add_argument("--dry-run", action="store_true", help="resolve and report the plan; never download or write")
    ap.add_argument("--no-download", action="store_true", help="local copies only")
    ap.add_argument("--stats", action="store_true", help="print the plan as JSON on stdout and write no items")
    a = ap.parse_args(argv)

    try:
        manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read manifest {a.manifest}: {exc}") from exc
    key = a.suite or manifest["primary"]
    if key not in manifest["suites"]:
        raise SuiteError(f"no suite {key!r} in {a.manifest}; known: {', '.join(manifest['suites'])}")
    spec = manifest["suites"][key]
    only = {x.strip() for x in a.datasets.split(",") if x.strip()} or None

    # a dry run resolves and verifies what is already on disk, but never reaches for the network
    items, stats = build(spec, manifest, cap_scale=a.cap_scale, only=only, allow_test=a.allow_test,
                         dry_run=False, allow_download=not (a.no_download or a.dry_run))
    report(stats, manifest.get("budget", {}).get("assumed_prefill_tokens_per_second", 90), None if a.stats else a.out)
    if a.stats or a.dry_run:
        json.dump(stats, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if a.out == "-":
        for item in items:
            sys.stdout.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SuiteError as error:
        print(f"load_suite: {error}", file=sys.stderr)
        sys.exit(2)
