#!/usr/bin/env python3
"""Model store: one short name -> one checkpoint, backed by the Hugging Face cache.

Weights are not kept in the repo. `$HF_HOME/hub` (default `~/.cache/hf/hub`) is the
single copy; this module turns an alias or an `org/repo[:file]` ref into a filesystem
path that the engines can open. Nothing is copied, so `hf cache` stays authoritative
and a checkpoint cannot drift between two directories.

    python3 benchmarks/common/model_store.py list
    python3 benchmarks/common/model_store.py resolve qwen38-27b-oq4e-mtp
    python3 benchmarks/common/model_store.py resolve clef-flash-q4km
    python3 benchmarks/common/model_store.py import models/clef-flash-4bit-mlx mlx-community/clef-flash-4bit

Ref forms accepted by `resolve`:

    qwen38-27b-oq4e-mtp                 alias from ALIASES
    Jundot/Qwen3.8-27B-oQ4e-mtp         a Hub repo id (whole snapshot)
    ggml-org/Clef-Flash-GGUF:Clef-Flash-Q4_K_M.gguf
                                        a Hub repo plus one file
    org/repo@<revision>                 pin a revision
    models/clef-flash-4bit-mlx          an existing local path (still works)

Resolution is cache-only unless `MODEL_STORE_ONLINE=1` (or `resolve --online`) is set:
a benchmark must never download inside a measured window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Short names are the ones used in READMEs, docs and shell scripts. Keep them stable:
# a row's `model` field has to be reproducible years later.
ALIASES: dict[str, tuple[str, str | None]] = {
    # resident text/vision backbone (text, image, video, audio tracks)
    "qwen38-27b-oq4e-mtp": ("Jundot/Qwen3.8-27B-oQ4e-mtp", None),
    # text track, removed from disk but reproducible from the Hub on demand
    "qwen38-27b-gguf-q4km": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
    "qwen38-27b-mtp": ("unsloth/Qwen3.8-27B-GGUF", "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"),
    # decision track
    "clef-flash-4bit": ("mlx-community/clef-flash-4bit", None),
    "clef-flash-8bit": ("mlx-community/clef-flash-8bit", None),
    "clef-flash-q4km": ("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf"),
    "kev-9b-q4km": ("ggml-org/Kev-9B-GGUF", "Kev-9B-Q4_K_M.gguf"),
    "d3-flash-q4km": ("ggml-org/vllm-d3-flash-GGUF", "vllm-d3-flash-Q4_K_M.gguf"),
}

# Values >= this are LFS objects on the Hub, so the blob name is the content sha256.
# Smaller files are stored under their git blob id (sha1 of "blob <len>\0" + bytes).
LFS_THRESHOLD = 5 * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parents[2]


def hub_cache() -> Path:
    """The Hub cache root, honouring HF_HUB_CACHE / HF_HOME like huggingface_hub does."""
    from huggingface_hub.constants import HF_HUB_CACHE  # pyright: ignore[reportMissingImports]

    return Path(HF_HUB_CACHE)


def online_default() -> bool:
    return os.environ.get("MODEL_STORE_ONLINE", "0").lower() in {"1", "true", "yes"}


def split_ref(ref: str) -> tuple[str, str | None, str | None]:
    """`org/repo@rev:file` -> (repo_id, filename, revision)."""
    revision = None
    if "@" in ref:
        ref, revision = ref.rsplit("@", 1)
    filename = None
    if ":" in ref:
        ref, filename = ref.split(":", 1)
    return ref, filename, revision


def resolve(ref: str, *, online: bool | None = None, quiet: bool = False) -> str:
    """Return a real path for an alias, a Hub ref, or an existing local path."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    if online is None:
        online = online_default()

    # 1. an existing path always wins: it is explicit and needs no network
    candidate = Path(ref).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    # 2. alias
    if ref in ALIASES:
        repo_id, filename = ALIASES[ref]
        revision = None
    else:
        repo_id, filename, revision = split_ref(ref)
        if repo_id.count("/") != 1 or not repo_id.split("/")[1]:
            raise SystemExit(
                f"unknown model ref {ref!r}: not a path, not an alias, not org/repo[:file].\n"
                f"known aliases: {', '.join(sorted(ALIASES))}"
            )

    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[filename] if filename else None,
            local_files_only=not online,
        )
    except Exception as exc:  # noqa: BLE001 - surface the cache miss, never a traceback
        raise SystemExit(
            f"cannot resolve {ref!r} -> {repo_id}{':' + filename if filename else ''} "
            f"from {hub_cache()}: {type(exc).__name__}: {str(exc)[:300]}\n"
            f"hint: import local weights with "
            f"`{sys.executable} benchmarks/common/model_store.py import <dir> {repo_id}`"
        ) from exc

    resolved = Path(path) / filename if filename else Path(path)
    if not resolved.exists():
        raise SystemExit(f"{repo_id}: snapshot resolved to {path} but {filename} is absent")
    if not quiet:
        print(f"[model] {ref} -> {resolved}", file=sys.stderr)
    return str(resolved)


# --------------------------------------------------------------------------- import


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_tree(src: Path) -> tuple[str | None, dict[str, dict]]:
    """Download metadata written by `hf download --local-dir`, if present."""
    trees = sorted((src / ".cache" / "huggingface" / "trees").glob("*.json"))
    if not trees:
        return None, {}
    try:
        data = json.loads(trees[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] ignoring unreadable {trees[0]}: {exc}", file=sys.stderr)
        return None, {}
    return trees[0].stem, data.get("files", {})


def _walk(src: Path) -> list[Path]:
    skip_dirs = {".cache", "__pycache__", ".eval_results", ".git"}
    out: list[Path] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(src)
        if rel.parts[0] in skip_dirs or rel.name == ".DS_Store":
            continue
        out.append(path)
    return out


def import_dir(
    src: Path,
    repo_id: str,
    *,
    revision: str | None = None,
    online: bool = True,
    remove: bool = False,
) -> Path:
    """Adopt a plain `--local-dir` checkpoint into the Hub snapshot cache.

    Bytes are hard-linked, not copied or downloaded: `models/x` and
    `$HF_HUB_CACHE/models--org--repo` then share one inode, so the move costs no disk
    and no network. `remove=True` drops the source directory once every file is linked.
    """
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")

    from huggingface_hub import HfApi  # pyright: ignore[reportMissingImports]

    local_rev, tree = _local_tree(src)
    info = None
    if online:
        try:
            info = HfApi().model_info(repo_id, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - offline import is still useful
            print(f"[warn] {repo_id}: metadata unavailable ({str(exc)[:120]}); using size/hash", file=sys.stderr)

    rev = revision or (info.sha if info else None) or local_rev
    if not rev:
        raise SystemExit(f"{src}: cannot determine a revision for {repo_id}; pass --revision")

    remote = {s.rfilename: s for s in (info.siblings or [])} if info else {}
    cache = hub_cache()
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    blobs = repo_dir / "blobs"
    snapshot = repo_dir / "snapshots" / rev
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)

    linked: list[Path] = []
    for path in _walk(src):
        rel = path.relative_to(src).as_posix()
        if remote and rel not in remote:
            print(f"[skip] {rel}: not a file in {repo_id}", file=sys.stderr)
            continue

        size = path.stat().st_size
        remote_size = getattr(remote.get(rel), "size", None)
        if remote_size is not None and remote_size != size:
            raise SystemExit(f"{rel}: local {size} B != Hub {remote_size} B; refusing to adopt")

        meta = tree.get(rel, {})
        sha = meta.get("lfs_sha256")
        if size >= LFS_THRESHOLD:
            sha = sha or _sha256(path)
            remote_sha = None
            lfs = getattr(remote.get(rel), "lfs", None)
            if lfs is not None:
                remote_sha = lfs.sha256
            if remote_sha and remote_sha != sha:
                raise SystemExit(f"{rel}: sha256 {sha[:12]} != Hub {remote_sha[:12]}; refusing to adopt")
            blob_name = sha
        else:
            # Small files are content-addressed too. `hf download --local-dir` records the
            # Hub blob id; without it a sha256 is just as unique for a local-only cache.
            blob_name = meta.get("blob_id") or _sha256(path)

        blob = blobs / blob_name
        if not blob.exists():
            try:
                os.link(path, blob)
            except OSError:
                shutil.copy2(path, blob)
            linked.append(blob)

        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        # The blob lives at <repo>/blobs; a snapshot file sits <depth> path parts below
        # <repo>/snapshots/<rev>, so it needs depth+1 ".." hops.
        up = Path(*([".."] * (len(Path(rel).parts) + 1)))
        target.symlink_to(up / "blobs" / blob_name)

    ref_file = repo_dir / "refs" / "main"
    if not ref_file.exists():
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(rev)

    print(f"[import] {src} -> {repo_id} @ {rev[:12]} ({len(linked)} new blobs, {len(_walk(src))} files)")
    if remove:
        for target_dir in (src / ".cache", src):
            try:
                shutil.rmtree(target_dir)
            except OSError as exc:
                print(f"[warn] cannot remove {target_dir}: {exc}", file=sys.stderr)
        print(f"[import] removed {src}")
    return snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="aliases and whether the cache has them")
    p_list.add_argument("--online", action="store_true")

    p_res = sub.add_parser("resolve", help="print a filesystem path for a ref")
    p_res.add_argument("ref")
    p_res.add_argument("--online", action="store_true", default=None, help="allow a Hub download")
    p_res.add_argument("--quiet", action="store_true")

    p_imp = sub.add_parser("import", help="adopt a --local-dir checkpoint into the Hub cache")
    p_imp.add_argument("dir")
    p_imp.add_argument("repo_id")
    p_imp.add_argument("--revision", default=None)
    p_imp.add_argument("--offline", action="store_true", help="do not query the Hub for metadata")
    p_imp.add_argument("--remove", action="store_true", help="delete the source dir afterwards")

    args = ap.parse_args()

    if args.cmd == "list":
        online = args.online or online_default()
        for alias, (repo_id, filename) in sorted(ALIASES.items()):
            try:
                path = resolve(alias, online=online, quiet=True)
                state = "cached"
            except SystemExit:
                path, state = "-", "MISSING"
            target = f"{repo_id}:{filename}" if filename else repo_id
            print(f"{alias:24} {state:8} {target:44} {path}")
        return 0

    if args.cmd == "resolve":
        print(resolve(args.ref, online=args.online, quiet=args.quiet))
        return 0

    snapshot = import_dir(
        Path(args.dir),
        args.repo_id,
        revision=args.revision,
        online=not args.offline,
        remove=args.remove,
    )
    print(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
