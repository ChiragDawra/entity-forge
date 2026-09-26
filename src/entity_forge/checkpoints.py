"""Durable checkpoints: manifests, atomic writes, validation, legacy adoption.

Every expensive artifact gets a manifest written *after* it is complete::

    <file>.parquet               -> <file>.parquet.manifest.json
    <dir>/part-*.parquet         -> <dir>/_MANIFEST.json

A manifest records the stage, partition key, a hash of the *semantic*
configuration that produced it (never thread counts or chunk sizes), row
count, schema, file sizes and a timestamp. An artifact is reused only if its
manifest exists, the configuration hash matches and every recorded file still
has its recorded size. Parts are written atomically (temp file + rename), so a
crash never leaves a half-written file that looks valid.

Artifacts created before manifests existed can be *adopted*: the stage
validates schema/row counts cheaply and, if they pass, writes a manifest
marked ``adopted``. Nothing in this module deletes data files except parts
inside a stage's own partition directory that were produced under a different
(stale) configuration or by an interrupted run without a plan.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import polars as pl

log = logging.getLogger("entity_forge")

MANIFEST = "_MANIFEST.json"
PLAN = "_PLAN.json"


class StaleArtifactError(RuntimeError):
    """An expensive artifact exists but was built with a different configuration."""


def fingerprint(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def is_dir_target(target: Path) -> bool:
    return target.suffix == ""


def manifest_path(target: Path) -> Path:
    return target / MANIFEST if is_dir_target(target) else target.with_name(target.name + ".manifest.json")


def data_files(target: Path) -> list[Path]:
    if is_dir_target(target):
        return sorted(target.glob("part-*.parquet")) if target.is_dir() else []
    return [target] if target.is_file() else []


def parquet_rows(paths: list[Path]) -> int:
    if not paths:
        return 0
    return int(pl.scan_parquet(paths).select(pl.len()).collect().item())


def parquet_schema(path: Path) -> dict[str, str]:
    return {k: str(v) for k, v in pl.read_parquet_schema(path).items()}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` so that ``path`` either does not exist or is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def write_manifest(target: Path, stage: str, key: str, config_hash: str, *, rows: int | None = None,
                   adopted: bool = False, extra: dict | None = None) -> dict:
    files = data_files(target)
    if not files:
        raise FileNotFoundError(f"cannot write manifest: no data files at {target}")
    is_parquet = files[0].suffix == ".parquet"
    manifest = {
        "stage": stage,
        "key": key,
        "config_hash": config_hash,
        "rows": (parquet_rows(files) if is_parquet else 0) if rows is None else rows,
        "schema": parquet_schema(files[0]) if is_parquet else {},
        "files": {p.name: p.stat().st_size for p in files},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "adopted": adopted,
        "extra": extra or {},
    }
    _atomic_text(manifest_path(target), json.dumps(manifest, indent=2))
    return manifest


def read_manifest(target: Path) -> dict | None:
    path = manifest_path(target)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("unreadable manifest %s: treating artifact as incomplete", path)
        return None


def check(target: Path, config_hash: str) -> tuple[bool, str]:
    """(is_valid, reason) for ``target`` under ``config_hash``."""
    m = read_manifest(target)
    if m is None:
        return False, "no manifest"
    if m.get("config_hash") != config_hash:
        return False, f"config changed ({m.get('config_hash')} -> {config_hash})"
    for name, size in m.get("files", {}).items():
        p = (target / name) if is_dir_target(target) else target.with_name(name)
        if not p.is_file():
            return False, f"missing file {p.name}"
        if p.stat().st_size != size:
            return False, f"file {p.name} changed size"
    return True, "ok"


def is_complete(target: Path, config_hash: str) -> bool:
    return check(target, config_hash)[0]


def reset_partition(directory: Path) -> None:
    """Make a part directory recompute from scratch (forced stage)."""
    invalidate(directory)
    plan = directory / PLAN
    if plan.is_file():
        plan.unlink()


def invalidate(target: Path) -> bool:
    """Forget that ``target`` is complete (the data stays; it will be recomputed)."""
    path = manifest_path(target)
    if path.is_file():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Partitions written as parts, resumable part by part
# ---------------------------------------------------------------------------


def part_path(directory: Path, i: int) -> Path:
    return directory / f"part-{i:05d}.parquet"


def load_or_create_plan(directory: Path, config_hash: str, make_plan) -> list:
    """Chunk plan of a partition, persisted so that a resume reuses the same boundaries.

    A stored plan from another configuration means the partition's parts are
    stale: its ``part-*.parquet`` files are removed and a fresh plan is made.
    Parts present without any plan come from an interrupted legacy run and are
    recomputed the same way.
    """
    path = directory / PLAN
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored.get("config_hash") == config_hash:
            return stored["chunks"]
        stale = list(directory.glob("part-*.parquet"))
        log.warning("%s: configuration changed; discarding %d stale parts", directory, len(stale))
        for p in stale:
            p.unlink()
        invalidate(directory)
    elif directory.is_dir() and any(directory.glob("part-*.parquet")):
        log.warning("%s: parts without a plan (interrupted old run); recomputing them", directory)
        for p in directory.glob("part-*.parquet"):
            p.unlink()
    chunks = make_plan()
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps({"config_hash": config_hash, "chunks": chunks}))
    return chunks


def remove_tree(path: Path) -> None:
    """Delete a *disposable* scratch directory."""
    shutil.rmtree(path, ignore_errors=True)


def disk_free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def require_disk(path: Path, need_gb: float, margin_gb: float, what: str) -> None:
    free = disk_free_gb(path)
    if free - need_gb < margin_gb:
        raise OSError(
            f"{what}: needs ~{need_gb:.1f} GB of disk, only {free:.1f} GB free at {path} "
            f"(keeping {margin_gb:.1f} GB spare). Free space (e.g. the disposable "
            "artifacts/tmp) or point EF_WORK_DIR at a bigger volume; completed work is kept."
        )
