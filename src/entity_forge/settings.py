"""Run settings shared by every notebook and the pipeline runner.

Precedence: dataclass defaults < resource profile (``EF_PROFILE``, auto-detected
from RAM) < ``EF_<FIELD>`` environment variables < explicit ``overrides``.

Memory-heavy knobs default to 0, meaning "take it from the profile" (see
``resources.PROFILES``: 16gb / 32gb / 64gb). Everything that changes *results*
(top-K per channel, candidate cap, folds, seeds) is independent of the profile,
so the same candidates are produced on any machine; only speed and peak RAM change.

``dev_mode`` samples a small, consistent slice of the data (a fraction of S1
plus a fraction of distractor targets) into *separate* ``*_dev`` directories,
so an end-to-end smoke test fits on a laptop without touching real artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

from .resources import cpu_count, resolve_profile


def _find_root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "entity_forge").is_dir():
            return p
    return here


@dataclass(frozen=True)
class Settings:
    root: Path = _find_root()
    data_dir: Path = Path("dataset")
    work_dir: Path = Path("artifacts")
    output_dir: Path = Path("output")
    team_name: str = "entity_forge"

    seed: int = 42
    n_folds: int = 5

    # Candidate generation: top-K per channel (see stages.make_channels, EXP-002..006).
    top_k_addr: int = 40
    top_k_both: int = 40
    top_k_cross: int = 20
    top_k_name: int = 10
    top_k_noaddr: int = 10
    top_k_translit: int = 20
    # Stage-0 pruning: keep this many union candidates per S1 (measured in notebook 02).
    max_candidates: int = 60
    stage0_train_frac: float = 0.15  # upper bound; stage0_max_rows also caps it

    # Resources (0 = from the profile; see resources.py).
    profile: str = "auto"
    n_threads: int = 0
    retrieval_chunk: int = 0
    candidate_chunk_rows: int = 0
    feature_chunk: int = 0
    stage0_max_rows: int = 0
    stage1_max_rows: int = 0
    stage2_max_rows: int = 0
    min_free_gb: float = 0.0
    min_free_disk_gb: float = 5.0

    # Smoke-test mode.
    dev_mode: bool = False
    dev_s1_frac: float = 0.02
    dev_target_frac: float = 0.10

    def __post_init__(self) -> None:
        for name in ("data_dir", "work_dir", "output_dir"):
            p = Path(getattr(self, name))
            object.__setattr__(self, name, p if p.is_absolute() else self.root / p)
        prof = resolve_profile(self.profile)
        object.__setattr__(self, "profile", prof.name)
        if not self.n_threads:
            object.__setattr__(self, "n_threads", cpu_count())
        for name in ("retrieval_chunk", "candidate_chunk_rows", "feature_chunk", "stage0_max_rows",
                     "stage1_max_rows", "stage2_max_rows", "min_free_gb"):
            if not getattr(self, name):
                object.__setattr__(self, name, getattr(prof, name))

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from defaults < profile < ``EF_*`` environment variables < ``overrides``."""
        types = {f.name: f.type for f in fields(cls)}
        values: dict = {}
        for name, typ in types.items():
            raw = os.environ.get(f"EF_{name.upper()}")
            if raw is None or raw == "":
                continue
            if typ == "bool":
                values[name] = raw.lower() in ("1", "true", "yes")
            elif typ == "int":
                values[name] = int(raw)
            elif typ == "float":
                values[name] = float(raw)
            elif typ == "Path":
                values[name] = Path(raw)
            else:
                values[name] = raw
        values.update(overrides)
        s = cls(**values)
        if s.dev_mode and not s.work_dir.name.endswith("_dev"):
            s = replace(
                s,
                work_dir=s.work_dir.with_name(s.work_dir.name + "_dev"),
                output_dir=s.output_dir.with_name(s.output_dir.name + "_dev"),
            )
        return s

    def describe(self) -> str:
        return (
            f"profile={self.profile} threads={self.n_threads} retrieval_chunk={self.retrieval_chunk} "
            f"candidate_chunk_rows={self.candidate_chunk_rows} feature_chunk={self.feature_chunk} "
            f"stage0/1/2_max_rows={self.stage0_max_rows}/{self.stage1_max_rows}/{self.stage2_max_rows} "
            f"min_free_gb={self.min_free_gb} dev_mode={self.dev_mode} work_dir={self.work_dir}"
        )

    # Layout ---------------------------------------------------------------
    # File paths are single Parquet files; directory paths hold part-*.parquet
    # files plus a _MANIFEST.json written when the partition is complete.

    def raw(self, split: str, source: int) -> Path:
        return self.data_dir / split / f"{split}_source{source}.tsv"

    @property
    def ground_truth(self) -> Path:
        return self.data_dir / "train" / "train_ground_truth.tsv"

    def norm(self, split: str, source: int) -> Path:
        return self.work_dir / "norm" / f"{split}_s{source}.parquet"

    def candidates_raw(self, split: str, ckey: str) -> Path:
        return self.work_dir / "candidates_raw" / split / f"{safe_name(ckey)}.parquet"

    def candidates(self, split: str, ckey: str) -> Path:
        """Pruned candidates (dir of parts)."""
        return self.work_dir / "candidates" / split / safe_name(ckey)

    def features(self, split: str, ckey: str) -> Path:
        """Stage-1 features (dir of parts, 1:1 with pruned candidate parts)."""
        return self.work_dir / "features" / split / safe_name(ckey)

    def stage2_features(self, ckey: str) -> Path:
        """Stage-2 features for train (dir of parts)."""
        return self.work_dir / "stage2_features" / "train" / safe_name(ckey)

    def scores(self, split: str, stage: int, ckey: str) -> Path:
        """Per-pair scores (dir of parts): q, t, [label], p1[, p]."""
        return self.work_dir / "scores" / split / f"stage{stage}" / safe_name(ckey)

    @property
    def stage0_model(self) -> Path:
        return self.work_dir / "models" / "stage0.txt"

    def model(self, stage: int, fold: int) -> Path:
        return self.work_dir / "models" / f"stage{stage}_fold{fold}.txt"

    @property
    def decision_file(self) -> Path:
        return self.work_dir / "models" / "decision.json"

    @property
    def reports(self) -> Path:
        return self.work_dir / "reports"

    @property
    def tmp(self) -> Path:
        """Disposable scratch space (safe to delete when no run is active)."""
        return self.work_dir / "tmp"

    @property
    def logs(self) -> Path:
        return self.work_dir / "logs"


def safe_name(ckey: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ckey) or "unknown"
