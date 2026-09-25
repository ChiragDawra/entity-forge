"""Run settings shared by every notebook.

Defaults target a full-size training machine (e.g. SageMaker ml.r5.4xlarge or
any 64 GB+ box). Every field can be overridden with an ``EF_<FIELD>``
environment variable, so the same notebooks run unchanged on a laptop, a
friend's machine or SageMaker.

``dev_mode`` samples a small, consistent slice of the data (a fraction of S1
plus a fraction of distractor targets) into *separate* ``*_dev`` directories,
so an end-to-end smoke test fits on an 8 GB laptop without touching real
artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path


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

    n_threads: int = os.cpu_count() or 8
    seed: int = 42
    n_folds: int = 5

    # Candidate generation: top-K per channel (see stages.make_channels, EXP-002..005).
    top_k_addr: int = 40
    top_k_both: int = 40
    top_k_cross: int = 20
    top_k_name: int = 10
    top_k_noaddr: int = 10
    top_k_translit: int = 20
    # Stage-0 pruning: keep this many union candidates per S1 (measured in notebook 02).
    max_candidates: int = 60
    stage0_train_frac: float = 0.15

    # Rows per chunk when computing pair features.
    feature_chunk: int = 3_000_000

    # Smoke-test mode.
    dev_mode: bool = False
    dev_s1_frac: float = 0.02
    dev_target_frac: float = 0.10

    def __post_init__(self) -> None:
        for name in ("data_dir", "work_dir", "output_dir"):
            p = Path(getattr(self, name))
            object.__setattr__(self, name, p if p.is_absolute() else self.root / p)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from defaults < ``EF_*`` environment variables < ``overrides``."""
        types = {f.name: f.type for f in fields(cls)}
        values: dict = {}
        for name, typ in types.items():
            raw = os.environ.get(f"EF_{name.upper()}")
            if raw is None:
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
        if s.dev_mode:
            s = replace(
                s,
                work_dir=s.work_dir.with_name(s.work_dir.name + "_dev"),
                output_dir=s.output_dir.with_name(s.output_dir.name + "_dev"),
            )
        return s

    # Layout ---------------------------------------------------------------

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
        return self.work_dir / "candidates" / split / f"{safe_name(ckey)}.parquet"

    def features(self, split: str, ckey: str) -> Path:
        return self.work_dir / "features" / split / f"{safe_name(ckey)}.parquet"

    @property
    def stage0_model(self) -> Path:
        return self.work_dir / "models" / "stage0.txt"

    def model(self, stage: int, fold: int) -> Path:
        return self.work_dir / "models" / f"stage{stage}_fold{fold}.txt"

    def scores(self, split: str, stage: int) -> Path:
        return self.work_dir / "scores" / f"{split}_stage{stage}.parquet"

    @property
    def decision_file(self) -> Path:
        return self.work_dir / "models" / "decision.json"

    @property
    def reports(self) -> Path:
        return self.work_dir / "reports"


def safe_name(ckey: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ckey) or "unknown"
