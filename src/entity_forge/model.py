"""LightGBM pair classifier with S1-grouped folds."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Columns that are identifiers or labels, never model inputs.
NON_FEATURES: frozenset[str] = frozenset({"q", "t", "s1_id", "t_id", "label", "fold", "p", "ckey"})

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": 8,
    "seed": 13,
    "deterministic": True,
    "verbose": -1,
}


_M32 = 1 << 32


def id_hash(col: pl.Expr, salt: int = 0) -> pl.Expr:
    """Deterministic 32-bit hash of an entity id's numeric suffix.

    Integer mixing (xor-shift-multiply) so that different ``salt`` values give
    independent streams: a 15 % sample of a 2 % sample really is 15 % of it.
    Machine and library-version independent (unlike ``pl.Expr.hash``).
    """
    x = (col.str.extract(r"(\d+)$").cast(pl.UInt64) + salt * 1_000_000_007) % _M32
    for _ in range(2):
        x = (x.xor(x // 65536) * 0x45D9F3B) % _M32
    return x.xor(x // 65536)


def fold_expr(col: pl.Expr, n_folds: int) -> pl.Expr:
    """Fold assignment grouped by S1 id: every pair of one S1 lands in one fold."""
    return (id_hash(col, salt=0) % n_folds).cast(pl.Int8)


def sample_expr(col: pl.Expr, frac: float, salt: int = 1) -> pl.Expr:
    """Deterministic Bernoulli(frac) keep-mask by id, independent across salts."""
    return (id_hash(col, salt) % 1_000_000) < int(frac * 1_000_000)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURES]


def to_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def train(
    frame: pl.DataFrame,
    columns: list[str],
    params: dict | None = None,
    rounds: int = 600,
    valid: pl.DataFrame | None = None,
    early_stopping: int = 40,
) -> lgb.Booster:
    params = {**DEFAULT_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(
        to_matrix(frame, columns), label=frame["label"].to_numpy(), feature_name=columns
    )
    callbacks = [lgb.log_evaluation(100)]
    valid_sets = []
    if valid is not None:
        dvalid = lgb.Dataset(to_matrix(valid, columns), label=valid["label"].to_numpy(), reference=dtrain)
        valid_sets = [dvalid]
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    log.info(
        "trained %d rounds on %d rows",
        booster.best_iteration or booster.current_iteration(), frame.height,
    )
    return booster


def predict(booster: lgb.Booster, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    it = booster.best_iteration or None
    return booster.predict(to_matrix(frame, columns), num_iteration=it).astype(np.float32)


def save(booster: lgb.Booster, columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path), num_iteration=booster.best_iteration or None)
    path.with_suffix(".features.json").write_text(json.dumps(columns))


def load(path: Path) -> tuple[lgb.Booster, list[str]]:
    booster = lgb.Booster(model_file=str(path))
    columns = json.loads(path.with_suffix(".features.json").read_text())
    if booster.feature_name() != columns:
        raise ValueError(f"{path}: feature schema mismatch")
    return booster, columns


def importance(booster: lgb.Booster) -> pl.DataFrame:
    return pl.DataFrame(
        {"feature": booster.feature_name(), "gain": booster.feature_importance("gain")}
    ).sort("gain", descending=True)
