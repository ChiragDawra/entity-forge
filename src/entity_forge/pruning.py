"""Stage 0: learned re-ranking of the channel union, keeping the top-N per S1.

The union of several top-K channels is large (each channel contributes up to K
pairs per S1). Scoring all of it with the expensive string features wastes
compute on obvious non-matches, so a small LightGBM on the *free* retrieval
features (per-channel cosine and rank, query and target context) re-ranks the
union and the best ``N`` per S1 survive. ``N`` is chosen from measured recall.
"""

from __future__ import annotations

import polars as pl

from . import model

STAGE0_PARAMS: dict = {"num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 500}


def _channels(frame: pl.DataFrame) -> list[str]:
    return [c.removeprefix("cos_") for c in frame.columns if c.startswith("cos_")]


def stage0_features(cands: pl.DataFrame) -> pl.DataFrame:
    """Cheap context features from retrieval scores only (whole-country frame)."""
    chans = _channels(cands)
    exprs = [
        pl.len().over("q").cast(pl.Int16).alias("s0_q_n"),
        pl.len().over("t").cast(pl.Int16).alias("s0_t_n"),
        pl.sum_horizontal([(pl.col(f"rank_{c}") >= 0).cast(pl.Int8) for c in chans]).alias("s0_n_channels"),
    ]
    for c in chans:
        exprs += [
            (pl.col(f"cos_{c}") - pl.col(f"cos_{c}").max().over("q")).alias(f"s0_qgap_{c}"),
            (pl.col(f"cos_{c}") - pl.col(f"cos_{c}").max().over("t")).alias(f"s0_tgap_{c}"),
        ]
    return cands.with_columns(exprs)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(("cos_", "rank_", "s0_"))]


def train_stage0(labeled: pl.DataFrame, n_threads: int, rounds: int = 400):
    """Fit the pruner on labeled candidates that already carry ``stage0_features``."""
    columns = feature_columns(labeled)
    es = labeled.select(model.sample_expr(pl.col("s1_id"), 0.1, salt=7)).to_series()
    booster = model.train(
        labeled.filter(~es),
        columns,
        {**STAGE0_PARAMS, "num_threads": n_threads},
        rounds,
        valid=labeled.filter(es),
        early_stopping=30,
    )
    return booster, columns


def prune(cands: pl.DataFrame, booster, columns: list[str], max_per_q: int) -> pl.DataFrame:
    """Keep the ``max_per_q`` best pairs per S1 by stage-0 score (column ``score0``)."""
    scored = cands.with_columns(pl.Series("score0", model.predict(booster, cands, columns)))
    keep = pl.col("score0").rank("ordinal", descending=True).over("q") <= max_per_q
    return scored.filter(keep).drop([c for c in scored.columns if c.startswith("s0_")])
