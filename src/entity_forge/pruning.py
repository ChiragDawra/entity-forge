"""Stage 0: learned re-ranking of the channel union, keeping the top-N per S1.

The union of several top-K channels is large (each channel contributes up to K
pairs per S1). Scoring all of it with the expensive string features wastes
compute on obvious non-matches, so a small LightGBM on the *free* retrieval
features (per-channel cosine and rank, query and target context) re-ranks the
union and the best ``N`` per S1 survive. ``N`` is chosen from measured recall.

Streaming: query-context features only look inside one S1's rows, so any chunk
of whole S1s computes them exactly. Target-context features look across all
S1s, so they come from ``target_aggregates`` computed once per country with a
streaming group-by. ``stage0_features(chunk, t_agg)`` therefore equals the
whole-partition computation for every chunk.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import model

STAGE0_PARAMS: dict = {"num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 500}


def _channels(columns: list[str]) -> list[str]:
    return [c.removeprefix("cos_") for c in columns if c.startswith("cos_")]


def target_aggregates(source: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Per target: number of union pairs and max cosine per channel (whole country)."""
    lf = source.lazy()
    chans = _channels(lf.collect_schema().names())
    agg = lf.group_by("t").agg(
        pl.len().cast(pl.Int16).alias("s0_t_n"),
        *[pl.col(f"cos_{c}").max().alias(f"_tmax_{c}") for c in chans],
    )
    return agg.collect(engine="streaming")


def target_aggregates_from_files(paths: list[Path] | Path) -> pl.DataFrame:
    return target_aggregates(pl.scan_parquet(paths))


def stage0_features(cands: pl.DataFrame, t_agg: pl.DataFrame | None = None) -> pl.DataFrame:
    """Retrieval-only context features for complete S1 groups.

    ``t_agg`` must come from the whole country (``target_aggregates``); when
    omitted, ``cands`` itself is taken as the whole population.
    """
    chans = _channels(cands.columns)
    if t_agg is None:
        t_agg = target_aggregates(cands)
    q_exprs = [
        pl.len().over("q").cast(pl.Int16).alias("s0_q_n"),
        pl.sum_horizontal([(pl.col(f"rank_{c}") >= 0).cast(pl.Int8) for c in chans]).alias("s0_n_channels"),
    ]
    q_exprs += [(pl.col(f"cos_{c}") - pl.col(f"cos_{c}").max().over("q")).alias(f"s0_qgap_{c}") for c in chans]
    out = cands.with_columns(q_exprs).join(t_agg, on="t", how="left", maintain_order="left")
    out = out.with_columns([(pl.col(f"cos_{c}") - pl.col(f"_tmax_{c}")).alias(f"s0_tgap_{c}") for c in chans])
    ordered = ["s0_q_n", "s0_t_n", "s0_n_channels"]
    for c in chans:
        ordered += [f"s0_qgap_{c}", f"s0_tgap_{c}"]
    return out.select(cands.columns + ordered)


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


def prune(cands: pl.DataFrame, booster, columns: list[str], max_per_q: int, n_threads: int = 0) -> pl.DataFrame:
    """Keep the ``max_per_q`` best pairs per S1 by stage-0 score (column ``score0``)."""
    scored = cands.with_columns(pl.Series("score0", model.predict(booster, cands, columns, n_threads)))
    keep = pl.col("score0").rank("ordinal", descending=True).over("q") <= max_per_q
    return scored.filter(keep).drop([c for c in scored.columns if c.startswith("s0_")])
