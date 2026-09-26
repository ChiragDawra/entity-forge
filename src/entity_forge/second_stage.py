"""Stage-2 features built from out-of-fold stage-1 probabilities (p1).

Stage 1 scores each pair in isolation. Stage 2 lets the model see the *set*:

- **Query context**: how p1 compares with the S1's other candidates
  (max, second, sum, counts above cut-offs, rank, gap).
- **Target competition**: the best p1 any *other* S1 gives this S2/S3 record.
  Training ground truth never assigns one record to two S1 entities, so a
  record that another S1 claims more strongly is almost never a match.
- **Cluster consistency**: S2/S3 are not deduplicated, and an S1's true matches
  are noisy copies of each other. Similarity of the candidate to the S1's most
  confident candidates (top-1 / top-2 by p1) recovers matches whose name
  drifted from S1 but not from its sibling copies.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def query_context(scores: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Per-S1 aggregates of p1. ``scores`` has q, t, p1 for a whole country."""
    ranked_t = pl.col("t").sort_by("p1", descending=True)
    return (
        scores.lazy()
        .group_by("q")
        .agg(
            pl.col("p1").max().alias("q_p1_max"),
            pl.col("p1").top_k(2).min().alias("_p1_second"),
            pl.len().alias("_n"),
            pl.col("p1").sum().alias("q_p1_sum"),
            (pl.col("p1") >= 0.5).sum().cast(pl.Int16).alias("q_p1_n50"),
            (pl.col("p1") >= 0.2).sum().cast(pl.Int16).alias("q_p1_n20"),
            ranked_t.first().alias("q_top1_t"),
            ranked_t.get(1, null_on_oob=True).alias("q_top2_t"),
        )
        .with_columns(
            pl.when(pl.col("_n") > 1).then(pl.col("_p1_second")).otherwise(0.0).alias("q_p1_second")
        )
        .drop("_p1_second", "_n")
        .collect()
    )


def target_context(scores: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Per-target aggregates of p1 across all S1 entities that retrieved it."""
    return (
        scores.lazy()
        .group_by("t")
        .agg(
            pl.col("p1").max().alias("t_p1_max"),
            pl.col("p1").top_k(2).min().alias("_second"),
            pl.len().alias("_n"),
            pl.col("q").sort_by("p1", descending=True).first().alias("t_best_q"),
            (pl.col("p1") >= 0.2).sum().cast(pl.Int16).alias("t_p1_n20"),
        )
        .with_columns(pl.when(pl.col("_n") > 1).then(pl.col("_second")).otherwise(0.0).alias("t_p1_second"))
        .drop("_second", "_n")
        .collect()
    )


def _sibling_similarity(values: pl.Series, t_idx: np.ndarray, ref: pl.Series, workers: int = -1) -> np.ndarray:
    """token_set_ratio between each candidate and a reference target (NaN if none)."""
    has_ref = ref.is_not_null().to_numpy()
    ref_idx = ref.fill_null(0).to_numpy()
    out = process.cpdist(
        values.gather(t_idx).to_list(),
        values.gather(ref_idx).to_list(),
        scorer=fuzz.token_set_ratio,
        workers=workers,
        dtype=np.float32,
    ) / np.float32(100)
    out[~has_ref] = np.nan
    return out


def stage2_features(
    part: pl.DataFrame, q_ctx: pl.DataFrame, t_ctx: pl.DataFrame, targets: pl.DataFrame, workers: int = -1
) -> pl.DataFrame:
    """Add stage-2 columns to ``part`` (must contain every row of its q's, plus q, t, p1).

    ``targets`` holds the country's normalized S2/S3 records; row position = t.
    """
    df = part.join(q_ctx, on="q", how="left", maintain_order="left").join(
        t_ctx, on="t", how="left", maintain_order="left")
    df = df.with_columns(
        (pl.col("p1") - pl.col("q_p1_max")).alias("q_p1_gap"),
        (pl.col("p1") / pl.col("q_p1_sum").clip(lower_bound=1e-6)).alias("q_p1_share"),
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Int16).alias("q_p1_rank"),
        (pl.col("t_best_q") == pl.col("q")).cast(pl.Int8).alias("t_is_best_q"),
        pl.when(pl.col("t_best_q") == pl.col("q"))
        .then(pl.col("t_p1_second"))
        .otherwise(pl.col("t_p1_max"))
        .alias("t_p1_best_other"),
        (pl.col("t") == pl.col("q_top1_t")).cast(pl.Int8).alias("is_top1"),
    ).with_columns((pl.col("p1") - pl.col("t_p1_best_other")).alias("t_p1_margin"))

    t_idx = df["t"].to_numpy()
    sib = {}
    for tag in ("top1", "top2"):
        ref = df[f"q_{tag}_t"]
        sib[f"sib_{tag}_name"] = _sibling_similarity(targets["name_core"], t_idx, ref, workers)
        sib[f"sib_{tag}_addr"] = _sibling_similarity(targets["addr_norm"], t_idx, ref, workers)
    df = df.with_columns([pl.Series(k, v) for k, v in sib.items()])
    return df.drop("q_top1_t", "q_top2_t", "t_best_q")
