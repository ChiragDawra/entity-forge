"""Turn pair probabilities into one match set per S1 (possibly empty).

Steps, each validated against the exact metric on out-of-fold predictions:

1. ``resolve_exclusive`` — each S2/S3 record keeps only its best S1. Justified
   by the training ground truth (0 of 7.6M matched records belong to more than
   one S1).
2. ``select_threshold`` — keep pairs with p >= tau.
3. ``select_expected_f`` — per S1, choose the top-k (k may be 0) that maximizes
   the expected F0.5 under the model probabilities.
"""

from __future__ import annotations

import polars as pl

from .metrics import BETA, macro_fbeta


def resolve_exclusive(pred: pl.DataFrame, score: str = "p") -> pl.DataFrame:
    """Keep, for every target ``t_id``, only the S1 with the highest score (ties: smaller s1_id)."""
    return pred.sort([score, "s1_id"], descending=[True, False]).unique(
        subset=["t_id"], keep="first", maintain_order=True
    )


def select_threshold(pred: pl.DataFrame, tau: float, score: str = "p") -> pl.DataFrame:
    return pred.filter(pl.col(score) >= tau)


def select_expected_f(
    pred: pl.DataFrame, missed: float = 0.0, floor: float = 0.02, score: str = "p", beta: float = BETA
) -> pl.DataFrame:
    """Per-S1 expected-F subset selection.

    For probabilities p1 >= p2 >= ... of one S1 and ``missed`` = expected number
    of true matches that blocking lost, the expected F-beta of predicting the
    top-k is approximated by

        (1 + b^2) * sum_{i<=k} p_i / (b^2 * (sum_i p_i + missed) + k)

    and predicting nothing scores P(no true match) ~ prod(1 - p_i), discounted
    for matches blocking may have lost. Candidates below ``floor`` are ignored.
    """
    b2 = beta * beta
    df = (
        pred.filter(pl.col(score) >= floor)
        .sort(["s1_id", score], descending=[False, True])
        .with_columns(
            pl.col(score).cum_sum().over("s1_id").alias("_cum"),
            pl.col(score).sum().over("s1_id").alias("_tot"),
            pl.int_range(1, pl.len() + 1).over("s1_id").alias("_k"),
            (1.0 - pl.col(score)).clip(1e-9, 1.0).log().sum().over("s1_id").exp().alias("_p_empty"),
        )
        .with_columns(
            ((1 + b2) * pl.col("_cum") / (b2 * (pl.col("_tot") + missed) + pl.col("_k"))).alias("_ef")
        )
        .with_columns(
            pl.col("_ef").max().over("s1_id").alias("_best"),
            (pl.col("_p_empty") / (1.0 + missed)).alias("_ef0"),
        )
    )
    best_k = (
        df.filter(pl.col("_ef") == pl.col("_best"))
        .group_by("s1_id")
        .agg(pl.col("_k").min().alias("_kstar"))
    )
    keep = df.join(best_k, on="s1_id").filter(
        (pl.col("_k") <= pl.col("_kstar")) & (pl.col("_best") > pl.col("_ef0"))
    )
    return keep.drop([c for c in keep.columns if c.startswith("_")])


def tune_threshold(
    pred: pl.DataFrame,
    truth: pl.DataFrame,
    s1_ids: pl.Series,
    grid: list[float] | None = None,
    score: str = "p",
) -> tuple[float, pl.DataFrame]:
    """Sweep tau on out-of-fold predictions with the exact macro metric."""
    grid = grid or [round(0.05 * i, 2) for i in range(2, 19)]
    rows = [
        {"tau": tau, "macro_f05": macro_fbeta(s1_ids, select_threshold(pred, tau, score), truth)}
        for tau in grid
    ]
    table = pl.DataFrame(rows)
    best = table.sort("macro_f05", descending=True).row(0, named=True)
    return float(best["tau"]), table
