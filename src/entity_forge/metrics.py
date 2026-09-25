"""Exact competition metric: macro F-beta (beta = 0.5) over all Source 1 entities.

Per S1 entity:
- true set empty and predicted set empty  -> 1.0
- exactly one of the two sets empty       -> 0.0
- otherwise F_beta of set precision/recall

The macro average runs over *every* evaluated S1, singletons included.
"""

from __future__ import annotations

import polars as pl

BETA = 0.5


def fbeta(tp: int, n_pred: int, n_true: int, beta: float = BETA) -> float:
    """F-beta for one entity from set sizes (scalar reference implementation)."""
    if n_true == 0 and n_pred == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p = tp / n_pred
    r = tp / n_true
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def per_entity_scores(
    s1_ids: pl.Series, pred: pl.DataFrame, truth: pl.DataFrame, beta: float = BETA
) -> pl.DataFrame:
    """Vectorized per-S1 F-beta.

    ``pred`` and ``truth`` hold (s1_id, t_id) pairs; duplicates are ignored.
    Returns columns s1_id, n_pred, n_true, tp, f.
    """
    pred = pred.select("s1_id", "t_id").unique()
    truth = truth.select("s1_id", "t_id").unique()
    tp = pred.join(truth, on=["s1_id", "t_id"], how="inner").group_by("s1_id").len("tp")
    n_pred = pred.group_by("s1_id").len("n_pred")
    n_true = truth.group_by("s1_id").len("n_true")
    b2 = beta * beta
    df = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(n_pred, on="s1_id", how="left")
        .join(n_true, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0))
    )
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise((1 + b2) * p * r / (b2 * p + r))
    )
    return df.with_columns(f.alias("f"))


def macro_fbeta(s1_ids: pl.Series, pred: pl.DataFrame, truth: pl.DataFrame, beta: float = BETA) -> float:
    return float(per_entity_scores(s1_ids, pred, truth, beta)["f"].mean())


def summary(s1_ids: pl.Series, pred: pl.DataFrame, truth: pl.DataFrame) -> dict[str, float]:
    """Macro F0.5 plus micro precision/recall and singleton accuracy."""
    df = per_entity_scores(s1_ids, pred, truth)
    tp, n_pred, n_true = df.select(pl.col("tp").sum(), pl.col("n_pred").sum(), pl.col("n_true").sum()).row(0)
    single = df.filter(pl.col("n_true") == 0)
    return {
        "macro_f05": float(df["f"].mean()),
        "micro_precision": tp / max(n_pred, 1),
        "micro_recall": tp / max(n_true, 1),
        "singleton_accuracy": float((single["n_pred"] == 0).mean()) if single.height else float("nan"),
        "n_s1": df.height,
    }
