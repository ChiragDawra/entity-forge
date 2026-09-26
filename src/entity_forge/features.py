"""Pairwise features for (S1, candidate) pairs.

Three groups:

1. **Pair similarity** (``pair_features``): RapidFuzz string similarities on the
   normalized name/address forms, token and house-number set overlap, flags.
   All scorers run through ``rapidfuzz.process.cpdist`` (paired, multithreaded
   C++), never a Python loop per pair.
2. **Query context** (``context_features``): how this candidate compares with
   the other candidates of the same S1 (max, gap, rank).
3. **Target competition** (``context_features``): how many S1 entities
   retrieved this S2/S3 record and whether this S1 is the best of them. The
   training ground truth has *zero* S2/S3 records shared by two S1 entities, so
   "someone else fits better" is strong negative evidence.

Features are language agnostic (ratios, ranks, flags); country is never a
feature because the test set adds an unseen country (France).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# (feature name, column, scorer, scale)
_STRING_SCORERS: tuple[tuple[str, str, object, float], ...] = (
    ("n_ratio", "name_core", fuzz.ratio, 0.01),
    ("n_tsort", "name_core", fuzz.token_sort_ratio, 0.01),
    ("n_tset", "name_core", fuzz.token_set_ratio, 0.01),
    ("n_partial", "name_core", fuzz.partial_ratio, 0.01),
    ("n_jw", "name_core", JaroWinkler.normalized_similarity, 1.0),
    ("n_full_ratio", "name_norm", fuzz.ratio, 0.01),
    ("n_alt_tset", "name_alt", fuzz.token_set_ratio, 0.01),
    ("n_phon_ratio", "name_phon", fuzz.ratio, 0.01),
    ("n_nospace_ratio", "name_nospace", fuzz.ratio, 0.01),
    ("a_ratio", "addr_norm", fuzz.ratio, 0.01),
    ("a_tset", "addr_norm", fuzz.token_set_ratio, 0.01),
    ("a_tsort", "addr_norm", fuzz.token_sort_ratio, 0.01),
)

RECORD_COLUMNS: tuple[str, ...] = (
    "entity_id", "name_norm", "name_core", "name_alt", "name_phon", "addr_norm",
    "addr_nums", "has_addr", "translit", "name_freq",
)


def with_derived(frame: pl.DataFrame) -> pl.DataFrame:
    """Columns the scorers need that are cheap to derive on the fly."""
    return frame.with_columns(
        pl.col("name_core").str.replace_all(" ", "", literal=True).alias("name_nospace")
    )


def _set_overlap(a: pl.Expr, b: pl.Expr, prefix: str) -> list[pl.Expr]:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return [
        inter.cast(pl.Int16).alias(f"{prefix}_common"),
        pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32).alias(f"{prefix}_jacc"),
    ]


def pair_features(
    pairs: pl.DataFrame, queries: pl.DataFrame, targets: pl.DataFrame, workers: int = -1
) -> pl.DataFrame:
    """Similarity features for ``pairs`` (columns ``q``, ``t`` index the frames).

    ``queries``/``targets`` must contain ``RECORD_COLUMNS``; row position is the
    index. Existing columns of ``pairs`` are kept.
    """
    qf = with_derived(queries[pairs["q"].to_numpy()])
    tf = with_derived(targets[pairs["t"].to_numpy()])

    scores = {
        name: process.cpdist(qf[col].to_list(), tf[col].to_list(), scorer=scorer, workers=workers,
                             dtype=np.float32) * np.float32(scale)
        for name, col, scorer, scale in _STRING_SCORERS
    }
    feats = pairs.with_columns([pl.Series(k, v) for k, v in scores.items()])

    lists = pl.DataFrame(
        {
            "qn": qf["name_core"].str.split(" "),
            "tn": tf["name_core"].str.split(" "),
            "qa": qf["addr_norm"].str.split(" "),
            "ta": tf["addr_norm"].str.split(" "),
            "qd": qf["addr_nums"],
            "td": tf["addr_nums"],
        }
    )
    sets = lists.select(
        *_set_overlap(pl.col("qn"), pl.col("tn"), "ntok"),
        *_set_overlap(pl.col("qa"), pl.col("ta"), "atok"),
        *_set_overlap(pl.col("qd"), pl.col("td"), "num"),
        pl.col("qd").list.len().cast(pl.Int16).alias("q_n_nums"),
        pl.col("td").list.len().cast(pl.Int16).alias("t_n_nums"),
        (pl.col("qd").list.first() == pl.col("td").list.first()).cast(pl.Int8).alias("num_first_eq"),
        (pl.col("qn").list.first() == pl.col("tn").list.first()).cast(pl.Int8).alias("ntok_first_eq"),
        pl.col("qn").list.len().cast(pl.Int16).alias("q_n_tokens"),
        pl.col("tn").list.len().cast(pl.Int16).alias("t_n_tokens"),
    ).with_columns(
        ((pl.col("q_n_nums") > 0) & (pl.col("t_n_nums") > 0) & (pl.col("num_common") == 0))
        .cast(pl.Int8)
        .alias("num_conflict"),
        ((pl.col("t_n_nums") > 0) & (pl.col("num_common") == pl.col("t_n_nums")))
        .cast(pl.Int8)
        .alias("num_t_subset"),
    )

    flags = pl.DataFrame(
        {
            "t_is_s3": tf["entity_id"].str.starts_with("S3-").cast(pl.Int8),
            "t_translit": tf["translit"].cast(pl.Int8),
            "t_has_addr": tf["has_addr"].cast(pl.Int8),
            "name_core_eq": (qf["name_core"] == tf["name_core"]).cast(pl.Int8),
            "addr_eq": (qf["addr_norm"] == tf["addr_norm"]).cast(pl.Int8),
            "q_name_freq": qf["name_freq"].cast(pl.Int32),
            "t_name_freq": tf["name_freq"].cast(pl.Int32),
            "len_diff": (
                qf["name_core"].str.len_chars().cast(pl.Int16)
                - tf["name_core"].str.len_chars().cast(pl.Int16)
            ).abs(),
        }
    )
    return feats.hstack(sets.get_columns() + flags.get_columns())


_CONTEXT_SCORES: tuple[str, ...] = ("cos_both", "cos_name", "cos_addr")
T_TOP = 8  # competing-S1 scores kept per target for rank / margin features


def target_context(source: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Per-target competition statistics over a *whole* country's candidates.

    Streaming group-by: memory is bounded by the number of distinct targets.
    """
    agg = (
        source.lazy()
        .group_by("t")
        .agg(
            pl.len().cast(pl.Int16).alias("t_n_q"),
            pl.col("cos_both").top_k(T_TOP).alias("_tops"),
            pl.col("cos_addr").max().alias("_tmax_addr"),
            pl.col("cos_name").max().alias("_tmax_name"),
        )
        .collect(engine="streaming")
    )
    tops = agg["_tops"].list.sort(descending=True)
    return agg.drop("_tops").with_columns(
        [tops.list.get(i, null_on_oob=True).alias(f"_top{i}") for i in range(T_TOP)]
    )


def context_features(cands: pl.DataFrame, t_ctx: pl.DataFrame | None = None) -> pl.DataFrame:
    """Query-context and target-competition features for complete S1 groups.

    Query context only looks inside one S1, so any chunk of whole S1s is exact.
    Target competition needs every S1 that retrieved the target: pass ``t_ctx``
    from ``target_context`` over the whole country (defaults to ``cands`` itself).
    ``t_rank_cos_both`` = 1 + number of competing S1s with a strictly higher
    ``cos_both`` (capped at ``T_TOP + 1``).
    """
    if t_ctx is None:
        t_ctx = target_context(cands)
    exprs: list[pl.Expr] = [pl.len().over("q").cast(pl.Int16).alias("q_n_cands")]
    for s in _CONTEXT_SCORES:
        exprs += [
            (pl.col(s) - pl.col(s).max().over("q")).alias(f"q_gap_{s}"),
            pl.col(s).rank("ordinal", descending=True).over("q").cast(pl.Int16).alias(f"q_rank_{s}"),
        ]
    out = cands.with_columns(exprs).join(t_ctx, on="t", how="left", maintain_order="left")
    higher = [(pl.col(f"_top{i}") > pl.col("cos_both")).fill_null(False).cast(pl.Int16) for i in range(T_TOP)]
    best = pl.col("cos_both") >= pl.col("_top0")
    out = out.with_columns(
        (1 + pl.sum_horizontal(higher)).cast(pl.Int16).alias("t_rank_cos_both"),
        (pl.col("cos_both") - pl.col("_top0")).alias("t_gap_cos_both"),
        (pl.col("cos_addr") - pl.col("_tmax_addr")).alias("t_gap_cos_addr"),
        (pl.col("cos_name") - pl.col("_tmax_name")).alias("t_gap_cos_name"),
        pl.when(best)
        .then(pl.col("cos_both") - pl.col("_top1").fill_null(0.0))
        .otherwise(pl.col("cos_both") - pl.col("_top0"))
        .alias("t_margin_cos_both"),
    )
    return out.drop([c for c in out.columns if c.startswith("_t")])


def name_frequency(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Count of records sharing each (ckey, name_core) across ``frames``."""
    return (
        pl.concat([f.select("ckey", "name_core") for f in frames])
        .group_by("ckey", "name_core")
        .len("name_freq")
    )
