"""Chunked (streaming) feature computation must equal the whole-partition computation."""

import numpy as np
import polars as pl
import pytest

from entity_forge.features import context_features, target_context
from entity_forge.pruning import stage0_features, target_aggregates
from entity_forge.second_stage import query_context, target_context as p1_target_context

CHANNELS = ("addr", "both", "cross", "name")


@pytest.fixture(scope="module")
def cands():
    rng = np.random.default_rng(3)
    n_q, n_t = 300, 150
    rows = [(q, t) for q in range(n_q) for t in rng.choice(n_t, size=rng.integers(1, 12), replace=False)]
    df = pl.DataFrame(rows, schema={"q": pl.Int32, "t": pl.Int32}, orient="row").sort("q", "t")
    cols = {}
    for c in CHANNELS:
        cols[f"rank_{c}"] = rng.integers(-1, 10, df.height).astype(np.int16)
        cols[f"cos_{c}"] = np.round(rng.random(df.height), 2).astype(np.float32)  # rounding creates ties
    return df.with_columns([pl.Series(k, v) for k, v in cols.items()])


def _chunks(df, size):
    for a in range(0, df["q"].max() + 1, size):
        yield df.filter((pl.col("q") >= a) & (pl.col("q") < a + size))


def test_stage0_features_chunked_equals_whole(cands):
    whole = stage0_features(cands)
    t_agg = target_aggregates(cands)
    chunked = pl.concat([stage0_features(c, t_agg) for c in _chunks(cands, 37)])
    assert chunked.equals(whole)


def test_context_features_chunked_equals_whole(cands):
    whole = context_features(cands)
    t_ctx = target_context(cands)
    chunked = pl.concat([context_features(c, t_ctx) for c in _chunks(cands, 41)])
    assert chunked.equals(whole)


def test_target_rank_and_margin_semantics():
    df = pl.DataFrame({"q": [0, 1, 2, 3], "t": [5, 5, 5, 5], "cos_both": [0.9, 0.9, 0.5, 0.1],
                       "cos_addr": [0.1] * 4, "cos_name": [0.2] * 4}).with_columns(
        pl.col("q", "t").cast(pl.Int32), pl.col("cos_both", "cos_addr", "cos_name").cast(pl.Float32))
    out = context_features(df).sort("q")
    assert out["t_rank_cos_both"].to_list() == [1, 1, 3, 4]  # ties share the best rank
    assert out["t_margin_cos_both"].to_list()[:2] == [0.0, 0.0]  # tied best: no margin
    assert out["t_n_q"].to_list() == [4] * 4


def test_stage2_aggregates_match_sort_based_reference(cands):
    scores = cands.select("q", "t", pl.Series("p1", np.linspace(0.01, 0.99, cands.height)[::-1].astype(np.float32)))
    q_ctx = query_context(scores).sort("q")
    ref = (scores.sort(["q", "p1"], descending=[False, True]).group_by("q", maintain_order=True)
           .agg(pl.col("p1").max().alias("q_p1_max"),
                pl.col("p1").get(1, null_on_oob=True).fill_null(0.0).alias("q_p1_second"),
                pl.col("t").first().alias("q_top1_t"), pl.col("t").get(1, null_on_oob=True).alias("q_top2_t")))
    for c in ("q_p1_max", "q_p1_second", "q_top1_t", "q_top2_t"):
        assert q_ctx[c].equals(ref[c]), c
    t_ctx = p1_target_context(scores).sort("t")
    ref_t = (scores.sort(["t", "p1"], descending=[False, True]).group_by("t", maintain_order=True)
             .agg(pl.col("p1").max().alias("t_p1_max"),
                  pl.col("p1").get(1, null_on_oob=True).fill_null(0.0).alias("t_p1_second"),
                  pl.col("q").first().alias("t_best_q")))
    for c in ("t_p1_max", "t_p1_second", "t_best_q"):
        assert t_ctx[c].equals(ref_t[c]), c
