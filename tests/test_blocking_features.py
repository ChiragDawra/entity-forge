import numpy as np
import polars as pl
import pytest

from entity_forge.blocking import Channel, candidate_recall, generate_candidates
from entity_forge.features import context_features, name_frequency, pair_features
from entity_forge.model import fold_expr, sample_expr
from entity_forge.normalize import normalize_records
from entity_forge.pruning import prune, stage0_features, train_stage0
from entity_forge.second_stage import query_context, stage2_features, target_context
from entity_forge.text_vectors import FieldSpec, fit_blocks, transform

SCHEMA = {"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String,
          "country": pl.String}

S1 = [
    ("S1-1", "Patterson Chadwick Corp", "1673 218th Avenue, Buckeye, AZ", "US"),
    ("S1-2", "Booker Urban Inc", "153 Honeywood Court, Aberdeen, NC", "US"),
    ("S1-3", "Design Academy", "Plot No 201 Mithila Nagar, Hyderabad, Telangana", "US"),
]
TARGETS = [
    ("S2-1", "PATTERSON CHADWICK (CORP)", "001673 218RD AVENUE, BUCKEYE, AZ", "US"),
    ("S3-1", "Patterson Corp (Chadwick)", "218rd Avenue, Buckeye, Arizona", "US"),
    ("S2-2", "INC BOOKER URNA", "153 HONEYWOOD CT, ABERDEEN, NC", "US"),
    ("S3-2", "Booker Urban", "", "US"),
    ("S2-3", "Design Service", "PLOT NO 201 MITHILA NAGAR, HYDERABAD, TG", "US"),
    ("S2-9", "Golden Foods", "12 Main Street, Springfield, IL", "US"),
]
TRUE_TARGETS = ["S2-1", "S3-1", "S2-2", "S3-2", "S2-3"]
CHANNELS = (
    Channel("name", (FieldSpec("name_core", "char3"),), top_k=3, max_df_frac=1.0),
    Channel("addr", (FieldSpec("addr_norm", "word"),), top_k=3, max_df_frac=1.0),
    Channel("both", (FieldSpec("name_core", "word", 0.5), FieldSpec("addr_norm", "word", 0.5)),
            top_k=3, max_df_frac=1.0),
    Channel("cross", (FieldSpec("name_core", "cross", column2="addr_norm"),), top_k=3, max_df_frac=1.0),
    Channel("noaddr", (FieldSpec("name_core", "word"),), top_k=3, max_df_frac=1.0, target_filter="no_addr"),
)


@pytest.fixture(scope="module")
def frames():
    q = normalize_records(pl.DataFrame(S1, schema=SCHEMA, orient="row"))
    t = normalize_records(pl.DataFrame(TARGETS, schema=SCHEMA, orient="row"))
    freq = name_frequency([q, t])
    q = q.join(freq, on=["ckey", "name_core"], how="left", maintain_order="left")
    t = t.join(freq, on=["ckey", "name_core"], how="left", maintain_order="left")
    return q, t


@pytest.fixture(scope="module")
def cands(frames):
    q, t = frames
    c = generate_candidates(q, t, CHANNELS, n_threads=1)
    return c.with_columns(
        q["entity_id"].gather(c["q"]).alias("s1_id"), t["entity_id"].gather(c["t"]).alias("t_id")
    )


def test_identical_text_has_cosine_one():
    df = pl.DataFrame({"x": ["acme tools", "acme tools", "zeta"]})
    blocks = fit_blocks(df, [FieldSpec("x", "char3")], max_df_frac=1.0)
    m = transform(df, blocks)
    sims = (m @ m.T).toarray()
    assert sims[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert sims[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_union_recovers_true_pairs(cands):
    truth = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3"], "t_id": TRUE_TARGETS})
    assert candidate_recall(cands, truth)["pair_recall"] == 1.0
    for c in ("name", "addr", "both", "cross", "noaddr"):
        assert {f"cos_{c}", f"rank_{c}"} <= set(cands.columns)
    in_range = cands.select(pl.all_horizontal(pl.col("^cos_.*$").is_between(-1e-5, 1 + 1e-5)).all())
    assert in_range.item()


def test_noaddr_channel_only_proposes_addressless_targets(cands):
    proposed = cands.filter(pl.col("rank_noaddr") >= 0)["t_id"].unique().to_list()
    assert proposed == ["S3-2"]


def test_pair_and_context_features(frames, cands):
    q, t = frames
    feats = pair_features(context_features(cands.sort("q", "t")), q, t)
    assert feats.height == cands.height
    row = feats.filter((pl.col("s1_id") == "S1-1") & (pl.col("t_id") == "S2-1")).row(0, named=True)
    assert row["n_tset"] == pytest.approx(1.0)
    assert row["num_first_eq"] == 1 and row["num_conflict"] == 0
    assert row["t_is_s3"] == 0
    # The best S1 for each target has a non-negative margin over its competitors.
    best = feats.filter(pl.col("t_rank_cos_both") == 1)
    assert (best["t_margin_cos_both"] >= 0).all()


def test_stage2_features(frames, cands):
    _, t = frames
    scored = cands.with_columns(pl.Series("p1", np.linspace(0.9, 0.1, cands.height).astype(np.float32)))
    out = stage2_features(scored, query_context(scored), target_context(scored), t)
    assert {"q_p1_gap", "t_p1_margin", "sib_top1_name", "is_top1"} <= set(out.columns)
    assert out.filter(pl.col("is_top1") == 1)["q_p1_gap"].max() == pytest.approx(0.0)
    assert out.height == cands.height


def test_fold_and_sample_are_deterministic():
    ids = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(1000)]})
    f1 = ids.select(fold_expr(pl.col("s1_id"), 5)).to_series()
    f2 = ids.select(fold_expr(pl.col("s1_id"), 5)).to_series()
    assert f1.equals(f2) and set(f1.unique().to_list()) == {0, 1, 2, 3, 4}
    frac = ids.select(sample_expr(pl.col("s1_id"), 0.3)).to_series().mean()
    assert 0.2 < frac < 0.4
    assert ids.select(sample_expr(pl.col("s1_id"), 1.0)).to_series().all()


def test_prune_keeps_top_n_per_query(cands):
    labeled = stage0_features(cands).with_columns(
        pl.col("t_id").is_in(TRUE_TARGETS).cast(pl.Int8).alias("label")
    )
    big = pl.concat([labeled.with_columns((pl.col("s1_id") + f"{i}").alias("s1_id")) for i in range(300)])
    booster, cols = train_stage0(big, n_threads=1, rounds=5)
    kept = prune(stage0_features(cands), booster, cols, max_per_q=2)
    assert kept.group_by("q").len()["len"].max() <= 2
    assert "score0" in kept.columns


def test_salted_samples_are_independent():
    ids = pl.DataFrame({"s1_id": [f"S1-{i * 7919 + 13}" for i in range(200_000)]})
    first = ids.filter(sample_expr(pl.col("s1_id"), 0.05, salt=11))
    kept = first.select(sample_expr(pl.col("s1_id"), 0.3, salt=12)).to_series().mean()
    assert 0.27 < kept < 0.33
    folds = first.select(fold_expr(pl.col("s1_id"), 5)).to_series().value_counts()["count"]
    assert folds.min() > 0.17 * first.height
