"""The memory-bounded retrieval must reproduce the original in-memory algorithm exactly."""

import numpy as np
import polars as pl
import pytest

from entity_forge.blocking import Channel, generate_candidates_to_file
from entity_forge.features import name_frequency
from entity_forge.normalize import normalize_records
from entity_forge.text_vectors import FieldSpec, fit_blocks, transform
from tests.reference import blocking_v1

SCHEMA = {"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String,
          "country": pl.String}
NAMES = ["Patterson Chadwick", "Booker Urban", "Design Academy", "Golden Foods", "Star Media",
         "Lotus International", "Keystone Schwab", "Southern Food"]
STREETS = ["Main St", "Honeywood Ct", "Mithila Nagar", "218th Avenue", "Church Road"]


def _records(prefix, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        name = NAMES[rng.integers(len(NAMES))] + ("" if rng.random() < 0.5 else " LLC")
        addr = "" if rng.random() < 0.1 else f"{rng.integers(1, 60)} {STREETS[rng.integers(len(STREETS))]}, Town, AZ"
        rows.append((f"{prefix}-{i}", name, addr, "US"))
    df = normalize_records(pl.DataFrame(rows, schema=SCHEMA, orient="row"))
    return df.join(name_frequency([df]), on=["ckey", "name_core"], how="left", maintain_order="left")


CHANNELS = (
    Channel("addr", (FieldSpec("addr_norm", "word"),), top_k=5, max_df_frac=0.5),
    Channel("both", (FieldSpec("name_core", "word", 0.5), FieldSpec("addr_norm", "word", 0.5)),
            top_k=5, max_df_frac=0.5),
    Channel("cross", (FieldSpec("name_core", "cross", column2="addr_norm"),), top_k=3, max_df_frac=0.5),
    Channel("noaddr", (FieldSpec("name_core", "word"),), top_k=3, max_df_frac=0.9, target_filter="no_addr"),
)


@pytest.fixture(scope="module")
def data():
    return _records("S1", 120, 1), _records("S2", 400, 2)


def test_streamed_candidates_equal_in_memory_reference(tmp_path, data):
    q, t = data
    ref = blocking_v1.generate_candidates(q, t, CHANNELS, n_threads=1)
    out = tmp_path / "cands.parquet"
    rows = generate_candidates_to_file(q, t, CHANNELS, out, tmp_path / "scratch", n_threads=2,
                                       retrieval_chunk=7, union_chunk_rows=50)
    new = pl.read_parquet(out)
    assert rows == ref.height == new.height
    assert new.columns == ref.columns + ["s1_id", "t_id"]
    assert new.select("q").to_series().is_sorted()
    for c in ref.columns:
        if c.startswith("cos_"):
            np.testing.assert_allclose(new[c].to_numpy(), ref[c].to_numpy(), rtol=0, atol=1e-6)
        else:
            assert new[c].equals(ref[c]), c
    assert new["s1_id"].to_list() == q["entity_id"].gather(new["q"]).to_list()


def test_resume_after_interruption_gives_same_file(tmp_path, data):
    q, t = data
    scratch = tmp_path / "scratch"
    first = tmp_path / "a.parquet"
    generate_candidates_to_file(q, t, CHANNELS, first, scratch, n_threads=1, retrieval_chunk=10,
                                union_chunk_rows=60)
    # Simulate a crash after the union step: drop cosine markers and final parts.
    for ch in CHANNELS:
        (scratch / ch.name / "cos.done").unlink()
    for p in (scratch / "final").glob("part-*.parquet"):
        p.unlink()
    second = tmp_path / "b.parquet"
    generate_candidates_to_file(q, t, CHANNELS, second, scratch, n_threads=1, retrieval_chunk=10,
                                union_chunk_rows=60)
    assert pl.read_parquet(first).equals(pl.read_parquet(second))


def test_chunked_transform_equals_single_pass(data):
    _, t = data
    specs = [FieldSpec("name_core", "char3", 0.5), FieldSpec("addr_norm", "word", 0.5)]
    blocks = fit_blocks(t, specs, 0.5, chunk_rows=37)
    assert all(np.array_equal(a.idf, b.idf) for a, b in zip(blocks, fit_blocks(t, specs, 0.5, chunk_rows=10**6)))
    small = transform(t, blocks, chunk_rows=37)
    whole = transform(t, blocks, chunk_rows=10**6)
    assert (small != whole).nnz == 0


def test_stale_scratch_from_other_inputs_is_not_reused(tmp_path, data):
    q, t = data
    scratch = tmp_path / "scratch"
    generate_candidates_to_file(q, t, CHANNELS, tmp_path / "a.parquet", scratch, n_threads=1,
                                retrieval_chunk=50, union_chunk_rows=100)
    q2 = q.head(60)  # different inputs, same scratch directory
    out = tmp_path / "b.parquet"
    generate_candidates_to_file(q2, t, CHANNELS, out, scratch, n_threads=1, retrieval_chunk=50,
                                union_chunk_rows=100)
    ref = blocking_v1.generate_candidates(q2, t, CHANNELS, n_threads=1)
    assert pl.read_parquet(out).height == ref.height
    assert pl.read_parquet(out)["q"].max() < 60
