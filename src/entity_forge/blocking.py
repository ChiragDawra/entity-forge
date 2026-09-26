"""Candidate generation: independent sparse-retrieval channels, unioned.

Each channel is a hashed TF-IDF representation of one or more fields. For every
query (S1) we keep the top-K targets (S2 ∪ S3 of the same country) by cosine,
computed with ``sparse_dot_topn`` in query chunks, so no dense all-pairs matrix
is ever built. Channels are unioned and every union pair then gets the exact
cosine from *every* channel (not only the channel that proposed it).

Memory model (``generate_candidates_to_file``): only one channel's vectors are
in RAM at a time.

1. per channel: vectorize → spill q/t vectors to disk → search (the transposed
   target matrix replaces the row-major one during search) → write hits → free;
2. union the channels' hits per S1 range (hits are sorted by q);
3. per channel: reload its vectors → exact cosines for every union part → free;
4. assemble parts with ids and stream them into one Parquet file.

Every step leaves a marker in a scratch directory, so an interrupted partition
resumes where it stopped. The result is identical to the in-memory algorithm
(``generate_candidates``), which is kept for small inputs and tests.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from .resources import log_mem
from .text_vectors import FieldSpec, fit_blocks, rowwise_dot, transform

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Channel:
    """One retrieval channel.

    ``target_filter`` restricts the *searched* pool to targets where that boolean
    column is true (e.g. ``no_addr``: name search among address-less records,
    where a generic name competes with far fewer look-alikes). Cosines for union
    pairs are still computed against every target.
    """

    name: str
    fields: tuple[FieldSpec, ...]
    top_k: int
    max_df_frac: float = 0.01
    min_score: float = 0.05
    target_filter: str | None = None


@dataclass
class ChannelIndex:
    """Fitted vectors for one channel (queries and targets of one country)."""

    channel: Channel
    q_vec: sp.csr_matrix
    t_vec: sp.csr_matrix | None
    searchable: np.ndarray | None = None  # target rows the search may return (None = all)
    fit_seconds: float = 0.0


def _target_mask(targets: pl.DataFrame, name: str | None) -> np.ndarray | None:
    if name is None:
        return None
    if name == "no_addr":
        return np.flatnonzero(~targets["has_addr"].to_numpy())
    return np.flatnonzero(targets[name].to_numpy())


def build_index(channel: Channel, queries: pl.DataFrame, targets: pl.DataFrame) -> ChannelIndex:
    t0 = time.time()
    searchable = _target_mask(targets, channel.target_filter)
    corpus = targets if searchable is None else targets[searchable]
    blocks = fit_blocks(corpus, list(channel.fields), channel.max_df_frac)
    t_vec = transform(targets, blocks)
    q_vec = transform(queries, blocks)
    return ChannelIndex(channel, q_vec, t_vec, searchable, fit_seconds=time.time() - t0)


_HITS_SCHEMA = {"q": pl.Int32, "t": pl.Int32, "score": pl.Float32, "rank": pl.Int16}


def _search_transposed(
    q_vec: sp.csr_matrix, bt: sp.csr_matrix, searchable: np.ndarray | None, k: int,
    min_score: float, chunk_rows: int, n_threads: int,
) -> pl.DataFrame:
    parts = []
    for start in range(0, q_vec.shape[0], chunk_rows):
        q = q_vec[start : start + chunk_rows]
        res = sp_matmul_topn(q, bt, top_n=k, threshold=min_score, sort=True, n_threads=n_threads).tocsr()
        counts = np.diff(res.indptr)
        rows = np.repeat(np.arange(res.shape[0], dtype=np.int32), counts) + start
        rank = (np.arange(len(res.indices)) - np.repeat(res.indptr[:-1], counts)).astype(np.int16)
        cols = res.indices if searchable is None else searchable[res.indices]
        parts.append(
            pl.DataFrame(
                {"q": rows, "t": cols.astype(np.int32), "score": res.data.astype(np.float32), "rank": rank}
            )
        )
    return pl.concat(parts) if parts else pl.DataFrame(schema=_HITS_SCHEMA)


def _transposed_pool(index: ChannelIndex) -> sp.csr_matrix:
    pool = index.t_vec if index.searchable is None else index.t_vec[index.searchable]
    return pool.T.tocsr()


def topk_search(
    index: ChannelIndex, top_k: int | None = None, chunk_rows: int = 100_000, n_threads: int = 8
) -> pl.DataFrame:
    """Top-K targets per query: frame (q, t, score, rank)."""
    bt = _transposed_pool(index)
    return _search_transposed(index.q_vec, bt, index.searchable, top_k or index.channel.top_k,
                              index.channel.min_score, chunk_rows, n_threads)


def pair_scores(index: ChannelIndex, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exact channel cosine for arbitrary (q, t) pairs."""
    return rowwise_dot(index.q_vec, index.t_vec, q, t)


def _union(found: list[pl.DataFrame], channels: tuple[Channel, ...]) -> pl.DataFrame:
    union = found[0]
    for other in found[1:]:
        union = union.join(other, on=["q", "t"], how="full", coalesce=True)
    return union.with_columns([pl.col(f"rank_{c.name}").fill_null(-1) for c in channels]).sort("q", "t")


def generate_candidates(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    channels: tuple[Channel, ...],
    n_threads: int = 8,
) -> pl.DataFrame:
    """In-memory union of all channels for one (small) country partition.

    Returns one row per unique (q, t) with, for every channel ``c``:
    ``cos_<c>`` (exact cosine) and ``rank_<c>`` (rank within that channel's
    top-K for the query, or -1 when the channel did not propose the pair).
    Use ``generate_candidates_to_file`` for full-size partitions.
    """
    found: list[pl.DataFrame] = []
    indexes: list[ChannelIndex] = []
    for ch in channels:
        idx = build_index(ch, queries, targets)
        hits = topk_search(idx, n_threads=n_threads)
        found.append(hits.select("q", "t", pl.col("rank").alias(f"rank_{ch.name}")))
        indexes.append(idx)
    union = _union(found, channels)
    q_arr = union["q"].to_numpy()
    t_arr = union["t"].to_numpy()
    return union.with_columns(
        [pl.Series(f"cos_{idx.channel.name}", pair_scores(idx, q_arr, t_arr)) for idx in indexes]
    )


# ---------------------------------------------------------------------------
# Memory-bounded, resumable version for full-size partitions
# ---------------------------------------------------------------------------


def _done(marker: Path) -> bool:
    return marker.is_file()


def _mark(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")


def _atomic_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def generate_candidates_to_file(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    channels: tuple[Channel, ...],
    out_path: Path,
    scratch: Path,
    *,
    n_threads: int,
    retrieval_chunk: int,
    union_chunk_rows: int,
) -> int:
    """Same result as ``generate_candidates`` (plus ``s1_id``/``t_id``), bounded memory.

    ``queries``/``targets`` must carry ``entity_id``. Returns the number of rows
    written to ``out_path``. ``scratch`` holds resumable intermediates; the
    caller removes it after the output is checkpointed.
    """
    n_q = queries.height
    # Scratch markers are only trusted for the same inputs and channels.
    meta = {
        "n_q": n_q, "n_t": targets.height,
        "ids": [queries["entity_id"].head(3).to_list(), queries["entity_id"].tail(3).to_list(),
                targets["entity_id"].head(3).to_list(), targets["entity_id"].tail(3).to_list()],
        "channels": [repr(c) for c in channels],
    }
    meta_path = scratch / "meta.json"
    if scratch.exists() and (not meta_path.is_file() or json.loads(meta_path.read_text()) != meta):
        log.warning("candidate scratch %s belongs to other inputs: starting it fresh", scratch)
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))

    # 1. Search channel by channel; keep only hits, spill vectors.
    for ch in channels:
        cdir = scratch / ch.name
        if _done(cdir / "search.done"):
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        log_mem(f"candidates: before channel {ch.name}")
        idx = build_index(ch, queries, targets)
        log_mem(f"candidates: vectorized {ch.name} (nnz q={idx.q_vec.nnz:,} t={idx.t_vec.nnz:,})")
        sp.save_npz(cdir / "q.npz", idx.q_vec, compressed=False)
        sp.save_npz(cdir / "t.npz", idx.t_vec, compressed=False)
        bt = _transposed_pool(idx)
        idx.t_vec = None  # the transposed copy replaces it during search
        gc.collect()
        log_mem(f"candidates: before top-k {ch.name}")
        t0 = time.time()
        hits = _search_transposed(idx.q_vec, bt, idx.searchable, ch.top_k, ch.min_score,
                                  retrieval_chunk, n_threads)
        log.info("channel %s: fit %.1fs search %.1fs, %d pairs", ch.name, idx.fit_seconds,
                 time.time() - t0, hits.height)
        _atomic_parquet(hits.select("q", "t", pl.col("rank").alias(f"rank_{ch.name}")), cdir / "hits.parquet")
        del idx, bt, hits
        gc.collect()
        _mark(cdir / "search.done")
        log_mem(f"candidates: after cleanup {ch.name}")

    # 2. Union per S1 range. Hits are sorted by q, so range filters read few row groups.
    total_hits = sum(pl.scan_parquet(scratch / c.name / "hits.parquet").select(pl.len()).collect().item()
                     for c in channels)
    q_per_part = max(1, int(union_chunk_rows / max(total_hits / max(n_q, 1), 1)))
    bounds = [(a, min(a + q_per_part, n_q)) for a in range(0, n_q, q_per_part)] or [(0, 0)]
    udir = scratch / "union"
    udir.mkdir(exist_ok=True)
    if not _done(udir / "done"):
        for i, (a, b) in enumerate(bounds):
            path = udir / f"part-{i:05d}.parquet"
            if path.is_file():
                continue
            found = [
                pl.scan_parquet(scratch / c.name / "hits.parquet")
                .filter((pl.col("q") >= a) & (pl.col("q") < b))
                .collect()
                for c in channels
            ]
            _atomic_parquet(_union(found, channels), path)
        _mark(udir / "done")
    log_mem("candidates: union written")

    # 3. Exact cosine of every channel for every union pair, one channel in RAM.
    for ch in channels:
        cdir = scratch / ch.name
        if _done(cdir / "cos.done"):
            continue
        q_vec = sp.load_npz(cdir / "q.npz").tocsr()
        t_vec = sp.load_npz(cdir / "t.npz").tocsr()
        log_mem(f"candidates: cosines {ch.name}")
        for i in range(len(bounds)):
            path = cdir / f"cos-{i:05d}.parquet"
            if path.is_file():
                continue
            pairs = pl.read_parquet(udir / f"part-{i:05d}.parquet", columns=["q", "t"])
            cos = rowwise_dot(q_vec, t_vec, pairs["q"].to_numpy(), pairs["t"].to_numpy())
            _atomic_parquet(pl.DataFrame({f"cos_{ch.name}": cos}), path)
        del q_vec, t_vec
        gc.collect()
        _mark(cdir / "cos.done")

    # 4. Assemble parts with ids, stream into the single output file.
    fdir = scratch / "final"
    fdir.mkdir(exist_ok=True)
    q_ids = queries["entity_id"]
    t_ids = targets["entity_id"]
    for i in range(len(bounds)):
        path = fdir / f"part-{i:05d}.parquet"
        if path.is_file():
            continue
        part = pl.read_parquet(udir / f"part-{i:05d}.parquet")
        cos = [pl.read_parquet(scratch / c.name / f"cos-{i:05d}.parquet").to_series() for c in channels]
        part = part.hstack(cos).with_columns(
            q_ids.gather(part["q"]).alias("s1_id"), t_ids.gather(part["t"]).alias("t_id")
        )
        _atomic_parquet(part, path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    pl.scan_parquet(sorted(fdir.glob("part-*.parquet"))).sink_parquet(tmp)
    os.replace(tmp, out_path)
    rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    log_mem(f"candidates: wrote {out_path.name} ({rows:,} rows)")
    return rows


def candidate_recall(cands: pl.DataFrame, truth: pl.DataFrame) -> dict[str, float]:
    """Pair recall of ``cands`` (s1_id, t_id) against ``truth`` (s1_id, t_id).

    ``truth`` must already be restricted to the evaluated S1 population.
    """
    hit = truth.join(cands.select("s1_id", "t_id").unique(), on=["s1_id", "t_id"], how="semi").height
    n_s1 = cands["s1_id"].n_unique()
    per_s1 = cands.group_by("s1_id").len()["len"]
    return {
        "pair_recall": hit / max(truth.height, 1),
        "true_pairs": truth.height,
        "candidates": cands.height,
        "avg_per_s1": cands.height / max(n_s1, 1),
        "p95_per_s1": float(per_s1.quantile(0.95) or 0),
        "p99_per_s1": float(per_s1.quantile(0.99) or 0),
    }
