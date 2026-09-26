"""Candidate generation: independent sparse-retrieval channels, unioned.

Each channel is a hashed TF-IDF representation of one or more fields. For every
query (S1) we keep the top-K targets (S2 ∪ S3 of the same country) by cosine,
computed with ``sparse_dot_topn`` in query chunks, so no dense all-pairs matrix
is ever built. Channels are unioned and every union pair then gets the exact
cosine from *every* channel (not only the channel that proposed it).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from entity_forge.text_vectors import FieldSpec, fit_blocks, rowwise_dot, transform

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
    t_vec: sp.csr_matrix
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


def topk_search(
    index: ChannelIndex, top_k: int | None = None, chunk_rows: int = 100_000, n_threads: int = 8
) -> pl.DataFrame:
    """Top-K targets per query: frame (q, t, score, rank)."""
    k = top_k or index.channel.top_k
    pool = index.t_vec if index.searchable is None else index.t_vec[index.searchable]
    bt = pool.T.tocsr()
    parts = []
    for start in range(0, index.q_vec.shape[0], chunk_rows):
        q = index.q_vec[start : start + chunk_rows]
        res = sp_matmul_topn(
            q, bt, top_n=k, threshold=index.channel.min_score, sort=True, n_threads=n_threads
        ).tocsr()
        counts = np.diff(res.indptr)
        rows = np.repeat(np.arange(res.shape[0], dtype=np.int32), counts) + start
        rank = (np.arange(len(res.indices)) - np.repeat(res.indptr[:-1], counts)).astype(np.int16)
        cols = res.indices if index.searchable is None else index.searchable[res.indices]
        parts.append(
            pl.DataFrame(
                {
                    "q": rows,
                    "t": cols.astype(np.int32),
                    "score": res.data.astype(np.float32),
                    "rank": rank,
                }
            )
        )
    del bt
    if not parts:
        return pl.DataFrame(
            schema={"q": pl.Int32, "t": pl.Int32, "score": pl.Float32, "rank": pl.Int16}
        )
    return pl.concat(parts)


def pair_scores(index: ChannelIndex, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exact channel cosine for arbitrary (q, t) pairs."""
    return rowwise_dot(index.q_vec, index.t_vec, q, t)


def generate_candidates(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    channels: tuple[Channel, ...],
    n_threads: int = 8,
) -> pl.DataFrame:
    """Union of all channels for one country partition.

    Returns one row per unique (q, t) with, for every channel ``c``:
    ``cos_<c>`` (exact cosine) and ``rank_<c>`` (rank within that channel's
    top-K for the query, or -1 when the channel did not propose the pair).
    """
    found: list[pl.DataFrame] = []
    indexes: list[ChannelIndex] = []
    for ch in channels:
        idx = build_index(ch, queries, targets)
        t0 = time.time()
        hits = topk_search(idx, n_threads=n_threads)
        log.info(
            "channel %s: fit %.1fs search %.1fs, %d pairs",
            ch.name, idx.fit_seconds, time.time() - t0, hits.height,
        )
        found.append(hits.select("q", "t", pl.col("rank").alias(f"rank_{ch.name}")))
        indexes.append(idx)

    union = found[0]
    for other in found[1:]:
        union = union.join(other, on=["q", "t"], how="full", coalesce=True)
    union = union.with_columns([pl.col(f"rank_{c.name}").fill_null(-1) for c in channels])

    q_arr = union["q"].to_numpy()
    t_arr = union["t"].to_numpy()
    union = union.with_columns(
        [pl.Series(f"cos_{idx.channel.name}", pair_scores(idx, q_arr, t_arr)) for idx in indexes]
    )
    return union.sort("q", "t")


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
