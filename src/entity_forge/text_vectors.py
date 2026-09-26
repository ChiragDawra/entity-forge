"""Hashed sparse TF-IDF vectors built with Polars (no Python loop per string).

A *field spec* describes one block of the vector: which column, how to split
it into features (``word`` tokens or ``charN`` word-bounded n-grams) and the
block weight. Blocks are L2-normalized separately and concatenated with
``sqrt(weight)`` scaling, so the cosine of two vectors is the weighted sum of
per-block cosines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

N_BITS = 22  # 4M hash buckets per block; collisions are negligible at this scale
_HASH_SEED = 20260925


@dataclass(frozen=True)
class FieldSpec:
    """One vector block.

    ``kind`` is ``word`` (tokens), ``charN`` (word-bounded character n-grams) or
    ``cross`` (every token of ``column`` paired with every token of
    ``column2``: rare conjunctions such as ``"sharma|kolhapur"`` single out one
    business even inside a building shared by many).
    """

    column: str
    kind: str
    weight: float = 1.0
    column2: str | None = None


def _feature_frame(texts: pl.Series, kind: str) -> pl.DataFrame:
    """Return (row, f) with one line per feature occurrence."""
    df = pl.DataFrame({"row": np.arange(len(texts), dtype=np.int32), "s": texts.fill_null("")})
    words = (
        df.select("row", pl.col("s").str.split(" ").alias("g"))
        .explode("g", empty_as_null=False)
        .filter(pl.col("g").str.len_chars() > 0)
    )
    if kind == "word":
        grams = words
    elif kind.startswith("char"):
        n = int(kind[4:])
        padded = words.with_columns(("<" + pl.col("g") + ">").alias("g"))
        padded = padded.with_columns(
            pl.int_ranges(0, (pl.col("g").str.len_chars() - n + 1).clip(lower_bound=1)).alias("p")
        ).explode("p", empty_as_null=False)
        grams = padded.select("row", pl.col("g").str.slice(pl.col("p"), n).alias("g"))
    else:
        raise ValueError(f"unknown feature kind: {kind}")
    return grams.select(
        "row", (pl.col("g").hash(seed=_HASH_SEED) % (1 << N_BITS)).cast(pl.Int32).alias("f")
    )


def _cross_frame(left: pl.Series, right: pl.Series) -> pl.DataFrame:
    """(row, f) for every (left token, right token) conjunction."""
    df = pl.DataFrame(
        {
            "row": np.arange(len(left), dtype=np.int32),
            "a": left.fill_null("").str.split(" "),
            "b": right.fill_null("").str.split(" "),
        }
    )
    df = df.with_columns(
        pl.col("a").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z][a-z0-9]+$"))),
        pl.col("b").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)),
    )
    pairs = df.explode("a", empty_as_null=False).explode("b", empty_as_null=False)
    return pairs.select(
        "row",
        ((pl.col("a") + "|" + pl.col("b")).hash(seed=_HASH_SEED) % (1 << N_BITS))
        .cast(pl.Int32)
        .alias("f"),
    )


def term_counts(
    texts: pl.Series, kind: str, chunk_rows: int = 500_000, texts2: pl.Series | None = None
) -> sp.csr_matrix:
    """Raw term-count CSR (rows x 2**N_BITS), built in chunks to bound memory."""
    blocks = []
    for start in range(0, len(texts), chunk_rows):
        part = texts.slice(start, chunk_rows)
        if kind == "cross":
            assert texts2 is not None, "cross features need a second column"
            frame = _cross_frame(part, texts2.slice(start, chunk_rows))
        else:
            frame = _feature_frame(part, kind)
        counts = frame.group_by("row", "f").len().sort("row", "f")
        mat = sp.csr_matrix(
            (
                counts["len"].to_numpy().astype(np.float32),
                (counts["row"].to_numpy(), counts["f"].to_numpy()),
            ),
            shape=(len(part), 1 << N_BITS),
        )
        blocks.append(mat)
    if not blocks:
        return sp.csr_matrix((0, 1 << N_BITS), dtype=np.float32)
    return sp.vstack(blocks, format="csr")


def _second(frame: pl.DataFrame, spec: FieldSpec) -> pl.Series | None:
    return frame[spec.column2] if spec.column2 else None


def document_frequency(counts: sp.csr_matrix) -> np.ndarray:
    return np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.int64)


def idf_weights(df: np.ndarray, n_docs: int, max_df: int | None) -> np.ndarray:
    """Smoothed IDF; features above ``max_df`` (or unseen in the corpus) get weight 0."""
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    idf[df == 0] = 0.0
    if max_df is not None:
        idf[df > max_df] = 0.0
    return idf.astype(np.float32)


def _l2_rows(mat: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags((1.0 / norms).astype(np.float32)) @ mat).astype(np.float32).tocsr()


def tfidf(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Sublinear TF x IDF, rows L2-normalized, zero entries removed."""
    mat = counts.copy()
    mat.data = (1.0 + np.log(mat.data)) * idf[mat.indices]
    mat.eliminate_zeros()
    return _l2_rows(mat)


@dataclass
class FittedBlock:
    spec: FieldSpec
    idf: np.ndarray


ROW_CHUNK = 500_000


def fit_blocks(corpus: pl.DataFrame, specs: list[FieldSpec], max_df_frac: float,
               chunk_rows: int = ROW_CHUNK) -> list[FittedBlock]:
    """Learn per-block IDF on the target corpus (document frequencies accumulated per chunk)."""
    n = corpus.height
    max_df = max(50, int(max_df_frac * n))
    fitted = []
    for spec in specs:
        df = np.zeros(1 << N_BITS, dtype=np.int64)
        for start in range(0, n, chunk_rows):
            part = corpus.slice(start, chunk_rows)
            counts = term_counts(part[spec.column], spec.kind, chunk_rows, texts2=_second(part, spec))
            df += np.bincount(counts.indices, minlength=counts.shape[1])
            del counts
        fitted.append(FittedBlock(spec, idf_weights(df, n, max_df)))
    return fitted


def transform(frame: pl.DataFrame, blocks: list[FittedBlock], chunk_rows: int = ROW_CHUNK) -> sp.csr_matrix:
    """Weighted concatenation of per-block TF-IDF vectors, L2-normalized overall.

    Built in row chunks so peak memory is about one chunk plus the result,
    instead of several full-size intermediate copies.
    """
    rows = []
    for start in range(0, frame.height, chunk_rows):
        part = frame.slice(start, chunk_rows)
        vecs = []
        for block in blocks:
            counts = term_counts(part[block.spec.column], block.spec.kind, chunk_rows,
                                 texts2=_second(part, block.spec))
            vecs.append(tfidf(counts, block.idf) * np.float32(np.sqrt(block.spec.weight)))
        mat = sp.hstack(vecs, format="csr") if len(vecs) > 1 else vecs[0]
        rows.append(_l2_rows(mat))
    if not rows:
        width = (1 << N_BITS) * max(len(blocks), 1)
        return sp.csr_matrix((0, width), dtype=np.float32)
    return sp.vstack(rows, format="csr") if len(rows) > 1 else rows[0]


def rowwise_dot(
    a: sp.csr_matrix, b: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 1_000_000
) -> np.ndarray:
    """Cosine for explicit pairs: out[k] = <a[ia[k]], b[ib[k]]>."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.asarray(a[ia[s:e]].multiply(b[ib[s:e]]).sum(axis=1)).ravel()
    return out
