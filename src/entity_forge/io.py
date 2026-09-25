"""Reading competition TSVs and writing submission files."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")

_READ_OPTS = dict(separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def read_tsv(path: str | Path, expected: tuple[str, ...], n_rows: int | None = None) -> pl.DataFrame:
    """Read a TSV as all-string columns and validate its header."""
    df = pl.read_csv(path, n_rows=n_rows, **_READ_OPTS)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {df.columns}")
    return df.select(expected)


def read_source(path: str | Path, n_rows: int | None = None) -> pl.DataFrame:
    df = read_tsv(path, SOURCE_COLUMNS, n_rows)
    if df["entity_id"].null_count():
        raise ValueError(f"{path}: null entity_id")
    if df["entity_id"].n_unique() != df.height:
        raise ValueError(f"{path}: duplicate entity_id")
    return df


def read_ground_truth_pairs(path: str | Path) -> pl.DataFrame:
    """Ground truth as one row per (s1, match) pair: columns ``s1_id``, ``t_id``."""
    gt = read_tsv(path, GT_COLUMNS)
    return (
        gt.select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t_id"),
        )
        .explode("t_id")
        .with_columns(pl.col("t_id").str.strip_chars())
        .filter(pl.col("t_id").is_not_null() & (pl.col("t_id") != ""))
        .unique(maintain_order=True)
    )


def write_id_lists(
    s1_ids: pl.Series,
    pairs: pl.DataFrame,
    path: str | Path,
    header: tuple[str, str],
) -> None:
    """Write one row per S1 (all of ``s1_ids``, in order) with its comma-joined ids.

    ``pairs`` has columns ``s1_id`` and ``t_id``; S1s without pairs get an empty list.
    """
    lists = (
        pairs.select("s1_id", "t_id")
        .unique(maintain_order=True)
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("t_id").str.join(","))
    )
    out = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(lists, on="s1_id", how="left", maintain_order="left")
        .with_columns(pl.col("t_id").fill_null(""))
        .rename({"s1_id": header[0], "t_id": header[1]})
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")
    log.info("wrote %s (%d rows)", path, out.height)
