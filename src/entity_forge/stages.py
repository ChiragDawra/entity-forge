"""Pipeline stages: checkpointed, resumable, memory-bounded.

Every stage reads its inputs from and writes its outputs to the paths in
``Settings``. Work is split into partitions (split × country) and, for the big
stages, into parts of whole S1 groups, so that

* peak RAM is set by the chunk sizes of the resource profile, never by the
  size of the data (no stage loads a full candidate/feature table);
* every finished partition / part / model is checkpointed (``checkpoints.py``)
  and is skipped on the next run; an interrupted stage resumes at the first
  missing part;
* a checkpoint is only reused if the *semantic* configuration that produced it
  is unchanged (config hash); expensive legacy artifacts without manifests are
  validated and adopted instead of recomputed.

Data flow (notebook in brackets)::

    dataset/*.tsv
      [01] run_normalize        norm/{split}_s{1,2,3}.parquet
      [02] run_candidates       candidates_raw/{split}/{country}.parquet
      [02] run_prune            candidates/{split}/{country}/part-*.parquet     (top-N per S1)
      [03] run_features         features/{split}/{country}/part-*.parquet
      [04] run_stage1_cv        models/stage1_fold*.txt, scores/train/stage1/{country}/
      [04] run_stage2_cv        stage2_features/train/{country}/, models/stage2_fold*.txt,
                                scores/train/stage2/{country}/
      [04] run_decision_tuning  models/decision.json
      [05] run_predict_test     scores/test/stage{1,2}/{country}/
      [05] run_write_submission output/{matching_results,candidate_pairs}.tsv

Forcing: ``EF_FORCE=stage[,stage]`` (or ``all``) makes the named stages
recompute even if their checkpoints are valid (``pipeline.py --force``).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import checkpoints as ck
from . import decision, model, pruning
from .blocking import Channel, generate_candidates_to_file
from .features import RECORD_COLUMNS, context_features, pair_features, target_context
from .io import CANDIDATE_HEADER, MATCH_HEADER, read_ground_truth_pairs, read_source, read_tsv, write_id_lists
from .metrics import summary
from .normalize import normalize_records
from .resources import GB, guard, log_mem
from .second_stage import query_context, stage2_features
from .second_stage import target_context as p1_target_context
from .settings import Settings
from .text_vectors import _HASH_SEED, N_BITS, FieldSpec

log = logging.getLogger("entity_forge")

SPLITS = ("train", "test")
NORM_CHUNK = 1_000_000
STAGES = ("normalize", "candidates", "prune", "features", "stage1", "stage2", "decision", "predict", "submit")

# Bump a stage's version when its output semantics change: its checkpoints (and
# everything downstream) then stop matching and are recomputed.
VERSIONS = {"normalize": 1, "candidates": 1, "stage0": 2, "prune": 2, "features": 2, "stage1": 2,
            "stage2": 2, "decision": 2}

# Measured bytes per row on disk (dev slice, EXP-008), used for disk checks.
_DISK_BYTES = {"raw": 22, "pruned": 26, "features": 73, "stage2": 105, "scores": 14}


# ---------------------------------------------------------------------------
# Logging and small helpers
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


class timer:
    """Context manager that logs and records elapsed seconds."""

    def __init__(self, what: str):
        self.what = what
        self.seconds = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        log.info("%s: %.1fs", self.what, self.seconds)


def forced(stage: str) -> bool:
    names = {x.strip() for x in os.environ.get("EF_FORCE", "").split(",") if x.strip()}
    return stage in names or "all" in names


def _skip(target: Path, config_hash: str, what: str) -> bool:
    ok, reason = ck.check(target, config_hash)
    if ok:
        log.info("skip %s: checkpoint valid", what)
    elif reason != "no manifest":
        log.info("%s: recompute (%s)", what, reason)
    return ok


def countries(s: Settings, split: str) -> list[str]:
    return sorted(pl.scan_parquet(s.norm(split, 1)).select("ckey").unique().collect()["ckey"].to_list())


def load_records(s: Settings, split: str, ckey: str | None = None,
                 columns: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(queries, targets) of one split, optionally one country. Row order is the index (q / t)."""
    cols = columns or ["ckey", *RECORD_COLUMNS]
    q = pl.scan_parquet(s.norm(split, 1))
    t = pl.concat([pl.scan_parquet(s.norm(split, src)) for src in (2, 3)])
    if ckey is not None:
        q = q.filter(pl.col("ckey") == ckey)
        t = t.filter(pl.col("ckey") == ckey)
    return q.select(cols).collect(), t.select(cols).collect()


def _n_queries(s: Settings, split: str, ckey: str) -> int:
    return int(pl.scan_parquet(s.norm(split, 1)).filter(pl.col("ckey") == ckey).select(pl.len()).collect().item())


# ---------------------------------------------------------------------------
# Configuration hashes (semantic settings only: never threads or chunk sizes)
# ---------------------------------------------------------------------------


def make_channels(s: Settings) -> tuple[Channel, ...]:
    """Retrieval channels, chosen from measured recall (docs/EXPERIMENTS.md, EXP-002..006)."""
    name_words = FieldSpec("name_core", "word")
    return (
        Channel("addr", (FieldSpec("addr_norm", "word"),), s.top_k_addr, max_df_frac=0.02),
        Channel("both", (FieldSpec("name_core", "word", 0.5), FieldSpec("addr_norm", "word", 0.5)),
                s.top_k_both, max_df_frac=0.02),
        Channel("cross", (FieldSpec("name_core", "cross", column2="addr_norm"),), s.top_k_cross,
                max_df_frac=0.01),
        Channel("name", (name_words,), s.top_k_name, max_df_frac=0.01),
        Channel("noaddr", (name_words,), s.top_k_noaddr, max_df_frac=0.05, target_filter="no_addr"),
        Channel("translit", (FieldSpec("name_phon", "cross", column2="addr_norm"),), s.top_k_translit,
                max_df_frac=0.05, target_filter="translit"),
    )


def norm_hash(s: Settings, split: str) -> str:
    dev = {"s1": s.dev_s1_frac, "t": s.dev_target_frac} if s.dev_mode else None
    return ck.fingerprint({"v": VERSIONS["normalize"], "split": split, "dev": dev})


def candidates_hash(s: Settings, split: str) -> str:
    return ck.fingerprint({
        "v": VERSIONS["candidates"], "norm": norm_hash(s, split),
        "channels": [dataclasses.asdict(c) for c in make_channels(s)], "bits": N_BITS, "seed": _HASH_SEED,
    })


def stage0_hash(s: Settings) -> str:
    return ck.fingerprint({"v": VERSIONS["stage0"], "raw": candidates_hash(s, "train"),
                           "params": pruning.STAGE0_PARAMS, "frac": s.stage0_train_frac})


def prune_hash(s: Settings, split: str) -> str:
    return ck.fingerprint({"v": VERSIONS["prune"], "stage0": stage0_hash(s),
                           "raw": candidates_hash(s, split), "max": s.max_candidates})


def features_hash(s: Settings, split: str) -> str:
    return ck.fingerprint({"v": VERSIONS["features"], "prune": prune_hash(s, split)})


def model_hash(s: Settings, stage: int) -> str:
    prev = model_hash(s, 1) if stage == 2 else None
    return ck.fingerprint({"v": VERSIONS[f"stage{stage}"], "features": features_hash(s, "train"),
                           "params": model.DEFAULT_PARAMS, "folds": s.n_folds, "seed": s.seed, "prev": prev})


def scores_hash(s: Settings, split: str, stage: int) -> str:
    prev = scores_hash(s, split, 1) if stage == 2 else None
    return ck.fingerprint({"model": model_hash(s, stage), "features": features_hash(s, split), "prev": prev})


def decision_hash(s: Settings) -> str:
    return ck.fingerprint({"v": VERSIONS["decision"], "scores": scores_hash(s, "train", 2),
                           "scores1": scores_hash(s, "train", 1)})


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------


def _dev_sample(s: Settings, split: str, raw: dict[int, pl.DataFrame]) -> dict[int, pl.DataFrame]:
    """Consistent small slice: S1 sample + (train) its true targets + random distractors."""
    s1 = raw[1].filter(model.sample_expr(pl.col("entity_id"), s.dev_s1_frac))
    keep = pl.Series("t_id", [], dtype=pl.String)
    if split == "train":
        gt = read_ground_truth_pairs(s.ground_truth)
        keep = gt.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")["t_id"]
    out = {1: s1}
    for src in (2, 3):
        mask = model.sample_expr(pl.col("entity_id"), s.dev_target_frac, salt=src) | pl.col(
            "entity_id").is_in(keep.implode())
        out[src] = raw[src].filter(mask)
    return out


def _tsv_rows(path: Path) -> int:
    return int(pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)
               .select(pl.len()).collect().item())


_NORM_COLUMNS = ["entity_id", "country", "ckey", *[c for c in RECORD_COLUMNS if c != "entity_id"]]


def _adopt_norm(s: Settings, split: str, h: str) -> bool:
    """Validate normalized files made before manifests existed; adopt them if they pass."""
    for src in (1, 2, 3):
        path = s.norm(split, src)
        schema = pl.read_parquet_schema(path)
        missing = [c for c in _NORM_COLUMNS if c not in schema]
        rows = ck.parquet_rows([path])
        raw_rows = _tsv_rows(s.raw(split, src))
        if missing or rows != raw_rows:
            log.warning("cannot adopt %s (missing columns %s, rows %d vs raw %d)", path, missing, rows, raw_rows)
            return False
    for src in (1, 2, 3):
        ck.write_manifest(s.norm(split, src), "normalize", f"{split}/s{src}", h, adopted=True)
    log.info("adopted legacy normalized files for %s (schema and row counts validated)", split)
    return True


def run_normalize(s: Settings, splits: tuple[str, ...] = SPLITS) -> pl.DataFrame:
    """Normalize all sources; add ``name_freq`` (records sharing the name_core, per country)."""
    stats = []
    for split in splits:
        h = norm_hash(s, split)
        targets = [s.norm(split, src) for src in (1, 2, 3)]
        if forced("normalize"):
            for t in targets:
                ck.invalidate(t)
        if all(ck.is_complete(t, h) for t in targets):
            log.info("skip normalize %s: checkpoints valid", split)
        elif (not s.dev_mode and not forced("normalize") and all(t.is_file() for t in targets)
              and all(ck.read_manifest(t) is None for t in targets) and _adopt_norm(s, split, h)):
            pass
        else:
            _normalize_split(s, split, h)
        for src, t in enumerate(targets, 1):
            m = ck.read_manifest(t) or {}
            stats.append({"split": split, "source": src, "rows": m.get("rows"), "adopted": m.get("adopted")})
    return pl.DataFrame(stats)


def _normalize_split(s: Settings, split: str, h: str) -> None:
    with timer(f"normalize {split}"):
        tmp = s.tmp / "norm"
        tmp.mkdir(parents=True, exist_ok=True)
        raw = {src: read_source(s.raw(split, src)) for src in (1, 2, 3)}
        if s.dev_mode:
            raw = _dev_sample(s, split, raw)
        for src in (1, 2, 3):
            df = raw.pop(src)
            normed = pl.concat([normalize_records(df.slice(i, NORM_CHUNK)) for i in range(0, df.height, NORM_CHUNK)])
            ck.atomic_write_parquet(normed, tmp / f"{split}_s{src}.parquet")
            del df, normed
            gc.collect()
            log_mem(f"normalized {split}/s{src}")
        parts = [tmp / f"{split}_s{src}.parquet" for src in (1, 2, 3)]
        freq = (pl.concat([pl.scan_parquet(p).select("ckey", "name_core") for p in parts])
                .group_by("ckey", "name_core").len("name_freq").collect(engine="streaming"))
        for src, p in enumerate(parts, 1):
            df = pl.read_parquet(p).join(freq, on=["ckey", "name_core"], how="left", maintain_order="left")
            ck.atomic_write_parquet(df, s.norm(split, src))
            ck.write_manifest(s.norm(split, src), "normalize", f"{split}/s{src}", h, rows=df.height)
            p.unlink()
            del df


# ---------------------------------------------------------------------------
# 2a. Candidate generation (retrieval)
# ---------------------------------------------------------------------------


def _expected_raw_schema(s: Settings) -> dict[str, pl.DataType]:
    chans = [c.name for c in make_channels(s)]
    schema: dict[str, pl.DataType] = {"q": pl.Int32, "t": pl.Int32}
    schema.update({f"rank_{c}": pl.Int16 for c in chans})
    schema.update({f"cos_{c}": pl.Float32 for c in chans})
    schema.update({"s1_id": pl.String, "t_id": pl.String})
    return schema


def validate_candidates_raw(s: Settings, split: str, ckey: str, path: Path) -> tuple[bool, str, dict]:
    """Cheap-but-thorough checks of a raw candidate file (one streaming pass over int columns)."""
    expected = _expected_raw_schema(s)
    schema = dict(pl.read_parquet_schema(path))
    if list(schema) != list(expected) or any(schema[k] != v for k, v in expected.items()):
        return False, f"schema {schema} != expected {expected}", {}
    chans = list(make_channels(s))
    n_q = _n_queries(s, split, ckey)
    stats = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"), pl.col("q").min().alias("q_min"), pl.col("q").max().alias("q_max"),
            pl.col("q").n_unique().alias("q_unique"),
            pl.sum_horizontal(pl.col("q", "t", "s1_id", "t_id").null_count()).alias("nulls"),
            *[pl.col(f"rank_{c.name}").max().alias(f"max_rank_{c.name}") for c in chans],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    problems = []
    if stats["rows"] == 0:
        problems.append("empty")
    if stats["nulls"]:
        problems.append(f"{stats['nulls']} nulls in key columns")
    if stats["q_min"] != 0 or stats["q_max"] != n_q - 1 or stats["q_unique"] != n_q:
        problems.append(f"q covers {stats['q_unique']} of {n_q} S1 (min {stats['q_min']}, max {stats['q_max']})")
    for c in chans:
        mx = stats[f"max_rank_{c.name}"]
        if mx is not None and mx > c.top_k - 1:
            problems.append(f"rank_{c.name} max {mx} exceeds top_k {c.top_k}")
    stats["n_queries"] = n_q
    return (not problems), "; ".join(problems) or "ok", stats


def run_candidates(s: Settings, split: str) -> pl.DataFrame:
    """Union of all channels, per country; one Parquet file per country (resumable)."""
    channels = make_channels(s)
    h = candidates_hash(s, split)
    stats = []
    for ckey in countries(s, split):
        path = s.candidates_raw(split, ckey)
        key = f"{split}/{ckey}"
        if forced("candidates"):
            ck.invalidate(path)
        elif _skip(path, h, f"candidates {key}"):
            stats.append({"split": split, "country": ckey, "rows": ck.read_manifest(path)["rows"], "status": "valid"})
            continue
        manifest = ck.read_manifest(path)
        if manifest is None and path.is_file() and not forced("candidates"):
            ok, why, info = validate_candidates_raw(s, split, ckey, path)
            if not ok:
                raise ck.StaleArtifactError(
                    f"{path} exists but failed validation ({why}). Nothing was deleted. Rebuild it with "
                    "`bash scripts/run_notebooks.sh --force candidates` (hours) after checking the cause.")
            ck.write_manifest(path, "candidates", key, h, rows=info["rows"], adopted=True,
                              extra={"n_queries": info["n_queries"], "validated": info})
            log.info("adopted legacy candidates %s: %d rows, all %d S1 present", key, info["rows"], info["n_queries"])
            stats.append({"split": split, "country": ckey, "rows": info["rows"], "status": "adopted"})
            continue
        if manifest is not None and not forced("candidates"):
            raise ck.StaleArtifactError(
                f"{path} was built with a different candidate configuration ({ck.check(path, h)[1]}). "
                "It is kept. Restore the previous top-K/channel settings, or rebuild with "
                "`bash scripts/run_notebooks.sh --force candidates` (hours).")
        queries, targets = load_records(s, split, ckey)
        n_q = queries.height
        ck.require_disk(s.work_dir, n_q * 120 * 3 * _DISK_BYTES["raw"] / GB, s.min_free_disk_gb, f"candidates {key}")
        scratch = s.tmp / "candidates" / split / ck.fingerprint([h, ckey])
        with timer(f"candidates {key} ({n_q} x {targets.height})") as tm:
            if targets.height == 0:
                log.warning("no S2/S3 records for country %r: its S1s get empty lists", ckey)
                ck.atomic_write_parquet(pl.DataFrame(schema=_expected_raw_schema(s)), path)
                rows = 0
            else:
                rows = generate_candidates_to_file(
                    queries, targets, channels, path, scratch, n_threads=s.n_threads,
                    retrieval_chunk=s.retrieval_chunk, union_chunk_rows=s.candidate_chunk_rows)
        ck.write_manifest(path, "candidates", key, h, rows=rows, extra={"n_queries": n_q, "seconds": tm.seconds})
        ck.remove_tree(scratch)
        stats.append({"split": split, "country": ckey, "rows": rows, "status": "built"})
        del queries, targets
        gc.collect()
    return pl.DataFrame(stats)


def _raw_ready(s: Settings, split: str, ckey: str) -> tuple[Path, dict]:
    path = s.candidates_raw(split, ckey)
    if not ck.is_complete(path, candidates_hash(s, split)):
        raise RuntimeError(f"raw candidates for {split}/{ckey} are missing or stale: run the candidates stage first")
    return path, ck.read_manifest(path)


# ---------------------------------------------------------------------------
# 2b. Stage-0 pruning
# ---------------------------------------------------------------------------


def _labeled(cands: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    return cands.join(
        gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "t_id"], how="left",
        maintain_order="left",
    ).with_columns(pl.col("label").fill_null(0))


def _train_stage0(s: Settings) -> tuple[lgb.Booster, list[str]]:
    h = stage0_hash(s)
    if forced("prune"):
        ck.invalidate(s.stage0_model)
    if _skip(s.stage0_model, h, "stage-0 model"):
        return model.load(s.stage0_model)
    gt = read_ground_truth_pairs(s.ground_truth)
    raw = {c: _raw_ready(s, "train", c) for c in countries(s, "train")}
    total = sum(m["rows"] for _, m in raw.values())
    frac = min(s.stage0_train_frac, s.stage0_max_rows / max(total, 1))
    guard("stage-0 training sample", frac * total * 400 / GB)
    samples = []
    for ckey, (path, _) in raw.items():
        t_agg = pruning.target_aggregates_from_files(path)
        sample = (pl.scan_parquet(path).filter(model.sample_expr(pl.col("s1_id"), frac, salt=3))
                  .collect(engine="streaming"))
        samples.append(_labeled(pruning.stage0_features(sample, t_agg), gt))
        log_mem(f"stage-0 sample {ckey}: {sample.height:,} rows")
        del sample, t_agg
    labeled = pl.concat(samples)
    del samples
    with timer(f"stage-0 training on {labeled.height:,} rows"):
        booster, columns = pruning.train_stage0(labeled, s.n_threads)
    model.save(booster, columns, s.stage0_model)
    ck.write_manifest(s.stage0_model, "stage0", "model", h, rows=labeled.height, extra={"frac": frac})
    return booster, columns


def _q_plan(n_q: int, rows: int, target_rows: int) -> list[list[int]]:
    per_q = max(rows / max(n_q, 1), 1.0)
    step = max(1, int(target_rows / per_q))
    return [[a, min(a + step, n_q)] for a in range(0, n_q, step)] or [[0, 0]]


def _q_range(path, a: int, b: int) -> pl.DataFrame:
    return pl.scan_parquet(path).filter((pl.col("q") >= a) & (pl.col("q") < b)).collect()


def run_prune(s: Settings) -> pl.DataFrame:
    """Fit the stage-0 re-ranker on train, keep the top ``max_candidates`` per S1 everywhere."""
    booster, columns = _train_stage0(s)
    stats = []
    for split in SPLITS:
        h = prune_hash(s, split)
        for ckey in countries(s, split):
            out_dir, key = s.candidates(split, ckey), f"{split}/{ckey}"
            if forced("prune"):
                ck.reset_partition(out_dir)
            elif _skip(out_dir, h, f"prune {key}"):
                stats.append({"split": split, "country": ckey, "kept_pairs": ck.read_manifest(out_dir)["rows"]})
                continue
            path, m = _raw_ready(s, split, ckey)
            n_q = m["extra"].get("n_queries") or _n_queries(s, split, ckey)
            ck.require_disk(s.work_dir, m["rows"] * _DISK_BYTES["pruned"] * 0.7 / GB, s.min_free_disk_gb,
                            f"prune {key}")
            chunks = ck.load_or_create_plan(out_dir, h, lambda: _q_plan(n_q, m["rows"], s.candidate_chunk_rows))
            with timer(f"prune {key} ({m['rows']:,} raw rows, {len(chunks)} parts)"):
                t_agg = pruning.target_aggregates_from_files(path)
                for i, (a, b) in enumerate(chunks):
                    part = ck.part_path(out_dir, i)
                    if part.is_file():
                        continue
                    guard(f"prune {key} part {i}", s.candidate_chunk_rows * 400 / GB)
                    chunk = _q_range(path, a, b)
                    kept = pruning.prune(pruning.stage0_features(chunk, t_agg), booster, columns,
                                         s.max_candidates, s.n_threads)
                    ck.atomic_write_parquet(kept.sort("q", "t"), part)
                    if i % 10 == 0:
                        log_mem(f"prune {key} part {i + 1}/{len(chunks)}")
                del t_agg
            man = ck.write_manifest(out_dir, "prune", key, h)
            stats.append({"split": split, "country": ckey, "raw_pairs": m["rows"], "kept_pairs": man["rows"],
                          "kept_per_s1": man["rows"] / max(n_q, 1)})
    return pl.DataFrame(stats)


def candidate_report(s: Settings, split: str = "train", which: str = "pruned",
                     cutoffs: tuple[int, ...] = ()) -> pl.DataFrame:
    """Pair recall of the union, of every channel alone, and (pruned) at stage-0 cut-offs.

    Streaming: only the ground-truth hits and per-S1 counts are materialized.
    """
    gt = read_ground_truth_pairs(s.ground_truth)
    rows = []
    for ckey in countries(s, split):
        source = [s.candidates_raw(split, ckey)] if which == "raw" else ck.data_files(s.candidates(split, ckey))
        if not source or not all(p.is_file() for p in source):
            continue
        s1 = pl.scan_parquet(s.norm(split, 1)).filter(pl.col("ckey") == ckey).select(
            pl.col("entity_id").alias("s1_id")).collect()
        truth = gt.join(s1, on="s1_id", how="semi")
        lf = pl.scan_parquet(source)
        chans = [c.removeprefix("rank_") for c in lf.collect_schema().names() if c.startswith("rank_")]
        per_q = lf.group_by("q").agg(pl.len().alias("union"),
                                     *[(pl.col(f"rank_{c}") >= 0).sum().alias(c) for c in chans]
                                     ).collect(engine="streaming")
        found = (lf.select("q", "s1_id", "t_id", *[f"rank_{c}" for c in chans])
                 .join(truth.lazy(), on=["s1_id", "t_id"], how="inner").collect(engine="streaming"))

        def row(label: str, hits: int, counts: pl.Series) -> dict:
            return {"country": ckey, "set": which, "channel": label, "pair_recall": hits / max(truth.height, 1),
                    "true_pairs": truth.height, "candidates": int(counts.sum()),
                    "avg_per_s1": float(counts.sum()) / max(s1.height, 1),
                    "p95_per_s1": float(counts.quantile(0.95) or 0), "p99_per_s1": float(counts.quantile(0.99) or 0)}

        rows.append(row("union", found.height, per_q["union"]))
        for c in chans:
            rows.append(row(c, int((found[f"rank_{c}"] >= 0).sum()), per_q[c]))
        if which == "pruned" and cutoffs:
            ranks = []
            for part in source:
                p = pl.read_parquet(part, columns=["q", "s1_id", "t_id", "score0"]).with_columns(
                    pl.col("score0").rank("ordinal", descending=True).over("q").alias("r0"))
                ranks.append(p.join(truth, on=["s1_id", "t_id"], how="inner").select("r0"))
            r0 = pl.concat(ranks)["r0"] if ranks else pl.Series("r0", [], dtype=pl.UInt32)
            for n in cutoffs:
                rows.append(row(f"top{n}", int((r0 <= n).sum()), per_q["union"].clip(upper_bound=n)))
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------


def feature_parts(s: Settings, split: str, ckey: str) -> list[Path]:
    return ck.data_files(s.features(split, ckey))


def _pruned_ready(s: Settings, split: str, ckey: str) -> list[Path]:
    d = s.candidates(split, ckey)
    if not ck.is_complete(d, prune_hash(s, split)):
        raise RuntimeError(f"pruned candidates for {split}/{ckey} are missing or stale: run the prune stage first")
    return ck.data_files(d)


def run_features(s: Settings, split: str) -> pl.DataFrame:
    """Stage-1 features for every pruned candidate pair; one feature part per candidate part."""
    h = features_hash(s, split)
    gt = read_ground_truth_pairs(s.ground_truth) if split == "train" else None
    stats = []
    for ckey in countries(s, split):
        out_dir, key = s.features(split, ckey), f"{split}/{ckey}"
        if forced("features"):
            ck.reset_partition(out_dir)
        elif _skip(out_dir, h, f"features {key}"):
            stats.append({"split": split, "country": ckey, "pairs": ck.read_manifest(out_dir)["rows"]})
            continue
        parts = _pruned_ready(s, split, ckey)
        rows = ck.read_manifest(s.candidates(split, ckey))["rows"]
        ck.require_disk(s.work_dir, rows * _DISK_BYTES["features"] / GB, s.min_free_disk_gb, f"features {key}")
        names = ck.load_or_create_plan(out_dir, h, lambda: [p.name for p in parts])
        with timer(f"features {key} ({rows:,} pairs, {len(names)} parts)"):
            guard(f"features {key}: load records", 1.0)
            queries, targets = load_records(s, split, ckey)
            t_ctx = target_context(pl.scan_parquet(parts))
            log_mem(f"features {key}: records + target context loaded")
            for i, name in enumerate(names):
                out = ck.part_path(out_dir, i)
                if out.is_file():
                    continue
                cands = context_features(pl.read_parquet(s.candidates(split, ckey) / name), t_ctx)
                if gt is not None:
                    cands = _labeled(cands, gt)
                pieces = []
                for a in range(0, max(cands.height, 1), s.feature_chunk):
                    guard(f"features {key} part {i}", s.feature_chunk * 600 / GB)
                    pieces.append(pair_features(cands.slice(a, s.feature_chunk), queries, targets, s.n_threads))
                ck.atomic_write_parquet(pl.concat(pieces), out)
                if i % 5 == 0:
                    log_mem(f"features {key} part {i + 1}/{len(names)}")
                del cands, pieces
            del queries, targets, t_ctx
            gc.collect()
        man = ck.write_manifest(out_dir, "features", key, h)
        stats.append({"split": split, "country": ckey, "pairs": man["rows"]})
    return pl.DataFrame(stats)


# ---------------------------------------------------------------------------
# 4. Cross-validated training (stage 1 and stage 2 share this)
# ---------------------------------------------------------------------------


def _train_folds(s: Settings, stage: int, dirs: list[Path], max_rows: int, rounds: int,
                 train_frac: float | None = None) -> None:
    """One model per fold on a row-budgeted sample of S1 groups: one LightGBM Dataset + subsets."""
    h = model_hash(s, stage)
    if forced(f"stage{stage}"):
        for f in range(s.n_folds):
            ck.invalidate(s.model(stage, f))
    todo = [f for f in range(s.n_folds) if not _skip(s.model(stage, f), h, f"stage-{stage} fold {f} model")]
    if not todo:
        return
    parts = [p for d in dirs for p in ck.data_files(d)]
    total = sum(ck.read_manifest(d)["rows"] for d in dirs)
    frac = train_frac if train_frac is not None else min(1.0, max_rows / max(total, 1))
    columns = [c for c in pl.read_parquet_schema(parts[0]) if c not in model.NON_FEATURES]
    with timer(f"stage-{stage} sample ({frac:.3f} of {total:,} rows)"):
        x, y, s1 = _training_matrix(parts, columns, frac, salt=stage, what=f"stage-{stage}")
    fold = s1.select(model.fold_expr(pl.col("s1_id"), s.n_folds)).to_series().to_numpy()
    params = {**model.DEFAULT_PARAMS, "num_threads": s.n_threads, "seed": s.seed + stage}
    ds = lgb.Dataset(x, label=y, feature_name=columns, params=params, free_raw_data=True)
    ds.construct()
    del x
    gc.collect()
    log_mem(f"stage-{stage} dataset built ({len(fold):,} rows x {len(columns)} features)")
    s.reports.mkdir(parents=True, exist_ok=True)
    for f in todo:
        es = s1.select(model.sample_expr(pl.col("s1_id"), 0.08, salt=100 + f)).to_series().to_numpy()
        tr = np.flatnonzero((fold != f) & ~es)
        va = np.flatnonzero((fold != f) & es)
        with timer(f"stage-{stage} fold {f} ({len(tr):,} train / {len(va):,} early-stop rows)"):
            booster = model.train_on_dataset(params, ds.subset(tr.tolist()), ds.subset(va.tolist()), rounds)
        model.save(booster, columns, s.model(stage, f))
        ck.write_manifest(s.model(stage, f), f"stage{stage}", f"fold{f}", h, rows=len(tr),
                          extra={"frac": frac, "best_iteration": booster.best_iteration})
        model.importance(booster).write_csv(s.reports / f"importance_stage{stage}_fold{f}.csv")
        del booster
        gc.collect()
    del ds


def _training_matrix(parts: list[Path], columns: list[str], frac: float, salt: int,
                     what: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Sampled rows as one preallocated float32 matrix, filled part by part.

    Peak memory = the matrix + one part (a DataFrame and its NumPy copy never
    coexist at full size).
    """
    mask = model.sample_expr(pl.col("s1_id"), frac, salt=salt)
    n = int(pl.scan_parquet(parts).filter(mask).select(pl.len()).collect(engine="streaming").item())
    guard(f"{what} training matrix ({n:,} rows x {len(columns)})", n * len(columns) * 5.5 / GB)
    x = np.empty((n, len(columns)), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    ids = []
    at = 0
    for part in parts:
        df = pl.scan_parquet(part).filter(mask).select("s1_id", "label", *columns).collect()
        m = df.height
        if m:
            x[at : at + m] = model.to_matrix(df, columns)
            y[at : at + m] = df["label"].to_numpy()
            ids.append(df.select("s1_id"))
            at += m
        del df
    if at != n:
        raise AssertionError(f"{what}: sampled {at} rows, expected {n}")
    return x, y, (pl.concat(ids) if ids else pl.DataFrame({"s1_id": pl.Series([], dtype=pl.String)}))


def _load_models(s: Settings, stage: int) -> list[tuple[lgb.Booster, list[str]]]:
    h = model_hash(s, stage)
    for f in range(s.n_folds):
        if not ck.is_complete(s.model(stage, f), h):
            raise RuntimeError(f"stage-{stage} fold {f} model missing or stale: run the stage{stage} stage first")
    return [model.load(s.model(stage, f)) for f in range(s.n_folds)]


def _predict(s: Settings, models, frame: pl.DataFrame, oof: bool) -> np.ndarray:
    """Out-of-fold (train) or fold-average (test) probabilities."""
    if not oof:
        p = np.zeros(frame.height, dtype=np.float32)
        for booster, columns in models:
            p += model.predict(booster, frame, columns, s.n_threads)
        return p / np.float32(len(models))
    fold = frame.select(model.fold_expr(pl.col("s1_id"), s.n_folds)).to_series().to_numpy()
    p = np.zeros(frame.height, dtype=np.float32)
    for f, (booster, columns) in enumerate(models):
        idx = np.flatnonzero(fold == f)
        if len(idx):
            p[idx] = model.predict(booster, frame[idx], columns, s.n_threads)
    return p


def _score_partition(s: Settings, stage: int, split: str, ckey: str, in_dir: Path, h: str, models,
                     make_frame=None) -> None:
    """Score every part of ``in_dir`` into scores/{split}/stage{stage}/{ckey} (1:1 parts)."""
    out_dir, key = s.scores(split, stage, ckey), f"{split}/{ckey}"
    if forced(f"stage{stage}") or (split == "test" and forced("predict")):
        ck.reset_partition(out_dir)
    elif _skip(out_dir, h, f"stage-{stage} scores {key}"):
        return
    parts = ck.data_files(in_dir)
    names = ck.load_or_create_plan(out_dir, h, lambda: [p.name for p in parts])
    keep = ["q", "t"] + (["label"] if split == "train" else [])
    with timer(f"stage-{stage} scoring {key} ({len(names)} parts)"):
        for i, name in enumerate(names):
            out = ck.part_path(out_dir, i)
            if out.is_file():
                continue
            frame = pl.read_parquet(in_dir / name)
            if make_frame is not None:
                frame = make_frame(i, frame)
            p = _predict(s, models, frame, oof=(split == "train"))
            cols = keep + (["p1"] if stage == 2 else [])
            ck.atomic_write_parquet(frame.select(cols).with_columns(pl.Series("p1" if stage == 1 else "p", p)), out)
    ck.write_manifest(out_dir, f"scores{stage}", key, h)


def run_stage1_cv(s: Settings, train_frac: float | None = None, rounds: int = 3000) -> pl.DataFrame:
    """Stage 1: fit fold models on a row-budgeted S1 sample, then OOF-score *every* train pair."""
    dirs = []
    for ckey in countries(s, "train"):
        d = s.features("train", ckey)
        if not ck.is_complete(d, features_hash(s, "train")):
            raise RuntimeError(f"train features for {ckey} missing or stale: run the features stage first")
        dirs.append(d)
    _train_folds(s, 1, dirs, s.stage1_max_rows, rounds, train_frac)
    models = _load_models(s, 1)
    h = scores_hash(s, "train", 1)
    for ckey in countries(s, "train"):
        _score_partition(s, 1, "train", ckey, s.features("train", ckey), h, models)
    return scores_summary(s, "train", 1)


def _stage2_frame_builder(s: Settings, split: str, ckey: str):
    """Returns f(i, features_part) -> stage-2 frame, using the stage-1 scores of the same part."""
    sdir = s.scores(split, 1, ckey)
    scores = pl.scan_parquet(ck.data_files(sdir)).select("q", "t", "p1")
    q_ctx = query_context(scores)
    t_ctx = p1_target_context(scores)
    _, targets = load_records(s, split, ckey, columns=["name_core", "addr_norm"])

    def build(i: int, frame: pl.DataFrame) -> pl.DataFrame:
        p1 = pl.read_parquet(ck.part_path(sdir, i), columns=["q", "t", "p1"])
        if not (p1["q"].equals(frame["q"]) and p1["t"].equals(frame["t"])):
            raise AssertionError(f"stage-1 scores part {i} of {split}/{ckey} is not aligned with its features")
        return stage2_features(frame.with_columns(p1["p1"]), q_ctx, t_ctx, targets, s.n_threads)

    return build


def run_stage2_cv(s: Settings, train_frac: float | None = None, rounds: int = 3000) -> pl.DataFrame:
    """Stage 2 on OOF stage-1 probabilities; OOF-scores every train pair."""
    h1 = scores_hash(s, "train", 1)
    h_feat = ck.fingerprint({"stage2_features": h1})
    for ckey in countries(s, "train"):
        if not ck.is_complete(s.scores("train", 1, ckey), h1):
            raise RuntimeError(f"stage-1 train scores for {ckey} missing or stale: run the stage1 stage first")
        out_dir, key = s.stage2_features(ckey), f"train/{ckey}"
        if forced("stage2"):
            ck.reset_partition(out_dir)
        elif _skip(out_dir, h_feat, f"stage-2 features {key}"):
            continue
        in_dir = s.features("train", ckey)
        rows = ck.read_manifest(in_dir)["rows"]
        ck.require_disk(s.work_dir, rows * _DISK_BYTES["stage2"] / GB, s.min_free_disk_gb, f"stage-2 features {key}")
        names = ck.load_or_create_plan(out_dir, h_feat, lambda: [p.name for p in ck.data_files(in_dir)])
        build = _stage2_frame_builder(s, "train", ckey)
        with timer(f"stage-2 features {key}"):
            for i, name in enumerate(names):
                out = ck.part_path(out_dir, i)
                if not out.is_file():
                    ck.atomic_write_parquet(build(i, pl.read_parquet(in_dir / name)), out)
        ck.write_manifest(out_dir, "stage2_features", key, h_feat)
        del build
        gc.collect()
    dirs = [s.stage2_features(c) for c in countries(s, "train")]
    _train_folds(s, 2, dirs, s.stage2_max_rows, rounds, train_frac)
    models = _load_models(s, 2)
    h = scores_hash(s, "train", 2)
    for ckey in countries(s, "train"):
        _score_partition(s, 2, "train", ckey, s.stage2_features(ckey), h, models)
    return scores_summary(s, "train", 2)


def scores_summary(s: Settings, split: str, stage: int) -> pl.DataFrame:
    col = "p1" if stage == 1 else "p"
    rows = []
    for ckey in countries(s, split):
        files = ck.data_files(s.scores(split, stage, ckey))
        if not files:
            continue
        lf = pl.scan_parquet(files)
        if split == "train":
            agg = lf.group_by("label").agg(pl.col(col).mean().alias(f"mean_{col}"), pl.len())
        else:
            agg = lf.select(pl.col(col).mean().alias(f"mean_{col}"), pl.len())
        rows.append(agg.collect(engine="streaming").with_columns(pl.lit(ckey).alias("country")))
    return pl.concat(rows, how="diagonal") if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# 5. Decision tuning on out-of-fold scores
# ---------------------------------------------------------------------------


def _scores_with_ids(s: Settings, split: str, stage: int, score_col: str, min_p: float) -> pl.DataFrame:
    """(s1_id, t_id, p) for pairs with p >= min_p, all countries; only this small set is materialized."""
    out = []
    for ckey in countries(s, split):
        files = ck.data_files(s.scores(split, stage, ckey))
        if not files:
            continue
        pairs = (pl.scan_parquet(files).filter(pl.col(score_col) >= min_p).select("q", "t", score_col)
                 .collect(engine="streaming"))
        q_ids, t_ids = load_records(s, split, ckey, columns=["entity_id"])
        out.append(pl.DataFrame({
            "s1_id": q_ids["entity_id"].gather(pairs["q"]),
            "t_id": t_ids["entity_id"].gather(pairs["t"]),
            "p": pairs[score_col],
        }))
    if not out:
        return pl.DataFrame(schema={"s1_id": pl.String, "t_id": pl.String, "p": pl.Float32})
    return pl.concat(out)


def decide(pred: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Apply a decision rule (see decision.py) to (s1_id, t_id, p)."""
    if cfg["exclusive"]:
        pred = decision.resolve_exclusive(pred)
    if cfg["method"] == "threshold":
        return decision.select_threshold(pred, cfg["tau"])
    return decision.select_expected_f(pred, missed=cfg["missed"], floor=cfg["floor"])


def run_decision_tuning(s: Settings, score_col: str = "p") -> tuple[dict, pl.DataFrame]:
    """Grid over decision rules on OOF train scores with the exact macro metric.

    ``score_col="p1"`` evaluates stage-1 only (ablation) and never overwrites the frozen rule.
    """
    stage = 2 if score_col == "p" else 1
    final = score_col == "p"
    h = decision_hash(s)
    grid_file = s.reports / f"decision_grid{'' if final else '_stage1'}.csv"
    if final and not forced("decision") and _skip(s.decision_file, h, "decision rule"):
        cfg = json.loads(s.decision_file.read_text())
        return cfg, pl.read_csv(grid_file) if grid_file.is_file() else pl.DataFrame()
    pred = _scores_with_ids(s, "train", stage, score_col, 0.01)
    s1_ids = pl.read_parquet(s.norm("train", 1), columns=["entity_id"])["entity_id"]
    truth = read_ground_truth_pairs(s.ground_truth).join(pl.DataFrame({"s1_id": s1_ids}), on="s1_id", how="semi")
    log_mem(f"decision tuning ({pred.height:,} scored pairs >= 0.01)")
    grid: list[dict] = [{"method": "threshold", "exclusive": ex, "tau": round(float(t), 3)}
                        for ex in (False, True) for t in np.arange(0.20, 0.86, 0.025)]
    grid += [{"method": "expected_f", "exclusive": True, "missed": m, "floor": 0.02} for m in (0.0, 0.05, 0.1, 0.2, 0.3)]
    resolved = {False: pred, True: decision.resolve_exclusive(pred)}
    rows = [{**cfg, **summary(s1_ids, decide(resolved[cfg["exclusive"]], {**cfg, "exclusive": False}), truth)}
            for cfg in grid]
    table = pl.DataFrame(rows).sort("macro_f05", descending=True)
    best = {k: v for k, v in table.row(0, named=True).items()
            if k in ("method", "exclusive", "tau", "missed", "floor") and v is not None}
    best["score_col"] = score_col
    best["oof_macro_f05"] = float(table["macro_f05"][0])
    s.reports.mkdir(parents=True, exist_ok=True)
    table.write_csv(grid_file)
    if final:
        s.decision_file.parent.mkdir(parents=True, exist_ok=True)
        s.decision_file.write_text(json.dumps(best, indent=2))
        ck.write_manifest(s.decision_file, "decision", "rule", h)
    return best, table


def _frozen_decision(s: Settings) -> dict:
    if not ck.is_complete(s.decision_file, decision_hash(s)):
        raise RuntimeError("decision rule missing or stale: run the decision stage first")
    return json.loads(s.decision_file.read_text())


def oof_report(s: Settings) -> pl.DataFrame:
    """Per-country OOF metrics of the frozen decision rule."""
    cfg = _frozen_decision(s)
    stage = 2 if cfg["score_col"] == "p" else 1
    chosen = decide(_scores_with_ids(s, "train", stage, cfg["score_col"], 0.01), cfg)
    s1 = pl.read_parquet(s.norm("train", 1), columns=["entity_id", "ckey"])
    gt = read_ground_truth_pairs(s.ground_truth)
    rows = []
    for ckey in ["ALL", *sorted(s1["ckey"].unique().to_list())]:
        ids = (s1 if ckey == "ALL" else s1.filter(pl.col("ckey") == ckey))["entity_id"]
        keep = pl.DataFrame({"s1_id": ids})
        rows.append({"country": ckey, **summary(ids, chosen.join(keep, on="s1_id", how="semi"),
                                                 gt.join(keep, on="s1_id", how="semi"))})
    return pl.DataFrame(rows)


def error_examples(s: Settings, n: int = 15, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Samples of OOF false positives (wrong merges) and false negatives, with raw text."""
    cfg = _frozen_decision(s)
    stage = 2 if cfg["score_col"] == "p" else 1
    scored = _scores_with_ids(s, "train", stage, cfg["score_col"], 0.01)
    chosen = decide(scored, cfg)
    gt = read_ground_truth_pairs(s.ground_truth)
    fp = chosen.join(gt, on=["s1_id", "t_id"], how="anti")
    fn = gt.join(chosen, on=["s1_id", "t_id"], how="anti").join(scored, on=["s1_id", "t_id"], how="left")
    fp, fn = (x.sample(min(n, x.height), seed=seed) for x in (fp, fn))
    cols = ["entity_id", "business_name", "business_address"]
    ids = pl.concat([fp.select("s1_id", "t_id"), fn.select("s1_id", "t_id")])
    s1 = (pl.scan_parquet(s.norm("train", 1)).select(cols)
          .filter(pl.col("entity_id").is_in(ids["s1_id"].implode())).collect())
    tg = (pl.concat([pl.scan_parquet(s.norm("train", i)).select(cols) for i in (2, 3)])
          .filter(pl.col("entity_id").is_in(ids["t_id"].implode())).collect())
    s1 = s1.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
    tg = tg.rename({"entity_id": "t_id", "business_name": "t_name", "business_address": "t_addr"})

    def view(pairs: pl.DataFrame) -> pl.DataFrame:
        return (pairs.join(s1, on="s1_id", how="left").join(tg, on="t_id", how="left")
                .select("p", "s1_name", "t_name", "s1_addr", "t_addr"))

    return view(fp), view(fn)


# ---------------------------------------------------------------------------
# 6. Test inference and submission
# ---------------------------------------------------------------------------


def run_predict_test(s: Settings) -> pl.DataFrame:
    """Stage-1 and stage-2 scores for every test candidate (fold-model average), part by part."""
    for ckey in countries(s, "test"):
        if not ck.is_complete(s.features("test", ckey), features_hash(s, "test")):
            raise RuntimeError(f"test features for {ckey} missing or stale: run the features stage first")
    models1 = _load_models(s, 1)
    for ckey in countries(s, "test"):
        _score_partition(s, 1, "test", ckey, s.features("test", ckey), scores_hash(s, "test", 1), models1)
    del models1
    models2 = _load_models(s, 2)
    for ckey in countries(s, "test"):
        _score_partition(s, 2, "test", ckey, s.features("test", ckey), scores_hash(s, "test", 2), models2,
                         make_frame=_stage2_frame_builder(s, "test", ckey))
    return scores_summary(s, "test", 2)


def _write_candidate_lists(s: Settings, s1_ids: pl.Series, path: Path) -> int:
    """candidate_pairs.tsv streamed part by part (never all pairs in RAM)."""
    tmp = path.with_name(path.name + ".tmp")
    written = []
    total = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\t".join(CANDIDATE_HEADER) + "\n")
        for ckey in countries(s, "test"):
            q_ids, t_ids = load_records(s, "test", ckey, columns=["entity_id"])
            for part in _pruned_ready(s, "test", ckey):
                pairs = pl.read_parquet(part, columns=["q", "t"])
                total += pairs.height
                grouped = (pairs.with_columns(t_ids["entity_id"].gather(pairs["t"]).alias("t_id"))
                           .group_by("q", maintain_order=True).agg(pl.col("t_id").str.join(",")))
                lists = pl.DataFrame({"s1": q_ids["entity_id"].gather(grouped["q"]), "t_id": grouped["t_id"]})
                fh.write(lists.write_csv(separator="\t", include_header=False, quote_style="never"))
                written.append(lists["s1"])
        done = pl.concat(written) if written else pl.Series("s1", [], dtype=pl.String)
        for sid in s1_ids.filter(~s1_ids.is_in(done.implode())).to_list():
            fh.write(f"{sid}\t\n")
    os.replace(tmp, path)
    return total


def run_write_submission(s: Settings) -> dict:
    """Apply the frozen decision rule and write both competition TSVs."""
    cfg = _frozen_decision(s)
    stage = 2 if cfg["score_col"] == "p" else 1
    h = scores_hash(s, "test", stage)
    for ckey in countries(s, "test"):
        if not ck.is_complete(s.scores("test", stage, ckey), h):
            raise RuntimeError(f"test stage-{stage} scores for {ckey} missing or stale: run the predict stage first")
        scored_rows = ck.read_manifest(s.scores("test", stage, ckey))["rows"]
        cand_rows = ck.read_manifest(s.candidates("test", ckey))["rows"]
        if scored_rows != cand_rows:
            raise AssertionError(f"{ckey}: {scored_rows} scored pairs but {cand_rows} candidates")
    # The raw file is authoritative for which S1 rows must exist (all of them).
    s1_ids = read_tsv(s.raw("test", 1), ("entity_id",))["entity_id"]
    normed = pl.read_parquet(s.norm("test", 1), columns=["entity_id"])["entity_id"]
    if not s.dev_mode and normed.len() != s1_ids.len():
        raise AssertionError(f"normalized test S1 has {normed.len()} rows, raw file {s1_ids.len()}")
    chosen = decide(_scores_with_ids(s, "test", stage, cfg["score_col"], 0.01), cfg)
    if cfg["exclusive"] and chosen["t_id"].n_unique() != chosen.height:
        raise AssertionError("exclusive decision produced a target under two S1s")
    unknown = chosen.join(pl.DataFrame({"s1_id": s1_ids}), on="s1_id", how="anti").height
    if unknown:
        raise AssertionError(f"{unknown} predictions for unknown S1 ids")
    s.output_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1_ids, chosen, s.output_dir / "matching_results.tsv", MATCH_HEADER)
    n_cands = _write_candidate_lists(s, s1_ids, s.output_dir / "candidate_pairs.tsv")
    with_matches = chosen["s1_id"].n_unique()
    return {"s1_rows": s1_ids.len(), "predicted_pairs": chosen.height, "s1_with_matches": with_matches,
            "empty_rate": 1 - with_matches / max(s1_ids.len(), 1), "candidate_pairs": n_cands, "decision": cfg}


def run_validator(s: Settings, check_ids: bool = False) -> str:
    """Run the official stdlib validator on the written files."""
    cmd = [
        sys.executable, str(s.root / "utils" / "validate_submission.py"),
        "--matching", str(s.output_dir / "matching_results.tsv"),
        "--candidate", str(s.output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(s.data_dir / "test"),
    ]
    if check_ids:
        cmd.append("--check-ids")
    res = subprocess.run(cmd, capture_output=True, text=True)
    text = res.stdout + res.stderr
    if res.returncode != 0:
        raise RuntimeError(text)
    return text


def build_submission_zip(s: Settings) -> Path:
    """``<team>_submission.zip`` in the layout the challenge asks for."""
    dest = s.root / "submission" / f"{s.team_name}_submission.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    code = "code/business_entity_resolution"
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(s.output_dir / name, f"output/{name}")
        for path in sorted((s.root / "src" / "entity_forge").glob("*.py")):
            z.write(path, f"{code}/src/entity_forge/{path.name}")
        for path in sorted((s.root / "notebooks").glob("*.ipynb")):
            z.write(path, f"{code}/src/notebooks/{path.name}")
        z.write(s.root / "scripts" / "run_notebooks.sh", f"{code}/src/scripts/run_notebooks.sh")
        z.write(s.root / "utils" / "validate_submission.py", f"{code}/src/utils/validate_submission.py")
        z.write(s.root / "README.md", f"{code}/README.md")
        z.write(s.root / "requirements.txt", f"{code}/requirements.txt")
        z.write(s.root / "docs" / "METHODOLOGY.md", "Documentation_template.md")
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_rows(s: Settings) -> list[dict]:
    """Checkpoint state of every stage partition (for ``pipeline.py status``)."""
    rows: list[dict] = []

    def add(stage: str, key: str, target: Path, h: str) -> None:
        ok, reason = ck.check(target, h)
        if ok:
            state = "done"
        elif reason != "no manifest":
            state = f"stale: {reason}"
        elif not ck.data_files(target):
            state = "missing"
        elif ck.is_dir_target(target):
            state = "partial (resumes)" if (target / ck.PLAN).is_file() else "old parts (recomputed)"
        elif stage in ("normalize", "candidates"):
            state = "legacy (validated + adopted on run)"
        else:
            state = "legacy (recomputed)"
        m = ck.read_manifest(target) or {}
        rows.append({"stage": stage, "partition": key, "state": state, "rows": m.get("rows"),
                     "adopted": bool(m.get("adopted", False))})

    splits_c: dict[str, list[str]] = {}
    for split in SPLITS:
        try:
            splits_c[split] = countries(s, split)
        except (FileNotFoundError, OSError, pl.exceptions.PolarsError):
            pass
    for split in SPLITS:
        for src in (1, 2, 3):
            add("normalize", f"{split}/s{src}", s.norm(split, src), norm_hash(s, split))
    for split, cs in splits_c.items():
        for c in cs:
            add("candidates", f"{split}/{c}", s.candidates_raw(split, c), candidates_hash(s, split))
    add("prune", "stage0 model", s.stage0_model, stage0_hash(s))
    for split, cs in splits_c.items():
        for c in cs:
            add("prune", f"{split}/{c}", s.candidates(split, c), prune_hash(s, split))
    for split, cs in splits_c.items():
        for c in cs:
            add("features", f"{split}/{c}", s.features(split, c), features_hash(s, split))
    for stage in (1, 2):
        for f in range(s.n_folds):
            add(f"stage{stage}", f"fold{f} model", s.model(stage, f), model_hash(s, stage))
        for c in splits_c.get("train", []):
            add(f"stage{stage}", f"train/{c} scores", s.scores("train", stage, c), scores_hash(s, "train", stage))
    add("decision", "rule", s.decision_file, decision_hash(s))
    for stage in (1, 2):
        for c in splits_c.get("test", []):
            add("predict", f"test/{c} stage{stage}", s.scores("test", stage, c), scores_hash(s, "test", stage))
    return rows
