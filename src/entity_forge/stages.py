"""Pipeline stages. Each notebook calls one or two of these.

Every stage reads its inputs from, and writes its outputs to, the paths in
``Settings``; stages are therefore resumable and can run on different machines
(e.g. normalize + candidates locally, features + training on SageMaker).

Data flow::

    dataset/*.tsv
      -> run_normalize        artifacts/norm/{split}_s{1,2,3}.parquet
      -> run_candidates       artifacts/candidates_raw/{split}/{country}.parquet
      -> run_prune            artifacts/candidates/{split}/{country}.parquet (top-N per S1)
      -> run_features         artifacts/features/{split}/{country}/part-*.parquet
      -> run_stage1_cv        artifacts/models/stage1_fold*.txt, scores/train_stage1.parquet
      -> run_stage2_cv        artifacts/models/stage2_fold*.txt, scores/train_stage2.parquet
      -> run_decision_tuning  artifacts/models/decision.json
      -> run_predict_test     scores/test_stage{1,2}.parquet
      -> run_write_submission output/{matching_results,candidate_pairs}.tsv
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from . import decision, model, pruning
from .blocking import Channel, candidate_recall, generate_candidates
from .features import RECORD_COLUMNS, context_features, name_frequency, pair_features
from .io import CANDIDATE_HEADER, MATCH_HEADER, read_ground_truth_pairs, read_source, read_tsv, write_id_lists
from .metrics import summary
from .normalize import normalize_records
from .second_stage import query_context, stage2_features, target_context
from .settings import Settings
from .text_vectors import FieldSpec

log = logging.getLogger("entity_forge")

SPLITS = ("train", "test")
NORM_CHUNK = 1_000_000


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )


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
            "entity_id"
        ).is_in(keep.implode())
        out[src] = raw[src].filter(mask)
    return out


def run_normalize(s: Settings, splits: tuple[str, ...] = SPLITS) -> pl.DataFrame:
    """Normalize all sources; add ``name_freq`` (records sharing the name_core, per country)."""
    stats = []
    for split in splits:
        with timer(f"normalize {split}"):
            raw = {src: read_source(s.raw(split, src)) for src in (1, 2, 3)}
            if s.dev_mode:
                raw = _dev_sample(s, split, raw)
            normed = {
                src: pl.concat(
                    [normalize_records(df.slice(i, NORM_CHUNK)) for i in range(0, df.height, NORM_CHUNK)]
                )
                for src, df in raw.items()
            }
            freq = name_frequency(list(normed.values()))
            for src, df in normed.items():
                df = df.join(freq, on=["ckey", "name_core"], how="left", maintain_order="left")
                path = s.norm(split, src)
                path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(path)
                stats.append(
                    {
                        "split": split,
                        "source": src,
                        "rows": df.height,
                        "countries": ",".join(sorted(df["ckey"].unique().to_list())),
                        "no_address": float((~df["has_addr"]).mean()),
                        "non_latin_name": float(df["translit"].mean()),
                    }
                )
    return pl.DataFrame(stats)


def load_records(s: Settings, split: str, ckey: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(queries, targets) of one split, optionally one country. Row order is the index."""
    cols = ["ckey", *RECORD_COLUMNS]
    q = pl.scan_parquet(s.norm(split, 1)).select(cols)
    t = pl.concat([pl.scan_parquet(s.norm(split, src)).select(cols) for src in (2, 3)])
    if ckey is not None:
        q = q.filter(pl.col("ckey") == ckey)
        t = t.filter(pl.col("ckey") == ckey)
    return q.collect(), t.collect()


def countries(s: Settings, split: str) -> list[str]:
    return sorted(
        pl.scan_parquet(s.norm(split, 1)).select("ckey").unique().collect()["ckey"].to_list()
    )


# ---------------------------------------------------------------------------
# 2. Candidate generation
# ---------------------------------------------------------------------------


def make_channels(s: Settings) -> tuple[Channel, ...]:
    """Retrieval channels, chosen from measured recall (docs/EXPERIMENTS.md, EXP-002..006)."""
    name_words = FieldSpec("name_core", "word")
    return (
        # Same place, any name.
        Channel("addr", (FieldSpec("addr_norm", "word"),), s.top_k_addr, max_df_frac=0.02),
        # Common names disambiguated by place.
        Channel(
            "both",
            (FieldSpec("name_core", "word", 0.5), FieldSpec("addr_norm", "word", 0.5)),
            s.top_k_both,
            max_df_frac=0.02,
        ),
        # One business inside a building shared by many.
        Channel("cross", (FieldSpec("name_core", "cross", column2="addr_norm"),), s.top_k_cross,
                max_df_frac=0.01),
        # Name only: safety net for reformatted addresses (and provides cos_name everywhere).
        Channel("name", (name_words,), s.top_k_name, max_df_frac=0.01),
        # Name search among the 3-4 % of targets without an address.
        Channel("noaddr", (name_words,), s.top_k_noaddr, max_df_frac=0.05, target_filter="no_addr"),
        # Phonetic name x address among non-Latin-script targets (Devanagari etc.).
        Channel(
            "translit",
            (FieldSpec("name_phon", "cross", column2="addr_norm"),),
            s.top_k_translit,
            max_df_frac=0.05,
            target_filter="translit",
        ),
    )


def run_candidates(s: Settings, split: str) -> pl.DataFrame:
    """Union of all channels, per country. Writes one Parquet per country."""
    channels = make_channels(s)
    stats = []
    for ckey in countries(s, split):
        queries, targets = load_records(s, split, ckey)
        path = s.candidates_raw(split, ckey)
        path.parent.mkdir(parents=True, exist_ok=True)
        with timer(f"candidates {split}/{ckey} ({queries.height} x {targets.height})") as tm:
            if targets.height == 0:
                log.warning("no S2/S3 records for country %r: its S1s get empty lists", ckey)
                cands = pl.DataFrame(schema={"q": pl.Int32, "t": pl.Int32})
            else:
                cands = generate_candidates(queries, targets, channels, n_threads=s.n_threads)
            cands = cands.with_columns(
                queries["entity_id"].gather(cands["q"]).alias("s1_id"),
                targets["entity_id"].gather(cands["t"]).alias("t_id"),
            )
            cands.write_parquet(path)
        stats.append(
            {
                "split": split,
                "country": ckey,
                "s1": queries.height,
                "targets": targets.height,
                "pairs": cands.height,
                "avg_per_s1": cands.height / max(queries.height, 1),
                "seconds": round(tm.seconds, 1),
            }
        )
    return pl.DataFrame(stats)


def _labeled(cands: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    return cands.join(
        gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "t_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0))


def run_prune(s: Settings) -> pl.DataFrame:
    """Fit the stage-0 re-ranker on train, keep the top ``max_candidates`` per S1 everywhere."""
    gt = read_ground_truth_pairs(s.ground_truth)
    sample = []
    for ckey in countries(s, "train"):
        raw = pl.read_parquet(s.candidates_raw("train", ckey))
        if raw.height:
            feats = pruning.stage0_features(raw)
            sample.append(_labeled(feats.filter(model.sample_expr(pl.col("s1_id"), s.stage0_train_frac, salt=3)), gt))
    with timer("stage0 training"):
        booster, columns = pruning.train_stage0(pl.concat(sample), s.n_threads)
    model.save(booster, columns, s.stage0_model)
    del sample
    stats = []
    for split in SPLITS:
        for ckey in countries(s, split):
            raw = pl.read_parquet(s.candidates_raw(split, ckey))
            path = s.candidates(split, ckey)
            path.parent.mkdir(parents=True, exist_ok=True)
            if raw.height == 0:
                raw.write_parquet(path)
                continue
            kept = pruning.prune(pruning.stage0_features(raw), booster, columns, s.max_candidates)
            kept.sort("q", "t").write_parquet(path)
            stats.append({"split": split, "country": ckey, "raw_pairs": raw.height,
                          "kept_pairs": kept.height, "kept_per_s1": kept.height / max(raw["q"].n_unique(), 1)})
    return pl.DataFrame(stats)


def candidate_report(s: Settings, split: str = "train", which: str = "pruned",
                     cutoffs: tuple[int, ...] = ()) -> pl.DataFrame:
    """Pair recall of the union, of every channel alone, and (pruned) at stage-0 cut-offs."""
    gt = read_ground_truth_pairs(s.ground_truth)
    rows = []
    for ckey in countries(s, split):
        path = s.candidates(split, ckey) if which == "pruned" else s.candidates_raw(split, ckey)
        cands = pl.read_parquet(path)
        s1 = (
            pl.scan_parquet(s.norm(split, 1))
            .filter(pl.col("ckey") == ckey)
            .select(pl.col("entity_id").alias("s1_id"))
            .collect()
        )
        truth = gt.join(s1, on="s1_id", how="semi")
        rank_cols = [c for c in cands.columns if c.startswith("rank_")]
        views = [("union", cands)] + [
            (c.removeprefix("rank_"), cands.filter(pl.col(c) >= 0)) for c in rank_cols
        ]
        if "score0" in cands.columns:
            rank0 = pl.col("score0").rank("ordinal", descending=True).over("q")
            views += [(f"top{n}", cands.filter(rank0 <= n)) for n in cutoffs]
        for label, view in views:
            rows.append({"country": ckey, "set": which, "channel": label,
                         **candidate_recall(view.select("s1_id", "t_id"), truth)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------


def _q_blocks(cands: pl.DataFrame, rows_per_block: int) -> list[tuple[int, int]]:
    """Split q-sorted candidates into row ranges that never cut one S1 in two."""
    q = cands["q"].to_numpy()
    bounds = [0]
    while bounds[-1] < len(q):
        end = min(bounds[-1] + rows_per_block, len(q))
        while end < len(q) and q[end] == q[end - 1]:
            end += 1
        bounds.append(end)
    return list(zip(bounds[:-1], bounds[1:]))


def feature_dir(s: Settings, split: str, ckey: str) -> Path:
    return s.features(split, ckey).with_suffix("")


def feature_parts(s: Settings, split: str, ckey: str) -> list[Path]:
    return sorted(feature_dir(s, split, ckey).glob("part-*.parquet"))


def run_features(s: Settings, split: str) -> pl.DataFrame:
    """Stage-1 features for every candidate pair; labels for train."""
    gt = read_ground_truth_pairs(s.ground_truth) if split == "train" else None
    stats = []
    for ckey in countries(s, split):
        out_dir = feature_dir(s, split, ckey)
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cands = pl.read_parquet(s.candidates(split, ckey))
        if cands.height == 0:
            continue
        queries, targets = load_records(s, split, ckey)
        with timer(f"features {split}/{ckey} ({cands.height} pairs)") as tm:
            cands = context_features(cands.sort("q", "t"))
            if gt is not None:
                cands = _labeled(cands, gt)
            for i, (a, b) in enumerate(_q_blocks(cands, s.feature_chunk)):
                part = pair_features(cands.slice(a, b - a), queries, targets)
                part.write_parquet(out_dir / f"part-{i:04d}.parquet")
        stats.append({"split": split, "country": ckey, "pairs": cands.height,
                      "seconds": round(tm.seconds, 1)})
    return pl.DataFrame(stats)


# ---------------------------------------------------------------------------
# 4. Cross-validated training (stage 1 and stage 2 share this)
# ---------------------------------------------------------------------------


def _lgb_params(s: Settings, stage: int) -> dict:
    return {"num_threads": s.n_threads, "seed": s.seed + stage}


def _train_folds(s: Settings, stage: int, frames: pl.LazyFrame, train_frac: float, rounds: int) -> list[str]:
    """Train one model per fold on the other folds (grouped by S1)."""
    lazy = frames.with_columns(model.fold_expr(pl.col("s1_id"), s.n_folds).alias("fold"))
    sampled = lazy.filter(model.sample_expr(pl.col("s1_id"), train_frac, salt=stage)).collect()
    columns = model.feature_columns(sampled.drop("fold"))
    log.info("stage %d: %d training rows, %d features", stage, sampled.height, len(columns))
    s.reports.mkdir(parents=True, exist_ok=True)
    for fold in range(s.n_folds):
        rest = sampled.filter(pl.col("fold") != fold)
        # Early stopping on a slice of the *training* folds, never on the OOF fold.
        es = rest.select(model.sample_expr(pl.col("s1_id"), 0.08, salt=100 + fold)).to_series()
        booster = model.train(rest.filter(~es), columns, _lgb_params(s, stage), rounds,
                              valid=rest.filter(es))
        model.save(booster, columns, s.model(stage, fold))
        model.importance(booster).write_csv(s.reports / f"importance_stage{stage}_fold{fold}.csv")
        del rest, booster
    return columns


def _predict_oof(s: Settings, stage: int, frame: pl.DataFrame) -> np.ndarray:
    fold = frame.select(model.fold_expr(pl.col("s1_id"), s.n_folds)).to_series().to_numpy()
    p = np.zeros(frame.height, dtype=np.float32)
    for f in range(s.n_folds):
        idx = np.flatnonzero(fold == f)
        if len(idx):
            booster, columns = model.load(s.model(stage, f))
            p[idx] = model.predict(booster, frame[idx], columns)
    return p


def _predict_mean(s: Settings, stage: int, frame: pl.DataFrame) -> np.ndarray:
    p = np.zeros(frame.height, dtype=np.float32)
    for f in range(s.n_folds):
        booster, columns = model.load(s.model(stage, f))
        p += model.predict(booster, frame, columns)
    return p / np.float32(s.n_folds)


def _score_stage1(s: Settings, split: str) -> pl.DataFrame:
    """Stage-1 scores for every feature part (OOF on train, fold average on test)."""
    keep = ["ckey", "q", "t", "s1_id", "t_id"] + (["label"] if split == "train" else [])
    out = []
    for ckey in countries(s, split):
        for path in feature_parts(s, split, ckey):
            frame = pl.read_parquet(path)
            p = _predict_oof(s, 1, frame) if split == "train" else _predict_mean(s, 1, frame)
            out.append(frame.with_columns(pl.lit(ckey).alias("ckey"), pl.Series("p1", p)).select(*keep, "p1"))
    scores = pl.concat(out)
    path = s.scores(split, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(path)
    return scores


def run_stage1_cv(s: Settings, train_frac: float = 0.3, rounds: int = 3000) -> pl.DataFrame:
    """Stage 1: fit fold models on a sample of S1s, then OOF-score *every* train pair."""
    parts = [p for ckey in countries(s, "train") for p in feature_parts(s, "train", ckey)]
    with timer("stage1 training"):
        _train_folds(s, 1, pl.scan_parquet(parts), train_frac, rounds)
    with timer("stage1 OOF scoring"):
        return _score_stage1(s, "train")


def _stage2_parts(s: Settings, split: str, scores1: pl.DataFrame):
    """Yield (country, stage-2 frame) = stage-1 features + p1 + set context, part by part."""
    for ckey in countries(s, split):
        p1 = scores1.filter(pl.col("ckey") == ckey).select("q", "t", "p1")
        if p1.height == 0:
            continue
        q_ctx = query_context(p1)
        t_ctx = target_context(p1)
        _, targets = load_records(s, split, ckey)
        for path in feature_parts(s, split, ckey):
            part = pl.read_parquet(path).join(p1, on=["q", "t"], how="left")
            yield ckey, stage2_features(part, q_ctx, t_ctx, targets)


def run_stage2_cv(s: Settings, train_frac: float = 0.4, rounds: int = 3000) -> pl.DataFrame:
    """Stage 2 on OOF stage-1 probabilities; OOF-scores every train pair."""
    scores1 = pl.read_parquet(s.scores("train", 1))
    tmp = s.work_dir / "stage2_features" / "train"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with timer("stage2 features (train)"):
        for i, (ckey, frame) in enumerate(_stage2_parts(s, "train", scores1)):
            frame.with_columns(pl.lit(ckey).alias("ckey")).write_parquet(tmp / f"part-{i:04d}.parquet")
    parts = sorted(tmp.glob("part-*.parquet"))
    with timer("stage2 training"):
        _train_folds(s, 2, pl.scan_parquet(parts), train_frac, rounds)
    out = []
    with timer("stage2 OOF scoring"):
        for path in parts:
            frame = pl.read_parquet(path)
            out.append(
                frame.select("ckey", "q", "t", "s1_id", "t_id", "label", "p1")
                .with_columns(pl.Series("p", _predict_oof(s, 2, frame)))
            )
    scores = pl.concat(out)
    scores.write_parquet(s.scores("train", 2))
    return scores


# ---------------------------------------------------------------------------
# 5. Decision tuning on out-of-fold scores
# ---------------------------------------------------------------------------


def decide(pred: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Apply a decision rule (see decision.py) to (s1_id, t_id, p)."""
    if cfg["exclusive"]:
        pred = decision.resolve_exclusive(pred)
    if cfg["method"] == "threshold":
        return decision.select_threshold(pred, cfg["tau"])
    return decision.select_expected_f(pred, missed=cfg["missed"], floor=cfg["floor"])


def run_decision_tuning(s: Settings, score_col: str = "p") -> tuple[dict, pl.DataFrame]:
    """Grid over decision rules on OOF train scores with the exact macro metric."""
    stage = 2 if score_col == "p" else 1
    scores = pl.read_parquet(s.scores("train", stage))
    pred = scores.select("s1_id", "t_id", pl.col(score_col).alias("p")).filter(pl.col("p") >= 0.01)
    s1_ids = pl.read_parquet(s.norm("train", 1), columns=["entity_id"])["entity_id"]
    truth = read_ground_truth_pairs(s.ground_truth).join(
        pl.DataFrame({"s1_id": s1_ids}), on="s1_id", how="semi"
    )
    grid: list[dict] = [
        {"method": "threshold", "exclusive": ex, "tau": round(float(t), 3)}
        for ex in (False, True)
        for t in np.arange(0.20, 0.86, 0.025)
    ]
    grid += [
        {"method": "expected_f", "exclusive": True, "missed": m, "floor": 0.02}
        for m in (0.0, 0.05, 0.1, 0.2, 0.3)
    ]
    # Exclusivity does not depend on the threshold: resolve it once.
    resolved = {False: pred, True: decision.resolve_exclusive(pred)}
    rows = [
        {**cfg, **summary(s1_ids, decide(resolved[cfg["exclusive"]], {**cfg, "exclusive": False}), truth)}
        for cfg in grid
    ]
    table = pl.DataFrame(rows).sort("macro_f05", descending=True)
    best = {
        k: v
        for k, v in table.row(0, named=True).items()
        if k in ("method", "exclusive", "tau", "missed", "floor") and v is not None
    }
    best["score_col"] = score_col
    best["oof_macro_f05"] = float(table["macro_f05"][0])
    s.decision_file.parent.mkdir(parents=True, exist_ok=True)
    s.decision_file.write_text(json.dumps(best, indent=2))
    s.reports.mkdir(parents=True, exist_ok=True)
    table.write_csv(s.reports / "decision_grid.csv")
    return best, table


def oof_report(s: Settings) -> pl.DataFrame:
    """Per-country OOF metrics of the frozen decision rule."""
    cfg = json.loads(s.decision_file.read_text())
    stage = 2 if cfg["score_col"] == "p" else 1
    scores = pl.read_parquet(s.scores("train", stage))
    s1 = pl.read_parquet(s.norm("train", 1), columns=["entity_id", "ckey"])
    gt = read_ground_truth_pairs(s.ground_truth)
    chosen = decide(scores.select("s1_id", "t_id", pl.col(cfg["score_col"]).alias("p")), cfg)
    rows = []
    for ckey in ["ALL", *sorted(s1["ckey"].unique().to_list())]:
        ids = (s1 if ckey == "ALL" else s1.filter(pl.col("ckey") == ckey))["entity_id"]
        keep = pl.DataFrame({"s1_id": ids})
        rows.append({
            "country": ckey,
            **summary(ids, chosen.join(keep, on="s1_id", how="semi"), gt.join(keep, on="s1_id", how="semi")),
        })
    return pl.DataFrame(rows)


def error_examples(s: Settings, n: int = 15, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Samples of OOF false positives (wrong merges) and false negatives, with raw text."""
    cfg = json.loads(s.decision_file.read_text())
    stage = 2 if cfg["score_col"] == "p" else 1
    scores = pl.read_parquet(s.scores("train", stage))
    chosen = decide(scores.select("s1_id", "t_id", pl.col(cfg["score_col"]).alias("p")), cfg)
    gt = read_ground_truth_pairs(s.ground_truth).join(
        scores.select("s1_id").unique(), on="s1_id", how="semi")
    fp = chosen.join(gt, on=["s1_id", "t_id"], how="anti")
    fn = gt.join(chosen, on=["s1_id", "t_id"], how="anti").join(
        scores.select("s1_id", "t_id", pl.col(cfg["score_col"]).alias("p")), on=["s1_id", "t_id"], how="left")
    cols = ["entity_id", "business_name", "business_address"]
    s1 = pl.read_parquet(s.norm("train", 1), columns=cols).rename(
        {"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
    tg = pl.concat([pl.read_parquet(s.norm("train", i), columns=cols) for i in (2, 3)]).rename(
        {"entity_id": "t_id", "business_name": "t_name", "business_address": "t_addr"})

    def view(pairs: pl.DataFrame) -> pl.DataFrame:
        if pairs.height == 0:
            return pairs
        return (pairs.sample(min(n, pairs.height), seed=seed)
                .join(s1, on="s1_id", how="left").join(tg, on="t_id", how="left")
                .select("p", "s1_name", "t_name", "s1_addr", "t_addr"))

    return view(fp), view(fn)


# ---------------------------------------------------------------------------
# 6. Test inference and submission
# ---------------------------------------------------------------------------


def run_predict_test(s: Settings) -> pl.DataFrame:
    """Stage-1 and stage-2 scores for every test candidate (fold-model average)."""
    with timer("test stage1 scoring"):
        scores1 = _score_stage1(s, "test")
    out = []
    with timer("test stage2 scoring"):
        for ckey, frame in _stage2_parts(s, "test", scores1):
            out.append(
                frame.select("q", "t", "s1_id", "t_id", "p1").with_columns(
                    pl.lit(ckey).alias("ckey"), pl.Series("p", _predict_mean(s, 2, frame))
                )
            )
    scores = pl.concat(out)
    scores.write_parquet(s.scores("test", 2))
    return scores


def run_write_submission(s: Settings) -> dict:
    """Apply the frozen decision rule and write both competition TSVs."""
    cfg = json.loads(s.decision_file.read_text())
    stage = 2 if cfg["score_col"] == "p" else 1
    scores = pl.read_parquet(s.scores("test", stage))
    # The raw file is authoritative for which S1 rows must exist (every one of
    # them, even if a stage skipped it); in dev mode the unsampled ones stay empty.
    s1_ids = read_tsv(s.raw("test", 1), ("entity_id",))["entity_id"]
    normed = pl.read_parquet(s.norm("test", 1), columns=["entity_id"])["entity_id"]
    if not s.dev_mode and normed.len() != s1_ids.len():
        raise AssertionError(f"normalized test S1 has {normed.len()} rows, raw file {s1_ids.len()}")
    chosen = decide(scores.select("s1_id", "t_id", pl.col(cfg["score_col"]).alias("p")), cfg)
    cands = pl.concat(
        [pl.read_parquet(s.candidates("test", c), columns=["s1_id", "t_id"]) for c in countries(s, "test")]
    )

    # Invariants: matches come from candidates, one S1 per target, known S1 ids only.
    stray = chosen.join(cands, on=["s1_id", "t_id"], how="anti").height
    if stray:
        raise AssertionError(f"{stray} predicted pairs are not in candidate_pairs")
    if cfg["exclusive"] and chosen["t_id"].n_unique() != chosen.height:
        raise AssertionError("exclusive decision produced a target under two S1s")
    unknown = chosen.join(pl.DataFrame({"s1_id": s1_ids}), on="s1_id", how="anti").height
    if unknown:
        raise AssertionError(f"{unknown} predictions for unknown S1 ids")

    s.output_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1_ids, chosen, s.output_dir / "matching_results.tsv", MATCH_HEADER)
    write_id_lists(s1_ids, cands, s.output_dir / "candidate_pairs.tsv", CANDIDATE_HEADER)
    with_matches = chosen["s1_id"].n_unique()
    return {
        "s1_rows": s1_ids.len(),
        "predicted_pairs": chosen.height,
        "s1_with_matches": with_matches,
        "empty_rate": 1 - with_matches / s1_ids.len(),
        "candidate_pairs": cands.height,
        "decision": cfg,
    }


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
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(s.output_dir / name, f"output/{name}")
        for path in sorted((s.root / "src" / "entity_forge").glob("*.py")):
            z.write(path, f"{code}/src/entity_forge/{path.name}")
        for path in sorted((s.root / "notebooks").glob("*.ipynb")):
            z.write(path, f"{code}/src/notebooks/{path.name}")
        z.write(s.root / "utils" / "validate_submission.py", f"{code}/src/utils/validate_submission.py")
        z.write(s.root / "README.md", f"{code}/README.md")
        z.write(s.root / "requirements.txt", f"{code}/requirements.txt")
        z.write(s.root / "docs" / "METHODOLOGY.md", "Documentation_template.md")
    return dest
