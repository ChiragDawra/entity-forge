"""Single production entry point (used by ``scripts/run_notebooks.sh``).

    python -m entity_forge.pipeline run   [--from STAGE] [--to STAGE] [--only STAGE[,..]]
                                          [--force STAGE[,..]|all] [--profile 16gb|32gb|64gb] [--dev]
    python -m entity_forge.pipeline status [--dev]
    python -m entity_forge.pipeline stage NAME          (one stage in this process)

Stages (notebook): normalize (01) · candidates, prune (02) · features (03) ·
stage1, stage2, decision (04) · predict, submit (05). ``--from``/``--to`` also
accept notebook numbers (``--from 03``).

``run`` executes each stage in its own Python process, so all memory is
returned to the OS between stages, and a crash in one stage leaves the
checkpoints of all earlier work intact. Every stage skips work whose checkpoint
is valid, so re-running after a failure resumes where it stopped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from . import stages
from .checkpoints import disk_free_gb
from .resources import available_gb, cpu_count, log_mem, total_memory_gb
from .settings import Settings

log = logging.getLogger("entity_forge")

STAGES = stages.STAGES
NOTEBOOK_STAGES = {
    "01": ("normalize", "normalize"),
    "02": ("candidates", "prune"),
    "03": ("features", "features"),
    "04": ("stage1", "decision"),
    "05": ("predict", "submit"),
}


def _stage_range(start: str | None, end: str | None) -> list[str]:
    def resolve(name: str | None, default: str, pick: int) -> str:
        if name is None:
            return default
        name = name.lower()
        if name in NOTEBOOK_STAGES:
            return NOTEBOOK_STAGES[name][pick]
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; stages: {', '.join(STAGES)} (or notebook numbers 01-05)")
        return name

    a = STAGES.index(resolve(start, STAGES[0], 0))
    b = STAGES.index(resolve(end, STAGES[-1], 1))
    if a > b:
        raise SystemExit(f"--from {start} comes after --to {end}")
    return list(STAGES[a : b + 1])


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ---------------------------------------------------------------------------
# One stage (child process)
# ---------------------------------------------------------------------------


def run_stage(name: str, s: Settings) -> object:
    if name == "normalize":
        return stages.run_normalize(s)
    if name == "candidates":
        return [stages.run_candidates(s, "train"), stages.run_candidates(s, "test")]
    if name == "prune":
        out = stages.run_prune(s)
        report = stages.candidate_report(s, "train", which="pruned", cutoffs=(10, 20, 30, 40, 50))
        s.reports.mkdir(parents=True, exist_ok=True)
        report.write_csv(s.reports / "candidate_recall_pruned.csv")
        return [out, report.select("country", "channel", "pair_recall", "avg_per_s1")]
    if name == "features":
        return [stages.run_features(s, "train"), stages.run_features(s, "test")]
    if name == "stage1":
        return stages.run_stage1_cv(s)
    if name == "stage2":
        return stages.run_stage2_cv(s)
    if name == "decision":
        best1, _ = stages.run_decision_tuning(s, score_col="p1")
        best, _ = stages.run_decision_tuning(s, score_col="p")
        report = stages.oof_report(s)
        report.write_csv(s.reports / "oof_report.csv")
        return [{"stage1_best": best1, "stage2_best": best}, report]
    if name == "predict":
        return stages.run_predict_test(s)
    if name == "submit":
        summary = stages.run_write_submission(s)
        # The official validator keeps every candidate id in Python sets (~10 GB for the full
        # test set); its optional --check-ids pass adds ~1 GB more, so only on bigger machines.
        print(stages.run_validator(s, check_ids=total_memory_gb() >= 24), flush=True)
        summary["zip"] = str(stages.build_submission_zip(s))
        return summary
    raise SystemExit(f"unknown stage {name!r}")


def _stage_main(name: str) -> int:
    s = Settings.from_env()
    log_file = Path(os.environ["EF_LOG_FILE"]) if os.environ.get("EF_LOG_FILE") else s.logs / f"stage-{name}.log"
    stages.setup_logging(log_file=log_file)
    log.info("=== stage %s | %s", name, s.describe())
    log_mem(f"stage {name}: start")
    t0 = time.time()
    result = run_stage(name, s)
    for item in result if isinstance(result, list) else [result]:
        print(json.dumps(item, indent=2, default=str) if isinstance(item, dict) else item, flush=True)
    log_mem(f"stage {name}: end ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# Orchestration (parent process)
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    if args.profile:
        os.environ["EF_PROFILE"] = args.profile
    if args.dev:
        os.environ["EF_DEV_MODE"] = "1"
    s = Settings.from_env()
    if args.only:
        todo = [x.strip() for x in args.only.split(",") if x.strip()]
        bad = [x for x in todo if x not in STAGES]
        if bad:
            raise SystemExit(f"--only: unknown stage(s) {bad}; stages: {', '.join(STAGES)}")
    else:
        todo = _stage_range(args.start, args.end)
    force = {x.strip() for x in (args.force or "").split(",") if x.strip()}
    unknown = force - set(STAGES) - {"all"}
    if unknown:
        raise SystemExit(f"--force: unknown stage(s) {sorted(unknown)}")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = s.logs / f"run-{stamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["EF_LOG_FILE"] = str(log_file)
    stages.setup_logging(log_file=log_file)
    log.info("run: stages %s | force %s | RAM %.1f GB (%.1f free) | %d CPUs | disk free %.1f GB | %s",
             ",".join(todo), ",".join(sorted(force)) or "-", total_memory_gb(), available_gb(), cpu_count(),
             disk_free_gb(s.work_dir), s.describe())
    for name in todo:
        env = dict(os.environ)
        env["EF_FORCE"] = "all" if "all" in force else (name if name in force else "")
        t0 = time.time()
        log.info(">>> stage %s", name)
        proc = subprocess.run([sys.executable, "-m", "entity_forge.pipeline", "stage", name], env=env)
        if proc.returncode != 0:
            log.error("stage %s FAILED (exit %d) after %.0fs. Completed checkpoints are kept; log: %s",
                      name, proc.returncode, time.time() - t0, log_file)
            if proc.returncode < 0 or proc.returncode == 137:
                log.error("the process was killed (signal %d): most likely out of memory. Use a smaller "
                          "profile (--profile 16gb) or smaller chunks (docs/SAGEMAKER.md).", abs(proc.returncode))
            log.error("resume with:  bash scripts/run_notebooks.sh --from %s", name)
            return proc.returncode if proc.returncode > 0 else 1
        log.info("<<< stage %s done in %.0fs", name, time.time() - t0)
    log.info("run finished: %s", ", ".join(todo))
    return 0


def _status(args: argparse.Namespace) -> int:
    if args.dev:
        os.environ["EF_DEV_MODE"] = "1"
    s = Settings.from_env()
    print(f"work_dir:  {s.work_dir}")
    print(f"resources: RAM {total_memory_gb():.1f} GB ({available_gb():.1f} free), {cpu_count()} CPUs, "
          f"disk free {disk_free_gb(s.work_dir):.1f} GB")
    print(f"settings:  {s.describe()}\n")
    _print_table(stages.status_rows(s))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m entity_forge.pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run / resume the pipeline")
    r.add_argument("--from", dest="start")
    r.add_argument("--to", dest="end")
    r.add_argument("--only", help="comma-separated stages to run (ignores --from/--to)")
    r.add_argument("--force", default="", help="comma-separated stages to recompute, or 'all'")
    r.add_argument("--profile", choices=["16gb", "32gb", "64gb", "auto"])
    r.add_argument("--dev", action="store_true", help="small consistent slice (artifacts_dev/)")
    st = sub.add_parser("status", help="show checkpoint state")
    st.add_argument("--dev", action="store_true")
    one = sub.add_parser("stage", help="run one stage in this process")
    one.add_argument("name", choices=list(STAGES))
    args = p.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "status":
        return _status(args)
    return _stage_main(args.name)


if __name__ == "__main__":
    sys.exit(main())
