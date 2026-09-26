"""Machine resources: memory profiles, thread limits, memory logging and guards.

The pipeline must run on anything from a 16 GB laptop to a 64 GB+ server
without code changes. A *profile* bounds every memory-heavy knob (retrieval
query chunk, rows per feature part, rows used to fit LightGBM, ...). It is
chosen from detected RAM or ``EF_PROFILE``; individual knobs can still be
overridden with their own ``EF_*`` variables (see ``settings.py``).

Profile values come from measured per-row costs on the dev slice
(``docs/EXPERIMENTS.md``, EXP-008): raw candidate row 68 B in RAM, pruned
candidate row 72 B, stage-1 feature row 206 B, stage-2 feature row 274 B;
LightGBM needs ~4 B per feature per row for the float32 matrix it bins.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
from dataclasses import dataclass

log = logging.getLogger("entity_forge")

GB = 1024**3


@dataclass(frozen=True)
class Profile:
    name: str
    retrieval_chunk: int  # S1 queries per sparse top-K call
    candidate_chunk_rows: int  # raw candidate rows per union / pruning chunk
    feature_chunk: int  # candidate pairs per feature part
    stage0_max_rows: int  # rows used to fit the stage-0 pruner
    stage1_max_rows: int  # rows used to fit the stage-1 fold models
    stage2_max_rows: int  # rows used to fit the stage-2 fold models
    min_free_gb: float  # refuse to start a big allocation below this free RAM


PROFILES: dict[str, Profile] = {
    "16gb": Profile("16gb", 20_000, 3_000_000, 1_000_000, 4_000_000, 6_000_000, 5_000_000, 1.5),
    "32gb": Profile("32gb", 50_000, 6_000_000, 2_000_000, 8_000_000, 16_000_000, 12_000_000, 3.0),
    "64gb": Profile("64gb", 100_000, 12_000_000, 3_000_000, 15_000_000, 40_000_000, 30_000_000, 6.0),
}


def total_memory_gb() -> float:
    """Physical RAM, or ``EF_MEMORY_LIMIT_GB`` when set (e.g. inside a container)."""
    override = os.environ.get("EF_MEMORY_LIMIT_GB")
    if override:
        return float(override)
    try:
        import psutil

        return psutil.virtual_memory().total / GB
    except ImportError:  # pragma: no cover
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB


def cpu_count() -> int:
    override = os.environ.get("EF_N_THREADS")
    if override:
        return max(1, int(override))
    try:
        return max(1, len(os.sched_getaffinity(0)))  # respects cgroup/affinity on Linux
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def auto_profile(mem_gb: float | None = None) -> str:
    mem_gb = total_memory_gb() if mem_gb is None else mem_gb
    if mem_gb < 24:
        return "16gb"
    if mem_gb < 56:
        return "32gb"
    return "64gb"


def resolve_profile(name: str | None = None) -> Profile:
    name = (name or os.environ.get("EF_PROFILE") or "auto").lower()
    if name == "auto":
        name = auto_profile()
    if name not in PROFILES:
        raise ValueError(f"unknown EF_PROFILE {name!r}; choose one of {sorted(PROFILES)} or 'auto'")
    return PROFILES[name]


_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS",
)


def configure_threads(n: int | None = None) -> int:
    """Cap every thread pool (BLAS, OpenMP, Polars/Rayon) at ``n``.

    Must run before numpy/polars are imported to take full effect; the package
    ``__init__`` calls it first thing. Values already set in the environment win.
    """
    n = n or cpu_count()
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))
    return n


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------


class MemoryBudgetError(MemoryError):
    """Raised *before* an allocation that would likely exhaust RAM."""


def rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / GB
    except ImportError:  # pragma: no cover
        return peak_rss_gb()


def available_gb() -> float:
    try:
        import psutil

        avail = psutil.virtual_memory().available / GB
    except ImportError:  # pragma: no cover
        return float("inf")
    limit = os.environ.get("EF_MEMORY_LIMIT_GB")
    if limit:  # a configured cap below physical RAM wins
        avail = min(avail, float(limit) - rss_gb())
    return avail


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / GB if sys.platform == "darwin" else peak * 1024 / GB


def log_mem(tag: str) -> None:
    log.info("[MEM] %-44s rss=%.2fGB peak=%.2fGB avail=%.2fGB", tag, rss_gb(), peak_rss_gb(), available_gb())


def guard(tag: str, need_gb: float = 0.0, min_free_gb: float | None = None) -> None:
    """Fail clearly if free RAM minus the expected allocation is below the safety margin."""
    if min_free_gb is None:
        env = os.environ.get("EF_MIN_FREE_GB")
        min_free_gb = float(env) if env else resolve_profile().min_free_gb
    margin = min_free_gb
    avail = available_gb()
    if avail - need_gb < margin:
        log_mem(f"{tag} (guard tripped)")
        raise MemoryBudgetError(
            f"{tag}: needs ~{need_gb:.1f} GB but only {avail:.1f} GB free (safety margin {margin:.1f} GB). "
            "Use a smaller profile (EF_PROFILE=16gb) or smaller chunks (EF_FEATURE_CHUNK, "
            "EF_CANDIDATE_CHUNK_ROWS, EF_STAGE1_MAX_ROWS), close other kernels, and re-run: "
            "completed work is kept."
        )
