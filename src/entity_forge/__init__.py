"""entity-forge: business entity resolution for the Amazon ML Challenge 2026."""

# 1. Cap BLAS / OpenMP / Polars thread pools at EF_N_THREADS (default: all
#    cores) before numpy or polars load, so libraries don't oversubscribe the CPU.
from .resources import configure_threads

configure_threads()

# 2. Load order matters on macOS: sparse_dot_topn and LightGBM each ship an
#    OpenMP runtime. If LightGBM's is loaded first, sparse_dot_topn's
#    multithreaded top-K search segfaults. Importing sparse_dot_topn first makes
#    both share one runtime.
import sparse_dot_topn  # noqa: E402,F401

__version__ = "1.1.0"
