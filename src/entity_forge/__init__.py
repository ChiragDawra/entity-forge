"""entity-forge: business entity resolution for the Amazon ML Challenge 2026."""

# Load order matters on macOS: sparse_dot_topn and LightGBM each ship an OpenMP
# runtime. If LightGBM's is loaded first, sparse_dot_topn's multithreaded top-K
# search segfaults. Importing sparse_dot_topn first makes both share one runtime.
import sparse_dot_topn  # noqa: F401  (must stay the first import)

__version__ = "1.0.0"
