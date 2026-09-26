#!/usr/bin/env bash
# Production entry point: runs (or resumes) the whole pipeline with checkpoints.
#
#   bash scripts/run_notebooks.sh                       # full run; skips everything already done
#   bash scripts/run_notebooks.sh --status              # what is done / missing / stale
#   bash scripts/run_notebooks.sh --from features       # start at a stage (or notebook number: --from 03)
#   bash scripts/run_notebooks.sh --from 04 --to 04     # only notebook 04's stages
#   bash scripts/run_notebooks.sh --only prune          # exactly these stages
#   bash scripts/run_notebooks.sh --force features      # recompute a stage even if checkpointed
#   bash scripts/run_notebooks.sh --profile 16gb        # 16gb | 32gb | 64gb (default: auto from RAM)
#   bash scripts/run_notebooks.sh --dev                 # small slice into artifacts_dev/ (smoke test)
#   bash scripts/run_notebooks.sh --notebooks [01 02]   # execute the notebooks instead (same checkpoints)
#
# Stages: normalize candidates prune features stage1 stage2 decision predict submit
# Settings come from EF_* environment variables (see src/entity_forge/settings.py and docs/SAGEMAKER.md).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY="python3"; fi

# One thread budget for every library (BLAS, OpenMP, Polars, LightGBM, RapidFuzz): no oversubscription.
if [ -z "${EF_N_THREADS:-}" ]; then
  EF_N_THREADS=$("$PY" -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())')
fi
export EF_N_THREADS
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS \
         POLARS_MAX_THREADS RAYON_NUM_THREADS; do
  export "$v"="${!v:-$EF_N_THREADS}"
done

case "${1:-}" in
  --status)
    shift
    exec "$PY" -m entity_forge.pipeline status "$@"
    ;;
  --notebooks)
    shift
    notebooks=("$@")
    if [ ${#notebooks[@]} -eq 0 ]; then notebooks=(01 02 03 04 05); fi
    out="$ROOT/artifacts/notebook_runs/$(date +%Y%m%d-%H%M%S)"
    if [ "${EF_DEV_MODE:-0}" = "1" ]; then out="$ROOT/artifacts_dev/notebook_runs/$(date +%Y%m%d-%H%M%S)"; fi
    mkdir -p "$out"
    for nb_id in "${notebooks[@]}"; do
      nb=$(ls "$ROOT"/notebooks/"${nb_id}"_*.ipynb)
      echo "=== $(date '+%H:%M:%S') executing $nb (output: $out)"
      "$PY" -m jupyter nbconvert --to notebook --execute --output-dir "$out" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
    done
    echo "=== $(date '+%H:%M:%S') notebooks done; executed copies in $out"
    ;;
  *)
    exec "$PY" -m entity_forge.pipeline run "$@"
    ;;
esac
