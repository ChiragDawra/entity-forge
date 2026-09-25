#!/usr/bin/env bash
# Execute the pipeline notebooks headless, in order, saving outputs in place.
#
#   bash scripts/run_notebooks.sh                  # stages 01..05
#   bash scripts/run_notebooks.sh 03 04            # only these stages
#   EF_DEV_MODE=1 bash scripts/run_notebooks.sh    # smoke test on a small slice
#
# Settings come from EF_* environment variables (see src/entity_forge/settings.py).
set -euo pipefail
cd "$(dirname "$0")/.."

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(01 02 03 04 05)
fi

for stage in "${stages[@]}"; do
  nb=$(ls notebooks/"${stage}"_*.ipynb)
  echo "=== $(date '+%H:%M:%S') running $nb"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
done
echo "=== $(date '+%H:%M:%S') done"
