# Running the pipeline (SageMaker or any machine: 16 / 32 / 64+ GB)

One command runs, and resumes, everything:

```bash
bash scripts/run_notebooks.sh
```

Every stage writes checkpoints. Re-running the command skips everything that
is already done and continues at the first missing piece, so a crash or a
stopped instance costs at most the part that was in progress.

## 1. Setup

```bash
cd ~/SageMaker
git clone https://github.com/ChiragDawra/entity-forge.git && cd entity-forge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name entity-forge      # only for interactive notebooks

# competition data, same layout as the challenge kit (never committed)
aws s3 cp --recursive s3://<your-bucket>/dataset/ dataset/
```

```
dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## 2. Everyday commands

| Goal | Command |
|---|---|
| full run / resume after any interruption | `bash scripts/run_notebooks.sh` |
| what is done, missing or stale | `bash scripts/run_notebooks.sh --status` |
| start at a stage | `bash scripts/run_notebooks.sh --from features` (or `--from 03`) |
| run only some stages | `bash scripts/run_notebooks.sh --only prune,features` |
| stop after a stage | `bash scripts/run_notebooks.sh --to stage1` |
| recompute a stage although it is checkpointed | `bash scripts/run_notebooks.sh --force features` |
| choose the memory profile | `bash scripts/run_notebooks.sh --profile 32gb` |
| smoke test on a small slice (`artifacts_dev/`) | `bash scripts/run_notebooks.sh --dev` |
| execute the notebooks instead (same checkpoints) | `bash scripts/run_notebooks.sh --notebooks 04 05` |

Run it detached so a closed browser tab doesn't matter:

```bash
nohup bash scripts/run_notebooks.sh > run.out 2>&1 &
tail -f artifacts/logs/run-*.log          # every run writes a timestamped log
```

Stages, in order (notebook in brackets): `normalize` [01] · `candidates`,
`prune` [02] · `features` [03] · `stage1`, `stage2`, `decision` [04] ·
`predict`, `submit` [05]. Each stage runs in its own Python process, so memory
is fully released between stages. A failing stage prints the exact command to
resume.

## 3. Checkpoints and resume

| Artifact | Checkpoint granularity |
|---|---|
| `norm/{split}_s{1,2,3}.parquet` | per file |
| `candidates_raw/{split}/{country}.parquet` | per country; inside a country per channel / union part / cosine pass (scratch in `artifacts/tmp/`) |
| `candidates/`, `features/`, `stage2_features/`, `scores/` | per part (a few million rows); `_PLAN.json` fixes part boundaries, `_MANIFEST.json` marks the partition complete |
| `models/stage{0,1,2}*.txt`, `models/decision.json` | per model / fold |

A checkpoint is reused only if its manifest exists, every file still has its
recorded size, and the **semantic configuration hash** matches (top-K per
channel, candidate cap, model parameters, folds, stage versions). Thread counts,
chunk sizes and the memory profile are *not* part of the hash, so you can
resume on another machine or with another profile.

- **Changed a result-affecting setting** (e.g. `EF_MAX_CANDIDATES`): the
  affected stage and everything downstream recompute automatically. The
  expensive raw candidates are the exception: if they were built with other
  top-K values, the run stops with a clear message instead of silently
  rebuilding them (`--force candidates` rebuilds, several hours).
- **Legacy artifacts** (created before checkpoints existed) in `norm/` and
  `candidates_raw/` are validated and adopted, never rebuilt. `norm/` checks
  schema and row counts against the raw TSVs. `candidates_raw/` checks schema,
  no nulls, every S1 of the country present, and ranks within top-K.
- Nothing ever deletes `norm/` or `candidates_raw/`. Parts of a derived stage
  are replaced only when that stage is recomputed.
- `artifacts/tmp/` is disposable scratch; delete it only when no run is active.

## 4. Memory profiles

The profile is picked from detected RAM (`auto`: < 24 GB → 16gb, < 56 GB → 32gb,
else 64gb) or set with `--profile` / `EF_PROFILE`. Individual knobs can be
overridden with their own variables.

| Knob (`EF_…`) | 16gb | 32gb | 64gb | What it bounds |
|---|---|---|---|---|
| `RETRIEVAL_CHUNK` | 20k | 50k | 100k | S1 queries per sparse top-K call |
| `CANDIDATE_CHUNK_ROWS` | 3M | 6M | 12M | raw candidate rows per union / pruning part |
| `FEATURE_CHUNK` | 1M | 2M | 3M | pairs per RapidFuzz feature batch |
| `STAGE0_MAX_ROWS` | 4M | 8M | 15M | rows used to fit the pruner |
| `STAGE1_MAX_ROWS` | 6M | 16M | 40M | rows used to fit the stage-1 fold models |
| `STAGE2_MAX_ROWS` | 5M | 12M | 30M | rows used to fit the stage-2 fold models |
| `MIN_FREE_GB` | 1.5 | 3 | 6 | a stage refuses to start a big allocation below this free RAM |

Other variables: `EF_N_THREADS` (default: all cores; one budget shared by
BLAS, OpenMP, Polars, LightGBM, RapidFuzz and sparse_dot_topn),
`EF_MEMORY_LIMIT_GB` (treat the machine as having this much RAM, e.g. in a
container), `EF_MIN_FREE_DISK_GB` (default 5), `EF_WORK_DIR` / `EF_DATA_DIR`
(move artifacts/data to a bigger volume), `EF_DEV_MODE`.

Result-affecting settings (`EF_TOP_K_*`, `EF_MAX_CANDIDATES`, `EF_N_FOLDS`,
`EF_SEED`) are the same on every profile. The only quality effect of a smaller
profile is fewer rows used to *fit* the LightGBM models. Candidates, features
and out-of-fold scoring always cover every pair.

**When memory is short** the stage stops *before* the allocation with a
`MemoryBudgetError` naming the knob to lower, instead of the kernel dying.
Lower the profile and re-run; finished parts are kept. `[MEM]` lines in the
log show RSS, peak RSS and free RAM at every expensive step.

## 5. Machines

| Machine | Profile | Notes |
|---|---|---|
| 4 vCPU / 32 GB SageMaker (reference target) | `32gb` (auto) | full pipeline; stages stream, peak RAM stays well below 32 GB |
| 16 GB laptop | `16gb` (auto) | works; smaller training samples, more parts |
| 64 GB+ | `64gb` (auto) | larger training samples and chunks |

Disk: ~60 GB for a full run (raw candidates ~9 GB, pruned ~6 GB, features
~17 GB, stage-2 features ~14 GB, scores ~4 GB, output ~2 GB). Each stage checks
free disk before starting and stops with a clear message if it is too low.

## 6. Expected behaviour on the 4 vCPU / 32 GB reference machine

Measured at medium scale (full S2/S3 pools, 3 % of S1) on an 8 GB laptop, where
every stage peaked at ≤ 2.5 GB RSS (EXP-009). Per-part memory does not grow with
the number of S1, so on 32 GB the peak is set by the profile, not the data:

| stage | peak RAM (32gb profile, estimate) | time on 4 vCPU (estimate) |
|---|---|---|
| normalize | ~3 GB | ~15 min (already done, adopted) |
| candidates | ~4–6 GB | ~6–8 h (already done, adopted) |
| prune | ~3 GB | ~20–30 min |
| features | ~4 GB | ~5 h (RapidFuzz, ~236M pairs) |
| stage1 (5 folds on 16M rows + OOF scoring) | ~7 GB | ~2–3 h |
| stage2 (features + 5 folds on 12M rows + scoring) | ~8 GB | ~2–3 h |
| decision | ~4 GB | ~15 min |
| predict | ~4 GB | ~30–45 min |
| submit (writer + official validator + zip) | ~12 GB (validator keeps all candidate ids in Python sets) | ~15 min |

Times are extrapolations; the `[MEM]` / timing lines in `artifacts/logs/`
show the real numbers. The run can be stopped and resumed at any point.

## 7. Cost hygiene

Stop the instance when nothing runs; artifacts stay on the EBS volume and the
next run resumes. Save what matters to S3:

```bash
aws s3 sync artifacts/models s3://<your-bucket>/entity-forge/models
aws s3 sync output s3://<your-bucket>/entity-forge/output
```
