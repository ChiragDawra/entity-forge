# Running the full pipeline on AWS SageMaker (or any big machine)

The notebooks are the entry points; the heavy logic lives in `src/entity_forge`.
Every stage writes Parquet artifacts under `artifacts/`, so you can stop, resume,
or move between machines at any stage boundary.

## 1. Machine

| Stage | Bottleneck | Recommendation |
|---|---|---|
| 01 normalize | ~24M rows, partly single-threaded | any |
| 02 candidates | sparse top-K search (multithreaded) | many cores |
| 03 features | RapidFuzz (multithreaded) | many cores |
| 04 training | LightGBM on tens of millions of rows | 64 GB+ RAM |
| 05 predict | LightGBM inference | any |

A single **`ml.r5.8xlarge`** (32 vCPU, 256 GB) or **`ml.m5.12xlarge`** (48 vCPU,
192 GB) notebook instance runs everything without tuning. No GPU needed.
Give the instance **≥ 200 GB EBS**: train + test artifacts take tens of GB.

On smaller machines (e.g. 32 GB) lower `train_frac` in notebook 04. That only
controls how many rows *fit* the models; out-of-fold scoring always covers every pair.

### Expected scale (full data)

| Stage | Volume | Rough time, 8 cores → 32 cores |
|---|---|---|
| 01 normalize | 24.2M records | 10 min |
| 02 candidates | 3.9M S1 queries × 6 channels, ~100 raw → 60 kept per S1 | ~8 h → ~2–2.5 h (the `both` channel dominates) |
| 03 features | ~235M pairs (train 132M, test 104M) | ~1.5 h → ~25 min |
| 04 training | stage 1 + stage 2, 5 folds each | depends on `train_frac`; start with the defaults |
| disk | features for train + test | ~40–60 GB Parquet |

Measured on the dev slice (Apple M1, 8 cores): normalize 80 s, candidates 2.5 min,
features 1 min, training 5 folds × 2 stages ≈ 10 min.

## 2. Setup (SageMaker notebook-instance terminal)

```bash
cd ~/SageMaker
git clone https://github.com/ChiragDawra/entity-forge.git
cd entity-forge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name entity-forge

# Data: the competition files, same layout as the challenge kit
aws s3 cp --recursive s3://<your-bucket>/dataset/ dataset/
```

Expected layout (never committed to git):

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## 3. Run

Interactive: open `notebooks/01_normalize.ipynb` … `05_predict_and_submit.ipynb`
in order with the `entity-forge` kernel.

Headless (survives a closed browser tab):

```bash
export EF_N_THREADS=$(nproc)
nohup bash scripts/run_notebooks.sh > run.log 2>&1 &
tail -f run.log
```

Resume from a stage: `bash scripts/run_notebooks.sh 04 05`.

## 4. Smoke test first (cheap)

```bash
EF_DEV_MODE=1 bash scripts/run_notebooks.sh
```

Runs every stage on a consistent ~2 % slice into `artifacts_dev/` and `output_dev/`.
Real artifacts are untouched.

## 5. Settings

Every field of `src/entity_forge/settings.py` can be overridden with an
`EF_<FIELD>` environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `EF_N_THREADS` | all cores | threads for search, RapidFuzz and LightGBM |
| `EF_MAX_CANDIDATES` | 60 | candidates kept per S1 after stage-0 pruning |
| `EF_TOP_K_ADDR`, `EF_TOP_K_BOTH`, `EF_TOP_K_CROSS`, … | 40 / 40 / 20 … | top-K per retrieval channel |
| `EF_N_FOLDS` | 5 | S1-grouped CV folds |
| `EF_DEV_MODE` | 0 | small-slice smoke test |
| `EF_DATA_DIR`, `EF_WORK_DIR` | `dataset`, `artifacts` | move data or artifacts (e.g. to a bigger volume) |

## 6. Cost hygiene

Stop the instance between stages you don't watch; artifacts stay on the EBS
volume. Save what matters to S3:

```bash
aws s3 sync artifacts/models s3://<your-bucket>/entity-forge/models
aws s3 sync output s3://<your-bucket>/entity-forge/output
```
