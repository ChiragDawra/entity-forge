# entity-forge

Business entity resolution for the **Amazon ML Challenge 2026**: for every
Source-1 business record, find all Source-2/3 records of the same business
(zero, one or many), scored by macro F0.5.

Five-stage pipeline, each stage a Jupyter notebook over a tested Python package:

```
normalize → multi-channel sparse retrieval → stage-0 pruning → pair features
          → 2-stage LightGBM (OOF) → exclusivity-aware decision → submission
```

## Why it is built this way

Measured on the training data (`notebooks/00_eda.ipynb`):

- **Exclusivity is perfect**: none of the 7.6M matched S2/S3 records belongs to two S1s.
  The decision layer gives each record to its best S1 only, and competition between
  S1s is a model feature.
- **No true pair crosses countries**: retrieval runs per country (open set: France is test-only).
- **Names are badly corrupted, addresses mostly intact**, and many businesses share a
  building. So retrieval is address-aware. On India, name-only retrieval reaches 0.54
  recall @50; address-aware channels plus targeted channels for address-less and
  Devanagari-romanized records reach **0.985** (US: **0.991**).
- **S2/S3 are not deduplicated** (3.46 matches per S1 on average), so the true matches
  of one S1 are copies of each other. Stage 2 uses similarity to the S1's most
  confident candidates.
- **5.6 % singletons**: "no match" is a tuned outcome, never a fallback.

Details: [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md). Evidence: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Status

- Retrieval measured on real data: recall **0.985** (India) / **0.991** (US) at ~100 raw candidates per S1 (`docs/EXPERIMENTS.md`, EXP-002…006).
- Whole pipeline verified end to end on the dev slice: all six notebooks run and the official validator passes (EXP-007).
- Pipeline is checkpointed, resumable and memory-bounded (every stage streams in parts); verified end to end on the dev slice and, with full S2/S3 pools, through features at ≤ 2.5 GB peak RSS (EXP-008, EXP-009). Realistic pruned recall: India 0.985, US 0.991.
- **Next: full run** with `bash scripts/run_notebooks.sh` (see `docs/SAGEMAKER.md`), then copy the OOF numbers from `artifacts/reports/` into `docs/METHODOLOGY.md`.

## Repository layout

```
entity-forge/
├── notebooks/                       run in order
│   ├── 00_eda.ipynb                 data facts behind every design decision
│   ├── 01_normalize.ipynb           raw TSV → normalized Parquet
│   ├── 02_candidates.ipynb          retrieval channels, union, stage-0 pruning, recall report
│   ├── 03_features.ipynb            pair / context / competition features
│   ├── 04_train.ipynb               stage-1 + stage-2 LightGBM (OOF), decision tuning
│   └── 05_predict_and_submit.ipynb  test inference, TSVs, validator, submission zip
├── src/entity_forge/                all logic (notebooks stay thin)
├── tests/                           pytest, synthetic data
├── utils/validate_submission.py     official validator (unchanged)
├── scripts/run_notebooks.sh         single entry point: run / resume / status / force
├── docs/                            design, experiments, methodology, SageMaker guide
├── dataset/                         competition data (git-ignored, read-only)
├── artifacts/                       generated: norm/ candidates/ features/ models/ scores/ reports/
└── output/                          generated: matching_results.tsv, candidate_pairs.tsv
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: brew install libomp (LightGBM)
pytest -q                                # unit + equivalence tests

# put the competition files in dataset/train and dataset/test, then:
bash scripts/run_notebooks.sh --dev      # ~2 % slice end to end (artifacts_dev/), minutes
bash scripts/run_notebooks.sh            # full run; re-run the same command to resume
bash scripts/run_notebooks.sh --status   # what is done / missing / stale
```

`scripts/run_notebooks.sh` is the single entry point. It runs the stages
`normalize → candidates → prune → features → stage1 → stage2 → decision →
predict → submit`, each in its own process. Every finished partition, part and
model is checkpointed and skipped on the next run, so an interrupted run
resumes where it stopped. Useful flags: `--from STAGE`, `--to STAGE`,
`--only STAGE[,..]`, `--force STAGE`, `--profile 16gb|32gb|64gb`,
`--notebooks` (execute the notebooks instead; same checkpoints).

**Hardware:** CPU only. A memory profile picked from detected RAM bounds every
chunk and training sample, so the same code runs on 16 GB, 32 GB (the 4-vCPU
SageMaker reference) and 64 GB+ machines without edits. Knobs, checkpoints and
resume rules: [`docs/SAGEMAKER.md`](docs/SAGEMAKER.md).

## Outputs

| File | Content |
|---|---|
| `output/matching_results.tsv` | final matches, one row per test S1 (uploaded to the leaderboard) |
| `output/candidate_pairs.tsv` | every pair the matching model scored (stage-0 output) |
| `submission/<team>_submission.zip` | `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + methodology |

Invariants asserted before writing: every test S1 exactly once, matches ⊆ candidates,
no S2/S3 record under two S1s. The official validator runs in notebook 05.

## Documentation

- [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md): the official challenge statement
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md): architecture, evidence, invariants, changes vs the original plan
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): measured results
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): the submission write-up (challenge template)
- [`docs/SAGEMAKER.md`](docs/SAGEMAKER.md): running on AWS
- [`AGENTS.md`](AGENTS.md): rules for contributors and coding agents
- [`docs/archive/`](docs/archive/): the original planning documents

## Fair play

Only the provided data. No external databases, geocoders or APIs; every
normalization dictionary is hand-written (`src/entity_forge/dictionaries.py`).
All dependencies are MIT / BSD / Apache-2.0 / ISC. No pretrained model is used.
