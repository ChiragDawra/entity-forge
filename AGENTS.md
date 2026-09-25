# Working on entity-forge (humans and coding agents)

Read `README.md` and `docs/SYSTEM_DESIGN.md` first. The problem statement is
`docs/PROBLEM_STATEMENT.md`; it is authoritative for task constraints.
`docs/archive/` holds the original planning documents, superseded by these files.

## Priorities

1. A valid, reproducible submission (validator passes, invariants hold).
2. Candidate recall: a pair lost in blocking can never be predicted.
3. Out-of-fold macro F0.5 measured with `entity_forge.metrics` (exact metric).
4. Everything else only when it moves (2) or (3) on measured evidence.

## Rules

- Never brute-force S1 × (S2+S3) and never build a dense all-pairs matrix.
- Never assume a fixed number of matches; empty sets are valid (5.6 % of S1).
- Exclusivity (one S1 per S2/S3 record) holds in training data: keep it in the decision layer; re-check it if the data changes.
- Country is an open set; never filter, hard-code or one-hot it. France exists only in test.
- No external data, gazetteers, geocoders or APIs. Dictionaries are hand-written in `dictionaries.py`; list any addition in `docs/METHODOLOGY.md`.
- Folds are grouped by S1 (`model.fold_expr`); never tune on the out-of-fold fold.
- Train and test go through the same functions. A new feature goes into `features.py` / `second_stage.py`, never into a notebook cell.
- Every predicted match must be in `candidate_pairs.tsv`; run the official validator before any upload.
- Raw files in `dataset/` are read-only. Generated files go to `artifacts/` and `output/` (git-ignored).

## Code

- Logic lives in `src/entity_forge/`; notebooks stay thin (settings, one stage call, a results table).
- Polars / NumPy / SciPy sparse; RapidFuzz `cpdist` for pair similarities; no Python loop per record or pair.
- Type hints on public functions; deterministic seeds; logging, not print, in library code.
- New dependency: necessary, permissively licensed (MIT/BSD/Apache/ISC), pinned in `requirements.txt`.
- Tests: `pytest` (pure, fast, synthetic data). Cover normal, missing, empty, duplicate and malformed input for new logic.

## Experiments

Log each meaningful experiment in `docs/EXPERIMENTS.md`:

```text
ID / hypothesis / change / data slice / recall (union, avg & p95 per S1) /
OOF macro F0.5 / runtime / keep or reject + reason
```

Change one major factor at a time. Compare only on the same folds and S1 sample.
Keep rejected experiments in the log.

## Running

```bash
EF_DEV_MODE=1 bash scripts/run_notebooks.sh   # smoke test on a small slice
bash scripts/run_notebooks.sh                 # full run (see docs/SAGEMAKER.md)
pytest -q
```
