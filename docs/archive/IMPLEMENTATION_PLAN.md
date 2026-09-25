# Implementation Plan

## Stage 0 — Repository setup

Create a clean structure:

```text
project/
├── README.md
├── PRD.md
├── SYSTEM_DESIGN.md
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
├── EXPERIMENT_PROTOCOL.md
├── CODING_STANDARDS.md
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── business_entity_resolution/
│       ├── data/
│       ├── normalization/
│       ├── blocking/
│       ├── features/
│       ├── models/
│       ├── decision/
│       ├── evaluation/
│       └── submission/
├── tests/
├── scripts/
└── output/
```

## Stage 1 — Evaluator

Implement exact macro F0.5 first.

Test:
- true empty / predicted empty
- true empty / predicted non-empty
- true non-empty / predicted empty
- partial overlap
- exact overlap
- duplicate predictions

This evaluator is the authority for local experiments.

## Stage 2 — Data

Implement:
- schema validation,
- chunk reader,
- Parquet conversion,
- training ground-truth loader.

## Stage 3 — Normalization

Implement deterministic transformations and unit tests.

Keep:
- raw,
- normalized,
- core,
- transliterated.

## Stage 4 — Blocking V1

Implement:
1. exact name-core
2. name/postal
3. postal/street
4. rare token

Measure recall.

## Stage 5 — TF-IDF

Implement sparse character n-gram retrieval.

Test K:
10/20/30/40/60/80/100.

Measure:
- recall,
- candidates,
- runtime,
- memory.

## Stage 6 — First submission baseline

Use a simple similarity decision.

Generate:
- matching_results.tsv
- candidate_pairs.tsv

Run official validator.

## Stage 7 — Features

Start with 20–30 features.

Create a reproducible feature schema.

## Stage 8 — LightGBM

Train grouped CV.

Use hard negatives.

Save:
- model,
- feature list,
- OOF probabilities,
- metrics.

## Stage 9 — Decision

Run:
- threshold sweep,
- threshold + top-k.

Add empty-set logic.

## Stage 10 — Advanced decision

Only now test:
- probability calibration,
- expected-F0.5 selection.

## Stage 11 — Post-processing

Test:
- exclusivity,
- competing-S1 context,
- second-stage LightGBM.

## Stage 12 — Optional upgrades

Test one at a time:
- embeddings,
- graph expansion,
- improved transliteration.

## Stage 13 — Final inference

Freeze configuration.

Run test inference.

Assert:
- every S1 appears exactly once,
- no invalid IDs,
- no duplicate matches,
- predicted matches ⊆ candidates.

Run official validator.

## Stage 14 — Final package

Include required:
- output files,
- source code,
- README,
- requirements,
- documentation.

Record final configuration and experiment evidence.
