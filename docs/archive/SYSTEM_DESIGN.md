# System Design — Business Entity Resolution

## 1. Architecture

```text
                  ┌──────────────────────────┐
                  │ train/test TSV datasets  │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Data ingestion            │
                  │ Parquet / chunked access  │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Normalization              │
                  │ raw / normalized / core   │
                  │ transliteration / address │
                  └────────────┬─────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │ Exact indexes   │          │ TF-IDF retrieval│
       │ name/postal/etc │          │ char n-grams    │
       └────────┬────────┘          └────────┬────────┘
                │                            │
                └──────────────┬─────────────┘
                               ▼
                    ┌────────────────────┐
                    │ Candidate union    │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Candidate recall   │
                    │ + budget selection │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Pair features      │
                    │ name/address/etc   │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ LightGBM            │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Match probabilities│
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Set decision       │
                    │ threshold / top-k  │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Optional context   │
                    │ / exclusivity      │
                    └─────────┬──────────┘
                              ▼
                  ┌───────────┴───────────┐
                  ▼                       ▼
        matching_results.tsv     candidate_pairs.tsv
                  │                       │
                  └───────────┬───────────┘
                              ▼
                     Official validator
```

## 2. Component responsibilities

### Data layer
Responsible for:
- reading TSV,
- schema validation,
- Parquet conversion,
- chunking.

### Normalization layer
Responsible for:
- Unicode normalization,
- case folding,
- punctuation handling,
- legal suffix handling,
- address token extraction,
- postal/house/unit extraction,
- transliteration.

Normalization must be deterministic.

### Candidate generation layer

Candidate channels are independent so that failure of one representation does not eliminate a true pair.

Recommended initial channels:
1. normalized name exact
2. name + postal
3. postal + street token
4. rare token
5. character TF-IDF top-K

Optional later:
6. multilingual embedding kNN
7. 2-hop candidate expansion

### Feature layer

Feature groups:
- name similarity,
- address similarity,
- numeric agreement/conflict,
- frequency/ambiguity,
- candidate rank,
- channel provenance.

### Model layer

LightGBM binary classifier.

Input:
`S1 × candidate S2/S3`

Output:
`P(match)`.

Validation must be grouped by S1.

### Decision layer

Initial:
- probability threshold sweep,
- optional top-k restriction.

Advanced:
- calibration,
- expected F0.5 subset selection.

The empty set is always a valid prediction.

### Post-processing

Potential:
- reverse exclusivity,
- competing-S1 context,
- second-stage LightGBM,
- graph/2-hop expansion.

Only enabled after empirical validation.

## 3. Data flow constraints

1. Candidate generation must never require internet access.
2. Candidate pairs must be traceable to the channel(s) that generated them.
3. Final matches must be a subset of final candidates.
4. All long-running stages should support checkpointing.
5. Intermediate files should have explicit schemas.

## 4. Scaling strategy

Use:
- country partitioning,
- S1 chunks,
- sparse matrices,
- vectorized similarity,
- Parquet/Arrow,
- DuckDB/Polars where useful.

Avoid:
- dense N×M matrices,
- Python object-heavy pair storage,
- full pairwise nested loops.

## 5. Failure containment

If an optional component fails:
- embedding failure → continue lexical pipeline,
- libpostal failure → continue custom normalization,
- graph failure → continue first-stage predictions.

The baseline must remain independently executable.
