# Amazon ML Challenge 2026 — Claude Master Instructions

## Role

You are the primary engineering agent helping build the Amazon ML Challenge 2026 Business Entity Resolution solution.

The team is under hackathon time pressure. Prioritize:
1. A working end-to-end submission.
2. High candidate recall with manageable candidate volume.
3. Precision-heavy macro F0.5 performance.
4. Reproducibility and validation.
5. Measured improvements over speculative complexity.

Do not continuously redesign the architecture. Implement the agreed architecture, measure it, and change it only when experiments justify the change.

## Source of truth

The project plan is documented in:
- `PRD.md`
- `SYSTEM_DESIGN.md`
- `AGENTS.md`
- `IMPLEMENTATION_PLAN.md`
- `EXPERIMENT_PROTOCOL.md`
- `CODING_STANDARDS.md`

The challenge statement and supplied dataset statistics are authoritative for task constraints.

## Core architecture

```text
Raw S1/S2/S3
    ↓
Normalization + derived fields
    ↓
Multi-channel candidate generation
    ├── exact normalized name
    ├── name/postal/street keys
    ├── rare-token retrieval
    └── character TF-IDF top-K
          └── optional embeddings later
    ↓
Candidate union
    ↓
Candidate recall evaluation
    ↓
Pairwise feature generation
    ↓
LightGBM binary classifier
    ↓
OOF probabilities
    ↓
Threshold / top-k decision
    ↓
Optional calibrated expected-F0.5 selection
    ↓
Optional exclusivity/context/graph post-processing
    ↓
matching_results.tsv
candidate_pairs.tsv
    ↓
Official validator
```

## Non-negotiable rules

- Never brute-force all S1 × (S2+S3) pairs.
- Never assume 30–40 candidates is correct. Measure candidate recall at multiple K values.
- Never force every S1 to have a match.
- Never assume one match per S1.
- Never enforce reverse exclusivity before checking training ground truth.
- Never start with a transformer.
- Do not let optional embeddings, graph logic, or libpostal installation block the baseline.
- Do not use external business/entity data, geocoders, government databases, commercial ER APIs, or internet data augmentation.
- Do not copy another team's implementation.
- Preserve raw and multiple normalized representations.
- Group validation by S1 entity.
- Include hard negatives produced by realistic blocking.
- Every final predicted match must be present in `candidate_pairs.tsv`.
- Run the official validator before submission.

## Working style

When implementing:
1. Inspect the current repository before changing it.
2. Prefer small, testable modules.
3. Add or update tests for important logic.
4. Run the narrowest relevant test after each change.
5. Record important experiment results.
6. Do not silently change data schemas.
7. Do not introduce a new dependency without checking license, Python compatibility, and necessity.
8. If a requested optimization conflicts with candidate recall or submission correctness, prioritize correctness.
9. If compute becomes a bottleneck, profile first and then optimize the bottleneck.
10. Keep the pipeline resumable; long jobs should write checkpoints/intermediates.

## Definition of done

A stage is done only when:
- its outputs are deterministic/reproducible,
- its tests pass,
- its runtime/memory behavior is known,
- and its output schema is documented.

Do not report a stage as complete merely because code was written.
