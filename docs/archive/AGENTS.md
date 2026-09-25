# Agent Instructions

## Agent hierarchy

### Orchestrator
Owns:
- task decomposition,
- integration,
- experiment selection,
- final validation.

Never changes architecture casually.

### Data Agent
Owns:
- schema checks,
- TSV/Parquet conversion,
- chunking,
- data statistics,
- intermediate storage.

Must not alter raw data.

### Normalization Agent
Owns:
- normalization functions,
- name core,
- address parsing,
- transliteration,
- normalization tests.

Must preserve raw fields.

### Blocking Agent
Owns:
- inverted indexes,
- exact blocking,
- TF-IDF retrieval,
- candidate union,
- recall measurements.

Primary KPI:
`pair recall vs candidate volume`.

### Feature Agent
Owns:
- pairwise feature generation,
- vectorized similarity,
- feature schemas.

Primary KPI:
feature correctness and runtime.

### Model Agent
Owns:
- LightGBM,
- grouped CV,
- hard-negative sampling,
- OOF predictions,
- probability calibration.

Primary KPI:
OOF macro F0.5.

### Decision Agent
Owns:
- threshold sweep,
- top-k logic,
- empty-set logic,
- optional expected-F0.5 selection.

Must evaluate exact competition metric.

### Postprocessing Agent
Owns:
- exclusivity analysis,
- context features,
- second-stage models,
- optional graph expansion.

Must not enforce assumptions without training-data evidence.

### Submission Agent
Owns:
- TSV generation,
- schema checks,
- subset invariant,
- official validator,
- packaging.

### Research/Upgrade Agent
Owns:
- embeddings,
- advanced retrieval,
- optional graph methods.

Cannot block baseline delivery.

## Agent communication contract

Every agent must report:

```text
Goal:
Inputs:
Changes:
Outputs:
Tests run:
Metrics:
Runtime:
Memory:
Known limitations:
Recommended next action:
```

Never report "done" without tests or measurable output.

## Parallel work rules

Safe to parallelize:
- normalization and baseline evaluator,
- candidate-channel implementations,
- feature implementation,
- documentation.

Do not parallelize competing changes to the same core pipeline without clear ownership.

## Stop conditions

An agent must stop and ask for integration review when:
- schema changes,
- a dependency is added,
- a candidate-recall regression occurs,
- a new architectural assumption is introduced,
- external data would be required,
- a proposed optimization risks submission correctness.

## Priority

P0: working submission
P1: candidate recall
P1: exact F0.5
P1: LightGBM
P2: feature improvements
P2: decision optimization
P2: validated post-processing
P3: embeddings
P3: graph methods
P4: experimental transformer work
