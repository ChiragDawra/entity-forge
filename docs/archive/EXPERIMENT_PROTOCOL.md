# Experiment Protocol

## Purpose

Prevent random tweaking during the hackathon.

Every meaningful experiment gets an ID.

Example:

`EXP-014`

## Required record

```text
Experiment:
Hypothesis:
Baseline:
Change:
Training data:
Validation split:
Candidate configuration:
Features:
Model:
Decision rule:
Candidate recall:
Average candidates:
P95 candidates:
OOF F0.5:
Runtime:
Memory:
Result:
Keep/Reject:
Reason:
```

## Primary metrics

### Candidate stage

1. Pair recall
2. Average candidates/S1
3. P95 candidates/S1
4. P99 candidates/S1
5. Runtime
6. Peak memory

### Matching stage

1. Macro F0.5
2. Precision
3. Recall
4. Empty-set accuracy
5. Per-country metrics where applicable

## Important experiments

### EXP-001
Exact normalized name blocking.

### EXP-002
Exact keys + postal/street.

### EXP-003
TF-IDF char 3-gram.

### EXP-004
Union of lexical + TF-IDF.

### EXP-005
Candidate K sweep:
10/20/30/40/60/80/100.

### EXP-006
Simple similarity baseline.

### EXP-007
LightGBM with initial features.

### EXP-008
Hard-negative strategy.

### EXP-009
Threshold sweep.

### EXP-010
Threshold + top-k.

### EXP-011
Probability calibration.

### EXP-012
Expected-F0.5 decision.

### EXP-013
Exclusivity.

### EXP-014
Second-stage context model.

### EXP-015
Embeddings.

### EXP-016
2-hop expansion.

## Rules

- Change one major factor at a time where practical.
- Never compare experiments using different validation splits.
- Never select a model using the leaderboard alone.
- Do not delete rejected experiments; record them.
- If an upgrade improves candidate recall but hurts final F0.5, investigate why before keeping it.
- If an upgrade improves final F0.5 but greatly increases runtime, record the tradeoff.
