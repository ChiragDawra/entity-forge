# Coding Standards

## General

- Python 3.11+ compatible where practical.
- Type hints for public functions.
- Small functions with explicit inputs/outputs.
- No hidden global state.
- Deterministic random seeds.
- Logging instead of print for long-running jobs.

## Data

- Never mutate raw input files.
- Explicit schemas.
- Use stable column names.
- Validate required columns.
- Prefer Arrow/Polars/DuckDB for large data.
- Avoid object-heavy pandas operations at full scale.

## Performance

Never do:

```python
for s1 in s1_rows:
    for s2 in s2_rows:
        ...
```

Never construct a dense all-pairs similarity matrix.

Prefer:
- sparse TF-IDF,
- vectorized operations,
- chunking,
- indexed lookup,
- batch RapidFuzz,
- Arrow/NumPy.

## Memory

Long-running jobs must:
- process chunks,
- release temporary matrices,
- write checkpoints,
- log peak/estimated memory.

## ML

- Training/validation split by S1 entity.
- Avoid leakage through duplicated entity groups.
- Keep feature generation identical between train and test.
- Save feature schema with model.
- Keep candidate generation deterministic.

## Testing

Every module should have tests for:
- normal case,
- missing value,
- empty string,
- duplicate/ambiguous values,
- malformed input.

Critical tests:
- normalization,
- candidate generation,
- F0.5,
- output validation,
- match⊆candidate invariant.

## Dependencies

Before adding a dependency:
1. Is it necessary?
2. Is it compatible with the environment?
3. Is its license allowed?
4. Does it create a deployment problem?
5. Is there a simpler built-in/installed alternative?

## Output

Never silently reorder or drop required S1 entities.

Always validate IDs before final write.
