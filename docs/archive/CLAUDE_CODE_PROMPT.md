# Initial Claude Code Prompt

You are working on the Amazon ML Challenge 2026 Business Entity Resolution project.

Read these files before making changes:

1. `PRD.md`
2. `SYSTEM_DESIGN.md`
3. `AGENTS.md`
4. `IMPLEMENTATION_PLAN.md`
5. `EXPERIMENT_PROTOCOL.md`
6. `CODING_STANDARDS.md`

Do not redesign the architecture.

## First task

Inspect the repository and identify:
- current files,
- dataset locations,
- existing code,
- Python environment,
- installed dependencies.

Then propose the smallest implementation sequence for Stage 1 and Stage 2 only.

Do NOT implement embeddings, transformers, graph propagation, or AWS infrastructure yet.

First implement:

1. exact macro F0.5 evaluator,
2. data schema validation,
3. chunked data loading,
4. training ground-truth loading,
5. deterministic normalization module,
6. unit tests for these components.

After implementation:
- run tests,
- show changed files,
- show commands executed,
- show remaining work.

Do not claim success without running the tests.

## Important

The hackathon is time constrained.

Prefer working, measurable code over speculative abstraction.

Do not ask for permission for every small implementation detail. Follow the project documents unless a requirement is ambiguous or a change would affect the architecture/schema.
