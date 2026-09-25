# PRD — Amazon ML Challenge 2026 Business Entity Resolution

## 1. Product

A reproducible machine-learning pipeline that resolves noisy business records from Source 2 and Source 3 to Source 1 reference entities.

## 2. Problem

For every Source 1 entity, identify all matching Source 2 and Source 3 records.

An S1 entity may have:
- zero matches,
- one match,
- multiple matches.

The final system must output:
- `matching_results.tsv`
- `candidate_pairs.tsv`

## 3. Data model

Each record contains:
- `entity_id`
- `business_name`
- `business_address`
- `country`

Source is identified by the source-specific ID/file.

Source 1 is the deduplicated reference source.

## 4. Known dataset facts

Training:
- S1: 2,206,821
- S2: 5,034,616
- S3: 5,285,603

Test:
- S1: 1,732,544
- S2: 4,887,273
- S3: 5,082,316

Training ground truth:
- S1 entities: 2,206,821
- zero-match S1: 123,247
- total matches: 7,638,365
- average matches/S1: 3.4613
- maximum matches/S1: 11

Test includes France in addition to US and India. Country must therefore be handled as an open-set string value.

## 5. Evaluation

Primary objective:
- maximize macro F0.5 over S1 entities.

F0.5 weights precision more heavily than recall.

Implications:
- false merges are costly,
- empty prediction is valid,
- the system must explicitly support no-match,
- the model must retrieve multiple matches where appropriate.

## 6. Functional requirements

### FR-1 Data ingestion
Load training/test TSVs reliably and support chunked processing.

### FR-2 Normalization
Create:
- raw representation,
- normalized representation,
- name core,
- optional transliterated representation,
- address-derived fields.

### FR-3 Candidate generation
Generate candidates using multiple independent channels:
- exact normalized name,
- name/postal,
- postal/street,
- rare tokens,
- character TF-IDF top-K.

Candidate channels must be unioned.

### FR-4 Candidate recall
Measure whether known ground-truth pairs are present in the candidate set.

Candidate budget must be data-driven.

### FR-5 Pair features
Compute name, address, numeric, ambiguity, and retrieval-context features.

### FR-6 Matching model
Train LightGBM binary classifier on candidate pairs.

### FR-7 Decision
Convert pair probabilities into a set of S2/S3 IDs per S1, including an empty set.

### FR-8 Output
Produce valid competition TSVs.

### FR-9 Validation
Run the official validator and enforce:
`predicted_matches ⊆ candidate_pairs`.

## 7. Non-functional requirements

- scalable to millions of records,
- chunked / resumable,
- deterministic where possible,
- no dense all-pairs similarity matrices,
- minimal unnecessary dependencies,
- license-compliant,
- reproducible from documented commands.

## 8. Out of scope for the baseline

- cross-encoder fine-tuning,
- custom contrastive training,
- MinHash/LSH tuning,
- libpostal troubleshooting,
- graph propagation,
- large-scale embedding infrastructure.

These are optional upgrades after the baseline works.

## 9. Success criteria

Baseline:
- complete end-to-end pipeline,
- valid output,
- candidate recall measured.

Strong system:
- LightGBM improves exact OOF macro F0.5 over similarity baseline,
- candidate recall remains high,
- runtime/memory remain practical.

Final:
- official validator passes,
- final output is reproducible,
- all selected upgrades have measured evidence.
