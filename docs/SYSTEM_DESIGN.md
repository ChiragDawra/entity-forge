# System design

Business entity resolution: for every Source-1 (S1) record, find all Source-2/3
(S2/S3) records of the same business. Metric: macro F0.5 over all S1 entities
(singletons included). See `PROBLEM_STATEMENT.md`.

## 1. Facts measured on the training data (notebook 00)

| # | Fact | Design consequence |
|---|---|---|
| F1 | 0 of 7,638,365 matched S2/S3 ids appears under two S1s | hard **exclusivity** in the decision layer; target-competition features |
| F2 | 0 true pairs cross countries | candidate generation runs **per country** (open set; France only in test) |
| F3 | 5.6 % singletons, same rate in US and India | empty prediction is a first-class outcome, tuned on the exact metric |
| F4 | avg 3.46, max 11 matches per S1; S2/S3 are not deduplicated | true matches are noisy copies of each other → **sibling-similarity** features in stage 2 |
| F5 | names heavily corrupted (reordering, legal-form noise, DBA, domains, OCR-style digits, `CENTER PATTERSON CORP` for `Patterson Chadwick Corp`); addresses mostly intact | address-aware retrieval; name-only retrieval is weak |
| F6 | ~5 % of S2 names and many state names in Indic scripts | anyascii transliteration (+ anusvara fix) and a phonetic skeleton |
| F7 | 3–4 % of S2/S3 records have no address | name retrieval restricted to address-less targets |
| F8 | ~30 % of S1 names are duplicated (chains, generic names) | `name_freq` features; name × address conjunction channel |

## 2. Pipeline

```text
dataset/*.tsv
   │  01 normalize (Polars, vectorized; transliteration on unique non-ASCII only)
   ▼
artifacts/norm/{split}_s{1,2,3}.parquet
   │  02 candidates, per country:
   │     channels (hashed sparse TF-IDF, top-K by cosine via sparse_dot_topn)
   │       addr    address words
   │       both    name words ⊕ address words
   │       cross   name-token × address-token conjunctions
   │       name    name words
   │       noaddr  name words, searched only among address-less targets
   │       translit phonetic name × address, searched only among non-Latin targets
   │     union → exact cosine of every channel for every union pair
   │     stage-0 LightGBM on retrieval features → keep top-N per S1
   ▼
artifacts/candidates/{split}/{country}.parquet          (= candidate_pairs.tsv)
   │  03 features: retrieval + query context + target competition
   │               + name/address string similarity (RapidFuzz cpdist) + numbers + flags
   ▼
artifacts/features/{split}/{country}/part-*.parquet
   │  04 stage 1: LightGBM, 5 S1-grouped folds → OOF p1 for every train pair
   │     stage 2: LightGBM on stage-1 features + p1 set context
   │              (S1 max/second/sum/rank, best competing S1 for the target,
   │               similarity to the S1's top-1/top-2 candidates) → OOF p
   │     decision grid on OOF with the exact metric:
   │        threshold | exclusivity + threshold | exclusivity + expected-F0.5 subset
   ▼
artifacts/models/{stage0,stage1_fold*,stage2_fold*}.txt, decision.json
   │  05 test: fold-average p1 → stage 2 → p → frozen decision
   │     assert predicted ⊆ candidates, one S1 per target, every S1 exactly once
   ▼
output/matching_results.tsv, output/candidate_pairs.tsv → official validator → submission zip
```

## 3. Components

| Module | Responsibility |
|---|---|
| `dictionaries.py` | hand-written legal forms, fillers, DBA markers, address abbreviations, US/India/France region codes, native-script Indian state names |
| `normalize.py` | `name_norm`, `name_core`, `name_alt`, `name_phon`, `addr_norm`, `addr_nums` |
| `text_vectors.py` | hashed TF-IDF blocks (`word`, `charN`, `cross`) built with Polars; exact pair cosines |
| `blocking.py` | channels, top-K search, union, recall measurement |
| `pruning.py` | stage-0 re-ranker and top-N cut |
| `features.py` | pair, query-context and target-competition features |
| `second_stage.py` | p1 set context and sibling similarity |
| `model.py` | LightGBM train/predict/save, deterministic S1-grouped folds |
| `decision.py` | exclusivity, threshold, expected-F0.5 subset selection |
| `metrics.py` | exact macro F0.5 (the authority for every experiment) |
| `stages.py` | resumable stage functions the notebooks call |
| `settings.py` | paths and knobs; `EF_*` env overrides; `dev_mode` |

## 4. Changes from the original plan (`docs/archive/`) and why

| Original plan | Now | Evidence |
|---|---|---|
| exact-key channels (name, name+postal, postal+street) | sparse TF-IDF channels incl. **name × address conjunctions** | S1 addresses rarely carry postal codes; exact keys break on the observed noise. `cross` alone: India recall 0.75 already at K=10 |
| name char-TF-IDF as the main fuzzy channel | address-aware channels first; targeted channels for address-less and non-Latin records | India, 30k S1: name char-3 0.54 @50; address 0.85 @40; addr ∪ both ∪ cross 0.964; + noaddr + translit **0.985**. US: 0.991 |
| "exclusivity is experimental" | exclusivity enforced (and used as features) | F1: zero violations in 7.6M pairs |
| optional 2-hop expansion | stage-2 sibling similarity to top-1/top-2 | F4; far cheaper than target-target kNN |
| fixed candidate cap | learned stage-0 re-rank; cap chosen from measured recall | union is ~110 pairs per S1 before pruning |
| design around an 8 GB laptop | defaults for a 64 GB+ box; `dev_mode` for laptops | requirement: best model, trained on SageMaker |
| scripts | numbered notebooks over a tested package | requirement: Jupyter format |

## 5. Execution model: bounded memory, checkpoints, resume

The full data (396M raw candidate rows, ~236M pruned pairs) never fits in RAM on
the 32 GB reference machine (a raw US partition alone is ~8 GB as a DataFrame),
so every stage streams:

| Stage | How memory stays bounded |
|---|---|
| retrieval | one channel's vectors in RAM at a time; vectors spilled to disk; transposed target matrix *replaces* the row-major one during search; union and exact cosines computed per S1 range; parts streamed into one file |
| vectorization | TF-IDF built in 500k-row chunks (document frequencies accumulated per chunk) |
| stage-0 pruning | per-target statistics (count, max cosine per channel) from one streaming group-by; then S1-range chunks |
| features | per-target competition statistics (count, top-8 `cos_both`, max `cos_addr`/`cos_name`) from one streaming group-by; one feature part per pruned part |
| training | S1-grouped sample capped by the profile's row budget; one LightGBM `Dataset`, folds via `Dataset.subset` (no per-fold matrix copies) |
| scoring / stage 2 / test | part by part; stage-2 set context from streaming group-bys over stage-1 scores |
| decision / submission | only pairs with p ≥ 0.01 are materialized; `candidate_pairs.tsv` is streamed part by part |

Chunked computations are proven equal to whole-partition computations by
`tests/test_retrieval_equivalence.py` and `tests/test_streaming_equivalence.py`;
the new retrieval reproduced the original dev candidates byte for byte (EXP-008).

Checkpoints (`checkpoints.py`): every artifact gets a manifest (stage, key,
semantic config hash, rows, schema, file sizes, timestamp) after it is complete.
Part files are written atomically, and `_PLAN.json` pins part boundaries so a
resume reuses them. Legacy `norm/` and `candidates_raw/` files are validated and
adopted. `pipeline.py` runs each stage in its own process and skips valid work.

## 6. Invariants (asserted in code)

1. Every test S1 appears exactly once in both output files (empty lists allowed).
2. Predicted matches ⊆ candidate pairs.
3. With exclusivity on, no S2/S3 id appears under two S1s.
4. Only known S1 ids; candidates come from the same country's pool.
5. Train and test share the same code path; the feature schema is saved with every model and checked on load.
6. Folds are grouped by S1 id (deterministic hash); early stopping never uses the out-of-fold fold.

## 7. Fair play

No external data, geocoders, gazetteers or APIs. Every dictionary is
hand-written in `dictionaries.py`. Libraries: Polars, NumPy, SciPy, RapidFuzz
(MIT), sparse_dot_topn (Apache-2.0), anyascii (ISC), LightGBM (MIT). No
pretrained model is used; the LightGBM models are far below the 8B-parameter cap.
