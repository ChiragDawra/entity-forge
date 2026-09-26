# Experiment log

All numbers are measured on the provided training data unless stated otherwise.
Recall experiments use a random sample of S1 (seed 42) retrieving against the
**full** S2 ∪ S3 pool of the same country, so the distractor density is real.
Timings: Apple M1, 8 cores, 8 GB RAM (two experiments sometimes shared the CPU).

## EXP-000 · Data facts (notebook 00)

| Fact | Value |
|---|---|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 (adds France: 259k S1) |
| true pairs | 7,638,365 (avg 3.46 per S1, max 11) |
| singleton S1 | 123,247 (5.58 %; US 5.58 %, India 5.59 %) |
| S2/S3 ids under > 1 S1 | **0** |
| true pairs crossing countries | **0** |
| S2 names with Devanagari | 5.3 % (non-ASCII 15 %: accents injected as noise) |
| S2/S3 without address | 3.0 % (India), 3.6 % (US) |
| India true pairs sharing < 2 address tokens | 3.8 %, all with an empty target address |

## EXP-001 · Normalization throughput

`normalize_records` on 1M S2 rows: 9.4 s. Full train + test (24.2M rows): ~9 min.

## EXP-002 · Single retrieval channels (India, 30k S1, 4.13M targets)

| Channel | max_df | recall @10 | @20 | @40 | @100 | search s |
|---|---|---|---|---|---|---|
| name char-3 | 2 % | 0.411 | 0.466 | ~0.52 | 0.601 | 335 |
| name char-3 | 0.3 % | 0.182 | 0.223 | 0.267 | 0.332 | 6 |
| phonetic char-3 | 0.3 % | 0.098 | 0.118 | 0.141 | 0.182 | 4 |
| name words | 1 % | 0.386 | 0.442 | 0.500 | 0.581 | 14 |
| address words | 0.5 % | 0.757 | 0.794 | 0.818 | 0.842 | 26 |
| address words | 2 % | 0.796 | 0.831 | 0.854 | 0.876 | 80 |
| name words ⊕ address words | 2 % | 0.772 | 0.801 | 0.813 | 0.846 | 129 |
| name × address conjunctions | 1 % | 0.748 | 0.750 | 0.752 | 0.753 | 35 |

Takeaways: India names are generic and duplicated, so name-only retrieval is
weak; hard df-pruning destroys char-gram channels; the conjunction channel is
nearly all-or-nothing (its hits rank at the very top).

## EXP-003 · Channel unions (India, same sample, K = 40 per channel)

| Union | recall | avg candidates / S1 |
|---|---|---|
| address(2 %) ∪ both | 0.950 | 74 |
| address(2 %) ∪ cross | 0.930 | 76 |
| address(2 %) ∪ both ∪ cross | **0.964** | 108 |
| + name char-3 | 0.965 | 127 |
| + name words | 0.964 | 134 |

Decision: address ∪ both ∪ cross is the backbone; other channels must earn their place.

## EXP-004 · Single retrieval channels (US, 20k S1, 6.19M targets)

| Channel | max_df | @10 | @20 | @40 | @100 |
|---|---|---|---|---|---|
| address words | 2 % | 0.841 | 0.883 | 0.906 | 0.921 |
| name words | 1 % | 0.541 | 0.612 | 0.665 | 0.720 |
| name × address conjunctions | 1 % | 0.868 | 0.869 | 0.869 | 0.869 |
| name char-3 | 1 % | 0.505 | 0.572 | 0.621 | 0.680 |
| name char-4 | 1 % | 0.542 | 0.610 | 0.659 | 0.718 |
| name words ⊕ address words | 2 % | 0.920 | 0.941 | **0.955** | 0.965 |

## EXP-005 · Channel unions (US, same sample)

| Union | K | recall | avg candidates / S1 |
|---|---|---|---|
| address ∪ both | 20 | 0.982 | 35 |
| address ∪ both | 40 | 0.988 | 73 |
| address ∪ cross | 40 | 0.949 | 76 |
| address ∪ both ∪ cross | 20 | 0.987 | 50 |
| address ∪ both ∪ cross | 40 | **0.991** | 106 |
| + name words | 40 | 0.991 | 130 |
| + name char-3 | 40 | 0.992 | 135 |

Decision: address (K=40) ∪ both (K=40) ∪ cross (K=20) plus a cheap name-word
channel (K=20). Char-gram name channels were dropped from retrieval: lower
recall than word tokens at 2–3× the search time. Remaining misses are
dominated by address-less targets (see EXP-006).

## EXP-006 · Closing the India gap (India, same 30k sample)

Misses of address ∪ both ∪ cross (3.6 % of true pairs) were inspected: only 7 %
had an empty target address. Most were **Devanagari names romanized without
vowels** (`laiph phainens` = life finance, `lots intrnesnl` = lotus
international, `vhait indo srvisej` = white indo services) next to a
truncated address (`102 2 bangalore ka`). Fix: a phonetic key that makes both
spellings converge (soft c/g, `tion`, `-j` plurals, retroflex d→t, vowels
dropped: "life finance" and "laiph phainens" → `lf fns`), then a phonetic
name × address conjunction channel **searched only among non-Latin-script
targets** (5 % of the pool → cheap).

| Union (address K40, both K40, cross K20, …) | recall | avg / S1 |
|---|---|---|
| address ∪ both ∪ cross | 0.963 | 89 |
| + name search among address-less targets (K10) | 0.966 | 97 |
| + phonetic × address among non-Latin targets (K20) | 0.984 | 107 |
| + both of the above | **0.985** | 109 |

Remaining 1.5 %: 22 % address-less, the rest mostly concatenated names
(`englishindia`) with a truncated address. Final channel set in
`stages.make_channels`: address, both, cross, name (K10, safety net + `cos_name`
feature), no-address name, transliteration.

## EXP-007 · End-to-end smoke test (dev slice — NOT representative)

`EF_DEV_MODE=1 bash scripts/run_notebooks.sh`: 2 % of S1 (train 43,955; test
34,930 incl. 5,163 France), their true targets, and 10 % of distractor targets.
Scores are **optimistic** (90 % of distractors removed); they only prove the
pipeline works end to end.

| Check | Result |
|---|---|
| runtime, Apple M1 8 cores | normalize 80 s · candidates 2.5 min · features 1 min · training 13.5 min · test + submit 3 min |
| union recall (raw → stage-0 top-60) | India 0.9935 → 0.9934 · US 0.9960 → 0.9960 (top-20 already 0.991 / 0.996) |
| OOF macro F0.5, stage 1 (exclusive + threshold 0.725) | 0.9795 |
| OOF macro F0.5, stage 2 (exclusive + expected-F0.5) | **0.9800** (India 0.977, US 0.982) |
| OOF precision / recall / singleton accuracy | 0.991 / 0.963 / 0.959 |
| calibration | mean p within ±0.02 of the hit rate in every decile |
| official validator (`--check-ids`) | **PASS** (1,732,544 rows, matches ⊆ candidates) |

Top stage-1 features by gain: stage-0 score, address token Jaccard, address
token-set ratio, `cos_both`, name partial ratio, full-name ratio, target
competition margin, house-number Jaccard.

Engineering issues found and fixed by this run:
1. `sparse_dot_topn`'s multithreaded search segfaults on macOS when LightGBM's
   OpenMP runtime is loaded first → `entity_forge/__init__.py` imports
   `sparse_dot_topn` first.
2. Salted id-sampling was a shifted copy of the same hash (a 15 % sample of a
   2 % sample kept 100 %) → proper xor-shift-multiply mixing, with a test.
3. Submission rows now come from the raw `test_source1.tsv`, so every S1 always
   gets a row.

## EXP-008 · Memory-bounded, checkpointed pipeline (32 GB target)

**Why.** On a 4 vCPU / 32 GB SageMaker instance, stages 01–02 finished
(396M raw candidate rows) but the kernel died in notebook 02. The old code
loaded whole candidate tables (`read_parquet`) and ran windows and filtered
copies on them. The US train partition alone is 121M rows ≈ 8 GB as a
DataFrame before any window. Retrieval also kept all six channels' sparse
matrices alive.

**Measured per-row costs (dev slice)** used to size profiles and disk checks:

| artifact | disk B/row | RAM B/row |
|---|---|---|
| normalized record | 91 | 220 |
| raw candidate | 22 | 68 |
| pruned candidate | 26 | 72 |
| stage-1 features | 73 | 206 |
| stage-2 features | 104 | 274 |

**Correctness of the refactor.**

| check | result |
|---|---|
| streamed retrieval vs original in-memory algorithm (dev train/india, 1.84M pairs) | identical: same pairs, ranks, ids; cosines equal (≤1e-6) |
| same, vs the candidate file produced by the original dev run | byte-identical frame |
| chunked vs whole stage-0 features, context features, stage-2 aggregates | identical (unit tests with ties) |
| pruned candidate recall (dev) | India 0.99339 / US 0.99602, identical to before |
| OOF macro F0.5 (dev), stage-1 rule / final | 0.97956 / **0.98015** (before: 0.97950 / 0.98000) |
| official validator (`--check-ids`) | PASS |

`t_rank_cos_both` was redefined as 1 + number of competing S1s with a strictly
higher `cos_both` (ties share a rank, capped at 9). It now comes from a bounded
per-target aggregate instead of a whole-table window. No model had been
trained on the old definition.

**Dev run on the new entry point** (`bash scripts/run_notebooks.sh --dev`,
Apple M1, 8 GB RAM, `16gb` profile):

| stage | time | peak RSS |
|---|---|---|
| normalize | 90 s | 1.2 GB |
| candidates (5 legacy files validated + adopted) | 3 s | 0.2 GB |
| prune (stage-0 training + 5 partitions) | 24 s | 1.5 GB |
| features | 90 s | 1.4 GB |
| stage 1 (5 folds + OOF scoring) | 520 s | 1.7 GB |
| stage 2 (features + 5 folds + OOF scoring) | 158 s | 1.2 GB |
| decision (stage-1 and stage-2 grids) | 18 s | 1.6 GB |
| predict (test stage 1 + 2) | 98 s | 1.1 GB |
| submit (TSVs + validator + zip) | 55 s | 0.6 GB |

**Resume and failure behaviour, observed:**
- A memory guard stopped `prune` *before* allocating, with a clear message.
  The re-run skipped normalize and candidates in 2–3 s and reused the stage-0
  model.
- Same for `stage2` (training-matrix estimate).
- After fixes, 27 checkpoints were skipped and only the missing work ran.
- A bug in `submit` was fixed and resumed with `--from submit`.

## EXP-009 · Medium scale: full target pools (memory realism for 32 GB)

`EF_DEV_S1_FRAC=0.03 EF_DEV_TARGET_FRAC=1.0`: 3 % of S1 (train 66k, test 52k)
against the **complete** S2/S3 pools (10.3M train, 10.0M test targets). Every
per-target structure is therefore real size: records, TF-IDF matrices,
per-target aggregates. Apple M1, 8 GB RAM, `16gb` profile, 8 threads.

| stage | time | peak RSS |
|---|---|---|
| normalize (all 24.2M records) | 491 s | 2.52 GB |
| candidates (5 partitions, 6 channels each) | 1320 s | 2.25 GB |
| prune | 176 s | 1.30 GB |
| features (9.1M pairs) | 317 s | 1.96 GB |

**Realistic recall** (full distractor density; top-60 after stage-0 pruning):

| | pruned union | top-10 | top-20 | top-40 | top-50 |
|---|---|---|---|---|---|
| India | **0.9846** | 0.9669 | 0.9783 | 0.9831 | 0.9840 |
| US | **0.9905** | 0.9791 | 0.9869 | 0.9899 | 0.9903 |

This matches the raw-union recall of EXP-005/006 (0.985 / 0.991), so stage-0
pruning to 60 per S1 loses ≤ 0.001, and the curve is flat after ~50.

**Old vs new retrieval on the full-pool India partition:** identical output
(3,007,433 pairs; ranks, ids, cosines ≤ 1e-6). Live sparse-matrix bytes
extrapolated to all S1 (from logged nnz):

| partition | old: all 6 channels | new: 1 channel + its transpose |
|---|---|---|
| train/india | 2.19 GB | 1.66 GB |
| train/us | 1.75 GB | 1.56 GB |
| test/india | 2.50 GB | 1.94 GB |

The bigger saving is after retrieval. The old union, id gathering and every
downstream stage held O(all rows): a 121M-row partition is ~8 GB as a bare
DataFrame (68 B/row), before join and window intermediates. The new code holds
O(one part) (≤ 6M rows on the 32gb profile). On the 8 GB Mac the old
retrieval took 584 s against 331 s for the new one on the same partition,
consistent with memory-pressure slowdown. macOS compresses pages, so its RSS
figures understate the old peak.

**Throughput for planning a full run on 4 vCPU** (extrapolated, M1 → ~2× slower on 4 vCPU):
features ≈ 35–40 µs/pair on 8 M1 cores → ~5 h for all ~236M pairs on 4 vCPU;
retrieval is already done on the SageMaker instance (adopted).

**Robustness fix found here:** an ad-hoc run reused a stale candidate scratch
directory built for other inputs. `generate_candidates_to_file` now stores an
input fingerprint in the scratch dir and starts fresh on mismatch (with a test).
