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
