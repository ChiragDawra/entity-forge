# ML Challenge 2026: Business Entity Resolution — Methodology

**Team Name:** entity-forge
**Team Members:** Chirag Dawra
**Submission Date:** _(fill in)_

> Numbers marked **(full run)** come from `notebooks/04_train.ipynb` /
> `05_predict_and_submit.ipynb` on the complete data. Every other number is
> measured and logged in `docs/EXPERIMENTS.md`.

---

## 1. Executive Summary

A three-stage entity-resolution pipeline: (1) per-country multi-channel sparse
retrieval whose channels were chosen from measured recall. It includes a
**name × address conjunction** channel that isolates one business inside a
building shared by many; a learned stage-0 re-ranker trims the union.
(2) A **two-stage LightGBM**; stage 2 sees each S1's whole candidate set,
competing S1s for the same record, and similarity to the S1's most confident
matches. (3) A decision layer tuned on out-of-fold predictions with the exact
macro F0.5. It enforces the **exclusivity** found in the training data (no
S2/S3 record ever belongs to two S1 entities) and treats "no match" as a
first-class outcome.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Exclusivity:** 0 of 7,638,365 matched S2/S3 ids appears under two S1 entities.
- **Country:** 0 true pairs cross countries; test adds France (unseen), so country is an open set and never a feature.
- **Singletons:** 5.6 % of S1 (same in US and India); each is worth a full 1.0 when predicted empty.
- **Set sizes:** 3.46 matches per S1 on average (max 11). S2/S3 are not deduplicated, so an S1's matches are noisy copies of each other.
- **Name noise:** reordering (`Patterson Corp (Chadwick)`), legal-form noise, DBA / "formerly" prefixes, domains (`brighttavern.com`, `| www.x.com`), OCR-style digits (`5arasaksh`), injected accents (`Índustries`), prefixes (`M/s`, `Sri`, `<<`, `--`), Devanagari transliterations (`टेक क्रिएटिव इंडस्ट्रीज` = Tech Creative Industries), and true matches whose name barely overlaps (`Design Service` for `Design Academy`).
- **Address noise:** abbreviations, component reordering, ordinal noise (`218rd`), zero padding (`001673`), placeholders (`NULL`), state names vs codes vs native script (`UT`/`Utah`, `MH`/`महाराष्ट्र`), city suffixes (`CDP`, `TOWNSHP`), 3–4 % missing.
- **Ambiguity:** ~30 % of S1 names are duplicated (chains, generic names), and many businesses share one building, so neither name nor address alone identifies a record.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage GBDT classifier + metric-aware set decision
**Core Innovation:** evidence-driven retrieval (name × address conjunctions,
address-less name search) plus set-level modelling that exploits verified
exclusivity and the near-duplicate structure of S2/S3.

---

## 3. Candidate Generation (Blocking)

Normalization (hand-written dictionaries only): anyascii transliteration with an
anusvara fix, legal-form/filler stripping into `name_core`, DBA extraction,
digit look-alike repair, a consonant-skeleton phonetic key, canonical address
tokens (abbreviations, US/India/France region codes incl. native-script Indian
states, ordinals, zero padding, placeholders), and house-number extraction.

Retrieval runs per country. Each channel is a hashed TF-IDF representation;
top-K by cosine uses `sparse_dot_topn` in chunks, never an all-pairs matrix:

| Channel | Representation | Role |
|---|---|---|
| address | address word tokens | same place, any name |
| both | name words ⊕ address words | common names disambiguated by place |
| cross | name-token × address-token conjunctions | one business inside a shared building |
| name | name word tokens | same business, reformatted or different address |
| no-address | name words, searched among address-less targets only | the 3–4 % of records with no address |
| transliteration | phonetic name × address, searched among non-Latin-script targets only | Devanagari names romanized without vowels |

- **Blocking keys used:** see table. Every union pair gets every channel's exact cosine and rank.
- **Stage-0 pruning:** LightGBM on retrieval-only features keeps the best N per S1.
- **Candidate pairs generated:** _(full run)_
- **How true matches were kept:** channel choice and K come from recall curves on 20–30k S1 samples against the full target pool (EXP-002…004). India: address alone 0.854 → address ∪ both ∪ cross 0.964 → + no-address + transliteration channels **0.985**. US: address ∪ both ∪ cross **0.991**. Final union / pruned recall: _(full run, notebook 02)_

---

## 4. Matching Model

**Features used (stage 1):**
- Retrieval: cosine and rank per channel, stage-0 score.
- Query context: number of candidates, gap to the S1's best and rank per channel.
- Target competition: number of S1s retrieving the record, rank and margin against the best competing S1.
- Name: RapidFuzz ratio, token-sort, token-set, partial, Jaro-Winkler on `name_core`; ratio on full name, DBA name, phonetic key, space-free name; token Jaccard, first-token match, lengths, name frequency of both sides.
- Address: ratio, token-set, token-sort; token Jaccard; house-number overlap, first-number match, conflict and subset flags; missing-address flags.
- Other: target is S3, target was non-Latin script.

**Stage 2** adds, from out-of-fold stage-1 probabilities: the S1's max / second / sum / count / rank of p1; the best p1 any other S1 gives the record (and the margin); similarity of the candidate to the S1's top-1 and top-2 candidates.

**Model type:** LightGBM binary classifiers (MIT). 5 folds grouped by S1 id; every train pair gets an out-of-fold score; test uses the fold average.
**Threshold selection method:** grid over threshold × (with / without exclusivity) and per-S1 expected-F0.5 subset selection, scored with the exact macro F0.5 on out-of-fold predictions.

---

## 5. Results & Error Analysis

| Metric | Value |
|---|---|
| Candidate pair recall (train, pruned union) | _(full run)_ |
| OOF macro F0.5, stage 1 + best rule | _(full run)_ |
| OOF macro F0.5, stage 2 + best rule | _(full run)_ |
| Singleton accuracy | _(full run)_ |
| Public leaderboard | _(fill in)_ |

- **Common false positives (wrong merges):** _(from notebook 04)_
- **Common false negatives (missed matches):** records with no address and a generic name; heavily corrupted names combined with a reformatted address.

---

## 6. Conclusion

_(2–3 sentences after the full run.)_

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/entity_forge/`: the package (normalization, retrieval, pruning, features, models, decision, metric, stages)
- `src/notebooks/01…05`: run in order to regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from `dataset/`
- `README.md` (exact commands), `requirements.txt` (pinned)

Headless reproduction: `pip install -r requirements.txt && bash scripts/run_notebooks.sh`.

### B. Compliance

- No external data, APIs, geocoders or gazetteers. Hand-written dictionaries (`dictionaries.py`): legal forms, fillers, DBA markers, name synonyms, address abbreviations, US state codes, Indian state codes and the native-script state names seen in the provided data, French region codes.
- Libraries: Polars, NumPy, SciPy (BSD), RapidFuzz (MIT), sparse_dot_topn (Apache-2.0), anyascii (ISC), LightGBM (MIT). No pretrained model; model size far below 8B parameters.
