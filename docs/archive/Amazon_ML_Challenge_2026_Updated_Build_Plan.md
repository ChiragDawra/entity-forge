# Winning the Amazon ML Challenge 2026 Business Entity Resolution Task: A 61-Hour Build Plan

Build the same three-stage pipeline that won the closest past
competition (Kaggle Foursquare Location Matching): (1) a union of cheap,
high-recall candidate generators --- normalized-key inverted indexes
plus TF-IDF character n-gram top-k retrieval --- (2) a LightGBM
classifier on roughly 50--100 string-similarity and "context" features,
and (3) a metric-aware decision layer that chooses each Source 1
entity's match set (including the empty set) to maximize expected
per-entity F0.5. Fine-tuned transformers are the optional upgrade to add
once the baseline is done. They are not the foundation.

> ## BUILD REVISION --- IMPORTANT CHANGES BEFORE IMPLEMENTATION
>
> This revision keeps the original architecture but removes several
> assumptions that should **not** be hard-coded before measurement. The
> implementation priority is now:
>
> **candidate recall → working baseline → LightGBM → exact F0.5 tuning →
> optional upgrades**
>
> ### 1. Do not hard-code a 30--40 candidate cap
>
> The original 30--40 recommendation is a starting hypothesis, not a
> requirement. Test:
>
> `K = 10, 20, 30, 40, 60, 80, 100`
>
> and measure pair recall, average candidates, p95/p99 candidates,
> runtime, and memory. A cap that loses true pairs is unacceptable
> because the matcher cannot recover discarded candidates.
>
> ### 2. Do not start with expected-F0.5 subset mathematics
>
> First implement a simple threshold sweep and then threshold + top-k
> selection against the exact competition metric. Only after that works
> should calibrated probabilities and expected-F0.5 subset selection be
> tested. The expected-value formula is a heuristic, not a guaranteed
> optimum because it depends on calibration and candidate-recall
> assumptions.
>
> ### 3. Exclusivity is experimental
>
> Before enforcing `one S2/S3 → one S1`, calculate how many S2/S3 IDs
> occur under multiple S1 IDs in training ground truth. Use hard
> exclusivity only if the data supports it. Otherwise use a soft
> contextual feature/penalty.
>
> ### 4. Start with \~20--30 strong features, not 100
>
> Build the strongest name, address, numeric, and candidate-context
> features first. Add feature groups only when out-of-fold F0.5 or
> candidate recall justifies them.
>
> ### 5. Hard negatives matter
>
> Do not train mainly on random negative pairs. The most useful
> negatives are candidates that look genuinely similar but are wrong:
> shared names, shared buildings, chain names, same postal code,
> transliteration collisions, and high-TF-IDF false matches.
>
> ### 6. Embeddings and graph expansion are optional upgrades
>
> E5 embeddings and 2-hop/graph candidate expansion should come only
> after a stable lexical + TF-IDF + LightGBM pipeline exists. Keep them
> only if measured out-of-fold performance or candidate recall improves
> enough to justify their cost.
>
> ### 7. Foursquare evidence is guidance, not a target
>
> Candidate counts, recall figures, feature importance, and scores from
> other competitions must not be treated as expected results for this
> dataset. This dataset's training ground truth determines the correct
> configuration.
>
> ### 8. The first milestone is a complete working submission
>
> Before advanced optimization, produce:
>
> `data → blocking → candidates → features → LightGBM/threshold baseline → matching_results.tsv → candidate_pairs.tsv → official validator`
>
> This protects the team from spending most of the competition on a
> component that never becomes a usable submission.

## TL;DR

-   **Blocking:** union several generators, all scoped by country: exact
    normalized-name keys, a postal-code + street-token key, a rare-token
    index, and TF-IDF char 3-gram cosine top-k via sparse_dot_topn. Do
    not hard-cap the union initially. Measure recall across
    K=10/20/30/40/60/80/100 and select a data-driven candidate budget.
    Measure pair recall on a held-out train slice before doing anything
    else. Foursquare's top teams reached a 0.98+ recall ceiling with
    12--40 candidates per record.
-   **Matching:** a LightGBM model on Jaro-Winkler, Levenshtein,
    token-set, TF-IDF-cosine, numeric-token and rank/context features.
    This is the pattern behind every top Foursquare solution.
    Gradient-boosted trees have no license or parameter-count issues,
    and they train in minutes. A small MIT-licensed
    multilingual-e5-small embedding (117.65M parameters per the
    Teradata/multilingual-e5-small Hugging Face card; 12 layers and
    384-dimensional embeddings per the intfloat model card) is the
    safest "neural" add-on, used as both a blocking channel and a
    feature.
-   **Decisions for F0.5:** don't use one global 0.5 threshold. For each
    S1 entity, sort candidates by calibrated probability and choose the
    top-k (k can be 0) that maximizes expected F0.5. Enforce "each S2/S3
    record belongs to at most one S1 entity" if train confirms it.
    Explicitly score the empty set so the \~5.6% singletons get a real
    1.0/0.0 decision. Tune everything on out-of-fold predictions using
    the exact macro metric.

## Key Findings

### 1. What actually won comparable competitions

**Kaggle Foursquare Location Matching (2022)** is the closest analogue:
noisy multi-source business/POI records, one-to-many matching, and a
per-record set metric (mean IoU). The metric punishes false merges much
as F0.5 does. It drew 22,050 submissions from 1,290 data scientists.
Every top solution used the same skeleton: multi-channel candidate
generation → GBDT pair classifier → optional transformer rerank → graph
post-processing.

-   **1st place (re:waiwai):** selected "100 candidates for each" of two
    methods: coordinate distance and name-embedding cosine. A LightGBM
    with limited features then kept "the top 20 candidates each, for a
    total of about 40 candidates". A second-stage LightGBM used
    Levenshtein and Jaro-Winkler distances, statistics/ratios and SVD
    name embeddings. BERT-family models were added in later stages.
-   **Team 2:30 (DeNA):** generated coarse candidates from geographic
    proximity and TF-IDF scores, added a transformer blocking stage, and
    ensembled LightGBM/XGBoost with BERT models. The host blog says
    balancing efficiency and effectiveness meant "using only important
    features".
-   **A GBDT-heavy top team:** according to the verbatim copy of its
    writeup on Zhihu, it kept "12 candidates per 1 id (about 7M
    candidates on test data), max IOU w/o postprocessing is about
    0.983", and "created about 120 features based on...
    foursquare-lightgbm-baseline".
-   **7th place (Future Architect):** retrieved 32 candidates from 5
    methods, for a max IoU of 0.9778 from retrieval alone and 0.9935
    after graph post-processing. Its best single retrieval channel was
    geographic distance, with precision@32 of 0.9160. The team's July
    20, 2022 blog post says each training sample was weighted by the IoU
    loss from mispredicting it (on average 0.8 for positives and 1.0 for
    negatives), because "False Positive... has a larger negative effect
    on the score than True Negative".
-   **Yuki Uehara:** going from 20 to 60 candidates per id raised his
    score from 0.907 to 0.924. GNN post-processing over 2-hop subgraphs
    lifted it to 0.946.
-   **4th place:** the key post-processing idea was "to adapt the
    thresholds to the sizes of the groups we were merging". Its most
    important features were text cosine similarities (normalized name,
    address), distance rank, name Jaccard, and matching-word counts.

**Implication.** Your data has no coordinates, so address tokens and
postal codes must stand in for the geographic channel, which was the
strongest recall source in Foursquare. Recall ceiling and
precision-aware post-processing moved scores more than model choice did.

**ACM SIGMOD Programming Contest 2022 (blocking for ER):** won by the
Mannheim WBSG team out of 55 teams. They embedded entity descriptions
with a contrastively pre-trained transformer, indexed them in FAISS, and
re-ranked retrieved pairs with a symbolic similarity metric. This
validates "embedding kNN + symbolic rerank" as a blocking pattern, but
it needed custom pre-training that you can't replicate in 61 hours.

**Ditto (VLDB 2021)** shows that fine-tuned cross-encoders beat
classical matchers on benchmark datasets by up to 29% F1. On a real
company-matching task with 789K and 412K records it reached 96.5% F1.
The catch is cost: a cross-encoder over \~50M test candidate pairs is
expensive. At this scale, use it only on the ambiguous probability band,
if at all.

**Shopee Price Match Guarantee** is a text+image product-matching
competition. One participant's baseline found TF-IDF title embeddings
"outperforming the other two by a huge margin" (fastText and BERT). Top
teams tuned similarity thresholds by sweeping them against the metric.
Cheap lexical retrieval is a strong baseline on short noisy strings: in
Foursquare, the 21st-place team (Bulian AI) wrote that "we tried TFIDF -
3 ngrams+char_analyzer and it immediately gave boost" over distance-only
blocking, which had held them at 84--85 on the leaderboard.

### 2. Blocking options compared for \~2.2M × \~10M records

  -------------------------------------------------------------------------------------------------
  Approach                   Build time     Expected recall role    Compute at this  Verdict
                                                                    scale            
  -------------------------- -------------- ----------------------- ---------------- --------------
  \(a\) Normalized exact     2--4 h         High for clean          Trivial          **Must have
  keys (name-core, postal                   variants; misses        (pandas/DuckDB   --- channel
  code + street token, rare                 typos/transliteration   joins); must     #1**
  name tokens, phonetic                                             skip oversized   
  key), country-scoped,                                             blocks           
  OR-combined                                                                        

  \(b\) MinHash/LSH on       3--6 h tuning  Similar to TF-IDF, but  Moderate; many   **Skip** ---
  n-grams                                   noisier candidate sets  hash tables for  dominated by
                                                                    high recall      (c)

  \(c\) TF-IDF char n-gram + 3--5 h         Best single fuzzy       Chunk 1M rows at **Must have
  top-k sparse product                      channel for typos,      a time; fits a   --- channel
  (sparse_dot_topn) or ANN                  suffixes, word order    Mac              #2**

  \(d\) Sentence-embedding   4--8 h         Catches cross-script    Encoding \~12M   **High reward
  kNN                                       and transliteration     strings is hours add-on**,
  (multilingual-e5-small +                  cases lexical methods   on CPU/MPS; GPU  after (a)+(c)
  FAISS HNSW)                               miss                    helps            are measured

  \(e\) Sorted neighborhood  1--2 h         Weak with word-order    Cheap            **Skip**, or
                                            and prefix noise                         use only as a
                                                                                     tiny safety
                                                                                     net
  -------------------------------------------------------------------------------------------------

Supporting evidence:

-   **Splink's documentation** recommends OR-combined blocking rules and
    says post-blocking comparisons are "often between 10 and 1,000 times
    higher" than input records. It advises starting under 10 million
    comparisons and scaling to "100s of millions (DuckDB on a laptop)".
    A Splink benchmark deduplicated 7 million records while evaluating
    over 1 billion comparisons with DuckDB.
-   **The BlockingPy paper (MIT license)** found that the LSH-based
    blocklib, while faster, "results in a significantly higher number of
    missed pairs and several orders of magnitude more pairs" than
    ANN-based blocking. This is why MinHash is deprioritized.
-   **ING's sparse_dot_topn** was built for company-name matching
    between very large datasets. The original write-up matched 663,000
    company names in 42 minutes on a dual-core laptop. The new version
    is up to 6× faster on an Apple M2 Pro and is designed to be chunked
    into \~1M-row blocks.

**Concrete blocking recipe:**

1.  Partition everything by country. Test adds France, so treat country
    as an open set and never hardcode it.
2.  Build these channels:
    -   exact normalized name-core (legal suffixes stripped);
    -   name-core first token + postal code;
    -   postal code + first street/landmark token;
    -   rare name tokens (low document frequency; ignore tokens
        appearing in more than N records);
    -   TF-IDF char 3-gram (word-boundary) cosine top-20 per S1 against
        S2 and S3 separately, on name and on name+address.
3.  Union the channels.
4.  Re-rank the union with a cheap score and evaluate several candidate
    budgets before choosing the final cap.
5.  Report pair recall and reduction ratio per channel and for the union
    (these numbers go straight into the methodology doc).
6.  Add a "2-hop" channel once the model exists: if S1→S2-a is
    high-confidence and S2-a has a close S3 neighbor, add that S3 record
    as a candidate. In Foursquare, 2-hop/graph post-processing raised
    the recall ceiling from 0.9778 to 0.9935.

### 3. Normalization and similarity features

**Normalization pipeline.** Apply one shared function to names and
addresses:

-   Unicode NFKC, then casefold.
-   Accent folding via NFKD and dropping combining marks. This handles
    French é/è.
-   Punctuation → space; "&" → "and"; collapse whitespace.
-   Script detection: flag Devanagari and other Indic ranges.
-   Transliteration of Indic scripts to Latin with
    `indic_transliteration` (or Aksharamukha). Keep both the original
    and the transliterated forms.
-   A hand-written abbreviation map applied at the token level:
    -   Business names: corp/corporation, pvt/private, ltd/limited,
        co/company, inc, llc.
    -   Street types: rd/road, st/street, ave/avenue, blvd/boulevard,
        nagar, marg.
    -   French: SARL, SAS, SA, EURL, "r."→rue, "bd"→boulevard,
        "av"→avenue, "st/ste"→saint/sainte.
-   Legal-suffix stripping into a separate "name_core" field (the
    MIT-licensed cleanco library covers many worldwide suffixes). Keep
    the suffix as its own feature.
-   Extract digit tokens: house number, postal code (US 5-digit, India
    6-digit PIN, France 5-digit), and unit numbers.
-   Strip landmark phrases ("near", "opp", "behind", "beside" +
    following tokens) into a separate landmark field.

**libpostal** supports address normalization "in over 60 languages", and
its parser reports 99.45% full-parse accuracy on held-out data. It's
also a C library with a heavy model download, which is a real
installation risk on a Mac under time pressure. Treat it as optional. A
hand-written regex/dictionary normalizer gets most of the value for
US/India/France in hours.

**Similarity features.** Compute all of them with RapidFuzz (MIT, C++,
batch `cpdist`) and scipy sparse operations. Never loop pairwise in
Python.

-   **Name:** Jaro-Winkler, normalized Levenshtein, token_set_ratio,
    token_sort_ratio, and partial_ratio on raw, normalized, name_core
    and transliterated forms. Also token Jaccard, TF-IDF char-3gram
    cosine, and word-level TF-IDF cosine. Add a rare-token overlap
    weighted by IDF, since shared rare tokens beat shared common tokens.
-   **Name structure:** suffix-equal flag, initials/acronym match ("ABC"
    vs "A B Corp"), and first-token equal.
-   **Address:** the same string metrics on the normalized address.
    Postal code exact/prefix match, house-number match/conflict, and
    street-token Jaccard. Landmark-stripped similarity, and a
    missing-address flag (S2/S3 have \~2--3% missing).
-   **Frequency/ambiguity** (critical because names are not unique): how
    many S1 records share this name_core, how many candidates this S1
    has above a cosine threshold, and the name's frequency in S2/S3.
-   **Rank/context** (the Foursquare 4th-place "distance rank"
    analogue): this candidate's rank within the S1's candidate list for
    each channel; the gap to the best candidate's score; and which
    channels proposed the pair (bitmask). Also the candidate's rank
    among all S1s that proposed it, which sets up the exclusivity step.
-   **Optional neural:** e5-small cosine between name+address
    embeddings.

Past work consistently ranks text cosine similarities (TF-IDF/embedding
on name and address), Jaccard and Jaro-Winkler/Levenshtein at the top.
The Foursquare 4th-place importance list and the 1st-place feature set
both led with these.

### 4. Matching model and decision layer for macro F0.5

**Model:** a LightGBM binary classifier.

-   Train on candidate pairs from a sample of train S1 entities. Start
    with a manageable, representative S1 sample (for example 300--500k),
    but verify that it contains enough singleton, multi-match, noisy,
    and high-frequency-name cases.
-   Use 5-fold cross-validation grouped by S1 entity.
-   Don't use country as a categorical feature, because France is
    unseen. Keep features language-agnostic (ratios, flags, ranks).
-   Optionally add a second-stage LightGBM that takes the first-stage
    probability plus set-level context (count of candidates above 0.5,
    the max/second-max probability among other S1s competing for the
    same S2/S3 record). This mirrors the 4th-place "3rd LGBM model
    taking the 2nd model's score as an input, as well as contextual
    features".

**Why not a transformer matcher first:** Ditto-style cross-encoders are
more accurate per pair but cost GPU hours over tens of millions of
pairs. The Foursquare winners only added them after the GBDT pipeline
was solid.

**Decision rule --- per-entity expected F0.5.** Lipton, Elkan and
Naryanaswamy ("Thresholding Classifiers to Maximize F1 Score",
arXiv:1402.1892, ECML-PKDD 2014) show that "if the classifier outputs
are well-calibrated conditional probabilities, then the optimal
threshold is half the optimal F1 score". The general Fβ form is
F*/(1+β²), i.e. about 0.8·F* for β=0.5. Because the leaderboard metric
is computed per S1 entity, do better than any single threshold:

1.  Calibrate the out-of-fold probabilities (isotonic regression).
2.  For each S1, sort candidates by probability p₁ ≥ p₂ ≥ ....
3.  Score each k ≥ 1 with this approximation:
    -   E\[F0.5 \| top-k\] ≈ 1.25·Σᵢ≤ₖ pᵢ / (0.25·(Σ_all pᵢ + m) + k)
    -   m is the expected number of true matches that blocking missed.
        Estimate it from measured blocking recall.
4.  Score k = 0 as P(no true match) ≈ Π(1−pᵢ) × P(blocking missed
    nothing). This is the only way to earn the singleton 1.0.
5.  Only after a simple threshold/top-k baseline is established, test
    calibrated expected-F0.5 subset selection and tune it directly
    against exact out-of-fold macro F0.5.

**Exclusivity.** S1 is a deduplicated reference, so each S2/S3 record
should belong to at most one S1 entity. Verify this on train by counting
S2/S3 ids that appear in more than one S1's ground-truth list. If the
count is effectively zero and the relationship is supported by training
data, test assigning each S2/S3 record only to its highest-probability
S1; otherwise use a soft exclusivity feature/penalty. This removes
chain-store and neighboring-business false positives, which are exactly
the errors F0.5 punishes most.

**Match count prior.** The train distribution peaks at 3--4 matches (max
11), so a predicted set size far outside that range is a warning sign.
Use the per-entity count of high-probability candidates as a feature,
not as a hard rule.

### 5. What reviewers reward beyond the leaderboard

**Amazon ML Challenge 2025 precedent:**

-   The top 10 teams presented to Amazon scientists at a virtual Grand
    Finale, chosen "based on the leaderboard results and the solution
    presented in the document".
-   One participant repository states that final rankings were "based on
    the full 75K test set and documentation quality".
-   The 2025 rules also required a 1-page methodology document and
    warned that "all submitted approaches, methodologies, and code
    pipelines will be thoroughly reviewed and verified". External
    lookups meant immediate disqualification.

This year's document has no page limit, so depth and evidence count.
Include:

-   A blocking table: recall and reduction ratio per channel and for the
    union, plus the recall ceiling this implies for F0.5.
-   An ablation table covering each channel, feature group, exclusivity
    and the decision layer, with delta macro F0.5 on out-of-fold data.
-   A calibration plot and the singleton confusion matrix (empty
    predicted vs empty true).
-   Per-country results and a cold-start experiment: train on US only
    and evaluate on India. This is a defensible proxy for how the model
    will handle France.
-   An error-analysis gallery (chains, shared buildings, DBA names,
    transliteration misses).
-   Runtime and cost table.
-   A license/parameter table: LightGBM MIT; multilingual-e5-small MIT
    at 117.65M parameters (per the Teradata/multilingual-e5-small
    Hugging Face card) if used.
-   An explicit "no external data" statement listing every dictionary
    you hand-wrote.
-   A one-command reproducibility script.

## Recommendations --- the 61-Hour Plan

**Roles.** Four people, in parallel:

  Person   Owns
  -------- ---------------------------------------------------------
  A        Blocking
  B        Normalization + features
  C        Model + decision layer + exact metric
  D        Infra, validation harness, submissions, methodology doc

**Sleep.** Schedule it in two staggered shifts. A 61-hour push without
sleep produces bugs in the final 10 hours, when they cost most.

  -------------------------------------------------------------------------
  Hours from now          Goal                      Exit criterion
  ----------------------- ------------------------- -----------------------
  0--4                    Load all data to Parquet. Scorer matches
                          Write the exact           hand-computed cases;
                          macro-F0.5 scorer         exclusivity stat known
                          (singleton rules          
                          included). Build a        
                          validation split: \~200k  
                          train S1 entities held    
                          out, blocked against the  
                          FULL S2/S3 pool. Check    
                          exclusivity. Normalizer   
                          v1.                       

  4--14                   Blocking v1: exact keys + Union pair recall ≥97%
                          postal/street keys +      at ≤40 candidates per
                          TF-IDF char 3-gram top-k  S1; first leaderboard
                          (sparse_dot_topn,         score
                          chunked). Recall@K curves 
                          per channel. **Baseline   
                          submission**: TF-IDF      
                          cosine threshold +        
                          singleton rule, to        
                          confirm formats and       
                          validate_submission.py.   

  14--30                  Features v1 (RapidFuzz    OOF macro F0.5 clearly
                          batch) → LightGBM grouped above baseline;
                          5-fold → calibration →    leaderboard confirms
                          expected-F0.5 decision +  
                          exclusivity. Submission   
                          v2.                       

  30--44                  Upgrades, in order of     Each upgrade kept only
                          expected value: (1) 2-hop if OOF improves
                          candidate channel; (2)    
                          rank/context +            
                          second-stage LightGBM;    
                          (3) e5-small embeddings   
                          as blocking channel +     
                          feature (one GPU job);    
                          (4) transliteration fixes 
                          from error analysis. Log  
                          every ablation.           

  44--52                  Final threshold/offset    Frozen config
                          tuning on OOF. France     
                          sanity checks:            
                          predicted-match-count     
                          histogram and singleton   
                          rate vs US/India.         
                          US→India cold-start       
                          experiment. **Code freeze 
                          at hour 52.**             

  52--58                  Full test inference.      Both TSVs validated
                          Assert matches ⊂          
                          candidate_pairs.tsv. Run  
                          the validator. Finish the 
                          methodology doc (tables   
                          are already logged).      

  58--61                  Buffer only.              ---
  -------------------------------------------------------------------------

**Compute and budget.** Do blocking, features and LightGBM on the Mac,
chunked by country and by S1 blocks of about 100k rows. If RAM is under
\~32 GB, rent a high-memory CPU instance for the full-test feature pass
instead of fighting swap. Use GPU only for the optional embedding pass.
Price quotes for a single-GPU ml.g5.xlarge conflict: one guide lists
\$1.01/hour for training, another \$2.03/hour for on-demand inference.
Either way, a few hours sits well inside \$200. Managed Spot Training
can cut this further (AWS's August 2019 launch notice says "by up to 90%
compared to on-demand instances"), but only if you checkpoint. Shut down
idle notebooks and endpoints.

**Risk ledger:**

  -----------------------------------------------------------------------
  Category                            Items
  ----------------------------------- -----------------------------------
  Safe/reliable                       Key + TF-IDF blocking; LightGBM;
                                      calibrated expected-F0.5 decision;
                                      exclusivity

  Medium risk, high reward            e5-small embedding channel
                                      (cross-script recall); 2-hop graph
                                      candidates; second-stage context
                                      model

  High risk                           Fine-tuning a cross-encoder
                                      (Ditto-style) or contrastive
                                      embedding in the last 30 hours;
                                      libpostal install; MinHash tuning
  -----------------------------------------------------------------------

Attempt high-risk items only on a branch, after submission v2 is
secured.

**Compliance.** Use only provided data plus hand-written dictionaries
and open-source code libraries. No geocoders and no downloaded business
or place lists. Declare every library and dictionary in the doc. A
public GitHub repo dated September 24, 2026 already describes a similar
pipeline for this exact task: TF-IDF kNN, inverted indexes, MinHash LSH,
107 features, LightGBM, and an "exclusivity + expected-F0.5 decision
layer". Its reported \~99.4% pair recall is on synthetic data only.
Treat it as confirmation that the pattern is converging, and do not copy
its code, since the code audit could flag it as plagiarism.

## Caveats

-   **Foursquare evidence is partly secondhand.** Several Foursquare
    figures come from the host's blog and from verbatim copies of Kaggle
    writeups aggregated on third-party pages. Attribution of some
    numbers to specific teams (e.g., the 12-candidate / 0.983 figure)
    could not be checked against the original Kaggle pages. Foursquare
    also had coordinates and a train/test leak that some top teams
    exploited, so absolute scores don't transfer.
-   **Some numbers are my estimates, not measurements on your data:**
    the ≥97% recall target, the 30--40 candidate cap, e5-small encoding
    times, and memory needs. Your first 14 hours exist to measure them.
-   **Expected-F0.5 subset selection is an optional heuristic; validate
    it against simpler threshold/top-k rules before adopting it.** It
    assumes roughly independent, calibrated probabilities. Treat it as a
    strong heuristic and let out-of-fold tuning against the exact metric
    have the final word.
-   **Exclusivity is an assumption until checked.** If train shows S2/S3
    records legitimately matching several S1 entities, relax it to a
    soft penalty feature.
-   **France is unvalidated.** Every choice for France is a transfer
    bet. Language-agnostic features and accent folding reduce the risk
    but can't eliminate it. Say so plainly in the methodology document.

------------------------------------------------------------------------

# FINAL GO/NO-GO CHECKLIST

Before moving to the next stage:

-   [ ] Data loads without corruption.
-   [ ] Exact macro F0.5 implementation is tested on hand-built cases.
-   [ ] Candidate generation can run in chunks without dense pairwise
    matrices.
-   [ ] Pair recall is measured on held-out training S1 entities.
-   [ ] Candidate K is selected from measured recall/runtime, not
    assumed from another competition.
-   [ ] First end-to-end baseline produces valid TSVs.
-   [ ] LightGBM validation is grouped by S1.
-   [ ] Hard negatives are included.
-   [ ] Empty predictions are allowed.
-   [ ] Threshold/top-k decisions are evaluated with the exact
    competition metric.
-   [ ] Exclusivity is measured before being enforced.
-   [ ] Every advanced upgrade is kept only after an out-of-fold
    improvement.
-   [ ] Final predicted matches are a subset of final candidate pairs.
-   [ ] Official submission validator passes.

**Implementation stance:** stop brainstorming once these decisions are
fixed. Build the baseline, measure it, and iterate from evidence.
