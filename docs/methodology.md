# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Queen's Code
**Team Members:** Vivek Prusty (Team Leader), Jatin Kumar Wahane
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our pipeline has five stages:

1. **Normalisation**, tuned to this data: 7 Indian scripts, aliases, domain-style names, look-alike digits and a sound key.
2. **Candidate generation.** Ten hashed blocking keys are merged with the nearest neighbours of a fine-tuned MiniLM bi-encoder, and a LightGBM ranker keeps only candidates with a real chance.
3. **A LightGBM pair matcher** with string, rarity-weighted, numeric and *relational* features.
4. **A second LightGBM stage** that rebuilds the relational context from the first stage's probabilities and adds the scores of two fine-tuned MiniLM cross-encoders, which read both records together.
5. **A decision step.** Each Source 2/3 record goes to at most one Source 1 record. A pair is a match above a threshold: 0.75 for the countries seen in training and a stricter 0.97 for France, which appears only in test.

**Result: val macro F0.5 0.9855, public leaderboard 0.977** (from 0.926 for our first submission).

The candidate set keeps **98.7 % of true matches with 8.1 candidates per Source 1 record on val (9.7 on test)**. The key-based top 20 kept 96.8 % with 19.4.

All models are small and permissively licensed: LightGBM (MIT) and three MiniLM models (22M–33M parameters, Apache-2.0). Only the provided data is used.

---

## 2. Methodology

### 2.1 Problem Analysis

What we found in the data, and what it meant for the design:

| Observation | Consequence |
| --- | --- |
| 2.2M train Source 1 records; 10.3M Source 2/3 records; ~3.5 true matches per Source 1 record; only **5.6 % singletons** | An empty prediction is almost always wrong (it scores 0), and precision is still worth twice recall |
| **Every Source 2/3 record belongs to at most one Source 1 record** (7.64M matched ids, none reused) | "One owner" rule; *competition* features between Source 1 records claiming the same record |
| ~26 % of Source 2/3 records match nothing (decoys) in train; about 38 % in test | The model must reject look-alikes, and more strictly on test |
| **Business names are heavily reused** across different businesses ("Primary Care Clinic", "Sweet Pilates LLC") | The address decides; a name key on its own forms huge blocks |
| Country is identical within every true match (100 %) | Blocking keys and bi-encoder search stay within the country |
| 7 Indian scripts appear in ~7 % of Source 2/3 names (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam); state names in native script | Transliteration, a transliterated business-word dictionary learned from the train data, and a sound key |
| Noise: word order swapped, typos, look-alike digits (`Appare1s`, `L0ndon`, `lnc`), `(ID: 123)` and `#123` tags, titles (Shri/Smt/Dr/M/s), area words (Midtown/Northside), aliases (dba/formerly/fka), domains and handles as names | Specific normalisation rules, each checked against real examples |
| Address noise: NULL / `<NULL>` / N/A, landmarks (near/opp), St/Street/**Saint** mix-ups, state names vs codes, `CITY` suffixes, extra leading zeros, **typos in the street name** | Every abbreviation maps to one short form; typo-tolerant number-and-word keys |
| House numbers are unreliable even for true matches (351 vs 35, 1304 vs 1551) | A number mismatch lowers the score but never rules a pair out |
| France appears **only** in test (15 % of test Source 1) | Country treated as an open label; only generic rules (French legal forms and street abbreviations included) |

### 2.2 Solution Strategy

**Approach type:** Hybrid candidate generation (keys + learned embeddings, re-ranked) → gradient-boosted pair classifier → a second stage with cross-encoder re-scoring → a constrained decision step.

**Core ideas:**

1. **Relational features.** Each pair "sees" its competition (other Source 1 records claiming the same record) and its siblings (other candidates of the same business). They are computed after blocking **every** train Source 1 record, so they look the same in training as on test.
2. **Two complementary candidate generators.** Keys and a bi-encoder miss *different* true matches, so their union is re-ranked by a learned model.
3. **Cross-encoders only where they matter.** They re-score the pairs the fast model is unsure about, where nearly all errors are.

---

## 3. Candidate Generation

### 3.1 Key blocking

- **Normalised text**: `name_core` (legal words, titles and stop words removed), `name_alt` (alias), `name_nospace`, `name_phon` (consonant-skeleton sound key: `bharat` and `bhart` both become `brt`), `addr_clean`, `addr_numbers`.
- **Blocking keys** (country-specific 64-bit hashes; "rarest" = lowest document frequency within the country):
  - name: the 2 rarest words; rarest sound word; first 8 letters without spaces; the full word set
  - address: number + rarest word; the 2 rarest words; **first number × each of the 4 rarest words** (typo-tolerant)
  - name + place: rarest name word × rarest / 2nd / 3rd address word; rarest name word × first number
- **Size limits**: a key is skipped if more than 500 Source 1 or 1,000 Source 2/3 records share it, or if it would form more than 50,000 pairs. The limits were chosen by a sweep on val: raw recall 0.970 at 253 pairs per Source 1 record. Looser limits added little (0.974 needs 533 pairs).
- **Ranker**: LightGBM on 20 cheap features (which keys matched, rapidfuzz name and address scores, gap to the best name). It is trained on 20k spare train Source 1 records outside the train/val split and keeps the top 20 per Source 1 record. With hand-set weights, recall@20 was 0.904; with the learned ranker it is 0.945 (v2 keys).
- **How we made sure true matches were not lost**: recall was measured on val at every stage (all keys → after the limits → top-K), and each missed pair was classified. 77 % of misses shared a key that had been capped, typically a street-name typo breaking both address keys while a common name formed a huge block. 21 % shared no key at all. The three v3 key types (number × several address words, name word × address word, name word × number) target these cases and raised **recall@20 from 0.945 to 0.968**.

### 3.2 Bi-encoder neighbours

The remaining misses share no reliable key: garbled names, translated or transliterated words, empty addresses. We fine-tuned `sentence-transformers/all-MiniLM-L6-v2` with an in-batch contrastive loss (one epoch) on true (Source 1, Source 2/3) pairs of the train split, with one pair per Source 1 record in a batch. Val records were held out. Each record is embedded as `name / alias ; address`. The model did not memorise its training pairs: the mean cosine of true pairs is 0.894 both on pairs it was trained on and on unseen val pairs. So its similarity is a feature that behaves the same in training as on test. For every Source 1 record we take its 10 nearest Source 2/3 records **within the same country** (exact cosine search on a GPU). On its own, the bi-encoder reaches val recall@10 0.975, against 0.962 for the key ranker at 10, and it misses different pairs.

### 3.3 Union and re-ranking

Per Source 1 record, the key top 20 and the bi-encoder top 10 are merged. A second LightGBM ranker scores every pair of the union: the key features, recomputed for bi-encoder pairs, plus the bi-encoder cosine and rank. It is trained on spare train records. A pair is kept if its ranker probability is ≥ 0.002.

| Candidate set | Val recall | Candidates / S1 (val) | Candidates / S1 (test) |
| --- | --- | --- | --- |
| key blocking, top 20 | 0.968 | 19.4 | 19.5 |
| key blocking, score ≥ 0.001 | 0.966 | 9.6 | 11.7 |
| **union, score ≥ 0.002 (final)** | **0.987** | **8.1** | **9.7** |

The cutoff is a score, not a fixed count, because a quarter of the businesses have five or more true matches. A fixed top 5 would keep only 92.6 % of true matches at the same size (4.8 per Source 1 record), while a score cutoff keeps 97.1 %. Test has about 1.9 times more unmatched decoy records per Source 1 record than train, so the same cutoff keeps a few more candidates there. **`candidate_pairs.tsv` is exactly this set, and every pair in it is scored by the matcher.**

---

## 4. Matching Model

**Features (~60):**

- Name features: token-set, token-sort and plain ratios; partial ratio and Jaro-Winkler on the name without spaces; sound-key overlap; rarity-weighted (IDF) Jaccard, the rarest shared word, and the share of the Source 1 name's information that is missing; word counts and lengths; similarity to the alias.
- Address features: token-set and token-sort ratios; IDF Jaccard and the rarest shared address word; number Jaccard; house number equal; any shared number; postcode and state agreement (1 / 0 / unknown).
- Candidate generation: which of the 10 key types matched, number of shared keys, ranker score and rank, bi-encoder cosine and rank.
- Context within the Source 1 record: number of candidates, number of strong candidates, margin to the best *other* candidate.
- **Competition**: how many Source 1 records claim this Source 2/3 record, and the margin to the best *other* claimant.
- **Group consistency**: similarity (name, address, numbers) to the best *other* candidate of the same Source 1 record.
- Record: source (S2/S3), Indian script present, alias present.

**Model type:** two-stage LightGBM, with cross-encoder scores in the second stage.

- *Stage 1*: the pair model on the features above. It is a binary LightGBM (127 leaves, early stopping on held-out Source 1 groups), trained on 500k train Source 1 records. For stage 2, it is **2-fold cross-fitted** (fold A's model scores fold B and vice versa), so every stage-1 score used in training comes from a model that never saw that pair.
- *Stage 2*: competition and group context rebuilt from the stage-1 **probability**, which is sharper than the ranker score:
  - the margin to the best *other* Source 1 claimant;
  - the margin to the best *other* candidate;
  - the rank within the Source 1 record;
  - the number of strong candidates, and the sum of probabilities.
- *Cross-encoders* (the `ce` and `ce12` features). `cross-encoder/ms-marco-MiniLM-L-6-v2` and `…-L-12-v2` read the two records' texts together, so they catch things the hand-made features cannot. For example, "vallee patrimoine … 126" vs "vallee primaire … 105" is a sister business, while "appare1s" vs "apparels" is a noisy copy.
  - Both are fine-tuned for match / no match on pairs of spare train records. The L6 model uses 590k uncertain pairs (stage-1 probability 0.005–0.999). The L12 model uses 1.2M pairs with stage-1 probability ≥ 0.002; the 339k negatives with probability ≥ 0.01 (confusable pairs such as sister businesses) are weighted ×2.
  - They score only pairs in the uncertain band; elsewhere the feature is empty. On val pairs in the band, the stage-1 AUC is 0.973, L6 reaches 0.989 and L12 0.990, and L12 nearly halves confident false positives (score > 0.9: 0.36 % vs 0.64 %).

**Decision:** each Source 2/3 record keeps only its highest-probability Source 1 record; pairs are kept if p ≥ threshold.

**Threshold selection method:** grid search maximising macro F0.5 on val (50k Source 1 records, singletons included), using our own implementation of the challenge metric, then leaderboard probes.

- **India and the US** use **0.75**. That is the optimum on val and on a "test-like" val in which a false match to a decoy counts 1.9 times, as test has 1.9 times more decoys per Source 1 record. On test, these countries behave like val: matches found only through the bi-encoder appear at the same rate (110 vs 103 per 1,000 Source 1 records in India, 20.8 vs 20.8 in the US) and with the same confidence.
- **France** uses **0.97**. It never appears in training, so its probabilities are the least calibrated: 90.7 % of its matches score ≥ 0.995, against about 95 % elsewhere. Two model versions also disagree on 19 % of French records, against 5–12 % elsewhere. With F0.5, which weights precision twice as much as recall, a stricter threshold is the safe choice where the model is unsure.
- **Leaderboard check** (same final model, India/US at 0.75 unless stated):
  - 0.90 for every country → 0.975;
  - France 0.95 → 0.976;
  - **France 0.97 → 0.977**;
  - France 0.985 → 0.977;
  - India/US 0.85, France 0.97 → 0.977;
  - India/US 0.80, France 0.9775 → 0.977.

  So the score sits on a flat top: it does not depend on the exact thresholds, which makes the result robust. We use India/US 0.75 (the val optimum) and the less extreme French value that reached it. Stricter French thresholds mostly remove extra, uncertain matches from businesses that keep their confident ones: the share of French records with at least one match only moves from 94.6 % to 94.4 %. Earlier, a uniform threshold of 0.90 had beaten 0.75 (0.967 vs 0.964), which this split explains: the gain came from France.

---

## 5. Results & Error Analysis

| Version | Val macro F0.5 | Leaderboard |
| --- | --- | --- |
| v1: 7 key types, 200k train Source 1, no competition/sibling features | 0.9352 | 0.926 |
| v2: v3 keys (recall@20 0.968), all train Source 1 blocked, competition + sibling features, 500k Source 1 | 0.9588 | |
| — v2 without sibling features | 0.9564 | |
| — v2 without competition features | 0.9461 | |
| v3: + stage 2 (context from cross-fitted stage-1 probabilities) | 0.9609 | 0.939 |
| v4: + MiniLM-L6 cross-encoder | 0.9766 | 0.962 |
| v5: smaller candidate set (score ≥ 0.001, 11.7/S1), cross-encoder v2 | 0.9772 | 0.964 |
| — v5 at threshold 0.90 | | **0.967** |
| — v5 + "sister business" cluster features | 0.9777 | 0.954 (rejected) |
| — v6 without the L12 cross-encoder (union candidates, L6 only), threshold 0.90 | 0.9840 | 0.967 |
| v6: union candidates (9.7/S1) + L6 and L12 cross-encoders, threshold 0.90 | 0.9855 | 0.975 |
| v6 with India/US 0.75, France 0.95 | 0.9855 | 0.976 |
| **v6 with India/US 0.75, France 0.97 (final)** | **0.9855** | **0.977** |

The val-to-leaderboard gap grew from 0.9 points (v1) to about 1.3 (v5) because test has more decoys and harder ones: several decoys that share a name and a street but have different house numbers. Features that describe such clusters helped on val and **hurt on test** (0.954 vs 0.964), so they were dropped. Decisions after that were checked with a leaderboard probe.

Tried and **not** used, because the gain was within noise or negative:

- **Separate thresholds for the US and India.** Both have the same optimum on val. Only France, which is absent from training, gets its own threshold.
- **Always keeping the top candidate** of a record that would otherwise get no match. A wrong guess on a true singleton costs a full 1.0.
- **Choosing each record's match set to maximise its expected F0.5** instead of a global threshold: +0.0003 on val.
- **Averaging stage-2 models** trained with different seeds: +0.0001 on val.
- **Training the final stage with false matches to decoys weighted ×1.9**, as test has 1.9 times more decoys: val 0.9854, and 0.976 on the leaderboard, the same as the unweighted model at the same French strictness.
- **"Sister business" cluster features:** val +0.0005, but the leaderboard fell from 0.964 to 0.954.

Where the v1 score was lost (val, 6.48 points), which guided the later versions:

| Cause | Source 1 records | Points |
| --- | --- | --- |
| True candidates rejected (the model was too cautious) | 18.7 % | 2.10 |
| Business had matches but none were predicted | 1.8 % | 1.84 |
| Wrong match predicted (look-alike business) | 3.9 % | 1.16 |
| True match not among the candidates (blocking) | 10.8 % | 1.15 |
| Singleton given a match | 0.2 % | 0.23 |

- **Common false positives:** the same business name at a different address (chains and reused names), especially when the Source 2/3 address is empty; renamed businesses ("X formerly Y") at a similar address; sister businesses on the same street.
- **Common false negatives:** heavily garbled names with empty addresses; Indian-script names whose words are not in the transliteration dictionary; typos in the street name combined with a changed house number.

---

## 6. Conclusion

The final pipeline scores **0.977** on the public leaderboard (val 0.9855), with **9.7 candidates per Source 1 record** on test that keep 98.7 % of true matches (val). Every step runs on a laptop CPU, except three small transformer models that were fine-tuned and run on a free Colab GPU. The biggest gains came from:

1. analysing *why* each true pair was lost and adding blocking keys aimed at those cases;
2. learned rankers instead of hand-set weights (recall@20 0.904 → 0.945), and later a union with bi-encoder neighbours (0.968 → 0.987, with fewer candidates);
3. relational features that use the one-owner structure of the data;
4. cross-encoders on the uncertain pairs (+2.3 points on the leaderboard), especially the L12 model trained with extra weight on look-alike negatives. It added only +0.15 on val, but +0.8 on the leaderboard (0.967 → 0.975), because test has many more look-alike decoys than val;
5. a threshold that is stricter only where the model is unsure: France, the country absent from training.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `run_all.sh` runs everything and stops before each GPU step (see `README.md`); `requirements.txt` pins the versions.

| File | Role |
| --- | --- |
| `src/config.py` | Paths and constants |
| `src/io_utils.py` | TSV → parquet, integer ids, integrity checks |
| `src/split.py` | train/val split |
| `src/metric.py` | Macro F0.5 exactly as the challenge defines it |
| `src/normalize.py` | Name and address cleaning, transliteration, sound key |
| `src/blocking.py` | Keys, size limits, key ranker, candidate files |
| `src/biencoder.py`, `colab/colab_biencoder_allinone.py` | Bi-encoder training data, fine-tuning and neighbour search (GPU) |
| `src/union.py` | Union of key and bi-encoder candidates, union ranker (final candidate set) |
| `src/features.py` | Pair, competition and sibling features |
| `src/model.py` | LightGBM training, one-owner rule, threshold search |
| `src/crossenc.py`, `colab/colab_crossencoder.py`, `colab/colab_ce_train_l12.py`, `colab/colab_ce_score_v5.py` | Cross-encoder data, fine-tuning and scoring (GPU) |
| `src/stage2.py` | Cross-fitted stage 1, context from its probabilities, final matcher, both TSVs |
| `src/predict.py`, `src/package.py` | TSV writing, validator, submission zip |

### B. Additional Results

Single-feature AUC on val (v1): ranker score 0.989, competition margin 0.987, within-group margin 0.969, address token-set 0.904, number Jaccard 0.883; name similarities only 0.72–0.77 (names are reused across businesses).
