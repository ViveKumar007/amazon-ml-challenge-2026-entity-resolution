# Business Entity Resolution: Amazon ML Challenge 2026 (team Queen's Code)

**Result:** macro F0.5 **0.977** on the public leaderboard, with 9.7 candidates per Source 1 record (98.7 % of true matches kept on val). The full methodology, analysis and results are in [docs/methodology.md](docs/methodology.md).

> The challenge dataset is not included: it belongs to the organisers. Point `ER_DATASET_DIR` to your copy (see Setup).

For every Source 1 record, this pipeline finds the Source 2 and Source 3 records that describe the same business. It writes `output/candidate_pairs.tsv` (every pair the model scores) and `matching_results.tsv` (the final matches, a subset of the candidates).

Only the provided data is used: no external lookups, no geocoding, no internet data. All models are trained here from the train set.

| Model | Role | Size | Licence |
| --- | --- | --- | --- |
| LightGBM | candidate rankers, stage-1/2/3 matchers | trees | MIT |
| `sentence-transformers/all-MiniLM-L6-v2`, fine-tuned | bi-encoder: extra candidates | 22M | Apache-2.0 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`, fine-tuned | pair re-scoring (feature `ce`) | 22M | Apache-2.0 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2`, fine-tuned | pair re-scoring (feature `ce12`) | 33M | Apache-2.0 |

## Setup

```bash
python -m pip install -r requirements.txt          # Python 3.12
export ER_DATASET_DIR=/path/to/student_resource/dataset
export ER_CACHE_DIR=/path/to/work/data        # intermediate files (~30 GB)
export ER_OUTPUT_DIR=/path/to/output          # the submission files
```

Default paths are set in `src/config.py`. The CPU steps were run on 8 cores and 16 GB of RAM. The three transformer models were trained and run on a free Google Colab T4 GPU, using the notebooks in `colab/`.

## Reproduce end to end

```bash
bash run_all.sh                    # stops before each GPU step and says what to run
bash run_all.sh --from union       # continue after a GPU step
```

The CPU steps take about 10 hours and the GPU notebooks about 4 hours. `run_all.sh` lists every step with its inputs and outputs.

## Method

**1. Normalisation** (`normalize.py`). The name and address cleaning:

- transliterates 7 Indian scripts;
- handles legal forms (US, India and France), aliases, domains and look-alike digits (`appare1s`);
- expands address abbreviations, state names and codes, NULLs and landmarks;
- adds a sound key.

**2. Candidate generation.** This is the set in `candidate_pairs.tsv`.

- **Key blocking** (`blocking.py`): 10 hashed key types. Examples are rare name words, the sound key, a no-space prefix, typo-tolerant house number × address word keys, and name + place keys. Each key has size limits. A LightGBM ranker keeps the top 20 per Source 1 record.
- **Bi-encoder** (`biencoder.py`, `colab/colab_biencoder_allinone.py`): a MiniLM model fine-tuned with in-batch contrastive loss. It gives the 10 nearest records per Source 1 record within the same country, and finds matches that share no key.
- **Union** (`union.py`): both lists are merged and re-ranked by a second LightGBM ranker (the key features plus bi-encoder cosine and rank). Pairs with ranker probability ≥ 0.002 are kept.

| Candidate set | Val recall | Candidates / S1 (val) | Candidates / S1 (test) |
| --- | --- | --- | --- |
| key blocking, top 20 | 0.968 | 19.4 | 19.5 |
| key blocking, score ≥ 0.001 | 0.966 | 9.6 | 11.7 |
| **union, score ≥ 0.002 (final)** | **0.987** | **8.1** | **9.7** |

The test set has about 1.9 times more unmatched "decoy" records per Source 1 record than train, so the same score cutoff keeps a few more candidates there.

**3. Pair features** (`features.py`). About 60 features:

- string and rarity-weighted overlaps of names and addresses;
- house number, postcode and state agreement;
- competition between Source 1 records claiming the same record;
- similarity to the best sibling candidate.

**4. Matching** (`stage2.py`).

- **Stage 1:** a LightGBM model, 2-fold cross-fitted on train Source 1 records, scores every pair.
- **Stage 2:** rebuilds the competition and group context from the stage-1 probabilities, for example the margin to the best competing Source 1 record.
- **Stage 3:** stage 2 plus two cross-encoder scores (`crossenc.py`, `colab/`). The MiniLM-L6 and MiniLM-L12 cross-encoders read both records' text together and are fine-tuned on uncertain pairs (stage-1 probability between 0.005 and 0.999). L12 is trained with confusable negatives, such as sister businesses at the same street, weighted ×2. Only pairs in that band are scored by the cross-encoders; the feature is empty elsewhere.
- **Decision:** each Source 2/3 record goes to at most one Source 1 record (the highest probability). A pair is a match above the threshold.

**Leakage control.** The rankers and cross-encoders are trained on train Source 1 records outside the 500k/50k train/val split. Stage-1 scores used for stage-2 training come from the fold model that did not see the pair. The official macro F0.5 is implemented in `metric.py`, singletons included.

## Results

| Version | Val macro F0.5 | Leaderboard |
| --- | --- | --- |
| stage 1 | 0.9352 | 0.926 |
| stage 2 | 0.9609 | 0.939 |
| + MiniLM-L6 cross-encoder, smaller key candidate set, threshold 0.90 | 0.9772 | 0.967 |
| union candidates + MiniLM-L6 only, threshold 0.90 | 0.9840 | 0.967 |
| union candidates + both cross-encoders, threshold 0.90 | 0.9855 | 0.975 |
| same, India/US 0.75 and France 0.95 | 0.9855 | 0.976 |
| **same, India/US 0.75 and France 0.97 (final)** | **0.9855** | **0.977** |

The thresholds were chosen with val and leaderboard probes. India and the US use 0.75, their val optimum. France never appears in training and its probabilities are the least calibrated, so it uses a stricter 0.97 (0.95 → 0.976, 0.97 → 0.977, 0.985 → 0.977 on the leaderboard). The final files are `output/final_threshold_0.75_France0.97/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Files

| Path | Purpose |
| --- | --- |
| `src/config.py` | paths, seeds, blocking and candidate limits |
| `src/io_utils.py`, `src/split.py`, `src/metric.py` | data loading, train/val split, the official metric |
| `src/normalize.py` | name and address normalisation |
| `src/blocking.py` | key blocking and candidate ranker |
| `src/biencoder.py`, `src/union.py` | bi-encoder data export, union candidate set |
| `src/features.py`, `src/model.py`, `src/stage2.py` | pair features, LightGBM matchers, final prediction |
| `src/crossenc.py` | cross-encoder data export and import (a CPU training path is included) |
| `src/predict.py`, `src/package.py` | writing and validating the TSVs, building the submission zip |
| `colab/colab_crossencoder.py`, `colab_ce_train_l12.py`, `colab_ce_score_v5.py` | cross-encoder training and scoring (GPU) |
| `colab/colab_biencoder_allinone.py` | bi-encoder training and neighbour search (GPU) |
