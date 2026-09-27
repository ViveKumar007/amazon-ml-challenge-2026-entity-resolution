#!/usr/bin/env bash
# End-to-end pipeline: raw TSVs -> output/final_threshold_0.75_France0.97/matching_results.tsv
#                                  + output/candidate_pairs.tsv
#
#   bash run_all.sh                  run from the start
#   bash run_all.sh --from union     start from a later step (earlier cached outputs are reused)
#
# Three steps need a GPU (Google Colab, T4). The script stops before each of them, says
# which notebook to run and which files to copy, and is then restarted with --from <next>.
#
# Steps (outputs under $ER_CACHE_DIR, default ../../data):
#   cache       io_utils.py    raw TSVs -> parquet with integer ids
#   split       split.py       train (500k) / val (50k) samples of train Source 1 records
#   normalize   normalize.py   cleaned names / addresses of every record
#   blocking    blocking.py    10 key types + LightGBM ranker -> top-20 per S1 (all train and test),
#                              kept if ranker score >= BLOCK_MIN_SCORE
#   features    features.py    ~60 pair features for val, train, test and all train S1
#   stage1      stage2.py      cross-fitted stage-1 matcher on the key-based candidates
#                              (its probabilities choose the cross-encoders' training pairs)
#   gpu1_export crossenc.py / biencoder.py   data for the three GPU notebooks
#   GPU 1       colab_crossencoder.py (MiniLM-L6), colab_ce_train_l12.py (MiniLM-L12),
#               colab_biencoder_allinone.py (bi-encoder neighbours)
#   union       union.py       key top-20 + bi-encoder top-10 -> union ranker -> final candidate set
#   features_u  features.py    pair features on the union candidates
#   stage1_u    stage2.py      stage-1 matcher on the union candidates
#   gpu2_export crossenc.py    ids of the uncertain pairs to score
#   GPU 2       colab_ce_score_v5.py (both cross-encoders)
#   final       stage2.py      stage 3 = stage 2 + both cross-encoder scores -> both TSVs, validator
# Logs go to $ER_CACHE_DIR/logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
DATA="${ER_CACHE_DIR:-../../data}"
THRESHOLD="${ER_THRESHOLD:-0.75}"             # final match threshold (val optimum)
STRICT="${ER_STRICT:-France=0.97}"          # stricter for the country absent from training
mkdir -p "$DATA/logs"

steps=(cache split normalize blocking features stage1 gpu1_export union features_u stage1_u gpu2_export final)
from=cache
[[ "${1:-}" == "--from" ]] && from="${2:?step name}"

run() {  # run <log name> <command...>
  local log=$1; shift
  echo "=== $(date +%H:%M:%S) $log: $*"
  "$@" 2>&1 | tee "$DATA/logs/$log.log"
}

gpu_stop() {  # print what to run on the GPU, then stop
  echo; echo "=== GPU step: $1"; echo "$2"; echo "Then run:  bash run_all.sh --from $3"; exit 0
}

started=0
for s in "${steps[@]}"; do
  [[ $s == "$from" ]] && started=1
  [[ $started == 1 ]] || continue
  case $s in
    cache)      run cache      python -u src/io_utils.py ;;
    split)      run split      python -u src/split.py ;;
    normalize)  run normalize  python -u src/normalize.py ;;
    blocking)   run blocking   python -u src/blocking.py --all-train --splits test ;;
    features)   run features   python -u src/features.py --splits val train test train_all ;;
    stage1)     run stage1     python -u src/stage2.py --baseline 0 ;;
    gpu1_export)
      run ce_export   python -u src/crossenc.py --export --band 0.005 0.999 --n-train 600000 --min-score 0.001
      run ce_hard     python -u src/crossenc.py --export-hard
      run bienc       python -u src/biencoder.py --export
      run bienc_full  python -u src/biencoder.py --export-full
      gpu_stop "three notebooks (colab/), each with a T4 GPU and Google Drive:" \
"  1. colab_crossencoder.py       input $DATA/colab/ce_data_v2.parquet (Colab Files panel)
                                  -> My Drive/ce_model_v2 and ce_scores_v2.parquet
     copy ce_scores_v2.parquet to $DATA/colab/, then: python src/crossenc.py --import-scores
  2. colab_ce_train_l12.py        input My Drive/ce_train_v4.parquet   -> My Drive/ce_model_l12
  3. colab_biencoder_allinone.py  input My Drive/bienc/ and bienc2/ (from $DATA/bienc, $DATA/bienc2)
     copy My Drive/bienc2/{train,test}_neighbors.parquet to $DATA/bienc2/" union ;;
    union)      run union      python -u src/union.py ;;
    features_u) run features_u python -u src/features.py --splits val train test train_all ;;
    stage1_u)   run stage1_u   python -u src/stage2.py --baseline 0 --tag _u ;;
    gpu2_export)
      run ce_v5_export python -u src/crossenc.py --export-missing --band 0.005 0.999
      gpu_stop "colab/colab_ce_score_v5.py (T4 GPU)" \
"  put $DATA/colab/ce_pairs_v5.parquet in My Drive (next to ce_model_l12, ce_model_v2, bienc, bienc2),
  run the notebook, copy My Drive/ce_scores_v5.parquet to $DATA/colab/,
  then: python src/crossenc.py --import-v5" final ;;
    final)
      run final   python -u src/stage2.py --reuse-stage1 --ce --ce12 --predict --baseline 0 --tag _final
      run final_t python -u src/stage2.py --variants "$THRESHOLD" --country-threshold "$STRICT" --tag _final ;;
  esac
done
echo "=== $(date +%H:%M:%S) done"
