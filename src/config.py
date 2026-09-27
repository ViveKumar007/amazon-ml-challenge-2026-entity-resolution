"""Paths and constants shared by every pipeline stage.

Every path can be overridden with an environment variable so the pipeline runs
from anywhere (e.g. from the unpacked submission zip).
"""
import os
from pathlib import Path

# Project root: src -> business_entity_resolution -> code -> <root>
WORK_DIR = Path(__file__).resolve().parents[3]

# Raw challenge data: the dataset/ folder from student_resource (holds train/ and test/).
DATASET_DIR = Path(os.environ.get(
    "ER_DATASET_DIR",
    WORK_DIR.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
))

# Parquet cache written by io_utils.py, read by every later stage.
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", WORK_DIR / "data"))

# Where matching_results.tsv and candidate_pairs.tsv are written.
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", WORK_DIR / "output"))

SEED = 42
N_WORKERS = os.cpu_count() or 4   # processes for CPU-heavy steps (normalisation)

# Train Source 1 entities used for model training / validation (split.py).
# dev = the first N_DEV ids of each split, for quick end-to-end runs while coding.
N_TRAIN = 500_000       # 100k -> 200k gave +0.17 F0.5 on val; more data still helps
N_VAL = 50_000
N_DEV = 5_000

# Blocking (blocking.py): a key is skipped when any of these limits is exceeded
BLOCK_CAP_S1 = 500           # Source 1 records sharing the key
BLOCK_CAP_OTHER = 1000       # Source 2/3 records sharing the key
BLOCK_MAX_PAIRS = 50_000     # pairs the key creates (S1 count x S2/S3 count)
# chosen with `blocking.py --sweep` on val (keys v3): raw recall 0.970 at ~253 pairs per S1;
# looser limits add little (0.974 needs 533 pairs per S1). v2 keys: 0.946 at 168.
TOP_K = 20               # candidates kept per Source 1 record (at most)
BLOCK_MIN_SCORE = 0.001  # ... and only those with ranker probability >= this. Halves the test
                         # candidate set (19.5 -> 11.7 per S1) for -0.0004 val F0.5; smaller
                         # candidate sets rank higher in the final evaluation (problem update)
UNION_MIN_SCORE = 0.002  # union.py (keys + bi-encoder): keep pairs with union-ranker probability >= this
BLOCK_CHUNK = 25_000     # Source 1 records joined per chunk (bounds memory: ~6M pairs)
