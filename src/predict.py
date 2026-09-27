"""Test predictions and submission files (step 7) for the stage-1-only model.

Its writers and validator call are used by stage2.py, which makes the final submission.

    1. score every test candidate pair with the saved matching model (in chunks)
    2. one owner per S2/S3 record, then the saved threshold (same as model.py on val)
    3. write output/matching_results.tsv  - final matches, one row per test S1 record
       and output/candidate_pairs.tsv     - the candidate list the model scored
    4. run the official validator and copy the files to submissions/ with a timestamp

Run:  python src/predict.py [--threshold 0.6]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from blocking import cand_path
from features import feat_path
from io_utils import int_to_ids, load_source
from model import meta_path, model_path, one_owner

VALIDATOR = config.DATASET_DIR.parent / "utils" / "validate_submission.py"
HEADER_MATCH = "source1_entity_id\tmatched_entity_ids\n"
HEADER_CAND = "source1_entity_id\tcandidate_entity_ids\n"


def score_test(cols, model, batch=2_000_000):
    """Probability for every test candidate pair -> DataFrame(id1, id2, p)."""
    pf = pq.ParquetFile(feat_path("test"))
    parts, t0 = [], time.time()
    for b in pf.iter_batches(batch_size=batch, columns=["id1", "id2"] + cols):
        df = b.to_pandas()
        p = model.predict(df[cols].to_numpy(np.float32)).astype(np.float32)
        parts.append(pd.DataFrame({"id1": df["id1"].to_numpy(), "id2": df["id2"].to_numpy(), "p": p}))
        print(f"  scored {sum(len(x) for x in parts):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def write_id_lists(path, header, s1_ids, pairs, col2="id2"):
    """Write one row per S1 id: '<S1 id>\\t<comma-separated S2/S3 ids>' (empty list allowed)."""
    lists = (pairs.assign(s=int_to_ids(pairs[col2].to_numpy()).to_numpy())
             .groupby("id1", sort=False)["s"].agg(",".join))
    ids_str = int_to_ids(s1_ids)
    vals = lists.reindex(s1_ids).fillna("").to_numpy()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.writelines(f"{a}\t{b}\n" for a, b in zip(ids_str, vals))


def write_candidates(path, s1_ids, batch=4_000_000):
    """candidate_pairs.tsv from test_candidates.parquet, grouped per S1 id in chunks."""
    lists = []
    for b in pq.ParquetFile(cand_path("test")).iter_batches(batch_size=batch, columns=["id1", "id2"]):
        df = b.to_pandas()
        lists.append(df.assign(s=int_to_ids(df["id2"].to_numpy()).to_numpy())
                     .groupby("id1", sort=False)["s"].agg(",".join))
    lists = pd.concat(lists)
    lists = lists.groupby(level=0, sort=False).agg(",".join)      # an S1 split across two batches
    ids_str = int_to_ids(s1_ids)
    vals = lists.reindex(s1_ids).fillna("").to_numpy()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER_CAND)
        f.writelines(f"{a}\t{b}\n" for a, b in zip(ids_str, vals))


def validate(match_path, cand_file):
    """Run the challenge's validator; returns True on PASS."""
    cmd = [sys.executable, str(VALIDATOR), "--matching", str(match_path),
           "--candidate", str(cand_file), "--test-dir", str(config.DATASET_DIR / "test")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())
    return res.returncode == 0


def main(argv=None):
    """Score test pairs, write both submission files, validate and archive them."""
    ap = argparse.ArgumentParser(description="Write the submission files")
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    args = ap.parse_args(argv)

    t0 = time.time()
    meta = json.loads(meta_path().read_text())
    cols, threshold = meta["features"], args.threshold or meta["threshold"]
    model = lgb.Booster(model_file=str(model_path()))
    print(f"model: {len(cols)} features, threshold {threshold:.3f}, val macro F0.5 {meta['val_macro_f05']:.4f}")

    pairs = score_test(cols, model)
    kept = one_owner(pairs)
    kept = kept[kept["p"] >= threshold]
    s1_ids = load_source("test", 1, columns=["id"])["id"].to_numpy()
    n_matched = kept["id1"].nunique()
    print(f"  {len(kept):,} matches for {n_matched:,} of {len(s1_ids):,} S1 "
          f"({kept.groupby('id1').size().mean():.2f} per matched S1; {1 - n_matched / len(s1_ids):.1%} empty)")

    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    match_file, cand_file = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    write_id_lists(match_file, HEADER_MATCH, s1_ids, kept)
    write_candidates(cand_file, s1_ids)
    print(f"  wrote {match_file} and {cand_file} ({time.time() - t0:.0f}s)")

    ok = validate(match_file, cand_file)
    stamp = datetime.now().strftime("%m%d_%H%M")
    arch = Path(config.WORK_DIR) / "submissions" / f"{stamp}_val{meta['val_macro_f05']:.4f}_t{threshold:.3f}"
    arch.mkdir(parents=True, exist_ok=True)
    shutil.copy(match_file, arch / match_file.name)
    shutil.copy(meta_path(), arch / meta_path().name)
    print(f"  {'PASS' if ok else 'VALIDATION FAILED'}; copy saved in {arch}")


if __name__ == "__main__":
    sys.exit(main())
