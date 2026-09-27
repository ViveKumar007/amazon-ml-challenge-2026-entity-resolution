"""Learned similarity search (bi-encoder) as an extra candidate generator: data export.

A small transformer maps every record's text ("name ; address") to a vector, trained so
that a Source 1 record lands close to its own Source 2/3 copies. Each Source 1 record's
nearest records (within its country) are candidates. This catches true matches that share
no blocking key (renamed businesses, heavy typos, transliterations).

The heavy work (training, encoding ~10M records, nearest-neighbour search) runs on a GPU
notebook: colab/colab_biencoder_allinone.py. This file prepares its inputs and reads its outputs.

Export (<cache>/bienc/):
    index.parquet   id, country, text   every train S2/S3 record (the search space)
    s1.parquet      id, country, text, split   train-split and val S1 records
    pairs.parquet   id1, id2, split     true matches of those S1 records

Run:  python src/biencoder.py --export
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from io_utils import load_gt_pairs
from normalize import norm_path

DATA = Path(config.CACHE_DIR)
OUT = DATA / "bienc"


def record_texts(split, source, ids=None):
    """id, country, text ('name / alias ; address') of one normalised source file."""
    filters = [("id", "in", list(map(int, ids)))] if ids is not None else None
    t = pq.read_table(norm_path(split, source), columns=["id", "country", "name_clean", "name_alt", "addr_clean"],
                      filters=filters).to_pandas()
    alt = np.where(t["name_alt"] != "", " / " + t["name_alt"], "")
    return pd.DataFrame({"id": t["id"].to_numpy(), "country": t["country"].astype(str).to_numpy(),
                         "text": (t["name_clean"] + alt + " ; " + t["addr_clean"]).to_numpy()})


def export():
    """Write the three files the notebook needs."""
    from split import load_split

    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    index = pd.concat([record_texts("train", s) for s in (2, 3)], ignore_index=True)
    index.to_parquet(OUT / "index.parquet", index=False, compression="zstd")
    tr, va = load_split("train"), load_split("val")
    s1 = record_texts("train", 1, np.r_[tr, va])
    s1["split"] = np.where(s1["id"].isin(va), "val", "train")
    s1.to_parquet(OUT / "s1.parquet", index=False, compression="zstd")
    gt = load_gt_pairs()
    pairs = gt[gt["id1"].isin(s1["id"])].copy()
    pairs["split"] = np.where(pairs["id1"].isin(va), "val", "train")
    pairs.to_parquet(OUT / "pairs.parquet", index=False, compression="zstd")
    size = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 1e6
    print(f"  index {len(index):,} records, S1 {len(s1):,}, pairs {len(pairs):,}; {size:.0f} MB in {OUT} "
          f"({time.time() - t0:.0f}s)")


def export_full():
    """Second export for using the bi-encoder in the pipeline: every train S1 record and the
    whole test split (queries + search space), in <cache>/bienc2/."""
    t0 = time.time()
    out = DATA / "bienc2"
    out.mkdir(exist_ok=True)
    record_texts("train", 1).to_parquet(out / "train_s1.parquet", index=False, compression="zstd")
    record_texts("test", 1).to_parquet(out / "test_s1.parquet", index=False, compression="zstd")
    pd.concat([record_texts("test", s) for s in (2, 3)], ignore_index=True).to_parquet(
        out / "test_index.parquet", index=False, compression="zstd")
    size = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
    print(f"  wrote {out}: {size:.0f} MB ({time.time() - t0:.0f}s)")


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Bi-encoder candidate generation (data side)")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-full", action="store_true", help="all train S1 + test split, for the pipeline")
    args = ap.parse_args(argv)
    if args.export:
        export()
    if args.export_full:
        export_full()


if __name__ == "__main__":
    sys.exit(main())
