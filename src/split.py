"""Split the train Source 1 entities into model-training and validation sets.

    train : config.N_TRAIN S1 ids the matching model learns from
    val   : config.N_VAL S1 ids used to pick the threshold and score the pipeline locally
    dev   : the first config.N_DEV ids of each split, for quick runs while writing code

The two splits are disjoint random samples (fixed seed, so the split is reproducible).
Candidates for every split are always drawn from the FULL train S2/S3 pool, decoys
included, so scoring val looks like scoring the test set.

Output: <cache>/split.parquet with columns id1, split, dev, country, n_matches.

Run:  python src/split.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from io_utils import load_ground_truth, load_source

SPLIT_FILE = "split.parquet"


def make_split(n_train=config.N_TRAIN, n_val=config.N_VAL, n_dev=config.N_DEV, seed=config.SEED):
    """Draw disjoint random train/val samples of train S1 ids and flag their dev subsets."""
    gt = load_ground_truth()[["id1", "n_matches"]]
    country = load_source("train", 1, columns=["id", "country"]).set_index("id")["country"]
    gt["country"] = gt["id1"].map(country).astype(str)
    if n_train + n_val > len(gt):
        raise ValueError(f"asked for {n_train + n_val:,} ids but train S1 has {len(gt):,}")

    order = np.random.default_rng(seed).permutation(len(gt))
    parts = []
    for name, idx in (("val", order[:n_val]), ("train", order[n_val:n_val + n_train])):
        part = gt.iloc[idx].copy()
        part["split"] = name
        part["dev"] = np.arange(len(part)) < n_dev   # order is random, so a prefix is a random subset
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[["id1", "split", "dev", "country", "n_matches"]]


def load_split(name, dev=False, cache_dir=config.CACHE_DIR):
    """Return the S1 integer ids of one split ("train" or "val"); dev=True gives its dev subset."""
    df = pd.read_parquet(Path(cache_dir) / SPLIT_FILE)
    mask = df["split"] == name
    if dev:
        mask &= df["dev"]
    return df.loc[mask, "id1"].to_numpy()


def summarize(df, full):
    """Print size, country mix, singleton rate and mean match count per split vs all of train."""
    def row(label, d):
        mix = d["country"].value_counts(normalize=True).sort_index()
        mix = ", ".join(f"{c} {v:.1%}" for c, v in mix.items())
        print(f"  {label:<10} {len(d):>9,}  {mix:<24} singletons {(d['n_matches'] == 0).mean():.2%}  "
              f"mean matches {d['n_matches'].mean():.2f}")

    row("all train", full)
    for name in ("train", "val"):
        row(name, df[df["split"] == name])
        row(name + " dev", df[(df["split"] == name) & df["dev"]])


def main():
    """Build, check, save and summarise the split."""
    df = make_split()
    assert df["id1"].is_unique, "train and val overlap"
    out = Path(config.CACHE_DIR) / SPLIT_FILE
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")

    gt = load_ground_truth()[["id1", "n_matches"]]
    country = load_source("train", 1, columns=["id", "country"]).set_index("id")["country"]
    summarize(df, gt.assign(country=gt["id1"].map(country).astype(str)))


if __name__ == "__main__":
    sys.exit(main())
