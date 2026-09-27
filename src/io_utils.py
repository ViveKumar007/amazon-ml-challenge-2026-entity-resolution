"""Load the challenge TSVs once and cache them as parquet.

Every record gets an integer id next to its string entity_id:

    S2-681193310  ->  2 * 10**10 + 681193310  =  20681193310

Integers are far cheaper than strings for the joins and group-bys in blocking and
matching. The string entity_id is kept for writing the submission files.

Cached files (in config.CACHE_DIR):
    {train,test}_source{1,2,3}.parquet  entity_id, business_name, business_address, country, id
    train_ground_truth.parquet          source1_entity_id, matched_entity_ids, id1, n_matches
    train_gt_pairs.parquet              id1, id2  (one row per true S1 -> S2/S3 match)

Run:  python src/io_utils.py [--dataset-dir DIR] [--cache-dir DIR] [--force]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

ID_BASE = 10**10
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def read_tsv(path):
    """Read a challenge TSV with every column as a plain string.

    quoting=3 (QUOTE_NONE) stops a stray quote character in a name from swallowing
    the following tabs/newlines; keep_default_na=False keeps empty fields as "" and
    names such as "NA" or "NULL" as text instead of NaN.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def count_data_rows(path):
    """Count the data rows (lines minus the header) of a text file, independently of pandas."""
    n_lines, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n_lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":  # final line has no trailing newline
        n_lines += 1
    return n_lines - 1


def ids_to_int(ids):
    """Convert a Series of 'S<k>-<digits>' ids to int64 k * 10**10 + digits.

    Raises ValueError on a malformed id, or on one that would not convert back to the
    same string (a leading zero), so int_to_ids() is always an exact inverse.
    """
    parts = ids.str.extract(r"^S([123])-(\d{1,10})$")
    bad = parts[0].isna()
    if bad.any():
        raise ValueError(f"malformed entity ids, e.g. {ids[bad].head(5).tolist()}")
    num = parts[1].astype(np.int64)
    if (num.astype(str) != parts[1]).any():
        raise ValueError("entity ids with leading zeros cannot round-trip through int")
    return (parts[0].astype(np.int64) * ID_BASE + num).to_numpy()


def int_to_ids(ints):
    """Inverse of ids_to_int: int64 array-like -> Series of 'S<k>-<digits>' strings."""
    s = pd.Series(np.asarray(ints, dtype=np.int64))
    return "S" + (s // ID_BASE).astype(str) + "-" + (s % ID_BASE).astype(str)


def source_of(ints):
    """Source number (1, 2 or 3) of each integer id."""
    return np.asarray(ints, dtype=np.int64) // ID_BASE


def _check(cond, msg):
    """Stop the caching run with a clear message when a data check fails."""
    if not cond:
        raise ValueError(msg)


def cache_source(split, source, dataset_dir, cache_dir):
    """Read one {split}_source{source}.tsv, add the integer id, validate and write parquet.

    Returns the integer ids so the ground-truth check can reuse them without re-reading.
    """
    path = Path(dataset_dir) / split / f"{split}_source{source}.tsv"
    df = read_tsv(path)
    _check(list(df.columns) == SOURCE_COLUMNS, f"{path.name}: unexpected columns {list(df.columns)}")

    expected = count_data_rows(path)
    _check(len(df) == expected, f"{path.name}: pandas read {len(df)} rows, file has {expected}")

    df["id"] = ids_to_int(df["entity_id"])
    _check((source_of(df["id"]) == source).all(), f"{path.name}: ids with the wrong S{source}- prefix")
    _check(df["id"].is_unique, f"{path.name}: duplicate entity ids")
    df["country"] = df["country"].astype("category")

    df.to_parquet(Path(cache_dir) / f"{split}_source{source}.parquet", index=False)
    print(f"  {path.name:<24} {len(df):>10,} rows (file: {expected:,})  "
          f"countries: {df['country'].value_counts().to_dict()}")
    return df["id"].to_numpy()


def cache_ground_truth(dataset_dir, cache_dir, train_ids):
    """Cache train_ground_truth.tsv plus an exploded (id1, id2) pair table.

    train_ids maps source number -> integer ids of that train source file; it is used
    to check that every S1 row is labelled once and every matched id exists.
    """
    path = Path(dataset_dir) / "train" / "train_ground_truth.tsv"
    gt = read_tsv(path)
    _check(list(gt.columns) == GT_COLUMNS, f"{path.name}: unexpected columns {list(gt.columns)}")

    expected = count_data_rows(path)
    _check(len(gt) == expected, f"{path.name}: pandas read {len(gt)} rows, file has {expected}")

    gt["id1"] = ids_to_int(gt["source1_entity_id"])
    _check(gt["id1"].is_unique, f"{path.name}: duplicate source1_entity_id rows")
    _check(set(gt["id1"]) == set(train_ids[1]), f"{path.name}: S1 ids differ from train_source1")

    pairs = (gt[["id1", "matched_entity_ids"]]
             .assign(m=gt["matched_entity_ids"].str.split(","))
             .explode("m"))
    pairs = pairs[pairs["m"].notna() & (pairs["m"] != "")]
    pairs = pd.DataFrame({"id1": pairs["id1"].to_numpy(), "id2": ids_to_int(pairs["m"])})

    _check(not np.isin(source_of(pairs["id2"]), [1]).any(), f"{path.name}: S1 id inside a match list")
    known = np.concatenate([train_ids[2], train_ids[3]])
    _check(np.isin(pairs["id2"], known).all(), f"{path.name}: matched id missing from train S2/S3")
    _check(not pairs.duplicated().any(), f"{path.name}: duplicate id inside a match list")

    gt["n_matches"] = pairs.groupby("id1").size().reindex(gt["id1"], fill_value=0).to_numpy()
    gt.to_parquet(Path(cache_dir) / "train_ground_truth.parquet", index=False)
    pairs.to_parquet(Path(cache_dir) / "train_gt_pairs.parquet", index=False)
    print(f"  {path.name:<24} {len(gt):>10,} rows (file: {expected:,})  "
          f"{len(pairs):,} true pairs, {(gt['n_matches'] == 0).mean():.1%} singletons")


def cache_all(dataset_dir=config.DATASET_DIR, cache_dir=config.CACHE_DIR, force=False):
    """Build the whole parquet cache; skipped when it already exists unless force=True."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = [cache_dir / f"{sp}_source{s}.parquet" for sp in SPLITS for s in SOURCES]
    targets += [cache_dir / "train_ground_truth.parquet", cache_dir / "train_gt_pairs.parquet"]
    if not force and all(p.exists() for p in targets):
        print(f"Cache already complete in {cache_dir} (use --force to rebuild)")
        return

    print(f"Caching {dataset_dir} -> {cache_dir}")
    for split in SPLITS:
        ids = {s: cache_source(split, s, dataset_dir, cache_dir) for s in SOURCES}
        if split == "train":
            cache_ground_truth(dataset_dir, cache_dir, ids)
        del ids


def load_source(split, source, columns=None, cache_dir=config.CACHE_DIR):
    """Load one cached source file, e.g. load_source("test", 2)."""
    return pd.read_parquet(Path(cache_dir) / f"{split}_source{source}.parquet", columns=columns)


def load_ground_truth(cache_dir=config.CACHE_DIR):
    """Load the cached train ground truth (one row per train S1 entity)."""
    return pd.read_parquet(Path(cache_dir) / "train_ground_truth.parquet")


def load_gt_pairs(cache_dir=config.CACHE_DIR):
    """Load the true (id1, id2) match pairs of the train set."""
    return pd.read_parquet(Path(cache_dir) / "train_gt_pairs.parquet")


def main(argv=None):
    """Command-line entry point: build the parquet cache."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args(argv)

    t0 = time.time()
    cache_all(args.dataset_dir, args.cache_dir, args.force)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
