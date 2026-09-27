"""Candidate union: key-based blocking + bi-encoder nearest neighbours, re-ranked.

Key-based blocking (blocking.py) and the learned similarity search (biencoder.py /
colab/colab_biencoder*.py) miss DIFFERENT true matches: val recall@10 is 0.962 for the
key-based ranker and 0.975 for the bi-encoder alone. This step

    1. takes, per Source 1 record, the key-based top 20 (blocking's full top-K file) and
       the bi-encoder's top 10 neighbours, as one candidate set;
    2. recomputes the ranker features for every pair of that set (key matches from the
       raw key blocks, rapidfuzz similarities) plus the bi-encoder cosine and rank;
    3. scores them with a new LightGBM ranker trained on spare train S1 records (outside
       the train/val split, as in blocking.py) and keeps the best, at most TOP_K and only
       those with ranker probability >= config.UNION_MIN_SCORE.

The output replaces {train_all,test}_candidates.parquet with the same columns as
blocking.py (so features.py and later stages run unchanged) plus bienc_cos / bienc_rank.

Needs: data/bienc2/{train,test}_neighbors.parquet (from colab/colab_biencoder_allinone.py) and
the key blocking's full candidate files {train_all,test}_candidates_top20.parquet.

Run:  python src/union.py            train the ranker, build train_all and test, report recall
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from blocking import (RANK_FEATURES, SplitIndex, _group_rank, _query_keys, cand_path, pair_features,
                      raw_pairs)
from io_utils import load_gt_pairs

DATA = Path(config.CACHE_DIR)
UNION_FEATURES = RANK_FEATURES + ["bienc_cos", "bienc_rank"]
RANKER_PATH = DATA / "union_ranker.txt"


def _top20(name):
    """Key-based candidates before BLOCK_MIN_SCORE (blocking's full top-K file)."""
    p = cand_path(name).with_name(f"{name}_candidates_top20.parquet")
    return pd.read_parquet(p if p.exists() else cand_path(name), columns=["id1", "id2"])


def _neighbours(name):
    """Bi-encoder neighbours: id1, id2, cos, rank."""
    return pd.read_parquet(DATA / "bienc2" / f"{'train' if name == 'train_all' else 'test'}_neighbors.parquet")


def union_features(ix, rows, qkeys, key_pairs, nb):
    """Ranker features for the union of key-based and bi-encoder pairs of some S1 rows.

    key_pairs / nb: DataFrames with q, o (row positions) [+ cos, rank for nb].
    """
    n_other = len(ix.o_ids)
    code_k = key_pairs["q"].to_numpy() * n_other + key_pairs["o"].to_numpy()
    code_b = nb["q"].to_numpy() * n_other + nb["o"].to_numpy()
    code = np.union1d(code_k, code_b)
    q, o = code // n_other, code % n_other

    # key matches (shared keys that survived the size limits) of every union pair
    rq, ro, rk, rm = raw_pairs(ix, rows, qkeys)
    raw = pd.Series(np.arange(len(rq)), index=rq * n_other + ro)
    pos = raw.reindex(code).to_numpy()
    has = ~np.isnan(pos)
    n_keys, mask = np.zeros(len(code), np.int16), np.zeros(len(code), np.int16)
    n_keys[has], mask[has] = rk[pos[has].astype(int)], rm[pos[has].astype(int)]

    X = pair_features(ix, q, o, n_keys, mask)
    b = pd.DataFrame({"cos": nb["cos"].to_numpy(np.float32), "rank": nb["rank"].to_numpy(np.float32)},
                     index=code_b).reindex(code)
    X["bienc_cos"] = b["cos"].to_numpy()                   # NaN: not among the neighbours
    X["bienc_rank"] = b["rank"].fillna(99).to_numpy()
    return q, o, mask, X


def _chunk_inputs(ix, name, rows_all, chunk):
    """Yield (rows, key_pairs, nb) per chunk of S1 rows, with row positions."""
    s1_pos = pd.Index(ix.s1_ids)
    o_pos = pd.Index(ix.o_ids)
    kp = _top20(name)
    kp = pd.DataFrame({"q": s1_pos.get_indexer(kp["id1"]), "o": o_pos.get_indexer(kp["id2"])})
    nb = _neighbours(name)
    nb = pd.DataFrame({"q": s1_pos.get_indexer(nb["id1"]), "o": o_pos.get_indexer(nb["id2"]),
                       "cos": nb["cos"].astype(np.float32), "rank": nb["rank"]})
    nb = nb[(nb["q"] >= 0) & (nb["o"] >= 0)]
    rows_all = np.sort(rows_all)
    kp = kp[kp["q"].isin(rows_all)].sort_values("q", kind="stable")
    nb = nb[nb["q"].isin(rows_all)].sort_values("q", kind="stable")
    kq, bq = kp["q"].to_numpy(), nb["q"].to_numpy()
    for s in range(0, len(rows_all), chunk):
        rows = rows_all[s:s + chunk]
        lo_k, hi_k = np.searchsorted(kq, rows[0]), np.searchsorted(kq, rows[-1], "right")
        lo_b, hi_b = np.searchsorted(bq, rows[0]), np.searchsorted(bq, rows[-1], "right")
        yield rows, kp.iloc[lo_k:hi_k], nb.iloc[lo_b:hi_b]


def train_ranker(ix, n_s1=20_000):
    """Train the union ranker on spare train S1 records (the same ones blocking.py uses)."""
    t0 = time.time()
    split_ids = pd.read_parquet(DATA / "split.parquet", columns=["id1"])["id1"].to_numpy()
    spare = np.setdiff1d(ix.s1_ids, split_ids)
    ids = np.random.default_rng(config.SEED).choice(spare, n_s1, replace=False)
    rows = np.sort(ix.rows_of(ids))
    qkeys = _query_keys(ix, rows)
    parts = []
    for r, kp, nb in _chunk_inputs(ix, "train_all", rows, len(rows)):
        q, o, _, X = union_features(ix, r, qkeys, kp, nb)
        X["code"] = q * len(ix.o_ids) + o
        parts.append(X)
    X = pd.concat(parts, ignore_index=True)
    gt = load_gt_pairs()
    gt = gt[gt["id1"].isin(ids)]
    gt_code = ix.rows_of(gt["id1"].to_numpy()) * len(ix.o_ids) + pd.Index(ix.o_ids).get_indexer(gt["id2"].to_numpy())
    y = np.isin(X["code"].to_numpy(), gt_code)
    print(f"  union ranker data: {len(X):,} pairs ({len(X) / n_s1:.1f}/S1), union recall on these S1 "
          f"{y.sum() / len(gt_code):.4f}", flush=True)
    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, random_state=config.SEED,
                               verbose=-1)
    model.fit(X[UNION_FEATURES], y)
    model.booster_.save_model(str(RANKER_PATH))
    print(f"  union ranker trained ({time.time() - t0:.0f}s) -> {RANKER_PATH.name}", flush=True)


def build(name, ranker, k=config.TOP_K, min_score=None, chunk=25_000):
    """Write the union candidate file for train_all or test."""
    min_score = config.UNION_MIN_SCORE if min_score is None else min_score
    t0 = time.time()
    ix = SplitIndex("train" if name == "train_all" else "test")
    rows_all = np.arange(len(ix.s1_ids))
    qkeys = _query_keys(ix, rows_all)
    out_path = cand_path(name)
    writer, n_out = None, 0
    try:
        for rows, kp, nb in _chunk_inputs(ix, name, rows_all, chunk):
            q, o, mask, X = union_features(ix, rows, qkeys, kp, nb)
            score = ranker.predict(X[UNION_FEATURES]).astype(np.float32)
            order = np.lexsort((-score, q))
            rank = _group_rank(q[order])
            top = (rank < k) & (score[order] >= min_score)
            keep, rank = order[top], rank[top]
            cand = X.iloc[keep].reset_index(drop=True)
            cand.insert(0, "id2", ix.o_ids[o[keep]])
            cand.insert(0, "id1", ix.s1_ids[q[keep]])
            cand["key_mask"] = mask[keep]
            cand["block_score"] = score[keep]
            cand["rank"] = rank.astype(np.int16)
            table = pa.Table.from_pandas(cand, preserve_index=False)
            writer = writer or pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            n_out += len(cand)
            print(f"  [{name}] {rows[-1] + 1:,}/{len(rows_all):,} S1, {n_out / (rows[-1] + 1):.2f} cand/S1 "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        if writer:
            writer.close()
    del ix
    gc.collect()


def sweep(ranker, taus=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05), k=config.TOP_K):
    """Val recall and candidates per S1 (val and a test sample) for several min-score cut-offs."""
    from split import load_split

    res = {}
    for name, split, ids in (("val", "train", load_split("val")), ("test", "test", None)):
        ix = SplitIndex(split)
        if ids is None:
            ids = np.random.default_rng(config.SEED).choice(ix.s1_ids, 50_000, replace=False)
        rows = np.sort(ix.rows_of(ids))
        qkeys = _query_keys(ix, rows)
        (r, kp, nb), = list(_chunk_inputs(ix, "train_all" if split == "train" else "test", rows, len(rows)))
        q, o, _, X = union_features(ix, r, qkeys, kp, nb)
        score = ranker.predict(X[UNION_FEATURES])
        order = np.lexsort((-score, q))
        rank = np.empty(len(q), int)
        rank[order] = _group_rank(q[order])
        if name == "val":
            gt = load_gt_pairs()
            gt = gt[gt["id1"].isin(ids)]
            gt_code = ix.rows_of(gt["id1"].to_numpy()) * len(ix.o_ids) + pd.Index(ix.o_ids).get_indexer(gt["id2"].to_numpy())
            y = np.isin(q * len(ix.o_ids) + o, gt_code)
            print(f"  union before cut-off: val recall {y.sum() / len(gt_code):.4f}, {len(q) / len(rows):.1f} pairs/S1")
        for t in taus:
            keep = (rank < k) & (score >= t)
            res.setdefault(t, {})[name] = (keep.sum() / len(rows), (y & keep).sum() / len(gt_code) if name == "val" else None)
        del ix
        gc.collect()
    print("  min score   val cand/S1   test cand/S1   val recall")
    for t in taus:
        print(f"  {t:<10}  {res[t]['val'][0]:10.2f}  {res[t]['test'][0]:12.2f}   {res[t]['val'][1]:.4f}")


def report():
    """Recall and size of the new candidate files for the val / train splits and test."""
    from split import load_split

    gt = load_gt_pairs()
    c = pd.read_parquet(cand_path("train_all"), columns=["id1", "id2", "rank"])
    for split in ("val", "train"):
        ids = load_split(split)
        g = gt[gt["id1"].isin(ids)]
        cc = c[c["id1"].isin(ids)]
        hit = g.merge(cc, on=["id1", "id2"])
        print(f"  [{split}] recall {len(hit) / len(g):.4f} with {len(cc) / len(ids):.2f} candidates per S1")
    t = pd.read_parquet(cand_path("test"), columns=["id1"])
    print(f"  [test] {len(t) / 1_732_544:.2f} candidates per S1")


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Union of key-based and bi-encoder candidates")
    ap.add_argument("--skip-ranker", action="store_true", help="reuse union_ranker.txt")
    ap.add_argument("--min-score", type=float, default=None, help="override config.UNION_MIN_SCORE")
    ap.add_argument("--sweep", action="store_true", help="only report recall / size for several cut-offs")
    args = ap.parse_args(argv)
    if not args.skip_ranker:
        ix = SplitIndex("train")
        train_ranker(ix)
        del ix
        gc.collect()
    ranker = lgb.Booster(model_file=str(RANKER_PATH))
    if args.sweep:
        return sweep(ranker)
    for name in ("train_all", "test"):
        build(name, ranker, min_score=args.min_score)
    report()


if __name__ == "__main__":
    sys.exit(main())
