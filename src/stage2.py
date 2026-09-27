"""Final matcher: stage 1 -> context from its probabilities -> stage 2 (+ cross-encoder scores).

The stage-1 pair model (model.py features) leans ~70% on o_margin, a competition feature
built from the blocking ranker's score. The stage-1 probability p1 is a much sharper
score, so stage 2 rebuilds the context from it:

    p1           stage-1 probability of the pair
    o_margin_p   p1 minus the best OTHER S1 record's p1 for the same S2/S3 record
    o_n_p50      S1 records claiming this S2/S3 record with p1 > 0.5
    g_margin_p   p1 minus the best OTHER candidate's p1 within the same S1 record
    g_n_p50      candidates of the S1 record with p1 > 0.5
    g_rank_p     rank of p1 within the S1 record (0 = best)
    g_sum_p      sum of p1 over the S1 record's candidates (expected number of matches)

Competition needs p1 for EVERY S1 record claiming a record, so stage 1 scores the
features of all train S1 records (train_all_features.parquet) and all test pairs.

Leakage control (cross-fitting): the train split's S1 records are split into folds A and
B. Model A (trained on A) scores fold B, model B scores fold A, and model A scores every
other pair (val, the train S1 records outside the split, test), none of which it saw.

Stage 3 (final model) = stage 2 plus two cross-encoder scores, --ce (MiniLM-L6) and --ce12
(MiniLM-L12), from crossenc.py. They exist only for pairs whose p1 was uncertain
(0.005-0.999) and are empty elsewhere, in train, val and test alike.

Decision: each S2/S3 record goes to its highest-probability S1 record (one owner), then a
threshold. --country-threshold sets a stricter one for a country absent from training
(France), whose probabilities are the least calibrated.

Needs: python src/features.py --splits val train test train_all

Run:  python src/stage2.py                    fit both stages, report val (writes nothing to output/)
      python src/stage2.py --predict          also write the submission files
      final submission (see run_all.sh):
      python src/stage2.py --reuse-stage1 --ce --ce12 --predict --baseline 0 --tag _final
      python src/stage2.py --variants 0.75 --country-threshold France=0.97 --tag _final
      -> output/final_threshold_0.75_France0.97/matching_results.tsv + output/candidate_pairs.tsv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from features import feat_path
from model import PARAMS, one_owner, train, tune, val_truth
from split import load_split

FAST = dict(PARAMS, learning_rate=0.1)          # fewer trees: stage 1 scores ~77M pairs
S2_COLS = ["p1", "o_margin_p", "o_n_p50", "g_margin_p", "g_n_p50", "g_rank_p", "g_sum_p"]
# "sister business" clusters: a house number that differs from the S1 record's but is shared
# by several of its candidates usually marks ANOTHER business (with its own noisy copies)
CLUSTER_COLS = ["n_same_num", "p_same_num", "n_s1_num", "alt_cluster", "alt_mass"]
CE_COLS = ["ce", "ce12"]                        # cross-encoder scores: MiniLM-L6 and MiniLM-L12
DATA = Path(config.CACHE_DIR)


def _p(name):
    """Paths of stage-2 artefacts."""
    return DATA / name


# --------------------------------------------------------------------------- stage 1

def fit_stage1(cols):
    """Train the two cross-fitting stage-1 models on the train split (folds A / B)."""
    tr = pd.read_parquet(feat_path("train"), columns=["id1", "id2", "label"] + cols)
    ids = tr["id1"].unique()
    fold_a = np.sort(np.random.default_rng(config.SEED + 1).permutation(ids)[: len(ids) // 2])
    in_a = tr["id1"].isin(fold_a).to_numpy()
    t0 = time.time()
    m_a = train(tr[in_a], cols, early_stop_share=0.1, params=FAST)
    m_b = train(tr[~in_a], cols, early_stop_share=0.1, params=FAST)
    m_a.save_model(str(_p("stage1_A.txt")))
    m_b.save_model(str(_p("stage1_B.txt")))
    np.save(_p("stage1_fold_a.npy"), fold_a)
    print(f"  stage 1: two fold models trained ({time.time() - t0:.0f}s)", flush=True)
    return m_a, m_b, fold_a


def score_file(name, cols, m_a, m_b, fold_a, batch=2_000_000):
    """Stage-1 probability of every pair of a feature file, in file order.

    Model A scores everything; pairs of fold-A S1 records are re-scored by model B.
    """
    t0 = time.time()
    parts = []
    for b in pq.ParquetFile(feat_path(name)).iter_batches(batch_size=batch, columns=["id1", "id2"] + cols):
        df = b.to_pandas()
        X = df[cols].to_numpy(np.float32)
        p = m_a.predict(X)
        in_a = np.isin(df["id1"].to_numpy(), fold_a)
        if in_a.any():
            p[in_a] = m_b.predict(X[in_a])
        parts.append(pd.DataFrame({"id1": df["id1"].to_numpy(), "id2": df["id2"].to_numpy(),
                                   "p1": p.astype(np.float32)}))
        print(f"    [{name}] scored {sum(len(x) for x in parts):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(_p(f"{name}_p1.parquet"), index=False)
    return out


# --------------------------------------------------------------------------- context from p1

def _group_stats(key, v):
    """Per element: best OTHER value in its key group, rank in group, count > 0.5, group sum."""
    n = len(v)
    order = np.lexsort((-v, key))
    ks, vs = key[order], v[order]
    start = np.ones(n, dtype=bool)
    start[1:] = ks[1:] != ks[:-1]
    gid_sorted = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, n])
    top1 = vs[first]
    top2 = np.where(size > 1, vs[np.minimum(first + 1, n - 1)], 0.0)
    n50 = np.bincount(gid_sorted, weights=(vs > 0.5).astype(np.float64))
    total = np.bincount(gid_sorted, weights=vs.astype(np.float64))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - first[gid_sorted]
    gid = np.empty(n, dtype=np.int64)
    gid[order] = gid_sorted
    best_other = np.where(rank == 0, top2[gid], top1[gid])
    return best_other, rank, n50[gid], total[gid]


def first_numbers(split):
    """First address number of every record of a split, hashed (0 = no number) -> Series by id."""
    from blocking import hstr
    from normalize import norm_path

    parts = []
    for s in (1, 2, 3):
        t = pq.read_table(norm_path(split, s), columns=["id", "addr_numbers"]).to_pandas()
        first = t["addr_numbers"].str.split(" ", n=1).str[0].fillna("")
        h = hstr(first.to_numpy(dtype=object)).astype(np.int64)
        h[first.to_numpy() == ""] = 0
        parts.append(pd.Series(h, index=t["id"].to_numpy()))
    return pd.concat(parts)


def cluster_features(p, nums):
    """Sister-business cluster features for all pairs (id1, id2, p1) of a split."""
    n1 = nums.reindex(p["id1"].to_numpy()).fillna(0).to_numpy(np.int64)
    nc = nums.reindex(p["id2"].to_numpy()).fillna(0).to_numpy(np.int64)
    d = pd.DataFrame({"id1": p["id1"].to_numpy(), "nc": nc, "p": p["p1"].to_numpy(np.float64)})
    g = d.groupby(["id1", "nc"], sort=False)["p"]
    same_n = g.transform("size").to_numpy() - 1                 # other candidates with my number
    same_p = g.transform("sum").to_numpy() - d["p"].to_numpy()  # their probability mass
    has = nc != 0
    same_n, same_p = np.where(has, same_n, 0), np.where(has, same_p, 0.0)
    s1_num = (nc == n1) & has & (n1 != 0)
    n_s1 = d.assign(s=s1_num).groupby("id1", sort=False)["s"].transform("sum").to_numpy()
    alt = has & (n1 != 0) & (nc != n1) & (same_n >= 1)
    return {"n_same_num": same_n, "p_same_num": same_p, "n_s1_num": n_s1,
            "alt_cluster": alt.astype(np.float32), "alt_mass": np.where(alt, same_p, 0.0)}


def context(p, nums=None):
    """Stage-2 context columns for a table (id1, id2, p1) holding ALL pairs of a split.

    With nums (first_numbers of the split) the sister-cluster columns are added too.
    """
    v = p["p1"].to_numpy().astype(np.float64)
    o_best, _, o_n50, _ = _group_stats(p["id2"].to_numpy(), v)
    g_best, g_rank, g_n50, g_sum = _group_stats(p["id1"].to_numpy(), v)
    out = pd.DataFrame({
        "id1": p["id1"].to_numpy(), "id2": p["id2"].to_numpy(), "p1": v.astype(np.float32),
        "o_margin_p": (v - o_best).astype(np.float32), "o_n_p50": o_n50.astype(np.float32),
        "g_margin_p": (v - g_best).astype(np.float32), "g_n_p50": g_n50.astype(np.float32),
        "g_rank_p": g_rank.astype(np.float32), "g_sum_p": g_sum.astype(np.float32),
    })
    if nums is not None:
        for k, col in cluster_features(p, nums).items():
            out[k] = np.asarray(col, dtype=np.float32)
    return out


def extra_set(n, cols, ctx_all):
    """Pairs of n more train S1 records outside the train/val split, for stage-2 training.

    Their stage-1 scores come from fold model A, which never saw them. The 20k records
    the blocking ranker was trained on are excluded (their block_score is in-sample).
    """
    from io_utils import load_source

    split_ids = pd.read_parquet(DATA / "split.parquet", columns=["id1"])["id1"].to_numpy()
    all_s1 = load_source("train", 1, columns=["id"])["id"].to_numpy()
    spare = np.setdiff1d(all_s1, split_ids)
    ranker_ids = np.random.default_rng(config.SEED).choice(spare, 20_000, replace=False)  # as blocking.train_ranker
    pool = np.setdiff1d(spare, ranker_ids)
    pick = np.random.default_rng(config.SEED + 2).choice(pool, n, replace=False)
    f = pd.read_parquet(feat_path("train_all"), columns=["id1", "id2", "label"] + cols,
                        filters=[("id1", "in", pick.tolist())])
    out = f.merge(ctx_all[ctx_all["id1"].isin(pick)], on=["id1", "id2"], how="left")
    if out["p1"].isna().any():
        raise ValueError("extra S1 pairs without stage-1 context")
    print(f"  extra stage-2 training data: {n:,} S1, {len(out):,} pairs", flush=True)
    return out


def add_ce(df, name, col="ce"):
    """Merge a cross-encoder score of each pair (NaN where the pair was not re-scored).

    col "ce" = MiniLM-L6 (<cache>/ce_{name}.parquet), "ce12" = MiniLM-L12 (ce12_{name}.parquet).
    """
    ce = pd.read_parquet(_p(f"{col}_{name}.parquet"), columns=["id1", "id2", col])
    out = df.merge(ce, on=["id1", "id2"], how="left")          # left merge keeps df's row order
    print(f"  [{name}] {col} score for {out[col].notna().mean():.1%} of pairs", flush=True)
    return out


def test_p1(cols=None, m_a=None):
    """Stage-1 scores of every test pair from fold model A, recomputed when older than the model
    or than the test feature file (a new candidate set)."""
    saved = _p("test_p1.parquet")
    if saved.exists() and saved.stat().st_mtime > max(_p("stage1_A.txt").stat().st_mtime,
                                                      feat_path("test").stat().st_mtime):
        return pd.read_parquet(saved)
    if cols is None:
        from model import EXCLUDE
        cols = [c for c in pq.read_schema(feat_path("val")).names if c not in EXCLUDE]
    if m_a is None:
        m_a = lgb.Booster(model_file=str(_p("stage1_A.txt")))
    return score_file("test", cols, m_a, m_a, np.array([], dtype=np.int64))


def decoy_weighted_f05(pairs, truth, threshold, r=1.9):
    """Val macro F0.5 under a test-like decoy prior: a wrong match to a decoy counts r times.

    Test has ~1.9x more decoy S2/S3 records (owned by no S1) per S1 record than train, so
    a decoy false positive is ~r times more likely there than on val.
    """
    from io_utils import load_gt_pairs

    owned = set(load_gt_pairs()["id2"].to_numpy())
    kept = one_owner(pairs)
    kept = kept[kept["p"] >= threshold]
    pred = kept.groupby("id1")["id2"].agg(set).to_dict()
    total = 0.0
    for k, tr in truth.items():
        pr = pred.get(k, set())
        if not tr:
            total += 0.0 if pr else 1.0
            continue
        tp = len(pr & tr)
        if tp == 0:
            continue
        wrong = pr - tr
        fp = sum(r if x not in owned else 1.0 for x in wrong)
        p, rc = tp / (tp + fp), tp / len(tr)
        total += 1.25 * p * rc / (0.25 * p + rc)
    return total / len(truth)


def labelled_set(name, cols, ctx_all):
    """Stage-1 features + stage-2 context for the val or train split."""
    f = pd.read_parquet(feat_path(name), columns=["id1", "id2", "label"] + cols)
    ctx = ctx_all[ctx_all["id1"].isin(load_split(name))]
    out = f.merge(ctx, on=["id1", "id2"], how="left")
    missing = out["p1"].isna().mean()
    if missing > 0:
        raise ValueError(f"{name}: {missing:.2%} of pairs have no stage-1 context")
    return out


# --------------------------------------------------------------------------- run

def main(argv=None):
    """Fit stage 1 (cross-fitted), build context, fit stage 2, report val; optionally predict."""
    ap = argparse.ArgumentParser(description="Second-stage matcher")
    ap.add_argument("--predict", action="store_true", help="write submission files if val improves")
    ap.add_argument("--baseline", type=float, default=0.9588, help="val macro F0.5 to beat")
    ap.add_argument("--reuse-stage1", action="store_true", help="load saved stage-1 models and p1 tables")
    ap.add_argument("--extra-s1", type=int, default=0,
                    help="add this many train S1 records outside the split to stage-2 training")
    ap.add_argument("--clusters", action="store_true", help="add the sister-business cluster features (tested; not used: they hurt on the leaderboard)")
    ap.add_argument("--ce", action="store_true",
                    help="add the cross-encoder score (crossenc.py: ce_{val,train,test}.parquet; NaN outside its band)")
    ap.add_argument("--ce12", action="store_true",
                    help="add the MiniLM-L12 cross-encoder score (ce12_{val,train,test}.parquet)")
    ap.add_argument("--country-threshold", nargs="+", default=None, metavar="COUNTRY=T",
                    help="with --variants: another threshold for these countries, e.g. France=0.97")
    ap.add_argument("--decoy-weight", type=float, default=1.0,
                    help="training weight of false pairs whose S2/S3 record belongs to no S1 (a decoy); "
                         "test has ~1.9x more decoys per S1 than train")
    ap.add_argument("--lr", type=float, default=FAST["learning_rate"], help="stage-2 learning rate")
    ap.add_argument("--tag", default="",
                    help="suffix for this experiment's files, e.g. _clusters; its test output goes to "
                         "output/<tag>/ instead of the main output")
    ap.add_argument("--test-only", action="store_true", help="score test with the saved stage-2 model")
    ap.add_argument("--variants", type=float, nargs="+", default=None,
                    help="write test files at these thresholds from saved test scores")
    args = ap.parse_args(argv)
    if args.test_only:
        test_only(args.tag)
    if args.variants:
        overrides = {c: float(t) for c, t in (x.split("=") for x in args.country_threshold or [])}
        write_variants(args.variants, args.tag, overrides)
    if args.test_only or args.variants:
        return

    from model import EXCLUDE

    t0 = time.time()
    # every feature column of the feature files (incl. bienc_* when union.py candidates are used)
    cols = [c for c in pq.read_schema(feat_path("val")).names if c not in EXCLUDE]
    print(f"stage 1: {len(cols)} features", flush=True)
    if args.reuse_stage1:
        m_a, m_b = lgb.Booster(model_file=str(_p("stage1_A.txt"))), lgb.Booster(model_file=str(_p("stage1_B.txt")))
        fold_a = np.load(_p("stage1_fold_a.npy"))
        p_train_all = pd.read_parquet(_p("train_all_p1.parquet"))
    else:
        m_a, m_b, fold_a = fit_stage1(cols)
        p_train_all = score_file("train_all", cols, m_a, m_b, fold_a)
    ctx_all = context(p_train_all, first_numbers("train") if args.clusters else None)
    del p_train_all

    va, tr = labelled_set("val", cols, ctx_all), labelled_set("train", cols, ctx_all)
    if args.extra_s1:
        tr = pd.concat([tr, extra_set(args.extra_s1, cols, ctx_all)], ignore_index=True)
    del ctx_all
    ce_cols = [c for c, on in zip(CE_COLS, (args.ce, args.ce12)) if on]
    for c in ce_cols:
        va, tr = add_ce(va, "val", c), add_ce(tr, "train", c)
    truth = val_truth()

    # stage 1 alone on val (sanity: should be close to model.py's 0.9588)
    t1, best1, _ = tune(va[["id1", "id2"]].assign(p=va["p1"]), truth)
    print(f"  stage 1 (fold model A) on val: macro F0.5 {best1['macro_f05']:.4f} (threshold {t1:.3f})")

    cols2 = cols + S2_COLS + (CLUSTER_COLS if args.clusters else []) + ce_cols
    if args.decoy_weight != 1.0:
        from io_utils import load_gt_pairs
        owned = load_gt_pairs()["id2"].unique()
        decoy = (tr["label"].to_numpy() == 0) & ~tr["id2"].isin(owned).to_numpy()
        tr["weight"] = np.where(decoy, args.decoy_weight, 1.0).astype(np.float32)
        print(f"  decoy false pairs weighted x{args.decoy_weight}: {decoy.sum():,} of {len(tr):,} training pairs",
              flush=True)
    model2 = train(tr, cols2, early_stop_share=0.1, params=dict(FAST, learning_rate=args.lr))
    pairs = va[["id1", "id2"]].assign(p=model2.predict(va[cols2].to_numpy(np.float32)))
    pairs.to_parquet(_p(f"val_p2{args.tag}.parquet"), index=False)
    t2, best2, table = tune(pairs, truth)
    print(f"  STAGE 2 on val: macro F0.5 {best2['macro_f05']:.4f}  (P {best2['precision']:.3f} "
          f"R {best2['recall']:.3f}, threshold {t2:.3f})   baseline {args.baseline:.4f}  [{time.time() - t0:.0f}s]")
    tw = {t: decoy_weighted_f05(pairs, truth, t) for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)}
    print("  test-like val (decoy false matches x1.9): "
          + "  ".join(f"t{t}={s:.4f}" for t, s in tw.items()), flush=True)
    print(table.iloc[::2][["threshold", "macro_f05", "precision", "recall"]].to_string(index=False))
    imp = pd.Series(model2.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    print("  top features: " + ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(10).items()))
    model2.save_model(str(_p(f"stage2{args.tag}_model.txt")))
    _p(f"stage2{args.tag}.json").write_text(json.dumps({"stage1_features": cols, "features": cols2, "threshold": t2,
                                             "val_macro_f05": best2["macro_f05"]}, indent=2))

    if not args.predict:
        return
    if best2["macro_f05"] <= args.baseline + 0.001:
        print("  stage 2 does not beat the baseline by > 0.001: NOT writing submission files")
        return
    del tr, va
    predict_test(cols, cols2, m_a, model2, t2, best2["macro_f05"], tag=args.tag)


def predict_test(cols, cols2, m_a, model2, threshold, val_score, batch=2_000_000, tag=""):
    """Stage 1 + context + stage 2 on test, then both TSVs, validator and archive."""
    import shutil
    from datetime import datetime
    from io_utils import load_source
    from predict import write_candidates

    p_test = test_p1(cols, m_a)
    ctx = context(p_test, first_numbers("test") if CLUSTER_COLS[0] in cols2 else None)
    for c in CE_COLS:
        if c in cols2:
            ctx = add_ce(ctx, "test", c)                         # row order (= feature file order) kept
    del p_test
    t0, parts, start = time.time(), [], 0
    for b in pq.ParquetFile(feat_path("test")).iter_batches(batch_size=batch, columns=["id1", "id2"] + cols):
        df = b.to_pandas()
        c = ctx.iloc[start:start + len(df)]
        if not (np.array_equal(c["id1"].to_numpy(), df["id1"].to_numpy())
                and np.array_equal(c["id2"].to_numpy(), df["id2"].to_numpy())):
            raise ValueError("test context is not aligned with the feature file")
        X = np.hstack([df[cols].to_numpy(np.float32), c[cols2[len(cols):]].to_numpy(np.float32)])
        parts.append(pd.DataFrame({"id1": df["id1"].to_numpy(), "id2": df["id2"].to_numpy(),
                                   "p": model2.predict(X).astype(np.float32)}))
        start += len(df)
        print(f"    [test stage 2] {start:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    pairs = pd.concat(parts, ignore_index=True)
    pairs.to_parquet(_p(f"test_p2{tag}.parquet"), index=False)   # reused by write_variants()
    out = Path(config.OUTPUT_DIR) / tag.strip("_") if tag else Path(config.OUTPUT_DIR)
    match_file, cand_file = out / "matching_results.tsv", Path(config.OUTPUT_DIR) / "candidate_pairs.tsv"
    s1_ids = load_source("test", 1, columns=["id"])["id"].to_numpy()
    from blocking import cand_path
    if not cand_file.exists() or cand_file.stat().st_mtime < cand_path("test").stat().st_mtime:
        write_candidates(cand_file, s1_ids)                     # the candidate set changed: rewrite
    ok = _write_matches(pairs, threshold, match_file, cand_file, s1_ids)
    arch = Path(config.WORK_DIR) / "submissions" / f"{datetime.now():%m%d_%H%M}_stage2{tag}_val{val_score:.4f}_t{threshold:.3f}"
    arch.mkdir(parents=True, exist_ok=True)
    shutil.copy(match_file, arch / match_file.name)
    shutil.copy(_p(f"stage2{tag}.json"), arch / f"stage2{tag}.json")
    print(f"  {'PASS' if ok else 'VALIDATION FAILED'}; copy saved in {arch}")


def _write_matches(pairs, threshold, match_file, cand_file, s1_ids, overrides=None):
    """One owner + threshold -> matching_results.tsv at match_file; returns validator PASS.

    overrides: {country: threshold} for S1 records of those countries (e.g. a stricter one
    for a country absent from training, whose probabilities are least calibrated).
    """
    from io_utils import load_source
    from predict import HEADER_MATCH, validate, write_id_lists

    kept = one_owner(pairs)
    thr = np.full(len(kept), threshold, dtype=np.float64)
    if overrides:
        s1 = load_source("test", 1, columns=["id", "country"])
        country = kept["id1"].map(pd.Series(s1["country"].astype(str).to_numpy(), index=s1["id"].to_numpy()))
        for c, t in overrides.items():
            thr[(country == c).to_numpy()] = t
    kept = kept[kept["p"].to_numpy() >= thr]
    extra = "".join(f", {c} {t:.3f}" for c, t in (overrides or {}).items())
    print(f"  threshold {threshold:.3f}{extra}: {len(kept):,} matches for {kept['id1'].nunique():,} of {len(s1_ids):,} S1")
    Path(match_file).parent.mkdir(parents=True, exist_ok=True)
    write_id_lists(match_file, HEADER_MATCH, s1_ids, kept)
    return validate(match_file, cand_file)


def write_variants(thresholds, tag="", overrides=None):
    """Write test files for other thresholds from the saved stage-2 test scores (no rescoring).

    Files go to <output>/<tag>_threshold_<t>[_<country><t>]/matching_results.tsv; the main
    output is untouched.
    """
    from io_utils import load_source

    pairs = pd.read_parquet(_p(f"test_p2{tag}.parquet"))
    s1_ids = load_source("test", 1, columns=["id"])["id"].to_numpy()
    cand_file = Path(config.OUTPUT_DIR) / "candidate_pairs.tsv"
    suffix = "".join(f"_{c}{t:g}" for c, t in (overrides or {}).items())
    for t in thresholds:
        f = (Path(config.OUTPUT_DIR) / f"{tag.strip('_') + '_' if tag else ''}threshold_{t:.2f}{suffix}"
             / "matching_results.tsv")
        ok = _write_matches(pairs, t, f, cand_file, s1_ids, overrides)
        print(f"  -> {f}  {'PASS' if ok else 'VALIDATION FAILED'}", flush=True)


def test_only(tag=""):
    """Re-run stage 2 on test with the SAVED models (stage2_model.txt), saving test_p2.parquet."""
    meta = json.loads(_p(f"stage2{tag}.json").read_text())
    m_a = lgb.Booster(model_file=str(_p("stage1_A.txt")))
    model2 = lgb.Booster(model_file=str(_p(f"stage2{tag}_model.txt")))
    predict_test(meta["stage1_features"], meta["features"], m_a, model2, meta["threshold"],
                 meta["val_macro_f05"], tag=tag)


if __name__ == "__main__":
    sys.exit(main())
