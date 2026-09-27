"""Matching model (step 6): LightGBM on pair features + one-owner assignment + threshold.

Its training, one-owner and threshold functions are used by stage2.py (the final matcher);
running this file on its own gives the stage-1-only model of the first submissions.

Pipeline for a set of candidate pairs:
    1. p = model probability that the pair is a true match
    2. one owner: each S2/S3 record is kept only for the S1 record with the highest p
       (in the data every S2/S3 record matches at most one S1 record)
    3. keep pairs with p >= threshold; the threshold maximises macro F0.5 on val

Training data: train split pairs (train_features.parquet). Evaluation: val split,
scored with metric.py exactly like the leaderboard, over ALL val S1 records
(including singletons and records that got no candidates).

Competition features (o_n_claims, o_margin) compare the S1 records claiming the same
S2/S3 record. They are only valid when every train S1 record was blocked
(train_all_candidates.parquet, as on test). Without that file they are left out: with
only ~11% of S1 records present the model leaned 52% on o_margin for no val gain
(first submission, 0.9352 val / 0.926 leaderboard, was trained without them).

Output: <cache>/match_model.txt (LightGBM) and <cache>/match_model.json (features, threshold).

Run:  python src/model.py                 train on the train split, tune on val, save
      python src/model.py --compare       also report data-size / feature-set checks
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from features import load_features
from io_utils import load_gt_pairs
from metric import evaluate, sets_from_pairs
from split import load_split

EXCLUDE = {"id1", "id2", "label", "key_mask"}
# without full context (no train_all candidates) the o_* competition features differ between
# train/val and test, so they are left out; with it they are used (see features.py)
EXCLUDE_PARTIAL_CONTEXT = {"o_n_claims", "o_margin"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=config.SEED, verbose=-1, num_threads=config.N_WORKERS)
N_ROUNDS = 600            # used when no early-stopping set is given
MAX_ROUNDS = 4000         # upper bound with early stopping
THRESHOLDS = np.round(np.arange(0.20, 0.96, 0.025), 3)


def model_path(cache_dir=config.CACHE_DIR):
    """Saved LightGBM matching model."""
    return Path(cache_dir) / "match_model.txt"


def meta_path(cache_dir=config.CACHE_DIR):
    """Feature list and chosen threshold of the saved model."""
    return Path(cache_dir) / "match_model.json"


def feature_columns(df, drop=()):
    """Model inputs: every feature column except ids, label and the excluded ones."""
    return [c for c in df.columns if c not in EXCLUDE and c not in drop]


def train(df, cols, rounds=N_ROUNDS, early_stop_share=0.0, params=PARAMS):
    """Fit LightGBM on labelled pairs.

    With early_stop_share > 0, that share of the S1 records (whole groups, never val) is
    held out and training stops when its log-loss has not improved for 100 rounds.
    An optional "weight" column gives per-pair training weights.
    """
    w = df["weight"].to_numpy(np.float32) if "weight" in df else None
    if early_stop_share <= 0:
        ds = lgb.Dataset(df[cols].to_numpy(np.float32), label=df["label"].to_numpy(), weight=w, feature_name=cols)
        return lgb.train(params, ds, num_boost_round=rounds)
    ids = df["id1"].unique()
    held = np.random.default_rng(config.SEED).choice(ids, int(len(ids) * early_stop_share), replace=False)
    es = df["id1"].isin(held).to_numpy()
    dtr = lgb.Dataset(df.loc[~es, cols].to_numpy(np.float32), label=df.loc[~es, "label"].to_numpy(),
                      weight=None if w is None else w[~es], feature_name=cols)
    dva = lgb.Dataset(df.loc[es, cols].to_numpy(np.float32), label=df.loc[es, "label"].to_numpy(),
                      weight=None if w is None else w[es], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"    early stopping: {model.best_iteration} rounds", flush=True)
    return model


def one_owner(pairs):
    """Keep, for every S2/S3 record, only its highest-probability S1 record."""
    best = pairs.groupby("id2")["p"].transform("max")
    return pairs[pairs["p"] >= best]


def predict_sets(pairs, threshold, assign=True):
    """Final matches {id1: set(id2)} from scored pairs (columns id1, id2, p)."""
    kept = one_owner(pairs) if assign else pairs
    kept = kept[kept["p"] >= threshold]
    return sets_from_pairs(kept)


def tune(pairs, truth, assign=True, thresholds=THRESHOLDS):
    """Macro F0.5 for each threshold -> (best threshold, best result dict, table)."""
    base = one_owner(pairs) if assign else pairs
    rows = []
    for t in thresholds:
        r = evaluate(sets_from_pairs(base[base["p"] >= t]), truth)
        rows.append({"threshold": t, **r})
    table = pd.DataFrame(rows)
    best = table.loc[table["macro_f05"].idxmax()]
    return float(best["threshold"]), best.to_dict(), table


def val_truth():
    """True match sets of every val S1 record (empty set for singletons)."""
    val = load_split("val")
    gt = load_gt_pairs()
    return sets_from_pairs(gt[gt["id1"].isin(val)], id1s=val)


def fit_and_score(tr, va, truth, cols, label, early_stop_share=0.0):
    """Train on tr, tune the threshold on va, print and return (model, threshold, best)."""
    t0 = time.time()
    model = train(tr, cols, early_stop_share=early_stop_share)
    pairs = va[["id1", "id2"]].assign(p=model.predict(va[cols].to_numpy(np.float32)))
    t, best, _ = tune(pairs, truth)
    print(f"  {label:<38} macro F0.5 {best['macro_f05']:.4f}  (P {best['precision']:.3f} "
          f"R {best['recall']:.3f}, threshold {t:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    return model, t, best, pairs


def main(argv=None):
    """Train, tune and save the matching model; optionally run comparison checks."""
    ap = argparse.ArgumentParser(description="Train the matching model")
    ap.add_argument("--compare", action="store_true", help="also run data-size / feature-set checks")
    ap.add_argument("--drop-competition", action="store_true",
                    help="leave out o_* features (S2/S3 competition), whose train/test distributions differ")
    args = ap.parse_args(argv)

    from blocking import cand_path

    t0 = time.time()
    tr, va = load_features("train"), load_features("val")
    truth = val_truth()
    full_context = cand_path("train_all").exists()
    cols = feature_columns(tr, drop=() if full_context else EXCLUDE_PARTIAL_CONTEXT)
    if args.drop_competition:
        cols = [c for c in cols if not c.startswith("o_")]
    print(f"train {len(tr):,} pairs ({tr['label'].mean():.2%} true), val {len(va):,} pairs; "
          f"{len(cols)} features; full competition context: {full_context}; val S1 {len(truth):,} "
          f"({time.time() - t0:.0f}s)")
    upper = evaluate({k: v for k, v in sets_from_pairs(va[va["label"] == 1]).items()}, truth)
    print(f"  ceiling if the model were perfect on these candidates: macro F0.5 {upper['macro_f05']:.4f}")

    if args.compare:
        n_s1 = tr["id1"].nunique()
        sib = [c for c in cols if c.startswith("sib_")]
        ctx = [c for c in cols if c.startswith("o_")]
        fit_and_score(tr, va, truth, [c for c in cols if c not in sib], f"{n_s1 // 1000}k S1, no sibling feats")
        if ctx:
            fit_and_score(tr, va, truth, [c for c in cols if c not in ctx], f"{n_s1 // 1000}k S1, no competition feats")

    model, threshold, best, pairs = fit_and_score(tr, va, truth, cols, "all features, early stopping (final)",
                                                  early_stop_share=0.1)
    _, _, table = tune(pairs, truth)
    print("\n  threshold  macro_F0.5  precision  recall  singleton_F0.5  empty_on_matched")
    for _, r in table.iloc[::2].iterrows():
        print(f"  {r['threshold']:>9.3f}  {r['macro_f05']:>10.4f}  {r['precision']:>9.3f}  {r['recall']:>6.3f}"
              f"  {r['singleton_f05']:>14.3f}  {r['empty_on_matched']:>16.3f}")
    t_na, best_na, _ = tune(pairs, truth, assign=False)
    print(f"\n  without one-owner assignment: macro F0.5 {best_na['macro_f05']:.4f} (threshold {t_na:.3f})")

    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\n  top features by gain: " + ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(12).items()))

    model.save_model(str(model_path()))
    meta_path().write_text(json.dumps({"features": cols, "threshold": threshold,
                                       "val_macro_f05": best["macro_f05"]}, indent=2))
    print(f"\nSaved {model_path().name} + {meta_path().name} (threshold {threshold:.3f}, "
          f"val macro F0.5 {best['macro_f05']:.4f})")


if __name__ == "__main__":
    sys.exit(main())
