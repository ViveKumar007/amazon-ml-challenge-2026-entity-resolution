"""Macro F0.5, computed the way the leaderboard describes it.

For each Source 1 entity, F0.5 of its predicted match set against its true set:

    F0.5 = 1.25 * P * R / (0.25 * P + R)

Singletons (no true matches) score 1.0 for an empty prediction and 0.0 otherwise.
An entity with true matches but no correct prediction scores 0.0. The final score
is the plain average over every Source 1 entity being evaluated.

Run:  python src/metric.py      (self-test, plus baselines on val once split.py has run)
"""
import sys

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5, so beta^2 = 0.25
EMPTY = frozenset()


def f05(pred, true):
    """F0.5 of one Source 1 entity: predicted id set vs true id set."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred, truth):
    """Average f05 over every Source 1 id in truth.

    pred and truth map id1 -> set of matched ids. An id1 missing from pred counts as
    an empty prediction, exactly like an empty matched_entity_ids cell.
    """
    return float(np.mean([f05(pred.get(k, EMPTY), t) for k, t in truth.items()]))


def sets_from_pairs(pairs, id1s=None, col1="id1", col2="id2"):
    """Turn an (id1, id2) pair table into {id1: set of id2}.

    If id1s is given, the result has exactly those keys, with an empty set for ids
    that have no pairs (use this to build the truth dict so singletons are included).
    """
    out = pairs.groupby(col1)[col2].agg(set).to_dict() if len(pairs) else {}
    if id1s is not None:
        out = {k: out.get(k, set()) for k in id1s}
    return out


def evaluate(pred, truth):
    """Macro F0.5 plus a breakdown that helps error analysis.

    precision/recall are pooled over all pairs (micro); singleton_f05 is the share of
    singletons predicted empty; empty_on_matched is how often an entity with true
    matches got an empty prediction (each one costs a full 1.0 on that entity).
    """
    rows = []
    for k, t in truth.items():
        p = pred.get(k, EMPTY)
        rows.append((len(p), len(t), len(p & t), f05(p, t)))
    df = pd.DataFrame(rows, columns=["n_pred", "n_true", "tp", "f05"])
    single = df["n_true"] == 0
    return {
        "macro_f05": df["f05"].mean(),
        "precision": df["tp"].sum() / max(df["n_pred"].sum(), 1),
        "recall": df["tp"].sum() / max(df["n_true"].sum(), 1),
        "singleton_f05": df.loc[single, "f05"].mean() if single.any() else float("nan"),
        "matched_f05": df.loc[~single, "f05"].mean() if (~single).any() else float("nan"),
        "empty_on_matched": (df.loc[~single, "n_pred"] == 0).mean() if (~single).any() else float("nan"),
        "n_entities": len(df),
    }


def _self_test():
    """Check f05 / macro_f05 against the worked example and rules in the problem statement."""
    # Worked example from the PDF: 3 predicted, 2 correct, 2 true -> 0.714
    assert round(f05({"a", "b", "c"}, {"a", "c"}), 3) == 0.714
    assert f05(set(), set()) == 1.0          # singleton, predicted empty
    assert f05({"a"}, set()) == 0.0          # singleton, false merge
    assert f05(set(), {"a"}) == 0.0          # missed everything
    assert f05({"b"}, {"a"}) == 0.0          # only wrong ids
    assert f05({"a", "b"}, {"a", "b"}) == 1.0
    assert round(f05({"a"}, {"a", "b", "c", "d"}), 3) == 0.625   # 1 of 4, no errors
    truth = {"s1": {"a"}, "s2": set()}
    assert macro_f05({}, truth) == 0.5       # missing key = empty prediction
    assert macro_f05({"s1": {"a"}}, truth) == 1.0
    print("metric self-test passed (PDF example = 0.714)")


def _baselines():
    """Score trivial predictions on the val split to show what the metric rewards."""
    from io_utils import load_gt_pairs
    from split import load_split

    val = load_split("val")
    gt = load_gt_pairs()
    truth = sets_from_pairs(gt[gt["id1"].isin(val)], id1s=val)
    print(f"\nBaselines on val ({len(truth):,} S1 entities):")
    print(f"  perfect prediction  : {macro_f05(truth, truth):.4f}")
    print(f"  predict nothing     : {macro_f05({}, truth):.4f}   (= share of singletons)")
    one = {k: {min(t)} for k, t in truth.items() if t}   # exactly one correct id each
    print(f"  one correct id each : {macro_f05(one, truth):.4f}   (precision 1, low recall)")


if __name__ == "__main__":
    _self_test()
    try:
        _baselines()
    except FileNotFoundError:
        print("(run split.py first to see baselines on val)")
    sys.exit(0)
