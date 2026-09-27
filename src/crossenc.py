"""Cross-encoder re-scoring of the uncertain candidate pairs.

A small transformer reads BOTH records' text together ("name ; address" vs "name ;
address") and scores how likely they are the same business. It sees clues the hand-made
similarity features cannot, e.g. "vallee patrimoine ... 126" vs "vallee primaire ... 105"
(a sister business) against "appare1s" vs "apparels" (a noisy copy).

Only pairs the stage-1 model is unsure about (p1 inside BAND) are scored: about 3-5 % of
all pairs, which is where nearly all errors are, and cheap enough for a CPU.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters, Apache-2.0), fine-tuned
here for match / no match.

Leakage control: the cross-encoder is trained on uncertain pairs of train S1 records
OUTSIDE the train/val split (and outside the blocking ranker's 20k), so its scores on
train-split, val and test pairs all come from a model that never saw them.

Output: <cache>/ce_{train,val,test}.parquet with id1, id2, ce (probability).

Run (settings used for the final submission):
      GPU (Colab):  python src/crossenc.py --export --band 0.005 0.999 --n-train 600000 --min-score 0.001
                    -> run colab/colab_crossencoder.py on data/colab/ce_data_v2.parquet
                    -> python src/crossenc.py --import-scores   (reads data/colab/ce_scores_v2.parquet)
      After union.py:  python src/crossenc.py --export-missing --band 0.005 0.999
                    -> run colab/colab_ce_score_v5.py (L12 on all pairs, L6 on new pairs)
                    -> python src/crossenc.py --import-v5       (reads data/colab/ce_scores_v5.parquet)
      CPU only:     python src/crossenc.py --band 0.005 0.999 --n-train 600000 --min-score 0.001
                    (same model and recipe; many hours on a CPU)
      python src/crossenc.py --bench          speed check only
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from io_utils import load_gt_pairs
from normalize import norm_path

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BAND = (0.03, 0.97)          # stage-1 probability range treated as "uncertain"
MAX_LEN = 96
N_TRAIN_PAIRS = 160_000
DATA = Path(config.CACHE_DIR)
CE_DIR = DATA / "ce_model"


def _texts(split, source_ids):
    """'name ; address' text of the given records of a split -> Series by id."""
    parts = []
    for s in (1, 2, 3):
        ids = [int(i) for i in source_ids if int(i) // 10**10 == s]
        if not ids:
            continue
        t = pq.read_table(norm_path(split, s), columns=["id", "name_clean", "name_alt", "addr_clean"],
                          filters=[("id", "in", ids)]).to_pandas()
        alt = np.where(t["name_alt"] != "", " / " + t["name_alt"], "")
        parts.append(pd.Series((t["name_clean"] + alt + " ; " + t["addr_clean"]).to_numpy(), index=t["id"].to_numpy()))
    return pd.concat(parts)


def uncertain(p1_file, s1_filter=None):
    """Pairs with stage-1 probability inside BAND (optionally only some S1 records)."""
    p = pd.read_parquet(DATA / p1_file)
    if s1_filter is not None:
        p = p[p["id1"].isin(s1_filter)]
    return p[(p["p1"] >= BAND[0]) & (p["p1"] <= BAND[1])].reset_index(drop=True)


def _batches(tok, a, b, bs):
    """Tokenised batches of text pairs."""
    for i in range(0, len(a), bs):
        yield i, tok(list(a[i:i + bs]), list(b[i:i + bs]), truncation=True, max_length=MAX_LEN,
                     padding=True, return_tensors="pt")


def train_ce(pairs, text, epochs=1, bs=32, lr=3e-5):
    """Fine-tune the cross-encoder on labelled pairs (columns id1, id2, label)."""
    torch.manual_seed(config.SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    model.train()
    pairs = pairs.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    a, b = text.reindex(pairs["id1"]).to_numpy(), text.reindex(pairs["id2"]).to_numpy()
    y = torch.tensor(pairs["label"].to_numpy(np.float32))
    steps = epochs * ((len(pairs) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0, step = time.time(), 0
    for _ in range(epochs):
        for i, enc in _batches(tok, a, b, bs):
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, y[i:i + len(logits)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 500 == 0:
                print(f"    step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)
    return tok, model


@torch.inference_mode()
def score(tok, model, pairs, text, bs=256):
    """Cross-encoder probability for each (id1, id2) pair."""
    model.eval()
    a, b = text.reindex(pairs["id1"]).to_numpy(), text.reindex(pairs["id2"]).to_numpy()
    out, t0 = np.empty(len(pairs), dtype=np.float32), time.time()
    for i, enc in _batches(tok, a, b, bs):
        out[i:i + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1)).numpy()
        if (i // bs) % 400 == 0:
            print(f"    scored {i + bs:,}/{len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _ce_training_pairs(n):
    """Uncertain pairs of train S1 records outside the split and the ranker's 20k, with labels."""
    from io_utils import load_source

    split_ids = pd.read_parquet(DATA / "split.parquet", columns=["id1"])["id1"].to_numpy()
    all_s1 = load_source("train", 1, columns=["id"])["id"].to_numpy()
    spare = np.setdiff1d(all_s1, split_ids)
    ranker_ids = np.random.default_rng(config.SEED).choice(spare, 20_000, replace=False)
    pool = np.setdiff1d(spare, ranker_ids)
    p = uncertain("train_all_p1.parquet", pool)
    p = p.sample(min(n, len(p)), random_state=config.SEED)
    gt = load_gt_pairs().assign(label=1)
    p = p.merge(gt, on=["id1", "id2"], how="left").fillna({"label": 0})
    print(f"  cross-encoder training pairs: {len(p):,} ({p['label'].mean():.1%} true)", flush=True)
    return p


def _min_score_filter(p, cand_name, min_score):
    """Keep only pairs whose blocking-ranker score is >= min_score (the final candidate set)."""
    from blocking import cand_path

    if min_score <= 0:
        return p
    c = pd.read_parquet(cand_path(cand_name), columns=["id1", "id2", "block_score"])
    return p.merge(c[c["block_score"] >= min_score][["id1", "id2"]], on=["id1", "id2"])


def export_for_colab(n_train=400_000, min_score=0.0):
    """Write every pair the cross-encoder needs, with text, to one parquet for a GPU notebook.

    Rows: set = ce_train (fine-tuning pairs, labelled) / val / train / test (to score).
    With min_score > 0 only pairs in the reduced candidate set are exported.
    """
    from split import load_split

    t0 = time.time()
    gt = load_gt_pairs().assign(label=1)
    parts = [_min_score_filter(_ce_training_pairs(n_train), "train_all", min_score).assign(set="ce_train")]
    for name, p1_file, ids, cand in (("val", "train_all_p1.parquet", load_split("val"), "train_all"),
                                     ("train", "train_all_p1.parquet", load_split("train"), "train_all"),
                                     ("test", "test_p1.parquet", None, "test")):
        p = _min_score_filter(uncertain(p1_file, ids), cand, min_score).assign(set=name)
        if name != "test":
            p = p.merge(gt, on=["id1", "id2"], how="left").fillna({"label": 0})
        parts.append(p)
        print(f"  {name}: {len(p):,} uncertain pairs", flush=True)
    out = []
    for split, sets in (("train", ("ce_train", "val", "train")), ("test", ("test",))):
        d = pd.concat([p for p in parts if p["set"].iat[0] in sets], ignore_index=True)
        text = _texts(split, np.unique(np.r_[d["id1"], d["id2"]]))
        d["text_a"], d["text_b"] = text.reindex(d["id1"]).to_numpy(), text.reindex(d["id2"]).to_numpy()
        out.append(d)
    d = pd.concat(out, ignore_index=True)[["set", "id1", "id2", "label", "p1", "text_a", "text_b"]]
    d["label"] = d["label"].fillna(-1).astype(np.int8)
    path = DATA / "colab" / "ce_data_v2.parquet"
    path.parent.mkdir(exist_ok=True)
    d.to_parquet(path, index=False, compression="zstd")
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB, {len(d):,} rows) in {time.time() - t0:.0f}s")


def export_missing(min_score=0.0):
    """Pair list (ids only) for GPU scoring after the candidate set changed (union.py).

    Every pair inside BAND of the current p1 tables (val, train, test) is written, so the
    stronger MiniLM-L12 model scores all of them; need_l6 marks the pairs that have no
    MiniLM-L6 score yet (only those are re-scored by the L6 model, so its scores stay
    consistent). The notebook (colab/colab_ce_score_v5.py) joins the record texts from
    the bi-encoder text files already in Drive, so the upload stays small.
    """
    from split import load_split
    from stage2 import test_p1

    parts = []
    for name, p, ids in (("val", pd.read_parquet(DATA / "train_all_p1.parquet"), load_split("val")),
                         ("train", pd.read_parquet(DATA / "train_all_p1.parquet"), load_split("train")),
                         ("test", test_p1(), None)):
        p = p if ids is None else p[p["id1"].isin(ids)]
        p = p[(p["p1"] >= BAND[0]) & (p["p1"] <= BAND[1])][["id1", "id2"]]
        old = pd.read_parquet(DATA / f"ce_{name}.parquet", columns=["id1", "id2"])
        p = p.merge(old.assign(had=1), on=["id1", "id2"], how="left")
        p = p.assign(set=name, need_l6=p["had"].isna()).drop(columns="had")
        parts.append(p)
        print(f"  {name}: {len(p):,} pairs in band, {p['need_l6'].sum():,} without an L6 score", flush=True)
    d = pd.concat(parts, ignore_index=True)[["set", "id1", "id2", "need_l6"]]
    d["set"] = d["set"].astype("category")
    path = DATA / "colab" / "ce_pairs_v5.parquet"
    d.sort_values(["set", "id1", "id2"]).to_parquet(path, index=False, compression="zstd")
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB, {len(d):,} pairs)")


def import_v5():
    """Read the notebook's ce_scores_v5.parquet (set, id1, id2, ce12, ce6):
    ce12 -> ce12_{val,train,test}.parquet; new L6 scores are appended to ce_{val,train,test}."""
    s = pd.read_parquet(DATA / "colab" / "ce_scores_v5.parquet")
    for name in ("val", "train", "test"):
        part = s[s["set"] == name]
        part[["id1", "id2", "ce12"]].to_parquet(DATA / f"ce12_{name}.parquet", index=False)
        old = pd.read_parquet(DATA / f"ce_{name}.parquet")
        new = part.loc[part["ce6"].notna(), ["id1", "id2", "ce6"]].rename(columns={"ce6": "ce"})
        both = pd.concat([old, new.astype({"ce": old["ce"].dtype})], ignore_index=True)
        both = both.drop_duplicates(["id1", "id2"], keep="first")
        both.to_parquet(DATA / f"ce_{name}.parquet", index=False)
        print(f"  {name}: L12 {len(part):,} pairs; L6 {len(old):,} + {len(new):,} -> {len(both):,}")


def export_hard(n_pairs=1_200_000, min_p1=0.002, hard_p1=0.01):
    """Training data for a stronger cross-encoder (colab/colab_ce_train_l12.py).

    Pairs of train S1 records OUTSIDE the train/val split (and the rankers' 20k) with stage-1
    probability >= min_p1: nearly all true matches plus the negatives the models find
    confusable. Negatives with p1 >= hard_p1 (sister businesses and other look-alikes, the
    errors that cost most on test) get weight 2. Val pairs in BAND are added for evaluation.
    """
    from io_utils import load_source
    from split import load_split

    split_ids = pd.read_parquet(DATA / "split.parquet", columns=["id1"])["id1"].to_numpy()
    all_s1 = load_source("train", 1, columns=["id"])["id"].to_numpy()
    spare = np.setdiff1d(all_s1, split_ids)
    ranker_ids = np.random.default_rng(config.SEED).choice(spare, 20_000, replace=False)
    pool = np.setdiff1d(spare, ranker_ids)
    p = pd.read_parquet(DATA / "train_all_p1.parquet")
    gt = load_gt_pairs().assign(label=1)
    tr = p[p["id1"].isin(pool) & (p["p1"] >= min_p1)].merge(gt, on=["id1", "id2"], how="left").fillna({"label": 0})
    tr = tr.sample(min(n_pairs, len(tr)), random_state=config.SEED)
    tr["weight"] = np.where((tr["label"] == 0) & (tr["p1"] >= hard_p1), 2.0, 1.0)
    va = uncertain("train_all_p1.parquet", load_split("val")).merge(gt, on=["id1", "id2"], how="left").fillna({"label": 0})
    va["weight"] = 1.0
    d = pd.concat([tr.assign(set="train"), va.assign(set="val")], ignore_index=True)
    text = _texts("train", np.unique(np.r_[d["id1"], d["id2"]]))
    d["text_a"], d["text_b"] = text.reindex(d["id1"]).to_numpy(), text.reindex(d["id2"]).to_numpy()
    d["label"] = d["label"].astype(np.int8)
    path = DATA / "colab" / "ce_train_v4.parquet"
    d[["set", "id1", "id2", "label", "weight", "p1", "text_a", "text_b"]].to_parquet(path, index=False, compression="zstd")
    t = d[d["set"] == "train"]
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB): train {len(t):,} pairs ({t['label'].mean():.1%} true, "
          f"{(t['weight'] > 1).sum():,} hard negatives x2), val {(d['set'] == 'val').sum():,}")


def import_from_colab():
    """Split the notebook's ce_scores_v2.parquet into ce_{val,train,test}.parquet."""
    f = DATA / "colab" / "ce_scores_v2.parquet"
    s = pd.read_parquet(f if f.exists() else DATA / "colab" / "ce_scores.parquet")
    for name in ("val", "train", "test"):
        s.loc[s["set"] == name, ["id1", "id2", "ce"]].to_parquet(DATA / f"ce_{name}.parquet", index=False)
        print(f"  ce_{name}.parquet: {(s['set'] == name).sum():,} pairs")


def main(argv=None):
    """Train the cross-encoder and score the uncertain pairs of train, val and test."""
    from split import load_split

    ap = argparse.ArgumentParser(description="Cross-encoder re-scoring of uncertain pairs")
    ap.add_argument("--bench", action="store_true", help="only measure training / scoring speed")
    ap.add_argument("--reuse-model", action="store_true", help="load the saved fine-tuned model")
    ap.add_argument("--export", action="store_true", help="write data/colab/ce_data_v2.parquet for a GPU notebook")
    ap.add_argument("--import-scores", action="store_true", help="read data/colab/ce_scores_v2.parquet from the notebook")
    ap.add_argument("--band", type=float, nargs=2, default=None, help="uncertain p1 band, e.g. 0.005 0.999")
    ap.add_argument("--n-train", type=int, default=400_000, help="cross-encoder fine-tuning pairs")
    ap.add_argument("--min-score", type=float, default=0.0, help="only pairs with ranker score >= this")
    ap.add_argument("--export-missing", action="store_true",
                    help="pair ids in BAND (L12: all, L6: need_l6) -> data/colab/ce_pairs_v5.parquet")
    ap.add_argument("--import-v5", action="store_true", help="read data/colab/ce_scores_v5.parquet (L12 + L6)")
    ap.add_argument("--export-hard", action="store_true", help="training data for the stronger L12 cross-encoder")
    args = ap.parse_args(argv)
    if args.export_hard:
        if args.band:
            globals()["BAND"] = tuple(args.band)
        return export_hard()
    if args.band:
        global BAND
        BAND = tuple(args.band)
    if args.export:
        return export_for_colab(args.n_train, args.min_score)
    if args.import_scores:
        return import_from_colab()
    if args.export_missing:
        return export_missing()
    if args.import_v5:
        return import_v5()
    torch.set_num_threads(config.N_WORKERS)
    t0 = time.time()

    if args.bench:
        p = _ce_training_pairs(2_000)
        text = _texts("train", np.unique(np.r_[p["id1"], p["id2"]]))
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
        a, b = text.reindex(p["id1"]).to_numpy()[:512], text.reindex(p["id2"]).to_numpy()[:512]
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        t = time.time()
        for i, enc in _batches(tok, a, b, 32):
            model(**enc).logits.sum().backward(); opt.step(); opt.zero_grad()
        print(f"  training: {512 / (time.time() - t):.0f} pairs/s")
        t = time.time()
        score(tok, model, p.head(2000), text)
        print(f"  scoring: {2000 / (time.time() - t):.0f} pairs/s")
        return

    if args.reuse_model:
        tok, model = AutoTokenizer.from_pretrained(CE_DIR), AutoModelForSequenceClassification.from_pretrained(CE_DIR)
    else:
        tp = _min_score_filter(_ce_training_pairs(args.n_train), "train_all", args.min_score)
        text = _texts("train", np.unique(np.r_[tp["id1"], tp["id2"]]))
        tok, model = train_ce(tp, text, bs=64)
        print(f"  cross-encoder trained ({time.time() - t0:.0f}s)", flush=True)

    for name, p1_file, split, ids, cand in (("val", "train_all_p1.parquet", "train", load_split("val"), "train_all"),
                                            ("train", "train_all_p1.parquet", "train", load_split("train"), "train_all"),
                                            ("test", "test_p1.parquet", "test", None, "test")):
        p = _min_score_filter(uncertain(p1_file, ids), cand, args.min_score)
        text = _texts(split, np.unique(np.r_[p["id1"], p["id2"]]))
        p["ce"] = score(tok, model, p, text)
        p[["id1", "id2", "ce"]].to_parquet(DATA / f"ce_{name}.parquet", index=False)
        print(f"  [{name}] {len(p):,} uncertain pairs scored ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
