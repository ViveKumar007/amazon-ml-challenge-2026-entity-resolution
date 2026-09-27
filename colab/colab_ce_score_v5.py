# Score the uncertain pairs of the union candidate set with both cross-encoders on a Colab
# GPU (Runtime -> Change runtime type -> T4 GPU). About 40-60 minutes.
#   MiniLM-L12 (My Drive/ce_model_l12) scores ALL pairs            -> ce12
#   MiniLM-L6  (My Drive/ce_model_v2)  scores pairs with need_l6    -> ce6 (skipped if the model is absent)
#
# My Drive must hold (as files or "Add shortcut to My Drive"):
#   ce_pairs_v5.parquet                       (from: python src/crossenc.py --export-missing)
#   ce_model_l12/   ce_model_v2/
#   bienc/index.parquet                       (train S2/S3 texts)
#   bienc2/train_s1.parquet, bienc2/test_s1.parquet, bienc2/test_index.parquet
# Progress is saved per 1M pairs in My Drive/ce_v5_parts/: after a disconnect just run the cell again.
# Output: My Drive/ce_scores_v5.parquet (set, id1, id2, ce12, ce6) -> download it into work/data/colab/
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from google.colab import drive
from transformers import AutoModelForSequenceClassification, AutoTokenizer

drive.mount("/content/drive")
D = "/content/drive/MyDrive/"
PARTS = D + "ce_v5_parts/"
MODELS = {"ce12": D + "ce_model_l12", "ce6": D + "ce_model_v2"}
TEXT_FILES = {"train": [D + "bienc2/train_s1.parquet", D + "bienc/index.parquet"],
              "test": [D + "bienc2/test_s1.parquet", D + "bienc2/test_index.parquet"]}
CHUNK, BS, MAX_LEN = 1_000_000, 1024, 96

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again"
for f in [D + "ce_pairs_v5.parquet", MODELS["ce12"]] + TEXT_FILES["train"] + TEXT_FILES["test"]:
    assert os.path.exists(f), f"missing in My Drive: {f[len(D):]}"
if not os.path.exists(MODELS["ce6"]):
    print("WARNING: ce_model_v2 not in My Drive -> the L6 scores (ce6) are skipped")
    del MODELS["ce6"]
os.makedirs(PARTS, exist_ok=True)
dev = torch.device("cuda")
pairs = pd.read_parquet(D + "ce_pairs_v5.parquet")
pairs["set"] = pairs["set"].astype(str)
print(pairs["set"].value_counts().to_dict(), f"need_l6: {int(pairs['need_l6'].sum()):,}")


def texts(split, ids):
    """'name / alias ; address' text of the given record ids of one split -> Series by id."""
    want = pa.array(np.unique(ids))
    parts = []
    for f in TEXT_FILES[split]:
        t = pq.read_table(f, columns=["id", "text"])
        parts.append(t.filter(pc.is_in(t["id"], value_set=want)).to_pandas())
    return pd.concat(parts).drop_duplicates("id").set_index("id")["text"]


@torch.inference_mode()
def score(tok, model, a, b):
    """Match probability of each text pair (length-sorted batches, fp16)."""
    order = np.argsort(pd.Series(a).str.len().to_numpy() + pd.Series(b).str.len().to_numpy(), kind="stable")
    out = np.empty(len(a), dtype=np.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(a), BS):
            idx = order[i:i + BS]
            enc = tok(list(a[idx]), list(b[idx]), truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


t0 = time.time()
for split, sets in (("train", ("val", "train")), ("test", ("test",))):
    sub = pairs[pairs["set"].isin(sets)]
    tx = texts(split, np.r_[sub["id1"].to_numpy(), sub["id2"].to_numpy()])
    a, b = tx.reindex(sub["id1"]).to_numpy(), tx.reindex(sub["id2"]).to_numpy()
    miss = int(pd.isna(a).sum() + pd.isna(b).sum())
    assert miss == 0, f"{split}: {miss} record texts not found - wrong text files?"
    print(f"[{split}] {len(sub):,} pairs, texts ready ({time.time() - t0:.0f}s)")
    for col, mdir in MODELS.items():
        need = np.ones(len(sub), bool) if col == "ce12" else sub["need_l6"].to_numpy()
        rows = np.flatnonzero(need)
        tok = model = None
        for k in range(0, len(rows), CHUNK):
            f = f"{PARTS}{col}_{split}_{k // CHUNK:03d}.npy"
            if os.path.exists(f):
                continue
            if model is None:
                tok = AutoTokenizer.from_pretrained(mdir)
                model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
            r = rows[k:k + CHUNK]
            np.save(f + ".tmp.npy", score(tok, model, a[r], b[r]))
            os.replace(f + ".tmp.npy", f)                    # a disconnect never leaves half a chunk
            print(f"  {col} {split}: {min(k + CHUNK, len(rows)):,}/{len(rows):,} ({time.time() - t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

out = []
for split, sets in (("train", ("val", "train")), ("test", ("test",))):
    sub = pairs[pairs["set"].isin(sets)][["set", "id1", "id2"]].copy()
    for col in ("ce12", "ce6"):
        v = np.full(len(sub), np.nan, dtype=np.float32)
        if col in MODELS:
            need = np.ones(len(sub), bool) if col == "ce12" else pairs.loc[sub.index, "need_l6"].to_numpy()
            rows = np.flatnonzero(need)
            chunks = [np.load(f"{PARTS}{col}_{split}_{k // CHUNK:03d}.npy") for k in range(0, len(rows), CHUNK)]
            if chunks:
                v[rows] = np.concatenate(chunks)
        sub[col] = v.astype(np.float16)
    out.append(sub)
res = pd.concat(out, ignore_index=True)
res.to_parquet(D + "ce_scores_v5.parquet", index=False, compression="zstd")
print(f"DONE ({time.time() - t0:.0f}s): {len(res):,} pairs, mean ce12 {res['ce12'].astype(float).mean():.3f}"
      f" -> download ce_scores_v5.parquet from My Drive into work/data/colab/")
