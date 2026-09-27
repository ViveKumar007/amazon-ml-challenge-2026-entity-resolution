# ALL-IN-ONE bi-encoder run (use when the earlier session is gone): train, SAVE the model to
# Drive, then find neighbours for every train and test S1 record. Needs My Drive/bienc and
# My Drive/bienc2. Runtime -> T4 GPU. About 90 minutes.
#
# Upload the folder "bienc" (3 parquet files) to Google Drive -> My Drive, then run this
# cell. It fine-tunes a MiniLM bi-encoder so each Source 1 record sits close to its own
# Source 2/3 copies, encodes all 10.3M train S2/S3 records, finds every val S1 record's
# nearest records within its country and prints recall@k. Neighbours are saved to Drive.
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from google.colab import drive

drive.mount("/content/drive")
D = "/content/drive/MyDrive/bienc/"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # 22M parameters, Apache-2.0
MAX_LEN, BS, LR, SCALE, SEED, N_PAIRS, TOPK = 48, 256, 5e-5, 20.0, 42, 600_000, 50

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU"
dev = torch.device("cuda")
torch.manual_seed(SEED)
t0 = time.time()
index = pd.read_parquet(D + "index.parquet")
s1 = pd.read_parquet(D + "s1.parquet")
pairs = pd.read_parquet(D + "pairs.parquet")
print(f"index {len(index):,}  S1 {len(s1):,}  pairs {len(pairs):,}  ({time.time() - t0:.0f}s)")

tok = AutoTokenizer.from_pretrained(MODEL)
enc_model = AutoModel.from_pretrained(MODEL).to(dev)


def embed(texts, train=False):
    b = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        out = enc_model(**b).last_hidden_state
    mask = b["attention_mask"].unsqueeze(-1).float()
    v = (out.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(v, dim=-1)


# ---- fine-tune: S1 text vs its true copy, other copies in the batch are negatives
text1 = s1.set_index("id")["text"]
text2 = index.set_index("id")["text"]
tr = pairs[pairs["split"] == "train"].sample(min(N_PAIRS, (pairs["split"] == "train").sum()), random_state=SEED)
tr = tr.drop_duplicates("id1").sample(frac=1.0, random_state=SEED)       # one pair per S1: no false negatives in a batch
a, b = text1.reindex(tr["id1"]).to_numpy(), text2.reindex(tr["id2"]).to_numpy()
opt = torch.optim.AdamW(enc_model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")
enc_model.train()
steps = len(tr) // BS
for s in range(steps):
    qa, qb = embed(a[s * BS:(s + 1) * BS], True), embed(b[s * BS:(s + 1) * BS], True)
    logits = SCALE * qa @ qb.T
    labels = torch.arange(len(qa), device=dev)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if s % 200 == 0:
        print(f"step {s}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
enc_model.eval()


@torch.inference_mode()
def encode_all(texts, bs=2048):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty((len(texts), 384), dtype=torch.float16, device=dev)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        out[torch.as_tensor(idx, device=dev)] = embed(texts[idx]).half()
    return out

# ---- keep the trained bi-encoder (so a disconnect does not lose it)
enc_model.save_pretrained("/content/drive/MyDrive/bienc_model"); tok.save_pretrained("/content/drive/MyDrive/bienc_model")
print("bi-encoder saved to My Drive/bienc_model")


D2 = "/content/drive/MyDrive/bienc2/"
TOPK_FULL = 10
t1 = time.time()


def neighbours(queries, space, k=TOPK_FULL, QB=1024, EB=1_000_000):
    """Top-k most similar records of `space` for every query, within each country."""
    rows = []
    for country in queries["country"].unique():
        idx_c = space[space["country"] == country].reset_index(drop=True)
        q_c = queries[queries["country"] == country].reset_index(drop=True)
        if len(idx_c) == 0:
            continue
        E = encode_all(idx_c["text"].to_numpy())
        Q = encode_all(q_c["text"].to_numpy())
        print(f"  {country}: {len(q_c):,} queries x {len(idx_c):,} records ({time.time() - t1:.0f}s)")
        for i in range(0, len(Q), QB):
            vals, inds = [], []
            for j in range(0, len(E), EB):
                t = torch.topk(Q[i:i + QB] @ E[j:j + EB].T, min(k, len(E) - j), dim=1)
                vals.append(t.values); inds.append(t.indices + j)
            v, ix = torch.cat(vals, 1), torch.cat(inds, 1)
            top = torch.topk(v.float(), k, dim=1)
            best = torch.gather(ix, 1, top.indices).cpu().numpy()
            rows.append(pd.DataFrame({
                "id1": np.repeat(q_c["id"].to_numpy()[i:i + QB], k),
                "id2": idx_c["id"].to_numpy()[best.ravel()],
                "cos": top.values.cpu().numpy().ravel().astype(np.float16),
                "rank": np.tile(np.arange(k, dtype=np.int8), len(best))}))
        del E, Q
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True)


train_s1 = pd.read_parquet(D2 + "train_s1.parquet")
nb = neighbours(train_s1, index)                     # `index` = train S2/S3, loaded in step 1
nb.to_parquet(D2 + "train_neighbors.parquet", index=False, compression="zstd")
print(f"train neighbours saved: {len(nb):,} rows ({time.time() - t1:.0f}s)")
del nb, train_s1

test_s1 = pd.read_parquet(D2 + "test_s1.parquet")
test_index = pd.read_parquet(D2 + "test_index.parquet")
nb = neighbours(test_s1, test_index)
nb.to_parquet(D2 + "test_neighbors.parquet", index=False, compression="zstd")
print(f"test neighbours saved: {len(nb):,} rows ({time.time() - t1:.0f}s)")
print("DONE: download train_neighbors.parquet and test_neighbors.parquet from My Drive/bienc2")
