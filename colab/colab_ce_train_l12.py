# Stronger cross-encoder (MiniLM-L12, 33M parameters, Apache-2.0) trained on hard pairs,
# on a Google Colab GPU (Runtime -> T4 GPU). About 45-60 minutes.
#
# 1. Put ce_train_v4.parquet in Google Drive -> My Drive (or a shortcut to it there).
# 2. Run this cell. It asks for Drive access at the START.
# 3. The trained model is saved to My Drive/ce_model_l12 (used for scoring later).
import time

import numpy as np
import pandas as pd
import torch
from google.colab import drive
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

drive.mount("/content/drive")
MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"      # 33M parameters, Apache-2.0
OUT = "/content/drive/MyDrive/ce_model_l12"
MAX_LEN, BS, LR, SEED = 96, 64, 2e-5, 42

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again"
dev = torch.device("cuda")
torch.manual_seed(SEED)
d = pd.read_parquet("/content/drive/MyDrive/ce_train_v4.parquet")
print(d["set"].value_counts().to_dict())
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
scaler = torch.amp.GradScaler("cuda")


def enc(part):
    return tok(list(part["text_a"]), list(part["text_b"]), truncation=True, max_length=MAX_LEN,
               padding=True, return_tensors="pt").to(dev)


# ---- train 1 epoch; hard negatives (weight 2) count double in the loss
tr = d[d["set"] == "train"].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
steps = (len(tr) + BS - 1) // BS
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
model.train()
t0 = time.time()
for step in range(steps):
    part = tr.iloc[step * BS:(step + 1) * BS]
    y = torch.tensor(part["label"].to_numpy(np.float32), device=dev)
    w = torch.tensor(part["weight"].to_numpy(np.float32), device=dev)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc(part)).logits.squeeze(-1).float()
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
    if step % 1000 == 0:
        print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    if step and step % 6000 == 0:                          # checkpoint: a disconnect keeps progress
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print(f"model saved to My Drive/ce_model_l12 ({time.time() - t0:.0f}s)")

# ---- check on val pairs
model.eval()
va = d[d["set"] == "val"].reset_index(drop=True)
order = np.argsort((va["text_a"].str.len() + va["text_b"].str.len()).to_numpy(), kind="stable")
s = va.iloc[order].reset_index(drop=True)
scores = np.empty(len(va), dtype=np.float32)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for i in range(0, len(s), 1024):
        part = s.iloc[i:i + 1024]
        scores[order[i:i + len(part)]] = torch.sigmoid(model(**enc(part)).logits.squeeze(-1).float()).cpu().numpy()
va["ce"] = scores
print(f"val AUC  L12 cross-encoder {roc_auc_score(va['label'], va['ce']):.4f}   stage-1 p1 {roc_auc_score(va['label'], va['p1']):.4f}")
va[["id1", "id2", "ce"]].to_parquet("/content/drive/MyDrive/ce_l12_val_scores.parquet", index=False)
print("DONE")
