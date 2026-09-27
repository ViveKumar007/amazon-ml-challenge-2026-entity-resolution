# Cross-encoder for the uncertain pairs, run on a Google Colab GPU (Runtime -> T4 GPU).
# Same model and training recipe as src/crossenc.py (the CPU version).
#
# 1. Upload ce_data_v2.parquet with the Files panel (folder icon on the left).
# 2. Paste this whole file into one cell and run it (about 60-80 minutes on a T4).
#    It asks once for Google Drive access at the START, so it can save the result there.
# 3. Download ce_scores_v2.parquet from drive.google.com (My Drive).
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

for d in ("/content", "/"):                        # wherever the upload landed
    if os.path.exists(os.path.join(d, "ce_data_v2.parquet")):
        os.chdir(d)
from google.colab import drive                      # ask for Drive access now, not after an hour
drive.mount("/content/drive")

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # 22M parameters, Apache-2.0
MAX_LEN, BS_TRAIN, BS_SCORE, LR, SEED = 96, 64, 1024, 3e-5, 42

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again"
dev = torch.device("cuda")
torch.manual_seed(SEED)
d = pd.read_parquet("ce_data_v2.parquet")
print(d["set"].value_counts().to_dict())

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
scaler = torch.amp.GradScaler("cuda")


def batches(df, bs):
    for i in range(0, len(df), bs):
        part = df.iloc[i:i + bs]
        enc = tok(list(part["text_a"]), list(part["text_b"]), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        yield i, part, enc


# ---- fine-tune on the labelled training pairs (1 epoch, random order)
tr = d[d["set"] == "ce_train"].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
steps = (len(tr) + BS_TRAIN - 1) // BS_TRAIN
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
loss_fn = torch.nn.BCEWithLogitsLoss()
model.train()
t0 = time.time()
for step, (i, part, enc) in enumerate(batches(tr, BS_TRAIN)):
    y = torch.tensor(part["label"].to_numpy(np.float32), device=dev)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(**enc).logits.squeeze(-1).float(), y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
    if step % 1000 == 0:
        print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
model.save_pretrained("/content/drive/MyDrive/ce_model_v2")     # reused by colab_ce_score_v5.py
tok.save_pretrained("/content/drive/MyDrive/ce_model_v2")

# ---- score val / train / test pairs, in batches of similar length (much less padding)
model.eval()
todo = d[d["set"] != "ce_train"].reset_index(drop=True)
order = np.argsort((todo["text_a"].str.len() + todo["text_b"].str.len()).to_numpy(), kind="stable")
sorted_todo = todo.iloc[order].reset_index(drop=True)
scores = np.empty(len(todo), dtype=np.float32)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for i, part, enc in batches(sorted_todo, BS_SCORE):
        scores[order[i:i + len(part)]] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if (i // BS_SCORE) % 500 == 0:
            print(f"scored {i:,}/{len(todo):,} ({time.time() - t0:.0f}s)")
todo["ce"] = scores.astype(np.float16)

for s in ("val", "train"):
    v = todo[todo["set"] == s]
    print(f"{s}: AUC cross-encoder {roc_auc_score(v['label'], v['ce']):.4f}   vs stage-1 p1 {roc_auc_score(v['label'], v['p1']):.4f}")
out = todo[["set", "id1", "id2", "ce"]]
out.to_parquet("/content/drive/MyDrive/ce_scores_v2.parquet", index=False, compression="zstd")
print("DONE: ce_scores_v2.parquet is in your Google Drive (My Drive) - download it from drive.google.com")
