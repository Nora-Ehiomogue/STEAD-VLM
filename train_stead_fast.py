#!/usr/bin/env python
"""Fine-tune the STEAD-Fast head on cached X3D-L features (thesis Section 3.4).

REFERENCE IMPLEMENTATION. The thesis appendix (B.2) refers to a `ClipDataset`, an `evaluate`
function and an anchor/positive/negative sampler that were not included, so the triplet
mining here (batch-hard on the 32-d embedding) is a reconstruction, not the original code.
Stated settings are kept: 10 epochs, AdamW, lr 1e-4, batch size 8, loss = BCE + 0.5 * triplet.
The validation split is grouped by recording session (`group_id`), not by individual clip.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stead_vlm.config import SEED                              # noqa: E402
from stead_vlm.stead_fast import build_head                     # noqa: E402


class FeatureSet(Dataset):
    def __init__(self, df):
        self.p, self.y = df["feat_path"].tolist(), df["label"].astype("float32").tolist()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return torch.from_numpy(np.load(self.p[i]).astype(np.float32)), torch.tensor(self.y[i])


def batch_hard_triplet(emb, y, margin=1.0):
    """Hardest positive / hardest negative in the batch; anchors lacking either are skipped."""
    d = torch.cdist(emb, emb)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = d.masked_fill(~same | eye, float("-inf")).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not valid.any():
        return emb.new_zeros(())
    return torch.relu(pos[valid] - neg[valid] + margin).mean()


@torch.no_grad()
def evaluate(head, loader, dev, bce):
    head.eval()
    ys, ps, loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits, _ = head(x)
        logits = logits.reshape(-1)
        loss += bce(logits, y).item() * len(y); n += len(y)
        ys += y.cpu().tolist(); ps += torch.sigmoid(logits).cpu().tolist()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return loss / n, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="augmented manifest with feat_path")
    ap.add_argument("--stead-repo", required=True); ap.add_argument("--init", help="official STEAD-Fast weights")
    ap.add_argument("--out-dir", default="runs/stead_fast"); ap.add_argument("--arch", default="fast")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.train_csv)
    tr, va = next(GroupShuffleSplit(1, test_size=a.val_frac, random_state=SEED).split(df, groups=df["group_id"]))
    train_loader = DataLoader(FeatureSet(df.iloc[tr]), batch_size=a.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(FeatureSet(df.iloc[va]), batch_size=a.batch_size)

    head = build_head(a.stead_repo, a.arch)
    if a.init:
        head.load_state_dict(torch.load(a.init, map_location="cpu"))
    head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss()       # the head outputs logits; BCELoss would need a sigmoid
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        head.train(); tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            logits, emb = head(x)
            loss = bce(logits.reshape(-1), y) + a.lam * batch_hard_triplet(emb, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        vl, vauc = evaluate(head, val_loader, dev, bce)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, "val_auc": vauc})
        print(hist[-1])
        if vauc > best:
            best = vauc; torch.save(head.state_dict(), out / "stead_fast_best.pt")
    torch.save(head.state_dict(), out / "stead_fast_last.pt")
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
