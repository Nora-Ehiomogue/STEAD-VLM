#!/usr/bin/env python
"""Cache X3D-L features (192 x 16 x 10 x 10) for every clip of a manifest so STEAD-Fast can be
trained on features (as in the official STEAD code). Adds a `feat_path` column."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stead_vlm.preprocessing import to_model_tensor           # noqa: E402
from stead_vlm.stead_fast import X3D_MODEL_NAME, load_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-csv", required=True); ap.add_argument("--backbone", default=X3D_MODEL_NAME)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_backbone(a.backbone, dev)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df, paths = pd.read_csv(a.manifest), []
    with torch.no_grad():
        for p in df["npy_path"]:
            dst = out / (Path(p).stem + ".npy")
            if not dst.exists():
                x = to_model_tensor(np.load(p)).unsqueeze(0).to(dev)
                np.save(dst, net(x)[0].cpu().numpy().astype(np.float32))
            paths.append(str(dst))
    df["feat_path"] = paths
    df.to_csv(a.out_csv, index=False)


if __name__ == "__main__":
    main()
