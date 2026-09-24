#!/usr/bin/env python
"""Dataset preparation CLI (thesis Appendix A). Typical order:

  audit -> blur -> clips -> (label clips by hand: CSV with columns clip,label)
        -> npy -> manifest -> split -> augment

The footage, labels and clips are NOT distributed with this repository.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stead_vlm import data                                     # noqa: E402
from stead_vlm.config import SEED                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("audit");   p.add_argument("--raw-dir", required=True)
    p = sub.add_parser("blur");    p.add_argument("--raw-dir", required=True); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("clips");   p.add_argument("--video-dir", required=True); p.add_argument("--clip-dir", required=True)
    p = sub.add_parser("npy");     p.add_argument("--clip-dir", required=True); p.add_argument("--npy-dir", required=True)
    p.add_argument("--mode", default="crop_pad", choices=["crop_pad", "scale_crop"])

    p = sub.add_parser("manifest")
    p.add_argument("--labels-csv", required=True, help="columns: clip,label (0 normal / 1 anomalous)")
    p.add_argument("--clip-dir", required=True); p.add_argument("--npy-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ucf-dir"); p.add_argument("--ucf-classes", nargs="+",
                   default=["Shoplifting", "Stealing", "Burglary", "Vandalism"])
    p.add_argument("--ucf-normal-dir", default="Normal")
    p.add_argument("--mode", default="crop_pad", choices=["crop_pad", "scale_crop"])

    p = sub.add_parser("split")
    p.add_argument("--manifest", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--test-frac", type=float, default=0.30); p.add_argument("--seed", type=int, default=SEED)

    p = sub.add_parser("augment")
    p.add_argument("--train-csv", required=True); p.add_argument("--test-csv", required=True)
    p.add_argument("--aug-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--p-flip", type=float, default=0.5); p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jit", type=float, default=0.5)

    a = ap.parse_args()

    if a.cmd == "audit":
        print(json.dumps(data.audit_footage(a.raw_dir), indent=2))
    elif a.cmd == "blur":
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for v in data.list_videos(a.raw_dir):
            dst = Path(a.out_dir) / (v.stem + ".mp4")
            if not dst.exists():
                data.blur_faces_in_video(v, dst)
    elif a.cmd == "clips":
        n = sum(len(data.extract_clips(v, a.clip_dir, v.stem)) for v in data.list_videos(a.video_dir))
        print(f"{n} clips written to {a.clip_dir}")
    elif a.cmd == "npy":
        for c in sorted(Path(a.clip_dir).glob("*.mp4")):
            data.clip_to_npy(c, a.npy_dir, a.mode)
    elif a.cmd == "manifest":
        rows = data.local_records(a.labels_csv, a.clip_dir, a.npy_dir)
        if a.ucf_dir:
            rows += data.ucf_records(a.ucf_dir, a.clip_dir, a.npy_dir, tuple(a.ucf_classes),
                                     a.ucf_normal_dir, a.mode)
        m = pd.DataFrame(rows)
        m.to_csv(a.out, index=False)
        print(m.groupby(["source", "label"]).size())
    elif a.cmd == "split":
        train, test = data.group_stratified_split(pd.read_csv(a.manifest), a.test_frac, a.seed)
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / "train_split.csv", index=False)
        test.to_csv(out / "test_split.csv", index=False)
        for name, d in (("train", train), ("test", test)):
            print(f"{name}: {len(d)} clips / {d.group_id.nunique()} groups, "
                  f"anomalous fraction {d.label.mean():.3f}")
    elif a.cmd == "augment":
        train, test = pd.read_csv(a.train_csv), pd.read_csv(a.test_csv)
        aug = data.augment_training_set(train, a.aug_dir, a.seed, test_df=test,
                                        p_flip=a.p_flip, p_rot=a.p_rot, p_jit=a.p_jit)
        aug.to_csv(a.out, index=False)
        print(f"{len(train)} -> {len(aug)} training clips (expansion {len(aug) / len(train):.2f}x)")


if __name__ == "__main__":
    main()
