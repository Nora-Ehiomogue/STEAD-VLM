#!/usr/bin/env python
"""Evaluation (thesis Chapter 4).

  run   : push the held-out test clips through the cascade -> per-clip results CSV
  score : metrics, bootstrap AUC CIs and latency summary for one or more results CSVs
          (baselines only need columns clip,label,score,latency_ms)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stead_vlm.config import CLIP_LEN, THETA                    # noqa: E402
from stead_vlm.metrics import auc_ci, detection_metrics, latency_summary  # noqa: E402


def cmd_run(a):
    from stead_vlm.metrics import peak_vram
    from stead_vlm.pipeline import CascadePipeline
    from stead_vlm.preprocessing import read_frames
    from stead_vlm.stage0 import Stage0Filter
    from stead_vlm.stead_fast import SteadFastDetector

    explainer = None
    if a.vlm_base:
        from stead_vlm.vlm import MobileVLMExplainer
        explainer = MobileVLMExplainer(a.vlm_base, a.vlm_adapter)
    pipe = CascadePipeline(Stage0Filter(a.yolo_weights), SteadFastDetector(a.stead_repo, a.stead_ckpt),
                           explainer, theta=a.theta, crop_mode=a.crop_mode)
    test = pd.read_csv(a.test_csv)
    rows = []
    with peak_vram() as vram:
        for _, r in test.iterrows():
            res = pipe.process_clip(read_frames(r["clip_path"], CLIP_LEN))
            conf = (res.explanation or {}).get("confidence", float("nan"))
            rows.append({"clip": r["clip_path"], "label": int(r["label"]), "score": res.score,
                         "passed_stage0": res.passed_stage0, "escalated": res.escalated,
                         "latency_ms": res.latency_ms, "stage_ms": json.dumps(res.stage_ms),
                         "vlm_conf": conf, "explanation": json.dumps(res.explanation)})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"peak GPU memory allocated by PyTorch: {vram['peak_gb']:.2f} GB -> {a.out}")


def cmd_score(a):
    report = {}
    for spec in a.results:                       # name=path
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)
        m = detection_metrics(df["label"], df["score"], a.theta)
        lo, hi = auc_ci(df["label"], df["score"], a.n_boot)
        m["AUC 95% CI"] = [round(lo, 2), round(hi, 2)]
        if "latency_ms" in df:
            m.update(latency_summary(df["latency_ms"]))
        if "escalated" in df:
            m["Trigger rate (%)"] = float(df["escalated"].mean() * 100)
        report[name] = m
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--test-csv", required=True); r.add_argument("--out", required=True)
    r.add_argument("--stead-repo", required=True); r.add_argument("--stead-ckpt", required=True)
    r.add_argument("--yolo-weights", default="yolov8n.pt")
    r.add_argument("--vlm-base"); r.add_argument("--vlm-adapter")
    r.add_argument("--theta", type=float, default=THETA)
    r.add_argument("--crop-mode", default="crop_pad", choices=["crop_pad", "scale_crop"])
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score")
    s.add_argument("results", nargs="+", help="name=results.csv")
    s.add_argument("--theta", type=float, default=THETA); s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--out"); s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
