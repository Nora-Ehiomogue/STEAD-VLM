#!/usr/bin/env python
"""Run the cascade on a recorded video; optionally send Telegram alerts.

Credentials: export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID .
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stead_vlm.alerts import format_alert, send_alert           # noqa: E402
from stead_vlm.config import THETA                              # noqa: E402
from stead_vlm.pipeline import CascadePipeline                  # noqa: E402
from stead_vlm.stage0 import Stage0Filter                       # noqa: E402
from stead_vlm.stead_fast import SteadFastDetector              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--stead-repo", required=True); ap.add_argument("--stead-ckpt", required=True)
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--vlm-base"); ap.add_argument("--vlm-adapter")
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--cooldown", type=float, default=10.0, help="seconds of VIDEO time between alerts")
    ap.add_argument("--telegram", action="store_true", help="send alerts via Telegram")
    ap.add_argument("--crop-mode", default="crop_pad", choices=["crop_pad", "scale_crop"])
    a = ap.parse_args()

    explainer = None
    if a.vlm_base:
        from stead_vlm.vlm import MobileVLMExplainer
        explainer = MobileVLMExplainer(a.vlm_base, a.vlm_adapter)
    pipe = CascadePipeline(Stage0Filter(a.yolo_weights), SteadFastDetector(a.stead_repo, a.stead_ckpt),
                           explainer, theta=a.theta, crop_mode=a.crop_mode)

    last_alert, n_clips, n_pass, n_flag, n_sent = -1e9, 0, 0, 0, 0
    for t, res in pipe.iter_video(a.video):
        n_clips += 1; n_pass += res.passed_stage0
        tag = "filtered" if not res.passed_stage0 else ("ANOMALY" if res.escalated else "normal")
        print(f"t={t:7.1f}s  score={res.score:.4f}  {tag}")
        if not res.escalated:
            continue
        n_flag += 1
        if t - last_alert < a.cooldown:
            print("   (alert suppressed: cooldown)")
            continue
        last_alert = t
        desc = (res.explanation or {}).get("description") or res.raw_explanation or "(no explanation)"
        text = format_alert(res.score, desc, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(text)
        if a.telegram:
            n_sent += bool(send_alert(text, res.evidence_bgr))
    print(f"\nclips: {n_clips}  passed Stage 0: {n_pass}  flagged: {n_flag}  alerts sent: {n_sent}")


if __name__ == "__main__":
    main()
