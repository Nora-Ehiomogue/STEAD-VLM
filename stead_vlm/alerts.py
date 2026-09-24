"""Telegram alerting (thesis Section 3.8). Credentials come from the environment only."""
from __future__ import annotations

import os

import cv2
import requests


def format_alert(score: float, description: str, timestamp: str) -> str:
    bar = "-" * 20
    return (f"STEAD-VLM ANOMALY ALERT\n{bar}\nTime  : {timestamp}\nScore : {score:.4f}\n{bar}\n"
            f"{description}\n{bar}\nAction required by security personnel.")


def send_alert(text: str, image_bgr=None, token: str | None = None, chat_id: str | None = None,
               timeout: float = 15.0, http=requests) -> bool:
    """Send a text message (and optionally the Grad-CAM evidence image). Returns True if
    Telegram accepted every request. Never logs the token."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (never hard-code them).")
    base = f"https://api.telegram.org/bot{token}"
    ok = http.post(f"{base}/sendMessage", data={"chat_id": chat_id, "text": text},
                   timeout=timeout).ok
    if image_bgr is not None:
        good, buf = cv2.imencode(".jpg", image_bgr)
        if good:
            ok &= http.post(f"{base}/sendPhoto", data={"chat_id": chat_id,
                            "caption": "Grad-CAM evidence"},
                            files={"photo": ("evidence.jpg", buf.tobytes(), "image/jpeg")},
                            timeout=timeout).ok
    return bool(ok)
