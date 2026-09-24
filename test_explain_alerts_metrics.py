import math

import numpy as np
import pytest

from stead_vlm import alerts, metrics
from stead_vlm.explain import parse_explanation
from stead_vlm.overlay import overlay_heatmap


# ---------------------------------------------------------------- explanation parsing
def test_parse_clean_json():
    e = parse_explanation('{"anomaly_type": "theft", "description": "hides item", "confidence": 0.87}')
    assert e["parsed"] and e["anomaly_type"] == "theft" and e["confidence"] == 0.87


def test_parse_json_inside_prose_and_nested_braces():
    txt = 'Sure! {"description": "a {curly} note", "confidence": "0.9"} hope this helps'
    e = parse_explanation(txt)
    assert e["parsed"] and e["description"] == "a {curly} note" and e["confidence"] == 0.9


def test_parse_failure_never_returns_a_str_and_never_crashes():
    for bad in ["just prose about a person", '{"description": "unterminated', "", None]:
        e = parse_explanation(bad)
        assert isinstance(e, dict) and not e["parsed"] and math.isnan(e["confidence"])
    assert parse_explanation("just prose")["description"] == "just prose"


def test_non_numeric_confidence_becomes_nan():
    assert math.isnan(parse_explanation('{"confidence": "high"}')["confidence"])


# ------------------------------------------------------------------------- alerts
class FakeResp:
    def __init__(self, ok=True): self.ok = ok


class FakeHttp:
    def __init__(self, ok=True): self.calls, self.ok = [], ok
    def post(self, url, **kw): self.calls.append((url, kw)); return FakeResp(self.ok)


def test_alert_requires_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(RuntimeError):
        alerts.send_alert("x", http=FakeHttp())


def test_alert_sends_text_and_photo_with_timeout_and_keeps_token_out_of_payload(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    http = FakeHttp()
    text = alerts.format_alert(0.91234, "desc", "2026-01-01 10:00:00")
    assert alerts.send_alert(text, np.zeros((20, 20, 3), np.uint8), http=http) is True
    assert [u.rsplit("/", 1)[1] for u, _ in http.calls] == ["sendMessage", "sendPhoto"]
    assert all(kw["timeout"] > 0 for _, kw in http.calls) and "SECRET" not in text
    assert "0.9123" in text
    assert alerts.send_alert("x", http=FakeHttp(ok=False)) is False


# ------------------------------------------------------------------------ metrics
def test_detection_metrics_and_strict_threshold():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.9, 0.8])
    m = metrics.detection_metrics(y, s)
    assert m["AUC-ROC"] == 100 and m["Accuracy"] == 100 and m["F1-Score"] == 100
    assert metrics.detection_metrics(y, np.array([0.1, 0.2, 0.65, 0.8]))["Recall"] == 50.0  # 0.65 !> 0.65


def test_auc_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); s = y * 0.3 + rng.random(400) * 0.7
    lo, hi = metrics.auc_ci(y, s, n_boot=300)
    assert lo < metrics.detection_metrics(y, s)["AUC-ROC"] < hi


def test_latency_summary_makes_the_fps_definition_explicit():
    lat = metrics.latency_summary([35.2 * 16])                 # 35.2 ms per SAMPLED frame
    assert abs(lat["sampled_frames_per_s"] - 28.4) < 0.05
    assert lat["video_seconds_per_clip"] == 3.2
    assert abs(lat["real_time_factor"] - 28.4 / 5) < 0.01     # ~5.7x faster than real time
    assert abs(lat["camera_frames_per_s"] - 28.4 * 6) < 0.3   # thesis definition = 6x the sampled rate


# ------------------------------------------------------------------------ overlay
def test_overlay_shape_dtype_and_highlights_hot_region():
    frame = np.full((320, 320, 3), 128, np.uint8)
    cam = np.zeros((10, 10), np.float32); cam[:, 5:] = 1.0
    out = overlay_heatmap(frame, cam)
    assert out.shape == frame.shape and out.dtype == np.uint8
    hot, cold = out[160, 300].astype(int), out[160, 10].astype(int)      # BGR pixels
    assert hot[2] > hot[0] and cold[0] > cold[2]        # JET: high CAM -> red, low CAM -> blue
