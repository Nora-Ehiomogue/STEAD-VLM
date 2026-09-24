"""The three-stage cascade (thesis Section 3.2, Appendices B.5 and E).

Stage 0 (YOLOv8-nano)  -> discard clips with no person / relevant object
Stage 1 (STEAD-Fast)   -> anomaly probability; clip is *escalated* if score > theta
Stage 2 (Grad-CAM+VLM) -> evidence heat-map and natural-language explanation (explanatory
                          only: it never changes the Stage-1 score or decision)

This module does not import torch; models are injected, so it can be unit-tested with fakes.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CLIP_LEN, SAMPLE_FPS, TARGET, THETA
from .overlay import overlay_heatmap
from .preprocessing import fit_frame, prepare_clip


@dataclass
class ClipResult:
    score: float = 0.0
    passed_stage0: bool = True
    escalated: bool = False
    explanation: dict | None = None
    raw_explanation: str | None = None
    evidence_bgr: np.ndarray | None = None
    latency_ms: float = 0.0
    stage_ms: dict = field(default_factory=dict)


class ClipBuffer:
    """Turns a raw-FPS frame stream into 16-frame clips sampled at `sample_fps`, exactly like
    the training clips (Appendix A.4). The original demo instead fed 16 *consecutive* raw
    frames (~0.5 s at 30 FPS) to a model trained on ~3.2 s clips."""

    def __init__(self, native_fps: float, sample_fps: float = SAMPLE_FPS,
                 clip_len: int = CLIP_LEN, hop: int | None = None):
        self.step = max(int(round((native_fps or 30) / sample_fps)), 1)
        self.hop = clip_len if hop is None else hop       # hop == clip_len -> no overlap
        self._buf: deque = deque(maxlen=clip_len)
        self._i, self._since = -1, 0

    def push(self, frame_bgr):
        """Feed every raw frame; returns a list of `clip_len` frames when a clip is ready."""
        self._i += 1
        if self._i % self.step:
            return None
        self._buf.append(frame_bgr)
        self._since += 1
        if len(self._buf) == self._buf.maxlen and self._since >= self.hop:
            self._since = 0
            return list(self._buf)
        return None


class CascadePipeline:
    def __init__(self, stage0, detector, explainer=None, theta: float = THETA,
                 crop_mode: str = "crop_pad"):
        self.stage0, self.detector, self.explainer = stage0, detector, explainer
        self.theta, self.crop_mode = theta, crop_mode

    def process_clip(self, frames_bgr) -> ClipResult:
        t0 = time.perf_counter()
        res = ClipResult()

        if self.stage0 is not None and not self.stage0.clip_flag(frames_bgr):
            res.passed_stage0, res.score = False, 0.0          # as in the evaluation protocol
            res.stage_ms["stage0"] = res.latency_ms = (time.perf_counter() - t0) * 1e3
            return res
        t1 = time.perf_counter()

        arr = prepare_clip(frames_bgr, mode=self.crop_mode)
        feats = self.detector.features_array(arr)
        res.score = float(self.detector.score_features(feats))
        res.escalated = res.score > self.theta
        t2 = time.perf_counter()
        res.stage_ms.update(stage0=(t1 - t0) * 1e3, stage1=(t2 - t1) * 1e3)

        if res.escalated:
            cam = self.detector.cam_from_features(feats)
            # Overlay on the 320x320 view the model actually saw, so the CAM is aligned.
            view = fit_frame(frames_bgr[len(frames_bgr) // 2], TARGET, self.crop_mode)
            res.evidence_bgr = overlay_heatmap(view, cam)
            if self.explainer is not None:
                res.explanation, res.raw_explanation = self.explainer.explain(res.evidence_bgr)
            res.stage_ms["stage2"] = (time.perf_counter() - t2) * 1e3
        res.latency_ms = (time.perf_counter() - t0) * 1e3
        return res

    def iter_video(self, video_path, hop: int | None = None):
        """Yield (video_time_seconds, ClipResult) for a recorded video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        buf, n = ClipBuffer(fps, hop=hop), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            clip = buf.push(frame)
            if clip is not None:
                yield n / fps, self.process_clip(clip)
        cap.release()
