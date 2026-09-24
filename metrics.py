"""Detection and efficiency metrics (thesis Sections 3.9 and 4.3-4.5, Appendix C.1/C.3)."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .config import CLIP_LEN, SAMPLE_FPS, SEED, THETA


def detection_metrics(y, s, theta: float = THETA) -> dict:
    """AUC-ROC uses the raw scores; accuracy/precision/recall/F1 use `s > theta`. All in %."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    y, s = np.asarray(y), np.asarray(s)
    p = (s > theta).astype(int)
    return {"AUC-ROC": roc_auc_score(y, s) * 100, "Accuracy": accuracy_score(y, p) * 100,
            "Precision": precision_score(y, p, zero_division=0) * 100,
            "Recall": recall_score(y, p, zero_division=0) * 100,
            "F1-Score": f1_score(y, p, zero_division=0) * 100}


def auc_ci(y, s, n_boot: int = 2000, seed: int = SEED, alpha: float = 0.05):
    """Percentile bootstrap CI for AUC-ROC (in %). Resamples clips with replacement."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    rng, boots = np.random.default_rng(seed), []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    if not boots:
        raise ValueError("no valid bootstrap resamples (need both classes)")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo * 100, hi * 100


def latency_summary(latency_ms_per_clip, clip_len: int = CLIP_LEN,
                    sample_fps: float = SAMPLE_FPS, native_fps: float = 30.0) -> dict:
    """Report latency with explicit units.

    The evaluation code computes `ms/frame = clip latency / 16` and `FPS = 1000 / that`.
    Those count the 16 *sampled* frames (5 FPS), not camera frames. The thesis (Section 3.1.1)
    instead defines throughput as *camera frames consumed per second*; with 1 of every 6
    frames kept, that figure is `native_fps / sample_fps` (= 6x) larger. Both are returned,
    with the real-time factor (seconds of video processed per second of compute).
    """
    ms = float(np.mean(latency_ms_per_clip))
    clip_seconds = clip_len / sample_fps                  # video time covered by one clip
    rtf = clip_seconds * 1000.0 / ms                      # > 1 = faster than real time
    return {"ms_per_clip": ms, "ms_per_sampled_frame": ms / clip_len,
            "sampled_frames_per_s": 1000.0 * clip_len / ms,
            "camera_frames_per_s": rtf * native_fps,
            "video_seconds_per_clip": clip_seconds, "real_time_factor": rtf}


@contextmanager
def peak_vram():
    """Peak GPU memory allocated by PyTorch inside the block (GB). Run each system in its own
    process/session -- a single `nvidia-smi` reading of a shared GPU is not a per-system peak."""
    import torch
    out = {"peak_gb": float("nan")}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
