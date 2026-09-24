"""Frame / clip pre-processing (thesis Section 3.1.3.3, Appendix A.5).

One code path is used for BOTH the cached training arrays and inference, so train and
deployment inputs cannot drift apart.

Frames are BGR uint8 (OpenCV convention) everywhere in this package; conversion to RGB
happens inside `prepare_clip`.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import CLIP_LEN, TARGET


def read_frames(path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Read frames of a video file as BGR uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def center_crop_pad(frame: np.ndarray, size: int = TARGET) -> np.ndarray:
    """Zero-pad if smaller than `size`, then centre-crop to size x size (no resizing)."""
    h, w = frame.shape[:2]
    if h < size or w < size:
        pt = max((size - h) // 2, 0)
        pl = max((size - w) // 2, 0)
        frame = cv2.copyMakeBorder(
            frame, pt, max(size - h - pt, 0), pl, max(size - w - pl, 0),
            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = frame.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return frame[y0:y0 + size, x0:x0 + size]


def scale_short_side_crop(frame: np.ndarray, size: int = TARGET) -> np.ndarray:
    """Resize so the short side == size, then centre-crop (as in the official STEAD
    feature extractor: ShortSideScale + CenterCrop). Keeps the whole field of view."""
    h, w = frame.shape[:2]
    s = size / min(h, w)
    new_w, new_h = max(size, round(w * s)), max(size, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return center_crop_pad(cv2.resize(frame, (new_w, new_h), interpolation=interp), size)


def fit_frame(frame: np.ndarray, size: int = TARGET, mode: str = "crop_pad") -> np.ndarray:
    """mode='crop_pad'   -> exactly what Appendix A.4/A.5 do (native-resolution centre crop).
       mode='scale_crop' -> resize short side to `size` first, then centre-crop."""
    if mode == "crop_pad":
        return center_crop_pad(frame, size)
    if mode == "scale_crop":
        return scale_short_side_crop(frame, size)
    raise ValueError(f"unknown mode {mode!r}")


def minmax_normalise(clip: np.ndarray) -> np.ndarray:
    """Per-clip min-max normalisation to [0, 1]: (X - Xmin) / (Xmax - Xmin)."""
    clip = clip.astype(np.float32)
    return (clip - clip.min()) / (clip.max() - clip.min() + 1e-8)


def prepare_clip(frames_bgr: list[np.ndarray], size: int = TARGET,
                 mode: str = "crop_pad", clip_len: int = CLIP_LEN) -> np.ndarray:
    """BGR frames -> float32 array (T, H, W, 3) in [0, 1], RGB channel order."""
    if len(frames_bgr) < clip_len:
        raise ValueError(f"need {clip_len} frames, got {len(frames_bgr)}")
    rgb = [fit_frame(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size, mode)
           for f in frames_bgr[:clip_len]]
    return minmax_normalise(np.stack(rgb))


def to_model_tensor(arr: np.ndarray):
    """(T, H, W, 3) -> torch float tensor (3, T, H, W)."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(3, 0, 1, 2).float()
