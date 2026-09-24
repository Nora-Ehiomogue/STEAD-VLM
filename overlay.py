"""Grad-CAM heat-map overlay (pure OpenCV / numpy)."""
from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0, 1] CAM (any h x w) onto a BGR uint8 frame. The CAM must describe the SAME
    field of view as `frame_bgr` (for the crop-based models use the 320 x 320 model-input
    view, not the full frame)."""
    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    cam = cv2.resize(cam, (frame_bgr.shape[1], frame_bgr.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0)
