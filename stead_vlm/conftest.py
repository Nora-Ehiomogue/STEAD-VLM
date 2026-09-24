import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def make_video(tmp_path):
    """Write a synthetic mp4 whose frame i is a flat image with value (i % 250)."""
    def _make(name="v.mp4", n=200, fps=30, size=(260, 340)):     # (h, w)
        path = tmp_path / name
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]))
        for i in range(n):
            vw.write(np.full((size[0], size[1], 3), i % 250, np.uint8))
        vw.release()
        return path
    return _make
