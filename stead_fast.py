"""Stage 1: X3D-L backbone + STEAD-Fast head (thesis Section 3.4, Appendix B.2).

The head is the official STEAD `Model` (https://github.com/agao8/STEAD, MIT licence): it
consumes X3D-L feature maps of shape (B, 192, 16, 10, 10) and returns RAW LOGITS plus an
embedding. Apply a sigmoid to get the anomaly probability that is compared with theta.
Clone the repository and pass its path as `stead_repo`.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from .gradcam import GradCAM
from .preprocessing import to_model_tensor

X3D_MODEL_NAME = "x3d_l"     # backbone of the official STEAD feature extractor (320 x 320)


def _import_stead_model(stead_repo):
    """Import `Model` from a clone of the STEAD repo. Its utils.py calls
    `option.parse_args()` at import time, which would choke on this program's own CLI
    arguments, so sys.argv is neutralised during the import."""
    repo = Path(stead_repo).resolve()
    if not (repo / "model.py").is_file():
        raise FileNotFoundError(f"{repo} does not look like a clone of agao8/STEAD")
    old_argv, sys.argv = sys.argv, [sys.argv[0]]
    sys.path.insert(0, str(repo))
    try:
        for name in ("model", "utils", "option"):
            sys.modules.pop(name, None)
        Model = importlib.import_module("model").Model
    finally:
        sys.path.remove(str(repo))
        sys.argv = old_argv
        for name in ("model", "utils", "option"):     # do not leave generic names behind
            sys.modules.pop(name, None)
    return Model


def build_head(stead_repo, arch: str = "fast", **kwargs) -> torch.nn.Module:
    Model = _import_stead_model(stead_repo)
    if arch == "fast":
        return Model(ff_mult=1, dims=(32, 32), depths=(1, 1), **kwargs)   # as in STEAD main.py
    if arch == "base":
        return Model(**kwargs)
    raise ValueError("arch must be 'fast' or 'base'")


def load_head(stead_repo, ckpt, arch: str = "fast", device: str = "cpu") -> torch.nn.Module:
    head = build_head(stead_repo, arch)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        head.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"{ckpt} does not match the official STEAD-{arch} head (state-dict keys/shapes "
            "differ). Fine-tuned weights must be saved as head.state_dict().") from e
    return head.to(device).eval()


def load_backbone(name: str = X3D_MODEL_NAME, device: str = "cpu") -> torch.nn.Module:
    """X3D with its classification block removed -> (B, 192, T, 10, 10) features.
    Needs internet on first use (torch.hub downloads code and Kinetics weights)."""
    m = torch.hub.load("facebookresearch/pytorchvideo", name, pretrained=True)
    del m.blocks[-1]
    return m.to(device).eval()


class SteadFastDetector:
    """Array-in / probability-out wrapper used by the cascade."""

    def __init__(self, stead_repo, ckpt, arch: str = "fast", backbone: str = X3D_MODEL_NAME,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = load_backbone(backbone, self.device)
        self.head = load_head(stead_repo, ckpt, arch, self.device)
        self._cam = GradCAM(self.head)

    @torch.no_grad()
    def features_array(self, clip: np.ndarray) -> torch.Tensor:
        """clip: (T, H, W, 3) float32 in [0, 1] (output of prepare_clip)."""
        x = to_model_tensor(clip).unsqueeze(0).to(self.device)
        return self.backbone(x)

    @torch.no_grad()
    def score_features(self, feats: torch.Tensor) -> float:
        logits, _ = self.head(feats)
        return float(torch.sigmoid(logits.reshape(-1))[0])

    def cam_from_features(self, feats: torch.Tensor) -> np.ndarray:
        return self._cam(feats)
