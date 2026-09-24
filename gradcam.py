"""Grad-CAM on the STEAD-Fast head (thesis Section 3.6, Appendix B.4).

Works on the head's input features (B, 192, T, H, W) produced by the X3D backbone, i.e. a
10 x 10 spatial map for 320 x 320 input. `target_layer` defaults to the last stage, whose
output is channel-last (B, T, H, W, C).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, head: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.head = head
        self._acts = None
        self._grads = None
        layer = target_layer if target_layer is not None else head.stages[-1]
        # A forward hook that registers a tensor hook is more robust than
        # register_full_backward_hook (which needs inputs that require grad).
        layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self._acts = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g.detach()))

    def __call__(self, feats: torch.Tensor) -> np.ndarray:
        """feats: (1, 192, T, H, W). Returns a normalised CAM of shape (H, W)."""
        self._acts = self._grads = None
        with torch.enable_grad():
            self.head.zero_grad(set_to_none=True)
            logits, _ = self.head(feats.detach())
            logits.reshape(-1).sum().backward()
        if self._acts is None or self._grads is None:
            raise RuntimeError("Grad-CAM hooks did not fire; is the head in the graph?")
        w = self._grads.mean(dim=(1, 2, 3), keepdim=True)          # (B, 1, 1, 1, C)
        cam = F.relu((w * self._acts).sum(dim=-1)).mean(dim=1)[0]  # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()
