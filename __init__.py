"""STEAD-VLM: a three-stage cascade (YOLOv8-nano -> STEAD-Fast -> MobileVLM V2) for
explainable retail video anomaly detection.

Torch-dependent modules (stead_fast, gradcam, vlm, pipeline) import torch lazily or only
when used, so the data / metrics / pre-processing code works without a GPU stack.
"""
__version__ = "0.1.0"
