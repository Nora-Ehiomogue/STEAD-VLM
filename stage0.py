"""Stage 0: YOLOv8-nano rule-based frame/clip filter (thesis Section 3.3, Appendix B.1).

Ultralytics YOLO is licensed AGPL-3.0 -- see README "Licences".
"""
from __future__ import annotations

from collections.abc import Iterable

from .config import STAGE0_MIN_RELEVANT, YOLO_CONF

COCO_PERSON = 0
COCO_BAG = {24, 26, 28}           # backpack, handbag, suitcase
COCO_RETAIL = {39, 41, 67, 73}    # bottle, cup, cell phone, book (COCO has no cart/box class)


def frame_is_relevant(class_ids: Iterable[int]) -> bool:
    """Rules 1-4 of the thesis, applied to the class ids detected (already thresholded at
    YOLO_CONF, which is rule 5).

    rule1: >=1 person          rule2: >=2 persons (crowd)
    rule3: >=1 bag/retail obj  rule4: person AND object
    A frame passes if ANY rule fires. Rules 2 and 4 are logically subsumed by rules 1 and 3,
    so the outcome is simply "a person or a relevant object is present".
    """
    ids = list(class_ids)
    n_person = ids.count(COCO_PERSON)
    n_object = sum(1 for c in ids if c in COCO_BAG or c in COCO_RETAIL)
    rule1 = n_person >= 1
    rule2 = n_person >= 2
    rule3 = n_object >= 1
    rule4 = n_person >= 1 and n_object >= 1
    return any((rule1, rule2, rule3, rule4))


def clip_is_relevant(frame_flags: Iterable[bool], min_relevant: int = STAGE0_MIN_RELEVANT) -> bool:
    """A clip is retained if at least `min_relevant` of its frames pass."""
    return sum(bool(f) for f in frame_flags) >= min_relevant


class Stage0Filter:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = YOLO_CONF,
                 min_relevant: int = STAGE0_MIN_RELEVANT, device: str | None = None):
        from ultralytics import YOLO          # lazy: keeps the rest importable without it
        self.model = YOLO(weights)
        self.conf, self.min_relevant, self.device = conf, min_relevant, device

    def frame_flag(self, frame_bgr) -> bool:
        """`frame_bgr`: OpenCV BGR ndarray (what Ultralytics expects for ndarray input)."""
        res = self.model(frame_bgr, conf=self.conf, verbose=False, device=self.device)[0]
        return frame_is_relevant(res.boxes.cls.int().tolist())

    def clip_flag(self, frames_bgr) -> bool:
        return clip_is_relevant((self.frame_flag(f) for f in frames_bgr), self.min_relevant)
