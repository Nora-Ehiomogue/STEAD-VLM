import random

from stead_vlm.stage0 import COCO_BAG, COCO_RETAIL, clip_is_relevant, frame_is_relevant


def test_basic_rules():
    assert not frame_is_relevant([])
    assert frame_is_relevant([0])            # person
    assert frame_is_relevant([0, 0])         # crowd
    assert frame_is_relevant([24])           # backpack only
    assert frame_is_relevant([39])           # bottle only
    assert not frame_is_relevant([2, 7])     # car, truck


def test_rules_2_and_4_are_redundant():
    """The five-rule wording collapses to: a person OR a relevant object is present."""
    rng, pool = random.Random(0), [0, 2, 7, 24, 26, 28, 39, 41, 67, 73, 15, 56]
    objects = COCO_BAG | COCO_RETAIL
    for _ in range(500):
        ids = [rng.choice(pool) for _ in range(rng.randint(0, 6))]
        expected = (0 in ids) or any(c in objects for c in ids)
        assert frame_is_relevant(ids) == expected


def test_clip_needs_min_relevant_frames():
    assert clip_is_relevant([True] * 4 + [False] * 12)
    assert not clip_is_relevant([True] * 3 + [False] * 13)
    assert clip_is_relevant([True] * 2, min_relevant=2)
