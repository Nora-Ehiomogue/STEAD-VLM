from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from stead_vlm import data


def test_extract_clips_downsamples_to_5fps(make_video, tmp_path):
    v = make_video(n=200, fps=30)            # 34 sampled frames (every 6th) -> 2 clips of 16
    clips = data.extract_clips(v, tmp_path / "clips", "vid")
    assert len(clips) == 2
    cap = cv2.VideoCapture(clips[0])
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 16 and round(cap.get(cv2.CAP_PROP_FPS)) == 5
    cap.release()


def test_clip_to_npy_and_group_id(make_video, tmp_path):
    clip = data.extract_clips(make_video(n=100), tmp_path / "c", "storeA_cam3")[0]
    npy = data.clip_to_npy(clip, tmp_path / "npy")
    arr = np.load(npy)
    assert arr.shape == (16, 320, 320, 3) and arr.dtype == np.float32
    assert data.group_id_from_clip(Path(clip).name) == "storeA_cam3"


def test_local_records_carry_group_and_npy_path(tmp_path):
    pd.DataFrame({"clip": ["a_clip00001.mp4", "a_clip00002.mp4", "b_clip00001.mp4"],
                  "label": [0, 1, 0]}).to_csv(tmp_path / "l.csv", index=False)
    rows = data.local_records(tmp_path / "l.csv", "clips", "npy")
    assert [r["group_id"] for r in rows] == ["a", "a", "b"]
    assert rows[0]["npy_path"].endswith("a_clip00001.npy")
    rows = data.local_records(tmp_path / "l.csv", "clips", "npy", session_map={"a": "S1", "b": "S1"})
    assert {r["group_id"] for r in rows} == {"S1"}


def _manifest(n_groups=80, per=5, seed=0):
    rng = np.random.default_rng(seed)
    g_label = rng.random(n_groups) < 0.4
    rows = [{"clip_path": f"g{g}_c{i}.mp4", "npy_path": f"g{g}_c{i}.npy", "label": int(g_label[g]),
             "source": "x", "group_id": f"g{g}"} for g in range(n_groups) for i in range(per)]
    return pd.DataFrame(rows)


def test_split_is_group_disjoint_stratified_and_deterministic():
    m = _manifest()
    tr, te = data.group_stratified_split(m, 0.30, seed=1)
    assert set(tr.group_id).isdisjoint(te.group_id)
    assert len(tr) + len(te) == len(m)
    assert 0.2 < len(te) / len(m) < 0.4
    assert abs(te.label.mean() - m.label.mean()) < 0.1          # label ratio preserved
    tr2, te2 = data.group_stratified_split(m, 0.30, seed=1)
    assert te.clip_path.tolist() == te2.clip_path.tolist()


def _train_df(tmp_path, n=300, shape=(16, 24, 24, 3)):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = tmp_path / f"c{i}.npy"
        np.save(p, rng.random(shape).astype(np.float32))
        rows.append({"clip_path": f"c{i}.mp4", "npy_path": str(p), "label": i % 2, "source": "x",
                     "group_id": f"g{i}"})
    return pd.DataFrame(rows)


def test_augmentation_expansion_matches_thesis_2_5x(tmp_path):
    df = _train_df(tmp_path)
    aug = data.augment_training_set(df, tmp_path / "aug", seed=0)
    factor = len(aug) / len(df)
    assert 2.35 < factor < 2.65, factor
    assert (aug.variant == "original").sum() == len(df)
    a = np.load(aug[aug.variant == "jit"].npy_path.iloc[0])
    assert a.shape == (16, 24, 24, 3) and a.min() >= 0 and a.max() <= 1


def test_original_listing_behaviour_is_3_5x(tmp_path):
    df = _train_df(tmp_path)
    aug = data.augment_training_set(df, tmp_path / "aug", seed=0, p_flip=1.0, p_rot=1.0, p_jit=0.5)
    assert 3.35 < len(aug) / len(df) < 3.65


def test_augmentation_never_touches_heldout_clips(tmp_path):
    df = _train_df(tmp_path, n=20)
    heldout = pd.DataFrame({"clip_path": ["test1.mp4", "test2.mp4"]})
    aug = data.augment_training_set(df, tmp_path / "aug", test_df=heldout)
    assert set(aug.source_clip) <= set(df.clip_path)
    with pytest.raises(AssertionError):
        data.augment_training_set(df, tmp_path / "aug2", test_df=df.iloc[:3])


def test_horizontal_flip_and_rotate_shapes():
    clip = np.random.default_rng(0).random((16, 32, 32, 3)).astype(np.float32)
    assert np.array_equal(data.horizontal_flip(data.horizontal_flip(clip)), clip)
    assert data.rotate(clip, 10).shape == clip.shape
    assert data.jitter(clip, 1.2, 25).max() <= 1.0
