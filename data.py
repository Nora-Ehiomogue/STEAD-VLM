"""Dataset preparation (thesis Section 3.1, Appendix A): clip extraction, manifest,
group-aware split, training-set augmentation.

Only numpy / OpenCV / pandas / scikit-learn are needed here (MediaPipe is imported lazily
for face blurring).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CLIP_LEN, SAMPLE_FPS, SEED
from .preprocessing import prepare_clip, read_frames

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_CLIP_SUFFIX = re.compile(r"_clip\d+$")


def list_videos(directory) -> list[Path]:
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in VIDEO_EXTS)


# ----------------------------------------------------------------------------- A.2
def audit_footage(raw_dir) -> dict:
    """Count videos, duration and raw frames (used to verify the ~85 h / ~9.18 M frames)."""
    n_videos, frames, seconds = 0, 0, 0.0
    for v in list_videos(raw_dir):
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        n_videos += 1
        frames += n
        seconds += n / fps
    return {"videos": n_videos, "frames": frames, "hours": seconds / 3600}


# ----------------------------------------------------------------------------- A.3
def blur_faces_in_video(in_path, out_path, conf: float = 0.5) -> None:
    """Anonymise a video by Gaussian-blurring every MediaPipe face detection."""
    import mediapipe as mp                                   # lazy import
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"cannot write video: {out_path}")
    with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=conf) as fd:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for det in (res.detections or []):
                b = det.location_data.relative_bounding_box
                x1, y1 = max(int(b.xmin * w), 0), max(int(b.ymin * h), 0)
                x2, y2 = min(x1 + int(b.width * w), w), min(y1 + int(b.height * h), h)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
            out.write(frame)
    cap.release()
    out.release()


# ----------------------------------------------------------------------------- A.4
def extract_clips(video_path, out_dir, prefix: str, clip_len: int = CLIP_LEN,
                  sample_fps: float = SAMPLE_FPS) -> list[str]:
    """Temporal down-sampling to `sample_fps` and cutting into non-overlapping clips of
    `clip_len` frames, written at native resolution. Returns the clip paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(native_fps / sample_fps)), 1)      # 30 / 5 -> every 6th frame
    buf, idx, clip_id, saved = [], 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            buf.append(frame)
            if len(buf) == clip_len:
                path = Path(out_dir) / f"{prefix}_clip{clip_id:05d}.mp4"
                vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     sample_fps, (buf[0].shape[1], buf[0].shape[0]))
                for f in buf:
                    vw.write(f)
                vw.release()
                saved.append(str(path))
                buf, clip_id = [], clip_id + 1
        idx += 1
    cap.release()
    return saved


# ----------------------------------------------------------------------------- A.5
def clip_to_npy(clip_path, npy_dir, mode: str = "crop_pad") -> str:
    """Pre-process one clip and cache it as float32 (16, 320, 320, 3) in [0, 1]."""
    Path(npy_dir).mkdir(parents=True, exist_ok=True)
    arr = prepare_clip(read_frames(clip_path), mode=mode)
    out = Path(npy_dir) / (Path(clip_path).stem + ".npy")
    np.save(out, arr)
    return str(out)


# ------------------------------------------------------------------------- A.6 / A.7
def group_id_from_clip(clip_name: str) -> str:
    """'storeA_cam3_clip00012.mp4' -> 'storeA_cam3' (the source recording)."""
    return _CLIP_SUFFIX.sub("", Path(clip_name).stem)


def local_records(labels_csv, clip_dir, npy_dir, session_map: dict | None = None) -> list[dict]:
    """Records for the manually labelled local clips. `labels_csv` has columns clip,label.
    `session_map` optionally maps a source-recording id to a recording-session id when one
    session spans several files (the split is done at session level)."""
    rows = []
    for _, r in pd.read_csv(labels_csv).iterrows():
        gid = group_id_from_clip(r["clip"])
        gid = (session_map or {}).get(gid, gid)
        rows.append({"clip_path": str(Path(clip_dir) / r["clip"]),
                     "npy_path": str(Path(npy_dir) / (Path(r["clip"]).stem + ".npy")),
                     "label": int(r["label"]), "source": "local_benin", "group_id": gid})
    return rows


def ucf_records(ucf_dir, clip_dir, npy_dir, anomaly_classes=("Shoplifting", "Stealing",
                "Burglary", "Vandalism"), normal_dir: str = "Normal",
                mode: str = "crop_pad") -> list[dict]:
    """Cut the selected UCF-Crime videos into clips and label them at VIDEO level
    (anomalous class folder = 1, normal folder = 0). Note that video-level labels are weak:
    an anomalous video also contains normal moments."""
    rows = []
    for cls, label in [(c, 1) for c in anomaly_classes] + [(normal_dir, 0)]:
        folder = Path(ucf_dir) / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"UCF-Crime folder not found: {folder}")
        for video in list_videos(folder):
            gid = f"ucf_{cls}_{video.stem}"
            for clip in extract_clips(video, clip_dir, gid):
                rows.append({"clip_path": clip, "npy_path": clip_to_npy(clip, npy_dir, mode),
                             "label": label, "source": "ucf_crime", "group_id": gid})
    return rows


# ----------------------------------------------------------------------------- A.8
def group_stratified_split(manifest: pd.DataFrame, test_frac: float = 0.30,
                           seed: int = SEED, n_folds: int = 10):
    """Label-stratified, group-aware train/test split.

    `GroupShuffleSplit` (used in the original listing) ignores labels, so it is NOT
    stratified. StratifiedGroupKFold keeps each group (recording session / source video) in
    exactly one fold and balances the label ratio; `round(test_frac * n_folds)` folds form
    the test set.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    n_test = round(test_frac * n_folds)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_idx = []
    for k, (_, te) in enumerate(skf.split(manifest, manifest["label"], manifest["group_id"])):
        if k < n_test:
            test_idx.extend(te)
    mask = np.zeros(len(manifest), dtype=bool)
    mask[test_idx] = True
    train, test = manifest[~mask].reset_index(drop=True), manifest[mask].reset_index(drop=True)
    assert set(train["group_id"]).isdisjoint(test["group_id"]), "group leakage across split"
    return train, test


# ----------------------------------------------------------------------------- A.9
def horizontal_flip(clip: np.ndarray) -> np.ndarray:
    return clip[:, :, ::-1, :].copy()


def rotate(clip: np.ndarray, theta: float) -> np.ndarray:
    _, h, w = clip.shape[:3]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    return np.stack([cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                     for f in clip])


def jitter(clip: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """I' = alpha * I + beta/255 on [0, 1] data, clipped to [0, 1]."""
    return np.clip(alpha * clip.astype(np.float32) + beta / 255.0, 0.0, 1.0)


def make_variants(clip, rng, p_flip=0.5, p_rot=0.5, p_jit=0.5, theta_max=15.0,
                  alpha_rng=(0.8, 1.2), beta_rng=(-25.0, 25.0)):
    """Yield (tag, augmented_clip). Each operation is applied independently with the given
    probability. With p = 0.5 for all three the mean expansion factor is 1 + 1.5 = 2.5x
    (the value stated in thesis Section 3.1.3.4). The original listing applied flip and
    rotation to EVERY clip and jitter to half, which gives 3.5x -- pass p_flip=p_rot=1.0 to
    reproduce that."""
    if rng.random() < p_flip:
        yield "flip", horizontal_flip(clip)
    if rng.random() < p_rot:
        yield "rot", rotate(clip, float(rng.uniform(-theta_max, theta_max)))
    if rng.random() < p_jit:
        yield "jit", jitter(clip, float(rng.uniform(*alpha_rng)), float(rng.uniform(*beta_rng)))


def augment_training_set(train_df: pd.DataFrame, aug_dir, seed: int = SEED,
                         test_df: pd.DataFrame | None = None, **probs) -> pd.DataFrame:
    """Augment TRAINING clips only. Returns the training manifest with the originals plus
    the new variants (columns `source_clip`, `variant` added)."""
    Path(aug_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in train_df.iterrows():
        rows.append({**row.to_dict(), "source_clip": row["clip_path"], "variant": "original"})
        clip = np.load(row["npy_path"])
        for tag, aug in make_variants(clip, rng, **probs):
            out = Path(aug_dir) / f"{Path(row['npy_path']).stem}_{tag}.npy"
            np.save(out, aug.astype(np.float32))
            rows.append({**row.to_dict(), "npy_path": str(out),
                         "source_clip": row["clip_path"], "variant": tag})
    aug_df = pd.DataFrame(rows)
    if test_df is not None:
        assert set(aug_df["source_clip"]).isdisjoint(set(test_df["clip_path"])), \
            "augmentation leaked a held-out clip into the training set"
    return aug_df
