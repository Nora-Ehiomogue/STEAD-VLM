import numpy as np

from stead_vlm.pipeline import CascadePipeline, ClipBuffer


def test_clipbuffer_samples_every_6th_frame_at_30fps():
    buf, clips = ClipBuffer(30, hop=16), []
    for i in range(200):
        c = buf.push(np.full((2, 2, 3), i, np.uint8))
        if c is not None:
            clips.append([int(f[0, 0, 0]) for f in c])
    assert len(clips) == 2
    assert clips[0] == list(range(0, 96, 6))          # 16 frames spanning ~3.2 s of video


def test_clipbuffer_overlap_with_smaller_hop():
    buf, n = ClipBuffer(30, hop=8), 0
    for i in range(200):
        n += buf.push(np.zeros((2, 2, 3), np.uint8)) is not None
    assert n == 3


class FakeStage0:
    def __init__(self, ok): self.ok = ok
    def clip_flag(self, frames): return self.ok


class FakeDetector:
    def __init__(self, score): self.score, self.calls = score, 0
    def features_array(self, arr): self.calls += 1; return arr
    def score_features(self, feats): return self.score
    def cam_from_features(self, feats): return np.linspace(0, 1, 100).reshape(10, 10)


class FakeExplainer:
    def __init__(self): self.seen = None
    def explain(self, img):
        self.seen = img
        return {"description": "person hides item", "confidence": 0.8}, '{"description": "..."}'


def _frames(n=16):
    return [np.full((260, 340, 3), 10 * i, np.uint8) for i in range(n)]


def test_stage0_rejection_skips_everything_else():
    det = FakeDetector(0.99)
    r = CascadePipeline(FakeStage0(False), det).process_clip(_frames())
    assert not r.passed_stage0 and r.score == 0.0 and not r.escalated and det.calls == 0


def test_escalation_runs_gradcam_and_explainer_on_aligned_view():
    exp = FakeExplainer()
    r = CascadePipeline(FakeStage0(True), FakeDetector(0.9), exp).process_clip(_frames())
    assert r.escalated and r.explanation["confidence"] == 0.8
    assert r.evidence_bgr.shape == (320, 320, 3) and exp.seen is r.evidence_bgr
    assert set(r.stage_ms) == {"stage0", "stage1", "stage2"}


def test_threshold_is_strict_and_stage2_never_changes_the_score():
    exp = FakeExplainer()
    at = CascadePipeline(FakeStage0(True), FakeDetector(0.65), exp).process_clip(_frames())
    assert not at.escalated and exp.seen is None          # score == theta is NOT escalated
    hi = CascadePipeline(FakeStage0(True), FakeDetector(0.7), exp).process_clip(_frames())
    assert hi.score == 0.7


def test_works_without_explainer():
    r = CascadePipeline(FakeStage0(True), FakeDetector(0.9), None).process_clip(_frames())
    assert r.escalated and r.explanation is None and r.evidence_bgr is not None


def test_iter_video_yields_a_result_per_clip(make_video):
    v = make_video(n=200, fps=30)
    out = list(CascadePipeline(FakeStage0(True), FakeDetector(0.1)).iter_video(v))
    assert len(out) == 2 and out[0][0] < out[1][0]
