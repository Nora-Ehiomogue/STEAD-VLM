import numpy as np
import pytest

from stead_vlm.preprocessing import (center_crop_pad, fit_frame, minmax_normalise,
                                     prepare_clip, scale_short_side_crop)


def test_center_crop_takes_the_middle():
    f = np.arange(400 * 500 * 3, dtype=np.uint32).reshape(400, 500, 3) % 251
    f = f.astype(np.uint8)
    out = center_crop_pad(f, 320)
    assert out.shape == (320, 320, 3)
    assert np.array_equal(out, f[40:360, 90:410])


def test_small_frame_is_zero_padded_to_target():
    f = np.full((100, 200, 3), 255, np.uint8)
    out = center_crop_pad(f, 320)
    assert out.shape == (320, 320, 3)
    assert out[0, 0].sum() == 0 and out[160, 160].sum() == 255 * 3   # black border, image in the middle


def test_odd_sizes_still_give_exact_target():
    for h, w in [(319, 321), (321, 319), (50, 700), (700, 50)]:
        assert center_crop_pad(np.zeros((h, w, 3), np.uint8), 320).shape == (320, 320, 3)


def test_scale_crop_keeps_field_of_view_but_crop_pad_does_not():
    f = np.zeros((720, 1280, 3), np.uint8)
    f[:100] = 255                                    # bright band along the top of a 720p frame
    assert fit_frame(f, 320, "crop_pad").max() == 0          # centre crop misses the band entirely
    assert fit_frame(f, 320, "scale_crop")[:20].min() == 255  # rescaled view still contains it
    assert scale_short_side_crop(f, 320).shape == (320, 320, 3)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        fit_frame(np.zeros((10, 10, 3), np.uint8), 320, "bogus")


def test_minmax_range_and_constant_input():
    x = np.random.default_rng(0).integers(0, 255, (16, 8, 8, 3)).astype(np.uint8)
    y = minmax_normalise(x)
    assert y.dtype == np.float32 and y.min() == 0 and abs(y.max() - 1) < 1e-6
    assert np.all(minmax_normalise(np.full((2, 4, 4, 3), 7)) == 0)      # no NaN on flat clips


def test_prepare_clip_shape_dtype_and_rgb_order():
    frames = [np.zeros((400, 420, 3), np.uint8) for _ in range(16)]   # big enough: no padding
    for f in frames:
        f[..., 0] = 255                                # pure BLUE in OpenCV's BGR
    arr = prepare_clip(frames)
    assert arr.shape == (16, 320, 320, 3) and arr.dtype == np.float32
    assert arr[..., 2].min() > 0.99 and arr[..., 0].max() < 0.01   # blue is channel 2 in RGB


def test_prepare_clip_needs_16_frames():
    with pytest.raises(ValueError):
        prepare_clip([np.zeros((50, 50, 3), np.uint8)] * 15)
