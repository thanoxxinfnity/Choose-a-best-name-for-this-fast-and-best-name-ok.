"""Cutting the subject out of a moving clip without flicker or stepping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enhancers import _smooth_masks  # noqa: E402


def _mask(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


# --------------------------------------------------------- temporal median --

def test_a_single_bad_matte_is_removed():
    """A segmenter run per sampled frame occasionally drops a limb.

    Held or interpolated, that one frame becomes a visible pop.
    """
    keys = [(0, _mask(255)), (6, _mask(0)), (12, _mask(255)), (18, _mask(255))]
    smoothed = _smooth_masks(keys)
    # The dropout at index 6 is replaced by its neighbours' median.
    assert int(smoothed[1][1].mean()) == 255


def test_a_real_change_survives_the_median():
    """A change present in two frames running is the subject, not an error."""
    keys = [(0, _mask(0)), (6, _mask(255)), (12, _mask(255)), (18, _mask(255))]
    smoothed = _smooth_masks(keys)
    assert int(smoothed[1][1].mean()) == 255
    assert int(smoothed[2][1].mean()) == 255


def test_the_frame_indices_are_preserved():
    keys = [(0, _mask(10)), (7, _mask(20)), (14, _mask(30))]
    assert [index for index, _ in _smooth_masks(keys)] == [0, 7, 14]


def test_the_ends_are_kept_as_they_are():
    """There is no neighbour on the far side to vote with."""
    keys = [(0, _mask(5)), (6, _mask(200)), (12, _mask(9))]
    smoothed = _smooth_masks(keys)
    assert int(smoothed[0][1].mean()) == 5
    assert int(smoothed[-1][1].mean()) == 9


def test_too_few_masks_are_left_alone():
    keys = [(0, _mask(1)), (6, _mask(2))]
    assert _smooth_masks(keys) == keys
    assert _smooth_masks([]) == []


def test_masks_of_different_sizes_do_not_crash_the_median():
    """They should never differ in practice, but a mismatch must not raise.

    Each smoothed mask keeps its own key's shape; the neighbours voting on it
    are brought to that shape rather than the other way round.
    """
    keys = [(0, _mask(255, 16)), (6, _mask(0, 12)), (12, _mask(255, 16))]
    smoothed = _smooth_masks(keys)
    assert smoothed[1][1].shape == (12, 12)
    assert int(smoothed[1][1].mean()) == 255


# ------------------------------------------------------------ interpolation --

def _alpha_lookup(keys):
    """The blend the compositor uses, lifted out for testing."""
    import bisect

    import cv2

    indices = [index for index, _ in keys]

    def alpha_at(frame_index, shape):
        position = bisect.bisect_right(indices, frame_index) - 1
        position = min(max(position, 0), len(keys) - 1)
        left_index, left = keys[position]
        if position + 1 < len(keys) and frame_index > left_index:
            right_index, right = keys[position + 1]
            span = max(right_index - left_index, 1)
            weight = min(max((frame_index - left_index) / span, 0.0), 1.0)
            blended = left.astype(np.float32) * (1 - weight) + right.astype(np.float32) * weight
        else:
            blended = left.astype(np.float32)
        return cv2.resize(blended, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    return alpha_at


def test_the_matte_crosses_between_keys_rather_than_freezing():
    """Holding one matte until the next arrives freezes the cut-out edge for
    several frames and then jumps it - on a moving subject that reads as the
    character sliding around inside their own outline."""
    keys = [(0, _mask(0)), (10, _mask(255))]
    alpha_at = _alpha_lookup(keys)

    values = [float(alpha_at(i, (16, 16)).mean()) for i in range(11)]
    # Strictly increasing: every frame moves, none of them are held.
    assert all(b > a for a, b in zip(values, values[1:])), values
    assert values[0] == pytest.approx(0, abs=1)
    assert values[-1] == pytest.approx(255, abs=1)


def test_the_midpoint_is_halfway_between_its_neighbours():
    keys = [(0, _mask(0)), (10, _mask(200))]
    assert float(_alpha_lookup(keys)(5, (16, 16)).mean()) == pytest.approx(100, abs=2)


def test_before_the_first_key_the_first_matte_is_used():
    keys = [(4, _mask(90)), (10, _mask(200))]
    assert float(_alpha_lookup(keys)(0, (16, 16)).mean()) == pytest.approx(90, abs=1)


def test_after_the_last_key_the_last_matte_is_used():
    keys = [(0, _mask(90)), (10, _mask(200))]
    assert float(_alpha_lookup(keys)(50, (16, 16)).mean()) == pytest.approx(200, abs=1)


def test_the_matte_is_upscaled_to_the_frame():
    keys = [(0, _mask(128, 8)), (10, _mask(128, 8))]
    assert _alpha_lookup(keys)(5, (64, 48)).shape == (64, 48)


# ------------------------------------------------------- refusing to guess --

def _keys(*coverages, size=20):
    """Key masks whose opaque share matches each given fraction."""
    out = []
    for index, cover in enumerate(coverages):
        mask = np.zeros((size, size), np.uint8)
        rows = int(round(size * cover))
        if rows:
            mask[:rows] = 255
        out.append((index * 6, mask))
    return out


def test_a_steady_subject_reads_as_confident():
    from enhancers import matte_confidence

    coverage, swing = matte_confidence(_keys(0.30, 0.32, 0.29, 0.31))
    assert coverage == pytest.approx(0.30, abs=0.03)
    assert swing < 0.05


def test_a_subject_that_keeps_vanishing_reads_as_unstable():
    """What shredded output looks like before it is rendered: half the frame
    one sample, nothing the next."""
    from enhancers import matte_confidence

    _coverage, swing = matte_confidence(_keys(0.35, 0.0, 0.30, 0.0))
    assert swing > 0.2


def test_finding_almost_nothing_reads_as_no_subject():
    from enhancers import matte_confidence

    coverage, _swing = matte_confidence(_keys(0.01, 0.0, 0.01, 0.0))
    assert coverage < 0.02


def test_no_keys_is_the_least_confident_answer():
    from enhancers import matte_confidence

    assert matte_confidence([]) == (0.0, 1.0)


def test_a_single_key_has_nothing_to_swing_against():
    from enhancers import matte_confidence

    coverage, swing = matte_confidence(_keys(0.4))
    assert coverage == pytest.approx(0.4, abs=0.03)
    assert swing == 0.0


def test_an_unstable_matte_refuses_rather_than_shipping_a_shredded_subject(monkeypatch):
    """Every model tried on an already-composited edit returned a different
    answer on every frame; compositing that produces a torn character."""
    import enhancers
    from enhancers import MatteTooUnstable, ai_remove_background

    monkeypatch.setattr(
        enhancers, "_matte_keyframes",
        lambda *a, **k: (_keys(0.35, 0.0, 0.30, 0.0), 30.0, 120),
    )
    with pytest.raises(MatteTooUnstable, match="could not be tracked"):
        ai_remove_background(Path("in.mp4"), Path("out.mp4"))


def test_a_confident_matte_is_allowed_through(monkeypatch, tmp_path):
    import enhancers
    from enhancers import ai_remove_background

    monkeypatch.setattr(
        enhancers, "_matte_keyframes",
        lambda *a, **k: (_keys(0.30, 0.31, 0.30, 0.29), 30.0, 120),
    )
    seen = {}
    monkeypatch.setattr(
        enhancers, "_process_frames",
        lambda source, destination, process, ffmpeg=None: seen.setdefault("ran", True)
        or destination,
    )
    ai_remove_background(Path("in.mp4"), tmp_path / "out.mp4")
    assert seen.get("ran") is True


# ------------------------------------------- a moving background, not a still --

def _clip(path, colour, seconds=1.0, fps=10):
    """A tiny video of one flat colour, so a frame identifies its source."""
    import subprocess

    from puter_integration import ffmpeg_binary

    subprocess.run(
        [ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i",
         f"color=c={colour}:size=64x64:rate={fps}:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def test_a_still_background_still_works(tmp_path):
    import cv2

    from enhancers import _BackgroundSource

    path = tmp_path / "bg.png"
    cv2.imwrite(str(path), np.full((32, 32, 3), (200, 30, 30), np.uint8))
    source = _BackgroundSource(path, (0, 0, 0))
    frame = source.frame_at(0.0, (64, 48))
    assert frame.shape == (64, 48, 3)
    assert int(frame[..., 2].mean()) > 150  # BGR blue channel became RGB red


def test_a_video_background_serves_moving_frames(tmp_path):
    """A still was all this supported, which is the wrong shape for putting
    generated animation behind a character."""
    from enhancers import _BackgroundSource

    source = _BackgroundSource(_clip(tmp_path / "bg.mp4", "green", seconds=1.0), (0, 0, 0))
    assert source.capture is not None
    frame = source.frame_at(0.5, (32, 24))
    assert frame.shape == (32, 24, 3)
    assert int(frame[..., 1].mean()) > int(frame[..., 2].mean())
    source.close()


def test_the_background_is_followed_by_time_not_frame_count(tmp_path):
    """A 10fps background behind 60fps footage must not run at a sixth speed."""
    from enhancers import _BackgroundSource

    source = _BackgroundSource(_clip(tmp_path / "bg.mp4", "blue", seconds=2.0, fps=10),
                               (0, 0, 0))
    assert source.fps == pytest.approx(10.0, abs=0.5)
    # One second in is frame ten of a ten-fps clip, whatever the foreground runs at.
    source.frame_at(1.0, (16, 16))
    assert source._position == 10
    source.close()


def test_a_short_background_loops_under_longer_footage(tmp_path):
    from enhancers import _BackgroundSource

    source = _BackgroundSource(_clip(tmp_path / "bg.mp4", "red", seconds=1.0, fps=10),
                               (0, 0, 0))
    # Well past the end of a one second clip; it should wrap rather than fail.
    frame = source.frame_at(7.3, (16, 16))
    assert frame is not None and frame.shape == (16, 16, 3)
    source.close()


def test_no_background_is_the_fill_colour(tmp_path):
    from enhancers import _BackgroundSource

    frame = _BackgroundSource(None, (7, 9, 11)).frame_at(0.0, (8, 8))
    assert frame.shape == (8, 8, 3)
    assert tuple(int(v) for v in frame[0, 0]) == (7, 9, 11)


def test_something_that_is_neither_falls_back_to_the_fill(tmp_path):
    from enhancers import _BackgroundSource

    junk = tmp_path / "nope.mp4"
    junk.write_bytes(b"not a video")
    source = _BackgroundSource(junk, (5, 5, 5))
    assert source.capture is None and source.still is None
    assert tuple(int(v) for v in source.frame_at(0.0, (8, 8))[0, 0]) == (5, 5, 5)
