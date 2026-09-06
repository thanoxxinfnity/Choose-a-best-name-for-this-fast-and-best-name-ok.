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
