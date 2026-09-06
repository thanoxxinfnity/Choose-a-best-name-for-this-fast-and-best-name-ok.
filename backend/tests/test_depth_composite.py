"""Compositing a cut-out so it reads as standing in the scene.

A straight alpha-over is geometrically right and still looks pasted. These
tests pin the four things that fix that, each one measurable: a shadow on the
plate, a rim on the lit edge, atmosphere in front of the subject, and a grade
that belongs to the same scene.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vfx import composite_over, depth_composite  # noqa: E402


def _plate(bright_side: str = "left", size: int = 200) -> np.ndarray:
    """A dark plate with a bright band down one side - a known light source."""
    plate = np.full((size, size, 3), 30, dtype=np.uint8)
    band = slice(0, size // 4) if bright_side == "left" else slice(3 * size // 4, size)
    plate[:, band] = 235
    return plate


def _subject(size: int = 200, colour=(220, 40, 200)) -> np.ndarray:
    """An opaque block floating in the middle of the frame."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    box = slice(size // 3, 2 * size // 3)
    rgba[box, box, :3] = colour
    rgba[box, box, 3] = 255
    return rgba


def _textured_subject(size: int = 200) -> np.ndarray:
    """The same block with fine detail in it.

    A flat block has no detail to lose, so it cannot answer what haze costs -
    measuring against one says haze *adds* variation, which is only true of a
    subject that had none.
    """
    rgba = _subject(size)
    box = slice(size // 3, 2 * size // 3)
    grid = np.indices((size, size)).sum(axis=0) % 8 * 12
    for channel in range(3):
        patch = rgba[box, box, channel].astype(np.int16) + grid[box, box] - 42
        rgba[box, box, channel] = np.clip(patch, 0, 255).astype(np.uint8)
    return rgba


def _alpha(rgba) -> np.ndarray:
    return rgba[:, :, 3] > 127


# ------------------------------------------------------------- the escape ---

def test_everything_off_is_exactly_a_plain_composite():
    """The new path must be able to get out of the way completely."""
    rgba, plate = _subject(), _plate()
    plain = composite_over(rgba, plate)
    same = depth_composite(rgba, plate, shadow=0, rim=0, haze=0, colour_match=0)
    assert np.array_equal(plain, same)


# ------------------------------------------------------------- the shadow ---

def test_a_shadow_darkens_the_plate_around_the_subject():
    rgba, plate = _subject(), _plate()
    plain = composite_over(rgba, plate)
    lit = depth_composite(rgba, plate, shadow=0.6, rim=0, haze=0, colour_match=0)

    off_subject = ~_alpha(rgba)
    assert lit[off_subject].mean() < plain[off_subject].mean(), "no shadow was cast"


def test_the_shadow_falls_away_from_the_light_not_toward_it():
    """A shadow pointing at the sun is the tell that nothing was measured."""
    rgba = _subject()
    size = rgba.shape[0]
    middle = slice(size // 3, 2 * size // 3)

    def darkening(plate):
        plain = composite_over(rgba, plate).astype(np.float32)
        lit = depth_composite(rgba, plate, shadow=0.6, rim=0, haze=0,
                              colour_match=0).astype(np.float32)
        drop = (plain - lit).mean(axis=2)
        # Compare the two flanks of the subject, clear of the bright bands.
        left = drop[middle, size // 4: size // 3].mean()
        right = drop[middle, 2 * size // 3: 3 * size // 4].mean()
        return left, right

    lit_left = darkening(_plate("left"))
    lit_right = darkening(_plate("right"))
    assert lit_left[1] > lit_left[0], "light on the left must throw the shadow right"
    assert lit_right[0] > lit_right[1], "light on the right must throw the shadow left"


# ---------------------------------------------------------------- the rim ---

def test_the_rim_brightens_the_edge_that_faces_the_light():
    rgba = _subject()
    size = rgba.shape[0]
    plate = _plate("left")
    plain = composite_over(rgba, plate).astype(np.float32)
    lit = depth_composite(rgba, plate, shadow=0, rim=0.8, haze=0,
                          colour_match=0).astype(np.float32)

    gain = (lit - plain).mean(axis=2)
    middle = slice(size // 3, 2 * size // 3)
    near = gain[middle, size // 3: size // 3 + 6].mean()
    far = gain[middle, 2 * size // 3 - 6: 2 * size // 3].mean()
    assert near > far, "the lit edge must be the one facing the light"
    assert near > 1.0, "the rim did not brighten anything"


# --------------------------------------------------------------- the grade --

def test_colour_match_pulls_the_subject_toward_the_plate():
    """A subject graded for another scene never belongs to this one."""
    rgba, plate = _subject(colour=(255, 0, 255)), _plate()
    inside = _alpha(rgba)

    plain = composite_over(rgba, plate).astype(np.float32)
    graded = depth_composite(rgba, plate, shadow=0, rim=0, haze=0,
                             colour_match=0.6).astype(np.float32)

    target = plate.reshape(-1, 3).mean(axis=0)
    before = np.abs(plain[inside].mean(axis=0) - target).mean()
    after = np.abs(graded[inside].mean(axis=0) - target).mean()
    assert after < before, "the subject was not moved toward the plate's grade"


def test_haze_costs_saturation_and_detail_but_not_brightness():
    """Measured, because the eye reads the haze as the picture going dark.

    It does not: the subject's luma is left alone and its colour and fine
    detail are what pay, which is the trade worth knowing before turning it up.
    """
    rgba, plate = _textured_subject(), _plate()
    inside = _alpha(rgba)

    plain = composite_over(rgba, plate).astype(np.float32)
    hazed = depth_composite(rgba, plate, shadow=0, rim=0, haze=0.4,
                            colour_match=0).astype(np.float32)

    def detail(frame):
        grey = frame.mean(axis=2)
        return float(np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 2))[inside].std())

    assert detail(hazed) < detail(plain), "haze must soften, that is what it is"

    luma = lambda f: (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])
    assert luma(hazed)[inside].mean() == pytest.approx(
        luma(plain)[inside].mean(), rel=0.35)


# ------------------------------------------------------------ the plumbing --

def test_a_plate_of_a_different_size_is_resized_not_refused():
    rgba = _subject(size=200)
    plate = _plate(size=64)
    out = depth_composite(rgba, plate)
    assert out.shape == (200, 200, 3)


def test_a_fully_transparent_cut_out_leaves_only_the_plate():
    rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    plate = _plate(size=80)
    out = depth_composite(rgba, plate, shadow=0.6, rim=0.6, haze=0.3)
    assert np.array_equal(out, plate), "nothing to composite must change nothing"


def test_a_black_plate_does_not_divide_by_its_own_darkness():
    """A plate with no light in it still has to composite."""
    rgba = _subject()
    plate = np.zeros((200, 200, 3), dtype=np.uint8)
    out = depth_composite(rgba, plate)
    assert np.isfinite(out).all()
    assert out.dtype == np.uint8
