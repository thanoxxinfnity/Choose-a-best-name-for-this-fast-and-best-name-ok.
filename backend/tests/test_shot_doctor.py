"""Reading the footage itself: grade, repeats, and what to lead with."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puter_integration import ffmpeg_binary  # noqa: E402
from shot_doctor import (  # noqa: E402
    find_duplicates,
    grade_distance,
    hash_distance,
    match_colour,
    perceptual_hash,
    pick_thumbnail,
    score_frame,
)

FF = ffmpeg_binary()


def _noise(mean: float, spread: float, seed: int = 1, size=(80, 60)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, spread, (*size, 3)), 0, 255).astype(np.uint8)


# --------------------------------------------------------- colour match ----

def test_matching_moves_a_shot_onto_the_reference_grade():
    warm, cool = _noise(70, 25, 1), _noise(180, 55, 2)
    before = grade_distance(warm, cool)
    after = grade_distance(match_colour(warm, cool), cool)
    assert after < before / 4, f"{before:.1f} -> {after:.1f}"


def test_a_flat_shot_is_matched_too():
    """A solid background has no spread to rescale but still has a mean.

    Skipping the channel outright left flat footage at its original colour,
    which is the exact case a plain background hits.
    """
    warm = np.full((60, 40, 3), (40, 90, 190), dtype=np.uint8)
    cool = np.full((60, 40, 3), (190, 120, 50), dtype=np.uint8)
    assert grade_distance(match_colour(warm, cool), cool) < 1.0


def test_strength_scales_the_correction():
    a, b = _noise(70, 25, 3), _noise(180, 55, 4)
    full = grade_distance(match_colour(a, b, 1.0), b)
    half = grade_distance(match_colour(a, b, 0.5), b)
    none = grade_distance(match_colour(a, b, 0.0), b)
    assert full < half < none


def test_zero_strength_returns_the_frame_untouched():
    a, b = _noise(70, 25, 5), _noise(180, 55, 6)
    assert np.array_equal(match_colour(a, b, 0.0), a)


def test_matching_a_shot_to_itself_changes_almost_nothing():
    a = _noise(120, 40, 7)
    assert grade_distance(match_colour(a, a), a) < 0.5


def test_exposure_counts_as_part_of_the_grade():
    """A darker shot IS a cut that does not match, so the distance must see it.

    Two frames of random noise have nearly identical statistics whatever their
    colours, which is why the fixture uses real colour rather than noise.
    """
    base = np.full((60, 40, 3), (60, 120, 200), dtype=np.uint8)
    darker = np.clip(base.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    assert grade_distance(base, darker) > 5.0
    assert grade_distance(base, base) < 0.01


def test_a_matched_shot_ends_up_closer_than_an_unmatched_one():
    warm = np.full((60, 40, 3), (40, 90, 190), dtype=np.uint8)
    cool = np.full((60, 40, 3), (190, 120, 50), dtype=np.uint8)
    assert grade_distance(match_colour(warm, cool), cool) < grade_distance(warm, cool)


# ----------------------------------------------------------- duplicates ----

def test_a_frame_matches_itself_exactly():
    frame = _noise(120, 50, 10)
    assert hash_distance(perceptual_hash(frame), perceptual_hash(frame)) == 0


def _structured(size=(160, 120)) -> np.ndarray:
    """A frame with actual shapes in it.

    Random noise is the wrong fixture for a structural hash: downscaling
    averages noise into mush, so the hash legitimately changes. A hash that
    survived that would be measuring nothing.
    """
    import cv2

    frame = np.zeros((*size, 3), dtype=np.uint8)
    cv2.circle(frame, (size[1] // 2, size[0] // 2), size[0] // 4, (240, 240, 240), -1)
    cv2.rectangle(frame, (8, 8), (size[1] // 3, size[0] // 4), (200, 60, 60), -1)
    return frame


def test_a_rescaled_frame_still_matches():
    """The same shot at another size is still the same shot."""
    import cv2

    frame = _structured()
    smaller = cv2.resize(frame, (80, 60))
    assert hash_distance(perceptual_hash(frame), perceptual_hash(smaller)) <= 6


def test_two_different_shots_do_not_match():
    a, b = _noise(120, 50, 12), _noise(120, 50, 13)
    assert hash_distance(perceptual_hash(a), perceptual_hash(b)) > 6


def test_duplicates_are_reported_as_pairs():
    frame = _noise(120, 50, 14)
    hashes = [perceptual_hash(frame), 0, perceptual_hash(frame)]
    assert (0, 2) in find_duplicates(hashes)


def test_nothing_repeated_reports_nothing():
    hashes = [perceptual_hash(_noise(120, 50, seed)) for seed in (15, 16, 17)]
    assert find_duplicates(hashes) == []


# ----------------------------------------------------------- thumbnails ----

def test_a_black_frame_is_worth_nothing():
    assert score_frame(np.zeros((60, 40, 3), dtype=np.uint8))[0] == 0.0


def test_a_sharp_colourful_frame_beats_a_flat_one():
    flat = np.full((80, 60, 3), 120, dtype=np.uint8)
    lively = _noise(120, 60, 18)
    assert score_frame(lively)[0] > score_frame(flat)[0]


def test_a_face_helps_the_score():
    frame = _noise(120, 50, 19)
    assert score_frame(frame, faces=1)[0] > score_frame(frame, faces=0)[0]


def test_extra_faces_stop_helping_after_a_couple():
    frame = _noise(120, 50, 20)
    assert score_frame(frame, faces=9)[0] == score_frame(frame, faces=2)[0]


@pytest.mark.slow
def test_picking_a_thumbnail_skips_the_black_opening(tmp_path):
    """An edit that opens on black must not be led with a black frame."""
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [FF, "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
         "-f", "lavfi", "-i", "testsrc2=s=320x240:d=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    best = pick_thumbnail(clip, samples=24, detect_faces=False)
    assert best is not None
    assert best.seconds > 1.8, f"it led with the black opening at {best.seconds:.2f}s"
    assert best.score > 0.0


def test_a_missing_file_returns_nothing_rather_than_raising(tmp_path):
    assert pick_thumbnail(tmp_path / "nope.mp4") is None
