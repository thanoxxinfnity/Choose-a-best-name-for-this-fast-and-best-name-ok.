"""Building one long video out of many short generated clips."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sequence as sq  # noqa: E402
from sequence import (  # noqa: E402
    ASSUMED_SECONDS_PER_CLIP,
    SequenceState,
    Shot,
    _render_seconds_from,
    build_sequence,
    estimate,
    stitch,
)


class _Response:
    def __init__(self, code, payload):
        self.status_code = code
        self._payload = payload

    def json(self):
        return self._payload


# ------------------------------------------------------------ the estimate --

def test_the_render_rate_is_read_not_the_acceptance_rate():
    """Acceptance is the number a service advertises and the one that does not
    matter: ten a minute in, one every ten minutes out."""
    stats = {"data": {"capacity_rpm":
                      "unlimited acceptance (10+ RPM) · provider free tier "
                      "renders ~1 per 10 min, FIFO"}}
    with patch.object(sq.requests, "get", lambda *a, **k: _Response(200, stats)):
        result = estimate(96, 5.0)

    assert result.render_rate_seconds == pytest.approx(600.0)
    assert result.wall_clock_seconds == pytest.approx(96 * 600.0)
    assert "reported" in result.source


def test_eight_minutes_of_video_is_named_as_sixteen_hours_of_waiting():
    stats = {"data": {"capacity_rpm": "renders ~1 per 10 min"}}
    with patch.object(sq.requests, "get", lambda *a, **k: _Response(200, stats)):
        told = estimate(96, 5.0).describe()
    assert "8.0 minutes of video" in told
    assert "16.0 hours" in told


def test_an_unreadable_endpoint_falls_back_to_the_pessimistic_number():
    """A promise that comes in early is a good surprise; the reverse is not."""
    def boom(*_a, **_k):
        raise OSError("unreachable")

    with patch.object(sq.requests, "get", boom):
        result = estimate(10)
    assert result.render_rate_seconds == ASSUMED_SECONDS_PER_CLIP
    assert "assumed" in result.source


def test_prose_it_does_not_recognise_leaves_the_default_alone():
    """Better a pessimistic default than a number invented from a phrase."""
    stats = {"data": {"capacity_rpm": "very fast, honestly"}}
    with patch.object(sq.requests, "get", lambda *a, **k: _Response(200, stats)):
        assert estimate(10).render_rate_seconds == ASSUMED_SECONDS_PER_CLIP


@pytest.mark.parametrize("text,expected", [
    ("renders ~1 per 10 min, FIFO", 600.0),
    ("about 2 per minute", 30.0),
    ("1 per hour", 3600.0),
    ("30 per 60 sec", 2.0),
])
def test_rates_are_parsed_from_the_shapes_the_service_uses(text, expected):
    assert _render_seconds_from(text) == pytest.approx(expected)


def test_a_phrase_with_no_rate_in_it_parses_to_nothing():
    assert _render_seconds_from("unlimited acceptance") is None
    assert _render_seconds_from("") is None


# --------------------------------------------------------------- building ---

class _Provider:
    """Records what it was asked for, and can be told to fail."""

    def __init__(self, fail_on=()):
        self.made = []
        self.fail_on = set(fail_on)

    def _make(self, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"\x00" * 2048)

    def text_to_video(self, prompt, destination, seconds=5.0, **kwargs):
        if len(self.made) in self.fail_on:
            self.made.append(None)
            raise RuntimeError("provider said no")
        self.made.append(prompt)
        self._make(destination)

    def image_to_video(self, image, prompt, destination, seconds=5.0, **kwargs):
        self.made.append(f"i2v:{prompt}")
        self._make(destination)


def _shots(count):
    return [Shot(prompt=f"shot {i}", index=i) for i in range(count)]


def test_every_shot_is_rendered_in_order(tmp_path):
    provider = _Provider()
    state = build_sequence(_shots(4), tmp_path, provider)
    assert provider.made == ["shot 0", "shot 1", "shot 2", "shot 3"]
    assert len(state.done) == 4


def test_a_restart_does_not_remake_what_already_exists(tmp_path):
    """A job measured in hours will be interrupted, and re-rendering finished
    clips is the most expensive possible way to recover."""
    build_sequence(_shots(3), tmp_path, _Provider())

    again = _Provider()
    state = build_sequence(_shots(3), tmp_path, again)
    assert again.made == []
    assert len(state.done) == 3


def test_a_clip_whose_file_vanished_is_remade(tmp_path):
    build_sequence(_shots(2), tmp_path, _Provider())
    Path(tmp_path / "clips" / "shot_001.mp4").unlink()

    again = _Provider()
    build_sequence(_shots(2), tmp_path, again)
    assert again.made == ["shot 1"]


def test_one_bad_shot_does_not_lose_the_rest(tmp_path):
    provider = _Provider(fail_on={1})
    state = build_sequence(_shots(4), tmp_path, provider)
    assert len(state.done) == 3
    assert 1 in state.failed


def test_failures_in_a_row_stop_the_run(tmp_path):
    """That pattern is the service, not the shots, and pressing on wastes hours."""
    provider = _Provider(fail_on={0, 1, 2})
    with pytest.raises(RuntimeError, match="in a row"):
        build_sequence(_shots(6), tmp_path, provider, stop_after_failures=3)


def test_what_was_made_before_a_stop_is_kept(tmp_path):
    provider = _Provider(fail_on={1, 2, 3})
    with pytest.raises(RuntimeError):
        build_sequence(_shots(6), tmp_path, provider, stop_after_failures=3)

    state = SequenceState(tmp_path).load()
    assert 0 in state.done


def test_a_shot_with_a_still_uses_image_to_video(tmp_path):
    provider = _Provider()
    build_sequence([Shot(prompt="animate this", image="frame.png", index=0)],
                   tmp_path, provider)
    assert provider.made == ["i2v:animate this"]


def test_unreadable_state_starts_fresh_rather_than_failing(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "sequence_state.json").write_text("{not json", encoding="utf-8")
    assert SequenceState(tmp_path).load().done == {}


# -------------------------------------------------------------- stitching ---

def test_nothing_made_means_nothing_stitched(tmp_path):
    assert stitch(SequenceState(tmp_path), _shots(3), tmp_path / "out.mp4") is None


def test_only_clips_that_exist_are_joined(tmp_path):
    state = SequenceState(tmp_path)
    state.done = {0: str(tmp_path / "gone.mp4")}
    assert stitch(state, _shots(1), tmp_path / "out.mp4") is None
