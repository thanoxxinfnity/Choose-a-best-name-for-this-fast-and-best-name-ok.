"""The keyless Z.ai video endpoint, without touching the network."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_providers as vp  # noqa: E402
from video_providers import (  # noqa: E402
    DEFAULT_PROVIDER,
    ProviderError,
    VideoForgeProvider,
    get_provider,
)


class _Response:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.content = content
        self.text = str(self._payload)

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    """The rate limiter sleeps for real seconds; tests must not."""
    monkeypatch.setattr(vp.time, "sleep", lambda *_: None)
    monkeypatch.setattr(VideoForgeProvider, "_last_submit", 0.0)
    yield


# ------------------------------------------------------------- the basics ---

def test_it_needs_no_key_and_is_the_default():
    provider = get_provider("videoforge")
    assert provider.requires_key is False
    assert provider.is_configured is True
    assert DEFAULT_PROVIDER == "videoforge"


def test_it_offers_both_directions():
    info = get_provider("videoforge").info()
    assert info.text_to_video and info.image_to_video


# --------------------------------------------------------- what it accepts --

@pytest.mark.parametrize("asked,sent", [(1, 5), (4, 5), (5, 5), (7, 5), (8, 10), (10, 10), (30, 10)])
def test_clip_length_is_snapped_to_what_the_service_takes(asked, sent):
    """It rejects anything but 5 or 10 outright - a 4 second ask is a 400."""
    assert VideoForgeProvider._clamp_seconds(asked) == sent


def test_the_submitted_payload_carries_the_snapped_length(tmp_path):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json or {})
        return _Response(202, {"data": {"id": "task-1", "status": "QUEUED"}})

    provider = get_provider("videoforge")
    with patch.object(vp.requests, "post", fake_post), \
         patch.object(VideoForgeProvider, "_await", lambda *a, **k: None), \
         patch.object(VideoForgeProvider, "_download",
                      lambda self, tid, out: Path(out).write_bytes(b"x" * 2048)):
        provider.text_to_video("a storm", tmp_path / "o.mp4", seconds=4,
                               resolution="720x1280")

    assert captured["duration"] == 5
    assert captured["size"] == "720x1280"
    assert captured["watermark"] is False


# ------------------------------------------------------------ the task id ---

@pytest.mark.parametrize("payload", [
    {"data": {"id": "abc"}},
    {"task_id": "abc"},
    {"id": "abc"},
])
def test_the_task_id_is_found_wherever_it_is_reported(payload):
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "post", lambda *a, **k: _Response(202, payload)):
        assert provider._submit({"prompt": "x"}) == "abc"


def test_a_response_with_no_task_id_is_an_error():
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "post", lambda *a, **k: _Response(202, {"success": True})):
        with pytest.raises(ProviderError, match="no task id"):
            provider._submit({"prompt": "x"})


def test_a_rejected_submission_reports_the_reason():
    """The service explains itself; that explanation is worth passing on."""
    rejection = _Response(400, {"error": "duration must be 5 or 10 seconds."})
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "post", lambda *a, **k: rejection):
        with pytest.raises(ProviderError, match="duration must be 5 or 10"):
            provider._submit({"prompt": "x"})


# --------------------------------------------------------------- polling ----

def test_polling_ends_on_success():
    states = iter([
        _Response(200, {"data": {"status": "QUEUED"}}),
        _Response(200, {"data": {"status": "PROCESSING"}}),
        _Response(200, {"data": {"status": "SUCCESS"}}),
    ])
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "get", lambda *a, **k: next(states)):
        provider._await("t", lambda *_: None)


def test_a_failed_task_raises_with_the_services_own_message():
    failed = _Response(200, {"data": {"status": "FAIL", "error": "model refused"}})
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "get", lambda *a, **k: failed):
        with pytest.raises(ProviderError, match="model refused"):
            provider._await("t", lambda *_: None)


def test_polling_gives_up_rather_than_hanging_forever(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "videoforge_timeout", 0)
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "get",
                      lambda *a, **k: _Response(200, {"data": {"status": "QUEUED"}})):
        with pytest.raises(ProviderError, match="did not finish"):
            provider._await("t", lambda *_: None)


# -------------------------------------------------------------- download ----

def test_a_short_body_is_not_accepted_as_a_video(tmp_path):
    """An error page is a 200 with bytes in it; a video it is not."""
    provider = get_provider("videoforge")
    with patch.object(vp.requests, "get", lambda *a, **k: _Response(200, {}, b"nope")):
        with pytest.raises(ProviderError, match="no file"):
            provider._download("t", tmp_path / "o.mp4")


def test_the_file_route_is_used_rather_than_the_expiring_cdn_link(tmp_path):
    seen = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        return _Response(200, {}, b"\x00" * 4096)

    provider = get_provider("videoforge")
    with patch.object(vp.requests, "get", fake_get):
        provider._download("task-9", tmp_path / "o.mp4")
    assert seen["url"].endswith("/api/v1/file/task-9")
    assert (tmp_path / "o.mp4").stat().st_size == 4096


# ---------------------------------------------------------------- pacing ----

def test_submissions_are_spaced_out(monkeypatch):
    """The limit belongs to the service, so the spacing is shared, not per object."""
    from config import settings

    monkeypatch.setattr(settings, "videoforge_rpm", 10)
    slept = []
    monkeypatch.setattr(vp.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(vp.time, "time", lambda: 1000.0)
    VideoForgeProvider._last_submit = 1000.0  # a submission just happened

    get_provider("videoforge")._pace()
    assert slept and slept[0] == pytest.approx(6.0, abs=0.01)


def test_two_provider_objects_share_one_rate_limit():
    assert get_provider("videoforge")._pace.__func__ is \
           get_provider("videoforge")._pace.__func__
    assert "_submit_lock" in vars(VideoForgeProvider)
