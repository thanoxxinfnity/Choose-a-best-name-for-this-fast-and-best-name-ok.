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


def test_a_queue_that_never_starts_is_reported_as_that(monkeypatch):
    """Not as a flat timeout: a task the service never handed to the model is
    a different failure, with a different owner, from a slow render."""
    from config import settings

    monkeypatch.setattr(settings, "videoforge_timeout", 1)
    provider = get_provider("videoforge")
    queued = _Response(200, {"data": {"status": "QUEUED", "submitted_at": None}})
    clock = iter([0.0, 0.0, 0.5, 5.0, 5.0, 5.0])
    with patch.object(vp.requests, "get", lambda *a, **k: queued), \
         patch.object(vp.time, "time", lambda: next(clock)):
        with pytest.raises(ProviderError, match="never started it"):
            provider._await("t", lambda *_: None)


def test_a_render_that_started_but_ran_long_says_so(monkeypatch):
    """Polling has to actually happen for either claim to be made."""
    from config import settings

    monkeypatch.setattr(settings, "videoforge_timeout", 1)
    provider = get_provider("videoforge")
    running = _Response(200, {"data": {"status": "PROCESSING",
                                       "submitted_at": "2026-09-06T04:00:00Z"}})
    clock = iter([0.0, 0.0, 0.5, 5.0, 5.0, 5.0])
    with patch.object(vp.requests, "get", lambda *a, **k: running), \
         patch.object(vp.time, "time", lambda: next(clock)):
        with pytest.raises(ProviderError, match="did not finish"):
            provider._await("t", lambda *_: None)


def test_nothing_is_claimed_when_the_task_was_never_polled(monkeypatch):
    """With the budget already gone there is no reading to reason from."""
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


# --------------------------------------------- the endpoint is the user's ---

def test_the_endpoint_comes_from_the_request_not_a_constant():
    """It is set on the device and sent per request, like the keys are."""
    provider = get_provider("videoforge", base_url="https://mine.example/")
    assert provider._base() == "https://mine.example"


def test_without_one_the_server_default_is_used():
    from config import settings

    assert get_provider("videoforge")._base() == settings.videoforge_base_url.rstrip("/")


def test_the_endpoint_reaches_whichever_provider_is_chosen():
    """best_available picks the provider; the setting has to survive that."""
    from video_providers import best_available

    provider = best_available({}, need_text_to_video=True, base_url="https://mine.example")
    assert provider is not None
    assert provider._base() == "https://mine.example"


def test_credentials_carry_the_endpoint():
    from jobs import JobCredentials

    assert JobCredentials(video_endpoint="https://x.example/").resolved_video_endpoint() \
        == "https://x.example"


def test_the_wait_outlasts_the_services_own_retry_ladder():
    """90s, 4min, 10min then 15min: a task can sit queued for over half an
    hour with nothing wrong, and abandoning it at fifteen would be wrong."""
    from config import settings

    assert settings.videoforge_timeout > (90 + 240 + 600 + 900)


# --------------------------------------------------------- ModelScope -------

def test_modelscope_is_registered_and_asks_for_a_key():
    from video_providers import ModelScopeProvider, provider_keys

    assert "modelscope" in provider_keys()
    info = ModelScopeProvider(api_key="").info()
    assert info.text_to_video and info.image_to_video
    assert info.requires_key and not info.configured
    assert "free" in info.notes.lower()


def test_modelscope_says_what_is_missing_rather_than_calling_out():
    """Without a token it must not reach the network to find that out."""
    from video_providers import ModelScopeProvider, ProviderError

    provider = ModelScopeProvider(api_key="")
    with pytest.raises(ProviderError, match="free account token"):
        provider.text_to_video("anything", Path("/tmp/never-written.mp4"))


def test_modelscope_submits_asynchronously(monkeypatch, tmp_path):
    """A synchronous render holds the connection open and dies through a proxy."""
    import video_providers

    seen = {}

    class _Response:
        status_code = 200

        def __init__(self, payload=None, content=b""):
            self._payload = payload or {}
            self.content = content
            self.text = ""

        def json(self):
            return self._payload

    def _post(url, headers=None, json=None, timeout=None):
        seen["headers"] = headers or {}
        seen["payload"] = json or {}
        return _Response({"task_id": "t-1"})

    def _get(url, headers=None, timeout=None):
        if "/tasks/" in url:
            return _Response({"task_status": "SUCCEED",
                              "output_video_url": ["https://example.test/v.mp4"]})
        return _Response(content=b"x" * 4096)

    monkeypatch.setattr(video_providers.requests, "post", _post)
    monkeypatch.setattr(video_providers.requests, "get", _get)
    monkeypatch.setattr(video_providers.time, "sleep", lambda _s: None)

    provider = video_providers.ModelScopeProvider(api_key="token")
    result = provider.text_to_video("a red cube", tmp_path / "out.mp4")

    assert seen["headers"].get("X-ModelScope-Async-Mode") == "true"
    assert seen["payload"]["model"] == provider.DEFAULT_MODEL
    assert result.path.exists() and result.path.stat().st_size == 4096


def test_modelscope_reports_a_failed_task_with_its_reason(monkeypatch, tmp_path):
    import video_providers
    from video_providers import ProviderError

    class _Response:
        status_code = 200
        text = ""
        content = b""

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    monkeypatch.setattr(video_providers.requests, "post",
                        lambda *a, **k: _Response({"task_id": "t-9"}))
    monkeypatch.setattr(video_providers.requests, "get",
                        lambda *a, **k: _Response({"task_status": "FAILED",
                                                   "message": "quota exhausted"}))
    monkeypatch.setattr(video_providers.time, "sleep", lambda _s: None)

    provider = video_providers.ModelScopeProvider(api_key="token")
    with pytest.raises(ProviderError, match="quota exhausted"):
        provider.text_to_video("a red cube", tmp_path / "out.mp4")


# -------------------------------------------------------- Pollinations ------

def test_pollinations_is_registered_with_its_model_range():
    from video_providers import PollinationsProvider, provider_keys

    assert "pollinations" in provider_keys()
    info = PollinationsProvider(api_key="").info()
    assert info.text_to_video and info.image_to_video
    assert "wan-fast" in info.models and "veo" in info.models
    assert "enter.pollinations.ai" in info.notes


def test_pollinations_says_what_is_missing_rather_than_calling_out():
    from video_providers import PollinationsProvider, ProviderError

    with pytest.raises(ProviderError, match="enter.pollinations.ai"):
        PollinationsProvider(api_key="").text_to_video(
            "anything", Path("/tmp/never-written.mp4"))


def test_pollinations_asks_for_a_vertical_clip(monkeypatch, tmp_path):
    import video_providers

    seen = {}

    class _Response:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"\x00" * 8192
        text = ""

    def _get(url, params=None, headers=None, timeout=None):
        seen["url"] = url
        seen["params"] = params or {}
        seen["auth"] = (headers or {}).get("Authorization", "")
        return _Response()

    monkeypatch.setattr(video_providers.requests, "get", _get)
    provider = video_providers.PollinationsProvider(api_key="sk_test")
    result = provider.text_to_video("a red cube", tmp_path / "out.mp4",
                                    seconds=5, resolution="720x1280")

    assert "/video/" in seen["url"]
    assert seen["params"]["aspectRatio"] == "9:16"
    assert seen["params"]["duration"] == 5
    assert seen["auth"] == "Bearer sk_test"
    assert result.path.exists()


def test_pollinations_rejects_an_error_wearing_a_200(monkeypatch, tmp_path):
    """A JSON body with a 200 is a failure, not a video."""
    import video_providers
    from video_providers import ProviderError

    class _Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"error":"out of pollen"}'
        text = '{"error":"out of pollen"}'

    monkeypatch.setattr(video_providers.requests, "get",
                        lambda *a, **k: _Response())
    provider = video_providers.PollinationsProvider(api_key="sk_test")
    with pytest.raises(ProviderError, match="out of pollen"):
        provider.text_to_video("a red cube", tmp_path / "out.mp4")


def test_pollinations_image_to_video_needs_a_reachable_still(tmp_path):
    from video_providers import PollinationsProvider, ProviderError

    provider = PollinationsProvider(api_key="sk_test")
    with pytest.raises(ProviderError, match="public image URL"):
        provider.image_to_video(tmp_path / "still.png", "move it",
                                tmp_path / "out.mp4")


# ------------------------------------------- holding a character steady -----

def test_a_reference_url_is_stable_for_one_prompt_and_seed():
    """The same character must come back byte for byte, or it is not a reference."""
    from video_providers import PollinationsProvider

    a = PollinationsProvider.reference_url("a white haired sorcerer", seed=21)
    b = PollinationsProvider.reference_url("a white haired sorcerer", seed=21)
    c = PollinationsProvider.reference_url("a white haired sorcerer", seed=22)
    assert a == b and a != c
    assert a.startswith(PollinationsProvider.BASE)
    assert "seed=21" in a


def test_only_reference_capable_models_are_accepted(tmp_path):
    """wan-fast makes a different character every call; that is the whole bug."""
    from video_providers import PollinationsProvider, ProviderError

    provider = PollinationsProvider(api_key="sk_test")
    with pytest.raises(ProviderError, match="does not take reference media"):
        provider.image_to_video(tmp_path / "s.png", "move", tmp_path / "o.mp4",
                                model="wan-fast",
                                reference_image_url="https://example.test/a.png")


def test_a_reference_that_will_not_prime_is_refused(monkeypatch, tmp_path):
    """An unprimed URL reaches the model as a 401 and the character vanishes."""
    import video_providers
    from video_providers import ProviderError

    class _Denied:
        status_code = 401
        content = b""

    monkeypatch.setattr(video_providers.requests, "get", lambda *a, **k: _Denied())
    provider = video_providers.PollinationsProvider(api_key="sk_test")
    url = video_providers.PollinationsProvider.reference_url("someone", seed=3)

    with pytest.raises(ProviderError, match="could not be made public"):
        provider.image_to_video(tmp_path / "s.png", "move", tmp_path / "o.mp4",
                                model="wan-3.0", reference_image_url=url)


def test_a_primed_reference_goes_through(monkeypatch, tmp_path):
    import video_providers

    seen = {}

    class _Response:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"\x00" * 8192
        text = ""

    def _get(url, params=None, headers=None, timeout=None):
        if params is None:                     # the priming fetch
            return _Response()
        seen["params"] = params
        return _Response()

    monkeypatch.setattr(video_providers.requests, "get", _get)
    provider = video_providers.PollinationsProvider(api_key="sk_test")
    url = video_providers.PollinationsProvider.reference_url("someone", seed=4)
    result = provider.image_to_video(tmp_path / "s.png", "move", tmp_path / "o.mp4",
                                     model="wan-3.0", reference_image_url=url)

    assert seen["params"]["reference_images"] == url
    assert seen["params"]["model"] == "wan-3.0"
    assert result.path.exists()


def test_an_externally_hosted_reference_is_not_primed(monkeypatch, tmp_path):
    """Only this service's own URLs need warming; someone else's is already up."""
    import video_providers

    calls = []

    class _Response:
        status_code = 200
        headers = {"content-type": "video/mp4"}
        content = b"\x00" * 8192
        text = ""

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _Response()

    monkeypatch.setattr(video_providers.requests, "get", _get)
    provider = video_providers.PollinationsProvider(api_key="sk_test")
    provider.image_to_video(tmp_path / "s.png", "move", tmp_path / "o.mp4",
                            model="wan-3.0",
                            reference_image_url="https://example.test/hero.png")
    assert len(calls) == 1, "it primed a URL it does not own"
