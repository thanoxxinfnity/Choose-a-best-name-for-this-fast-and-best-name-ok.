"""Generating a batch of shots without running past the money.

The batch this replaces fired fourteen requests, made one, and failed thirteen
times on "Insufficient funds" - once per remaining shot, because nothing was
counting. Each rule below exists because of something that actually happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot_batch import (  # noqa: E402
    CLIP_PRICES,
    Shot,
    generate_batch,
    is_out_of_money,
    load_shots,
    price_of,
)


class _Provider:
    """Stands in for a video provider. Writes plausible files, counts calls."""

    def __init__(self, fail_on=(), broke_after=None, bytes_out=200_000):
        self.calls = []
        self.fail_on = set(fail_on)
        self.broke_after = broke_after
        self.bytes_out = bytes_out

    def text_to_video(self, prompt, output_path, seconds=4.0, resolution="720x1280",
                      **kwargs):
        self.calls.append(Path(output_path).stem)
        if self.broke_after is not None and len(self.calls) > self.broke_after:
            raise RuntimeError(
                "Insufficient balance. This request costs ~0.0500 pollen, but "
                "your available paid balance is 0.0000."
            )
        if Path(output_path).stem in self.fail_on:
            raise RuntimeError("the model hiccuped")
        Path(output_path).write_bytes(b"\x00" * self.bytes_out)


def _shots(count: int):
    return [Shot(f"s{i:02d}", f"prompt {i}") for i in range(count)]


# ---------------------------------------------------------------- pricing ---

def test_a_shorter_clip_costs_less():
    assert price_of("wan-fast", 2.0) < price_of("wan-fast", 4.0)
    assert price_of("wan-fast", 4.0) == pytest.approx(CLIP_PRICES["wan-fast"])


def test_an_unknown_model_still_gets_a_price():
    """A model with no quote must not be treated as free."""
    assert price_of("something-new", 4.0) > 0


# ----------------------------------------------------------- the budget -----

def test_the_batch_stops_before_going_over_budget(tmp_path):
    provider = _Provider()
    # $0.16 at 5 cents a clip is three clips, not four.
    result = generate_batch(_shots(10), provider, tmp_path, budget=0.16,
                            model="wan-fast")

    assert len(result.made) == 3
    assert len(provider.calls) == 3, "it must not call for a shot it cannot afford"
    assert result.spent <= 0.16
    assert "budget" in result.stopped_because


def test_a_budget_of_nothing_generates_nothing(tmp_path):
    provider = _Provider()
    result = generate_batch(_shots(5), provider, tmp_path, budget=0.0)
    assert result.made == [] and provider.calls == []
    assert len(result.not_attempted) == 5


def test_the_whole_list_runs_when_the_budget_allows(tmp_path):
    provider = _Provider()
    result = generate_batch(_shots(6), provider, tmp_path, budget=5.0,
                            model="wan-fast")
    assert len(result.made) == 6
    assert result.stopped_because == ""


# ------------------------------------------------------- running dry --------

def test_an_empty_account_stops_the_batch_at_once(tmp_path):
    """Thirteen identical refusals is thirteen wasted round trips."""
    provider = _Provider(broke_after=2)
    result = generate_batch(_shots(15), provider, tmp_path, budget=99.0)

    assert len(result.made) == 2
    assert len(provider.calls) == 3, "it asked once more, then stopped"
    assert "ran out of balance" in result.stopped_because
    assert len(result.not_attempted) == 13


def test_an_ordinary_failure_does_not_stop_the_batch(tmp_path):
    """One bad shot is not an empty wallet."""
    provider = _Provider(fail_on={"s02"})
    result = generate_batch(_shots(6), provider, tmp_path, budget=99.0)

    assert len(result.made) == 5
    assert result.failed == ["s02"]
    assert result.stopped_because == ""


def test_the_out_of_money_check_is_not_fooled_by_a_normal_error():
    assert is_out_of_money(RuntimeError("Insufficient balance. costs ~0.05"))
    assert is_out_of_money(RuntimeError("HTTP 402: payment required"))
    assert not is_out_of_money(RuntimeError("connection reset by peer"))
    assert not is_out_of_money(RuntimeError("the model hiccuped"))


# ------------------------------------------------------------- resuming -----

def test_a_second_run_skips_what_is_already_on_disk(tmp_path):
    first = _Provider()
    generate_batch(_shots(4), first, tmp_path, budget=99.0)

    second = _Provider()
    result = generate_batch(_shots(4), second, tmp_path, budget=99.0)

    assert second.calls == [], "it regenerated shots it already had"
    assert len(result.skipped) == 4
    assert result.spent == 0.0


def test_a_truncated_file_is_regenerated_not_trusted(tmp_path):
    (tmp_path / "s00.mp4").write_bytes(b"\x00" * 200)  # far too small
    provider = _Provider()
    result = generate_batch(_shots(1), provider, tmp_path, budget=99.0)
    assert provider.calls == ["s00"]
    assert len(result.made) == 1


def test_a_short_response_is_a_failure_and_leaves_no_file(tmp_path):
    provider = _Provider(bytes_out=500)
    result = generate_batch(_shots(1), provider, tmp_path, budget=99.0)
    assert result.made == [] and result.failed == ["s00"]
    assert not (tmp_path / "s00.mp4").exists(), "a stub must not be left behind"


# ------------------------------------------------------------ shot lists ----

def test_a_shot_list_round_trips_through_json(tmp_path):
    path = tmp_path / "shots.json"
    path.write_text('[{"name":"a","prompt":"one","seconds":5},'
                    ' {"name":"b","prompt":"two"}]')
    shots = load_shots(path)
    assert [s.name for s in shots] == ["a", "b"]
    assert shots[0].seconds == 5.0 and shots[1].seconds == 4.0
