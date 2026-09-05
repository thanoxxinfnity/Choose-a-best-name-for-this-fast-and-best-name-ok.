"""Learning from edits someone handed over, rather than from a search."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editing_skill import EditingSkill  # noqa: E402
from trend_research import (  # noqa: E402
    TrendReport,
    TrendVideo,
    derive_style,
    extract_video_ids,
    is_montage,
    study_references,
    summarise,
)


# ------------------------------------------------------------ the links ----

def test_ids_come_out_of_every_shape_of_link():
    assert extract_video_ids([
        "https://youtu.be/vqFHLvB5UtU",
        "https://www.youtube.com/watch?v=OOFKmfgggvM&t=30s",
        "https://youtube.com/shorts/BfMech28CkE",
        "https://www.youtube.com/embed/eygAX0hb2gA",
        "m9SIBiK-_Ro",
    ]) == ["vqFHLvB5UtU", "OOFKmfgggvM", "BfMech28CkE", "eygAX0hb2gA", "m9SIBiK-_Ro"]


def test_a_mangled_tracking_parameter_does_not_lose_the_id():
    """One of the references arrived with two 'si' values glued together."""
    assert extract_video_ids(
        ["https://youtu.be/vqFHLvB5UtU?si=vVMb-XCVIaQymNXZsi=PFVSKcxduuRUc49V"]
    ) == ["vqFHLvB5UtU"]


def test_duplicates_are_studied_once():
    assert extract_video_ids([
        "https://youtu.be/OOFKmfgggvM",
        "https://www.youtube.com/watch?v=OOFKmfgggvM",
    ]) == ["OOFKmfgggvM"]


def test_junk_is_ignored_rather_than_guessed_at():
    assert extract_video_ids(["", "   ", "not a link", "https://example.com/x"]) == []


def test_no_links_reports_instead_of_calling_the_api():
    report = study_references(["nonsense"], api_key="whatever")
    assert "no YouTube video ids" in report.error


def test_no_key_reports_instead_of_failing(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "youtube_api_key", "")
    report = study_references(["https://youtu.be/vqFHLvB5UtU"], api_key="")
    assert "no YouTube Data API key" in report.error


# --------------------------------------------------------- montage sense ---

def _sample(duration: float, tags, title="Gojo - Track [Edit/AMV]"):
    return TrendVideo(video_id="x" * 11, title=title, channel="c", published_at="",
                      duration=duration, views=1000, likes=50, tags=list(tags))


def test_an_edit_sample_is_recognised_as_a_montage():
    report = summarise("references", [
        _sample(52, ["amv", "edit", "jujutsu kaisen"]),
        _sample(48, ["amv", "edit", "anime"]),
    ], short_form_only=False)
    assert is_montage(report)


def test_a_talking_sample_is_not():
    report = summarise("cooking tips", [
        TrendVideo(video_id="a" * 11, title="How to poach an egg", channel="c",
                   published_at="", duration=45, views=900, tags=["cooking", "recipe"]),
        TrendVideo(video_id="b" * 11, title="How to boil rice", channel="c",
                   published_at="", duration=50, views=800, tags=["cooking", "kitchen"]),
    ], short_form_only=False)
    assert not is_montage(report)


def test_a_montage_gets_montage_cadence_whatever_its_length():
    """A 52s AMV and a 52s talking short want completely different shot lengths.

    Scaling cadence off duration alone gave the AMV three second shots, which
    is not a fast edit - it is a slideshow with good music.
    """
    montage = summarise("references", [
        _sample(52, ["amv", "edit"]), _sample(55, ["amv", "edit"]),
    ], short_form_only=False)
    style = derive_style(montage)
    assert style["montage"] is True
    assert style["segment_seconds"][1] <= 1.8
    assert style["suggested_cuts"] > 30


def test_a_long_talking_short_keeps_its_slower_cadence():
    report = summarise("interviews", [
        TrendVideo(video_id="a" * 11, title="A quiet conversation", channel="c",
                   published_at="", duration=55, views=900, tags=["talk", "interview"]),
        TrendVideo(video_id="b" * 11, title="Another conversation", channel="c",
                   published_at="", duration=52, views=800, tags=["talk", "podcast"]),
    ], short_form_only=False)
    style = derive_style(report)
    assert style["montage"] is False
    assert style["segment_seconds"][0] >= 1.5


def test_a_montage_may_run_longer_than_a_talking_short():
    """The short-form ceiling is about talking heads; a montage runs its track."""
    long_montage = summarise("references", [
        _sample(85, ["amv", "edit"]), _sample(90, ["amv", "edit"]),
    ], short_form_only=False)
    assert derive_style(long_montage)["target_duration"] > 60


def test_an_explicit_list_keeps_its_long_examples():
    """Trimming a list someone chose is overruling the person who chose it."""
    report = summarise("references", [
        _sample(30, ["amv"]), _sample(40, ["amv"]), _sample(105, ["amv"]),
    ], short_form_only=False)
    assert report.sampled == 3
    assert report.duration_range[1] == 105


# ------------------------------------------------- chosen beats searched ----

def _report(duration, sampled=10):
    return TrendReport(query="references", niche="anime_edit", sampled=sampled,
                       median_duration=duration, duration_range=(duration, duration),
                       median_views=500_000, title_keywords=["amv", "edit"],
                       common_tags=["amv", "edit"], hook_patterns=["an explicit 'edit' label"])


def test_chosen_references_replace_a_searched_playbook():
    """Averaging ten chosen edits into a keyword search dilutes the instruction."""
    skill = EditingSkill()
    skill.learn_from_research("anime_edit", _report(28, sampled=17),
                              {"segment_seconds": [1.2, 2.6], "suggested_cuts": 14})
    skill.learn_from_research("anime_edit", _report(52), 
                              {"segment_seconds": [0.7, 1.6], "suggested_cuts": 44},
                              chosen=True)

    playbook = skill.playbooks["anime_edit"]
    assert playbook.chosen is True
    assert playbook.median_duration == pytest.approx(52.0)
    assert playbook.sampled == 10


def test_a_later_search_cannot_water_down_what_the_user_picked():
    skill = EditingSkill()
    skill.learn_from_research("anime_edit", _report(52),
                              {"segment_seconds": [0.7, 1.6], "suggested_cuts": 44},
                              chosen=True)
    changed = skill.learn_from_research("anime_edit", _report(28, sampled=40),
                                        {"segment_seconds": [1.2, 2.6], "suggested_cuts": 14})

    assert changed is False
    assert skill.playbooks["anime_edit"].median_duration == pytest.approx(52.0)


def test_two_chosen_sets_are_averaged_like_any_other_pair():
    skill = EditingSkill()
    skill.learn_from_research("anime_edit", _report(60), {}, chosen=True)
    skill.learn_from_research("anime_edit", _report(40), {}, chosen=True)
    playbook = skill.playbooks["anime_edit"]
    assert playbook.chosen is True
    assert 40 < playbook.median_duration < 60


def test_a_chosen_playbook_says_where_it_came_from():
    skill = EditingSkill()
    skill.learn_from_research("anime_edit", _report(52), {}, chosen=True)
    assert "references you chose" in skill.playbooks["anime_edit"].to_line()

    other = EditingSkill()
    other.learn_from_research("gaming", _report(28), {})
    assert "winners" in other.playbooks["gaming"].to_line()


def test_a_chosen_playbook_is_offered_to_the_model_first():
    skill = EditingSkill()
    skill.learn_from_research("gaming", _report(28, sampled=99), {})
    skill.learn_from_research("anime_edit", _report(52), {}, chosen=True)
    top = skill._relevant_playbooks("", limit=1)[0]
    assert top.niche == "anime_edit"


def test_a_playbook_stored_before_chosen_existed_still_loads():
    legacy = {
        "version": 2, "craft": {}, "updated_at": 0.0,
        "playbooks": {"gaming": {
            "niche": "gaming", "query": "q", "sampled": 5, "median_duration": 30.0,
            "shot_seconds": [1.0, 2.0], "suggested_cuts": 12, "hooks": [],
            "keywords": [], "updated_at": 0.0, "observations": 1,
        }},
    }
    restored = EditingSkill.from_dict(legacy)
    assert restored.playbooks["gaming"].chosen is False
