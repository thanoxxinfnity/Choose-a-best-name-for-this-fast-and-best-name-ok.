"""The persistent editing skill Kimi carries into every render."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import editing_skill  # noqa: E402
from editing_skill import (  # noqa: E402
    CRAFT,
    MAX_PLAYBOOKS,
    EditingSkill,
    GenrePlaybook,
    load_skill,
    reset_skill,
    save_skill,
)
from trend_research import TrendReport, TrendVideo, derive_style, summarise  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Never touch the real skill file from a test."""
    monkeypatch.setattr(editing_skill.settings, "data_dir", tmp_path)
    monkeypatch.setattr(editing_skill, "_CACHE", None)
    yield


def _report(duration: float = 20.0, count: int = 6, query: str = "q") -> TrendReport:
    videos = [
        TrendVideo(video_id=str(i), title=f"Gojo EDIT {i} 🔥", channel="c",
                   published_at="", duration=duration, views=1000 * (i + 1),
                   likes=10, comments=5, tags=["jjk", "gojo"])
        for i in range(count)
    ]
    return summarise(query, videos)


# ------------------------------------------------------------------- craft --

def test_the_curated_craft_is_substantial_and_specific():
    assert len(CRAFT) >= 6
    total = sum(len(rules) for rules in CRAFT.values())
    assert total >= 20, "a skill this thin teaches nothing"
    for section, rules in CRAFT.items():
        assert rules, section
        for rule in rules:
            assert len(rule) > 40, f"{section}: '{rule}' is too vague to act on"


def test_system_block_contains_every_section():
    block = EditingSkill().to_system_block()
    for section in CRAFT:
        assert section in block


def test_system_block_has_no_playbooks_before_learning():
    assert "Measured genre playbooks" not in EditingSkill().to_system_block()


# ---------------------------------------------------------------- learning --

def test_learning_creates_a_playbook():
    skill = EditingSkill()
    report = _report(duration=19)
    assert skill.learn_from_research("anime_edit", report, derive_style(report))
    playbook = skill.playbooks["anime_edit"]
    assert playbook.median_duration == 19
    assert playbook.observations == 1
    assert playbook.sampled == 6
    assert "Measured genre playbooks" in skill.to_system_block()


def test_a_failed_report_teaches_nothing():
    skill = EditingSkill()
    assert not skill.learn_from_research("x", TrendReport(query="q", error="no key"), {})
    assert skill.playbooks == {}


def test_repeated_observations_average_rather_than_overwrite():
    """One unusual sample must not be able to rewrite a playbook."""
    skill = EditingSkill()
    first = _report(duration=20)
    skill.learn_from_research("anime_edit", first, derive_style(first))
    outlier = _report(duration=60)
    skill.learn_from_research("anime_edit", outlier, derive_style(outlier))

    playbook = skill.playbooks["anime_edit"]
    assert playbook.observations == 2
    assert 20 < playbook.median_duration < 60, playbook.median_duration
    assert playbook.median_duration == pytest.approx(40, abs=1)
    assert playbook.sampled == 12, "sample counts accumulate"


def test_a_well_observed_playbook_resists_a_single_outlier():
    skill = EditingSkill()
    for _ in range(9):
        report = _report(duration=20)
        skill.learn_from_research("anime_edit", report, derive_style(report))
    outlier = _report(duration=90)
    skill.learn_from_research("anime_edit", outlier, derive_style(outlier))
    # With nine prior observations the tenth moves it only a little.
    assert skill.playbooks["anime_edit"].median_duration < 30


def test_hooks_and_keywords_merge_without_duplicates():
    skill = EditingSkill()
    for _ in range(2):
        report = _report()
        skill.learn_from_research("anime_edit", report, derive_style(report))
    playbook = skill.playbooks["anime_edit"]
    assert len(playbook.hooks) == len(set(playbook.hooks))
    assert len(playbook.keywords) == len(set(playbook.keywords))


def test_playbooks_are_pruned_to_the_cap():
    skill = EditingSkill()
    for index in range(MAX_PLAYBOOKS + 8):
        report = _report(query=f"q{index}")
        skill.learn_from_research(f"niche_{index}", report, derive_style(report))
    assert len(skill.playbooks) <= MAX_PLAYBOOKS


def test_pruning_keeps_the_best_observed_niche():
    skill = EditingSkill()
    for _ in range(5):
        report = _report(query="core")
        skill.learn_from_research("core", report, derive_style(report))
    for index in range(MAX_PLAYBOOKS + 5):
        report = _report(query=f"one_off_{index}")
        skill.learn_from_research(f"one_off_{index}", report, derive_style(report))
    assert "core" in skill.playbooks, "a repeatedly observed niche must survive"


# -------------------------------------------------------------- relevance --

def test_the_matching_niche_is_listed_first():
    skill = EditingSkill()
    for niche in ("gaming", "nature", "anime_edit"):
        report = _report(query=niche)
        skill.learn_from_research(niche, report, derive_style(report))
    block = skill.to_system_block(niche="anime_edit", max_playbooks=1)
    assert "anime_edit:" in block
    assert "gaming:" not in block


# ------------------------------------------------------------ persistence --

def test_skill_round_trips_through_disk():
    skill = load_skill()
    report = _report()
    skill.learn_from_research("anime_edit", report, derive_style(report))
    save_skill(skill)

    reloaded = load_skill(refresh=True)
    assert "anime_edit" in reloaded.playbooks
    assert reloaded.playbooks["anime_edit"].median_duration == \
        skill.playbooks["anime_edit"].median_duration


def test_reset_drops_learning_but_keeps_craft():
    skill = load_skill()
    report = _report()
    skill.learn_from_research("anime_edit", report, derive_style(report))
    save_skill(skill)

    fresh = reset_skill()
    assert fresh.playbooks == {}
    assert sum(len(r) for r in fresh.craft.values()) >= 20


def test_an_older_stored_skill_keeps_new_curated_sections():
    """Upgrading the code must not silently drop craft the file predates."""
    path = editing_skill.skill_path()
    path.write_text(json.dumps({
        "version": 0,
        "craft": {"Where to cut": ["only this one old rule that is long enough to pass"]},
        "playbooks": {},
    }), encoding="utf-8")
    loaded = load_skill(refresh=True)
    assert "The first second" in loaded.craft, "missing sections must be restored"
    assert loaded.craft["Where to cut"] == [
        "only this one old rule that is long enough to pass"
    ], "the user's edited section must be preserved"


def test_a_corrupt_skill_file_falls_back_to_curated():
    editing_skill.skill_path().write_text("{ not json", encoding="utf-8")
    loaded = load_skill(refresh=True)
    assert sum(len(r) for r in loaded.craft.values()) >= 20


def test_playbook_line_is_readable():
    line = GenrePlaybook(
        niche="anime_edit", median_duration=19.0, shot_seconds=[0.8, 1.8],
        suggested_cuts=14, hooks=["emoji in the title"], keywords=["gojo"], sampled=17,
    ).to_line()
    assert "19s" in line and "0.8-1.8s" in line and "14 cuts" in line and "17 videos" in line


# ------------------------------------------- learning from its own review ---

def test_one_bad_draft_teaches_nothing():
    """A model has an off draft like anyone; naming it as a habit is noise."""
    from editing_skill import EditingSkill

    skill = EditingSkill()
    skill.learn_from_review(["metronome"])
    assert skill.recurring_habits() == []
    assert "Habits the review" not in skill.to_system_block()


def test_the_same_mistake_three_times_becomes_a_habit():
    from editing_skill import EditingSkill

    skill = EditingSkill()
    for _ in range(3):
        skill.learn_from_review(["metronome"])

    habits = skill.recurring_habits()
    assert [habit.rule for habit in habits] == ["metronome"]
    block = skill.to_system_block()
    assert "Habits the review keeps catching" in block
    assert "same length" in block


def test_habits_are_ranked_by_how_often_they_recur():
    from editing_skill import EditingSkill

    skill = EditingSkill()
    for _ in range(3):
        skill.learn_from_review(["wordy_overlay"])
    for _ in range(6):
        skill.learn_from_review(["metronome"])

    assert [habit.rule for habit in skill.recurring_habits()][0] == "metronome"


def test_a_rule_with_no_lesson_is_skipped_rather_than_paraphrased_badly():
    from editing_skill import EditingSkill

    skill = EditingSkill()
    assert skill.learn_from_review(["some_rule_that_does_not_exist"]) is False
    assert skill.habits == {}


def test_habits_survive_being_saved_and_loaded():
    from editing_skill import EditingSkill

    skill = EditingSkill()
    for _ in range(4):
        skill.learn_from_review(["voice_on_the_hit"])

    restored = EditingSkill.from_dict(skill.to_dict())
    assert restored.habits["voice_on_the_hit"].count == 4
    assert [h.rule for h in restored.recurring_habits()] == ["voice_on_the_hit"]


def test_a_skill_stored_before_habits_existed_still_loads():
    """Upgrading the code must not make an existing skill file unreadable."""
    from editing_skill import CRAFT, EditingSkill

    legacy = {"version": 1, "craft": {"Pacing": ["something"]}, "playbooks": {},
              "updated_at": 0.0}
    restored = EditingSkill.from_dict(legacy)
    assert restored.habits == {}
    # And the curated sections it never knew about are still there.
    assert set(CRAFT) <= set(restored.craft)


def test_the_habit_list_is_bounded():
    from editing_skill import HABIT_LESSONS, MAX_HABITS, EditingSkill

    skill = EditingSkill()
    for rule in HABIT_LESSONS:
        skill.learn_from_review([rule])
    assert len(skill.habits) <= MAX_HABITS
