"""Trend research and the exemplar style library."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import style_library  # noqa: E402
from schemas import EditPlan  # noqa: E402
from style_library import (  # noqa: E402
    CURATED,
    Exemplar,
    build_prompt_section,
    retrieve,
    validate_library,
)
from trend_research import (  # noqa: E402
    SHORT_FORM_CEILING_SECONDS,
    TrendReport,
    TrendVideo,
    derive_style,
    parse_iso_duration,
    summarise,
)


def _video(**kwargs) -> TrendVideo:
    base = dict(video_id="x", title="t", channel="c", published_at="", duration=20.0,
                views=1000, likes=10, comments=5, tags=[])
    base.update(kwargs)
    return TrendVideo(**base)


# ------------------------------------------------------------------ parsing --

@pytest.mark.parametrize("value,expected", [
    ("PT23S", 23.0), ("PT1M5S", 65.0), ("PT1H2M3S", 3723.0),
    ("PT4M", 240.0), ("", 0.0), ("garbage", 0.0),
])
def test_parse_iso_duration(value, expected):
    assert parse_iso_duration(value) == expected


# ------------------------------------------------------------- summarising --

def test_summarise_reports_the_shared_structure():
    videos = [
        _video(video_id="a", title="Gojo EDIT 🔥", duration=18, views=5_000_000,
               tags=["jjk", "gojo", "anime"]),
        _video(video_id="b", title="Sukuna edit vs Gojo", duration=22, views=3_000_000,
               tags=["jjk", "sukuna"]),
        _video(video_id="c", title="Yuta AMV edit", duration=26, views=1_000_000,
               tags=["jjk", "anime"]),
    ]
    report = summarise("jjk edit", videos)
    assert report.sampled == 3
    assert report.median_duration == 22
    assert report.median_views == 3_000_000
    assert "jjk" in report.common_tags
    assert "edit" in report.title_keywords
    assert report.examples[0].video_id == "a", "sorted by views"


def test_summarise_drops_long_form_outliers():
    """videoDuration=short means under 4 minutes, not 'a Short'."""
    videos = [_video(video_id=str(i), duration=20) for i in range(6)]
    videos.append(_video(video_id="long", duration=232, views=99_000_000))
    report = summarise("q", videos)
    assert report.duration_range[1] <= SHORT_FORM_CEILING_SECONDS
    assert report.sampled == 6


def test_summarise_keeps_long_form_when_that_is_the_niche():
    videos = [_video(video_id=str(i), duration=200) for i in range(6)]
    report = summarise("full amv", videos)
    assert report.sampled == 6, "a genuinely long-form niche must not be emptied"


def test_summarise_needs_usable_results():
    assert summarise("q", []).error
    assert summarise("q", [_video(duration=0)]).error


def test_engagement_is_per_thousand_views():
    assert _video(views=1000, likes=80, comments=20).engagement == pytest.approx(100.0)
    assert _video(views=0).engagement == 0.0


def test_hook_patterns_need_repetition():
    once = summarise("q", [
        _video(video_id="a", title="Why did he do that?", duration=20),
        _video(video_id="b", title="plain title here", duration=20),
        _video(video_id="c", title="another plain one", duration=20),
        _video(video_id="d", title="and one more", duration=20),
    ])
    assert "a question as the hook" not in once.hook_patterns

    twice = summarise("q", [
        _video(video_id="a", title="Why did he do that?", duration=20),
        _video(video_id="b", title="How is this real?", duration=20),
        _video(video_id="c", title="plain", duration=20),
        _video(video_id="d", title="plain two", duration=20),
    ])
    assert "a question as the hook" in twice.hook_patterns


def test_stopwords_do_not_become_keywords():
    videos = [_video(video_id=str(i), title="the best video of the shorts", duration=20)
              for i in range(4)]
    keywords = summarise("q", videos).title_keywords
    assert not ({"the", "of", "best", "video", "shorts"} & set(keywords))


# ----------------------------------------------------------- derived style --

def test_derive_style_scales_shot_length_to_the_target():
    short = derive_style(summarise("q", [_video(video_id=str(i), duration=15)
                                         for i in range(4)]))
    long = derive_style(summarise("q", [_video(video_id=str(i), duration=50)
                                        for i in range(4)]))
    assert short["segment_seconds"][1] < long["segment_seconds"][1]
    assert short["target_duration"] == 15 and long["target_duration"] == 50


def test_derive_style_of_a_failed_report_is_empty():
    assert derive_style(TrendReport(query="q", error="no key")) == {}


def test_derive_style_suggests_a_sane_cut_count():
    style = derive_style(summarise("q", [_video(video_id=str(i), duration=24)
                                         for i in range(4)]))
    assert 8 <= style["suggested_cuts"] <= 20


def test_prompt_block_survives_an_error():
    assert "unavailable" in TrendReport(query="q", error="no key").to_prompt_block()


# ------------------------------------------------------------- exemplars ----

def test_every_curated_exemplar_is_a_valid_plan():
    assert validate_library() == []


def test_exemplars_are_realistic_timelines():
    for exemplar in CURATED:
        plan = EditPlan.model_validate(exemplar.plan)
        assert len(plan.edit_timeline) >= 3, exemplar.key
        for segment in plan.edit_timeline:
            assert 0.3 <= segment.duration <= 8.0, (exemplar.key, segment.duration)
        assert exemplar.why.strip(), f"{exemplar.key} teaches nothing without a reason"


def test_retrieval_prefers_the_matching_content_type():
    assert retrieve("anime_edit", "high", "anime_edits", limit=1)[0].key == "anime_fight_fast"
    assert retrieve("nature", "low", "normal", limit=1)[0].key == "calm_nature_breath"
    assert retrieve("gaming", "high", "anime_edits", limit=1)[0].key == "gaming_montage_payoff"


def test_retrieval_still_returns_something_for_an_unknown_niche():
    assert len(retrieve("underwater_basket_weaving", "medium", "normal", limit=2)) == 2


def test_learned_exemplars_outrank_curated_at_equal_relevance():
    learned = Exemplar(
        key="learned_one", content_type="anime_edit", energy="high",
        theme="anime_edits", why="kept by the user",
        plan=CURATED[0].plan, source="learned", score=CURATED[0].score,
    )
    pool = list(CURATED) + [learned]
    assert retrieve("anime_edit", "high", "anime_edits", limit=1, pool=pool)[0].key == "learned_one"


def test_prompt_section_carries_the_lesson_and_the_json():
    section = build_prompt_section("anime_edit", "high", "anime_edits", limit=2)
    assert "Why it works" in section
    assert "edit_timeline" in section
    assert "do not copy" in section.lower(), "the model must not clone the timings"


def test_prompt_section_is_empty_without_a_library(monkeypatch):
    monkeypatch.setattr(style_library, "all_exemplars", lambda: [])
    assert build_prompt_section("anime_edit", "high", "anime_edits") == ""


def test_remember_plan_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(style_library.settings, "data_dir", tmp_path)
    path = style_library.remember_plan(
        CURATED[0].plan, content_type="anime_edit", energy="high", theme="anime_edits",
    )
    assert path and path.exists()
    stored = json.loads(path.read_text())
    assert stored["source"] == "learned"
    assert [e.key for e in style_library.load_learned()] == [stored["key"]]


def test_remember_plan_refuses_an_empty_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(style_library.settings, "data_dir", tmp_path)
    assert style_library.remember_plan({"edit_timeline": []}, "x", "high", "normal") is None


def test_learned_library_is_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(style_library.settings, "data_dir", tmp_path)
    monkeypatch.setattr(style_library, "MAX_LEARNED", 3)
    for index in range(6):
        style_library.remember_plan(
            CURATED[0].plan, content_type=f"t{index}", energy="high", theme="normal",
        )
    assert len(style_library.load_learned()) <= 3
