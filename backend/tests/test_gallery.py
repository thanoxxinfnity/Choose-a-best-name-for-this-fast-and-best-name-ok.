"""The in-app gallery: indexing, thumbnails, and deletion."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gallery  # noqa: E402
from gallery import (  # noqa: E402
    GalleryItem,
    delete_item,
    ensure_thumbnail,
    find_item,
    keep_generated,
    list_items,
    stats,
    thumbnail_path,
)
from puter_integration import ffmpeg_binary  # noqa: E402

FF = ffmpeg_binary()


def _clip(path: Path, seconds: float = 2.0, size: str = "180x320") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery.settings, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(gallery.settings), "jobs_dir",
        property(lambda self: tmp_path / "jobs"),
    )
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    yield


def _seed_render(job_id: str, prompt: str = "an edit", finished: float = None,
                 stage: str = "completed") -> Path:
    job = gallery.settings.jobs_dir / job_id
    video = _clip(job / "moja_ai_final.mp4")
    (job / "status.json").write_text(json.dumps({
        "job_id": job_id, "stage": stage,
        "finished_at": finished or time.time(),
        "prompt": prompt, "theme": "anime_edits", "export_preset": "1080p60",
        "duration_seconds": 2.0, "output_width": 180, "output_height": 320,
        "output_fps": 24.0,
    }), encoding="utf-8")
    return video


# ------------------------------------------------------------------ index --

def test_an_empty_gallery_is_empty():
    assert list_items() == []
    assert stats()["count"] == 0


def test_a_finished_render_appears():
    _seed_render("job1", prompt="High energy anime edit")
    items = list_items()
    assert len(items) == 1
    item = items[0]
    assert item.kind == "render"
    assert item.item_id == "render:job1"
    assert item.title == "High energy anime edit"
    assert (item.width, item.height) == (180, 320)
    assert item.theme == "anime_edits"


def test_an_unfinished_job_is_hidden():
    _seed_render("running", stage="rendering")
    assert list_items() == []


def test_a_job_with_no_output_is_hidden():
    (gallery.settings.jobs_dir / "empty").mkdir(parents=True)
    assert list_items() == []


def test_items_are_newest_first():
    now = time.time()
    _seed_render("old", prompt="older", finished=now - 500)
    _seed_render("new", prompt="newer", finished=now)
    assert [item.title for item in list_items()] == ["newer", "older"]


def test_geometry_is_filled_in_for_older_jobs():
    """Jobs written before the geometry fields existed still report their size."""
    job = gallery.settings.jobs_dir / "legacy"
    _clip(job / "moja_ai_final.mp4")
    (job / "status.json").write_text(
        json.dumps({"job_id": "legacy", "stage": "completed", "prompt": "old"}),
        encoding="utf-8",
    )
    item = list_items()[0]
    assert item.width == 180 and item.height == 320
    assert item.duration == pytest.approx(2.0, abs=0.3)


def test_stats_count_both_kinds(tmp_path):
    _seed_render("job1")
    keep_generated(_clip(tmp_path / "gen.mp4"), prompt="a clip", provider="local_motion")
    summary = stats()
    assert summary["count"] == 2
    assert summary["renders"] == 1 and summary["generated"] == 1
    assert summary["total_bytes"] > 0


def test_filtering_by_kind(tmp_path):
    _seed_render("job1")
    keep_generated(_clip(tmp_path / "gen.mp4"), prompt="a clip")
    assert len(list_items(kind="render")) == 1
    assert len(list_items(kind="generated")) == 1


# ------------------------------------------------------------- generated --

def test_keeping_a_generated_clip_records_its_metadata(tmp_path):
    item = keep_generated(
        _clip(tmp_path / "src.mp4"), prompt="cursed energy swirling",
        provider="puter", model="wan2.2-i2v-a14b", seconds=4.0,
    )
    assert item is not None and item.kind == "generated"
    assert item.provider == "puter"
    assert item.title == "cursed energy swirling"
    stored = list_items(kind="generated")[0]
    assert stored.prompt == "cursed energy swirling"


def test_keeping_an_empty_file_is_refused(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert keep_generated(empty) is None


def test_keeping_a_missing_file_is_refused(tmp_path):
    assert keep_generated(tmp_path / "nope.mp4") is None


# ------------------------------------------------------------ thumbnails --

def test_a_thumbnail_is_generated_and_cached():
    _seed_render("job1")
    item = list_items()[0]
    path = ensure_thumbnail(item)
    assert path is not None and path.exists() and path.stat().st_size > 0
    assert path.suffix == ".jpg"

    before = path.stat().st_mtime_ns
    assert ensure_thumbnail(item) == path
    assert path.stat().st_mtime_ns == before, "a cached thumbnail must not be rebuilt"


def test_thumbnail_of_a_missing_file_is_none(tmp_path):
    ghost = GalleryItem(
        item_id="generated:ghost", kind="generated", filename="ghost.mp4",
        path=str(tmp_path / "ghost.mp4"), size_bytes=0, created_at=time.time(),
    )
    assert ensure_thumbnail(ghost) is None


def test_listing_reports_whether_a_thumbnail_exists():
    _seed_render("job1")
    assert list_items()[0].has_thumbnail is False
    ensure_thumbnail(list_items()[0])
    assert list_items()[0].has_thumbnail is True


# --------------------------------------------------------------- lookup ----

def test_find_item_round_trips():
    _seed_render("job1")
    assert find_item("render:job1") is not None
    assert find_item("render:nope") is None


# --------------------------------------------------------------- delete ----

def test_deleting_a_render_removes_its_whole_workspace():
    _seed_render("job1")
    workspace = gallery.settings.jobs_dir / "job1"
    assert delete_item("render:job1") is True
    assert not workspace.exists()
    assert list_items() == []


def test_deleting_a_generated_clip_removes_its_sidecar_and_thumbnail(tmp_path):
    item = keep_generated(_clip(tmp_path / "gen.mp4"), prompt="x")
    video = Path(item.path)
    ensure_thumbnail(item)
    thumb = thumbnail_path(item)
    assert video.exists() and thumb.exists()

    assert delete_item(item.item_id) is True
    assert not video.exists()
    assert not video.with_suffix(".json").exists()
    assert not thumb.exists()


def test_deleting_an_unknown_item_is_false():
    assert delete_item("render:nothing") is False
