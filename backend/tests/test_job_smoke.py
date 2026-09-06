"""One real job, start to finish, with nothing stubbed but the network.

Every other test in this suite exercises a piece of the pipeline. None of them
ran JobManager._run, so an unbound local in it - a client built two dozen lines
above the variable it needed - reached a render before anything complained.
A job that actually completes is the only thing that catches that class of
mistake, and it is cheap enough to keep.

    cd backend && python -m pytest tests/test_job_smoke.py -q
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobs import JobCredentials, JobManager, JobRequest, JobStage  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402
from schemas import EditPlan, ProjectMeta, TimelineSegment  # noqa: E402

pytestmark = pytest.mark.slow
FF = ffmpeg_binary()


@pytest.fixture
def clip(tmp_path) -> Path:
    """Three seconds of moving colour with a tone under it."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [FF, "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


def _wait(manager: JobManager, job_id: str, seconds: float = 420.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        status = manager.get(job_id)
        assert status is not None, "the job vanished"
        if status.stage in (JobStage.COMPLETED, JobStage.FAILED):
            return status
        time.sleep(2)
    pytest.fail(f"job did not finish in {seconds:.0f}s")


def test_a_job_with_a_written_plan_renders_end_to_end(clip, tmp_path):
    plan = EditPlan(
        project_meta=ProjectMeta(resolution="360x640", fps=30),
        edit_timeline=[
            TimelineSegment(start_time="00:00:00.100", end_time="00:00:01.000",
                            cut_type="hard_cut", source_index=0),
            TimelineSegment(start_time="00:00:01.500", end_time="00:00:02.400",
                            cut_type="jump_cut", source_index=0),
        ],
    )
    manager = JobManager()
    status = manager.create(JobRequest(
        prompt="smoke test",
        youtube_url="",
        clip_paths=[clip],
        credentials=JobCredentials(),          # deliberately empty: no network
        plan_override=plan,
        theme="ae_hype",
        enable_captions=False,
        enable_voiceover=False,
        enable_sfx=True,
        enable_transitions=True,
        review_plan=True,
        auto_highlight=False,
        auto_beat_sync=False,
        export_preset="720p60",
    ))

    finished = _wait(manager, status.job_id)
    assert finished.stage is JobStage.COMPLETED, (
        f"render failed: {finished.message} {finished.warnings}"
    )
    assert finished.output_size_bytes and finished.output_size_bytes > 10_000, (
        f"no usable output: {finished.output_size_bytes} bytes"
    )
    assert (finished.output_width, finished.output_height) == (720, 1280)
    assert finished.download_url, "a finished job must be downloadable"


def test_a_job_survives_having_no_credentials_at_all(clip):
    """Missing keys must degrade, not crash: the local edit still has to run."""
    plan = EditPlan(
        project_meta=ProjectMeta(resolution="360x640", fps=30),
        edit_timeline=[TimelineSegment(start_time="00:00:00.100",
                                       end_time="00:00:01.200", source_index=0)],
    )
    manager = JobManager()
    status = manager.create(JobRequest(
        prompt="no keys",
        youtube_url="",
        clip_paths=[clip],
        credentials=JobCredentials(),
        plan_override=plan,
        enable_captions=False,
        enable_voiceover=False,
        enable_sfx=False,
        enable_transitions=False,
        auto_beat_sync=False,
        export_preset="720p60",
    ))
    finished = _wait(manager, status.job_id)
    assert finished.stage is JobStage.COMPLETED, finished.message
