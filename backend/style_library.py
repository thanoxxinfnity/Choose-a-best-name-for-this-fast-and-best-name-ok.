"""Exemplar edit plans used to teach the planner by example.

A hosted model cannot be fine-tuned, so the effective way to raise output
quality is to *show* it good work: retrieve the exemplars closest to the
footage in front of it and put them in the prompt. That is what this module
does - a small curated corpus, plus anything the user's own good renders add to
it over time, retrieved by content type and energy.

Exemplars are stored as the same JSON the planner has to emit, so every one of
them is also a schema test: :func:`validate_library` loads each through
``EditPlan`` and refuses to ship a broken example.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import settings

logger = logging.getLogger(__name__)

LEARNED_DIRNAME = "learned_styles"
MAX_LEARNED = 60


@dataclass
class Exemplar:
    key: str
    content_type: str          # anime_edit, gaming, vlog, nature, ...
    energy: str                # low | medium | high
    theme: str                 # the editing mode it suits
    why: str                   # what makes this edit work - the lesson
    plan: Dict[str, Any]       # a valid EditPlan payload
    source: str = "curated"    # curated | learned
    score: float = 0.0         # engagement or user signal, for ranking

    def to_prompt_block(self) -> str:
        return (
            f"# {self.content_type} / {self.energy} energy / {self.theme}\n"
            f"# Why it works: {self.why}\n"
            + json.dumps(self.plan, ensure_ascii=False, separators=(",", ":"))
        )


# ---------------------------------------------------------------------------
# Curated corpus
# ---------------------------------------------------------------------------


def _anime_fight() -> Dict[str, Any]:
    return {
        "project_meta": {"resolution": "1080x1920", "fps": 60},
        "audio": {
            "use_puter_tts": False,
            "voice_accent": "deep_dark",
            "tts_lines": [],
        },
        "captions": {"enabled": True, "highlight_color": "#00E5FF"},
        "edit_timeline": [
            {"start_time": "00:00:00", "end_time": "00:00:01.200", "cut_type": "hard_cut",
             "source_index": 0,
             "text_overlay": {"text": "WATCH THIS", "style": "3d_pop", "position": "center"}},
            {"start_time": "00:00:04", "end_time": "00:00:05.400", "cut_type": "zoom_punch",
             "source_index": 0,
             "puter_sticker": {"generate_prompt": "electric blue lightning bolt impact",
                               "position": "top_right", "animation": "pop_up"}},
            {"start_time": "00:00:08", "end_time": "00:00:09.600", "cut_type": "speed_ramp",
             "source_index": 0, "speed": 1.4},
            {"start_time": "00:00:12", "end_time": "00:00:13.500", "cut_type": "jump_cut",
             "source_index": 0,
             "text_overlay": {"text": "NO WAY", "style": "3d_pop", "position": "bottom_center"}},
            {"start_time": "00:00:16", "end_time": "00:00:18", "cut_type": "zoom_punch",
             "source_index": 0,
             "puter_sticker": {"generate_prompt": "glowing red impact slash",
                               "position": "center", "animation": "pop_up"}},
        ],
    }


def _gaming_montage() -> Dict[str, Any]:
    return {
        "project_meta": {"resolution": "1080x1920", "fps": 60},
        "audio": {"use_puter_tts": True, "voice_accent": "hype",
                  "tts_lines": [
                      {"text": "Yeh clip dekh ke yakeen nahi hoga", "start_time": "00:00:00.400"},
                      {"text": "Aur phir yeh hua", "start_time": "00:00:09"},
                  ]},
        "captions": {"enabled": True, "highlight_color": "#FFD400"},
        "edit_timeline": [
            {"start_time": "00:00:02", "end_time": "00:00:04", "cut_type": "jump_cut",
             "source_index": 0,
             "text_overlay": {"text": "WAIT FOR IT", "style": "3d_pop", "position": "top_center"}},
            {"start_time": "00:00:07", "end_time": "00:00:08.500", "cut_type": "speed_ramp",
             "source_index": 0, "speed": 0.6},
            {"start_time": "00:00:11", "end_time": "00:00:13", "cut_type": "zoom_punch",
             "source_index": 0,
             "puter_sticker": {"generate_prompt": "3D golden crown, glossy",
                               "position": "top_center", "animation": "pop_up"}},
            {"start_time": "00:00:15", "end_time": "00:00:17", "cut_type": "jump_cut",
             "source_index": 0,
             "text_overlay": {"text": "INSANE", "style": "3d_pop", "position": "center"}},
        ],
    }


def _calm_nature() -> Dict[str, Any]:
    return {
        "project_meta": {"resolution": "1080x1920", "fps": 60},
        "audio": {"use_puter_tts": True, "voice_accent": "deep_dark",
                  "tts_lines": [
                      {"text": "Kabhi kabhi bas ruk jaana chahiye", "start_time": "00:00:01"},
                      {"text": "Yahi hai asli sukoon", "start_time": "00:00:08"},
                  ]},
        "captions": {"enabled": True, "highlight_color": "#FFFFFF"},
        "edit_timeline": [
            {"start_time": "00:00:00", "end_time": "00:00:04", "cut_type": "crossfade",
             "source_index": 0,
             "text_overlay": {"text": "JUST BREATHE", "style": "3d_pop", "position": "center"}},
            {"start_time": "00:00:06", "end_time": "00:00:10", "cut_type": "crossfade",
             "source_index": 0},
            {"start_time": "00:00:13", "end_time": "00:00:17", "cut_type": "crossfade",
             "source_index": 0,
             "puter_sticker": {"generate_prompt": "soft glowing golden sun flare",
                               "position": "top_center", "animation": "fade_in"}},
        ],
    }


def _horror_hook() -> Dict[str, Any]:
    return {
        "project_meta": {"resolution": "1080x1920", "fps": 60},
        "audio": {"use_puter_tts": True, "voice_accent": "horror_whisper",
                  "tts_lines": [
                      {"text": "Andhere mein kuch hai", "start_time": "00:00:00.600"},
                      {"text": "Aur woh dekh raha hai", "start_time": "00:00:06.500"},
                  ]},
        "captions": {"enabled": True, "highlight_color": "#9B5CFF"},
        "edit_timeline": [
            {"start_time": "00:00:00", "end_time": "00:00:04", "cut_type": "crossfade",
             "source_index": 0,
             "text_overlay": {"text": "DO NOT LOOK", "style": "3d_pop", "position": "center"}},
            {"start_time": "00:00:06", "end_time": "00:00:09", "cut_type": "jump_cut",
             "source_index": 0,
             "puter_sticker": {"generate_prompt": "cursed purple skull, dark",
                               "position": "center", "animation": "fade_in"}},
            {"start_time": "00:00:12", "end_time": "00:00:15", "cut_type": "zoom_punch",
             "source_index": 0},
        ],
    }


CURATED: List[Exemplar] = [
    Exemplar(
        key="anime_fight_fast",
        content_type="anime_edit", energy="high", theme="anime_edits",
        why=(
            "Cuts land every 1.2-2s on impact frames, the hook text is on screen "
            "inside the first second, and stickers only appear on hits - never on "
            "filler. No voiceover, because the track carries it."
        ),
        plan=_anime_fight(), score=1.0,
    ),
    Exemplar(
        key="gaming_montage_payoff",
        content_type="gaming", energy="high", theme="anime_edits",
        why=(
            "Opens on a promise ('WAIT FOR IT'), slows to 0.6x exactly on the "
            "payoff so the best moment reads, then pays off with a reward sticker. "
            "Two short voice lines, spaced far apart, never over the action."
        ),
        plan=_gaming_montage(), score=0.9,
    ),
    Exemplar(
        key="calm_nature_breath",
        content_type="nature", energy="low", theme="normal",
        why=(
            "Long 4s crossfades and no shake: the edit gets out of the footage's "
            "way. One idea per shot, and the voice lines sit in the gaps rather "
            "than on top of the visuals."
        ),
        plan=_calm_nature(), score=0.8,
    ),
    Exemplar(
        key="horror_slow_reveal",
        content_type="other", energy="low", theme="haunted",
        why=(
            "Withholds. The first line is a threat, the reveal is six seconds "
            "later, and the only sticker arrives with it. Whisper voice, long "
            "shots, one zoom punch at the end."
        ),
        plan=_horror_hook(), score=0.8,
    ),
]


# ---------------------------------------------------------------------------
# Learned exemplars
# ---------------------------------------------------------------------------


def _learned_dir() -> Path:
    directory = settings.data_dir / LEARNED_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def remember_plan(
    plan: Dict[str, Any],
    content_type: str,
    energy: str,
    theme: str,
    why: str = "",
    score: float = 0.5,
) -> Optional[Path]:
    """Keep a plan the user was happy with so later renders can learn from it."""
    if not plan.get("edit_timeline"):
        return None
    key = f"{content_type}_{int(time.time())}"
    payload = {
        "key": key, "content_type": content_type, "energy": energy, "theme": theme,
        "why": why or "Kept from a render the user saved.",
        "plan": plan, "source": "learned", "score": score,
    }
    path = _learned_dir() / f"{key}.json"
    try:
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not store the learned style: %s", exc)
        return None
    _prune_learned()
    return path


def _prune_learned() -> None:
    files = sorted(_learned_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
    for stale in files[:-MAX_LEARNED]:
        stale.unlink(missing_ok=True)


def load_learned() -> List[Exemplar]:
    exemplars: List[Exemplar] = []
    for path in _learned_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            exemplars.append(Exemplar(**data))
        except Exception:
            logger.debug("skipping unreadable learned style %s", path.name)
    return exemplars


def all_exemplars() -> List[Exemplar]:
    return CURATED + load_learned()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _score(exemplar: Exemplar, content_type: str, energy: str, theme: str) -> float:
    score = exemplar.score
    if exemplar.content_type == content_type:
        score += 3.0
    if exemplar.energy == energy:
        score += 1.5
    if exemplar.theme == theme:
        score += 1.0
    # A style the user kept beats a generic one at equal relevance.
    if exemplar.source == "learned":
        score += 0.5
    return score


def retrieve(
    content_type: str = "",
    energy: str = "",
    theme: str = "",
    limit: int = 2,
    pool: Optional[Sequence[Exemplar]] = None,
) -> List[Exemplar]:
    """The exemplars closest to the footage being edited."""
    candidates = list(pool if pool is not None else all_exemplars())
    if not candidates:
        return []
    candidates.sort(
        key=lambda item: _score(item, content_type, energy, theme), reverse=True
    )
    return candidates[:max(0, limit)]


def build_prompt_section(
    content_type: str = "",
    energy: str = "",
    theme: str = "",
    limit: int = 2,
) -> str:
    """The few-shot block injected into the planner prompt."""
    chosen = retrieve(content_type, energy, theme, limit)
    if not chosen:
        return ""
    blocks = "\n\n".join(exemplar.to_prompt_block() for exemplar in chosen)
    return (
        "[EDITS THAT WORK - study the shape, do not copy the timings]\n"
        "These are real timelines for footage like this one. Match how they "
        "pace cuts, where they put text, and how sparingly they use stickers. "
        "Your timestamps must come from the clips you were given.\n\n"
        + blocks
    )


def validate_library() -> List[str]:
    """Every exemplar must parse as an EditPlan; a broken example teaches badly."""
    from schemas import EditPlan

    problems: List[str] = []
    for exemplar in all_exemplars():
        try:
            plan = EditPlan.model_validate(exemplar.plan)
        except Exception as exc:
            problems.append(f"{exemplar.key}: {exc}")
            continue
        if not plan.edit_timeline:
            problems.append(f"{exemplar.key}: empty timeline")
    return problems
