"""Compact emotion-state description for the response prompt (V0.5).

The companion's affect is a *state*, not a script. This module turns the
trusted :class:`EmotionState` into at most one short, human-readable sentence
for the response prompt, and returns an empty string when nothing about the
current affect is worth mentioning.

What used to live here — a numeric ``directness / warmth / playfulness /
questioning / seriousness`` stance conversion — is gone. The model is given
the state and left to choose its own wording through its stable persona; the
system never hands it five floats to obey or a per-turn "be more direct" rule.

``SpeechPlanner`` remains the separate TTS/delivery concern and is not driven
from here. Nothing in this module changes whether to reply, what is true, or
any permission.
"""

from __future__ import annotations

from companion.emotion.models import EmotionState
from companion.emotion.parameters import (
    DEFAULT_CONFIG,
    EmotionDynamicsConfig,
)

# Category -> a short, non-clinical Chinese descriptor. Used only in the
# compact state description; never rendered as a diagnosis or a human feeling.
_EMOTION_LABEL = {
    "JOY": "轻快、愉快",
    "EXCITEMENT": "兴奋、来劲",
    "RELIEF": "松了口气",
    "CURIOSITY": "好奇、想多了解",
    "CONCERN": "关切、有点不放心",
    "SADNESS": "低落",
    "FRUSTRATION": "有些不快",
    "SERIOUS": "认真、专注",
    "NEUTRAL": "平稳",
}

_FRAME_LABEL = {
    "PLAYFUL": "轻松玩笑",
    "CONFLICT": "有些对立",
    "REPAIR": "在缓和关系",
    "SUPPORT": "在互相支持",
    "SERIOUS": "认真的交谈",
}


def _intensity_band(value: float) -> str:
    if value >= 0.65:
        return "强烈"
    if value >= 0.35:
        return "明显"
    return "轻微"


def compile_affect_context(
    state: EmotionState,
    *,
    interaction_frame: str | None = None,
    frame_changed: bool = False,
    config: EmotionDynamicsConfig | None = None,
) -> str:
    """A compact, human-readable state description for the response prompt.

    Returns an empty string when nothing about the current affect is worth
    mentioning. It never emits a forced "emotion is calm" line just to fill
    the prompt, and it never repeats a disclaimer about simulated emotion
    (that generic rule lives in the trusted system prompt). Only a significant
    current emotion, a meaningful mood, a lingering relationship tension or a
    known interaction frame produces text. Nothing here changes whether to
    reply, what is true, or any permission.
    """
    cfg = config or DEFAULT_CONFIG
    minimum = cfg.text_expression_min_intensity
    parts: list[str] = []

    acute_active = float(state.intensity or 0.0) >= minimum
    mood_active = float(state.mood_intensity or 0.0) >= minimum
    if acute_active:
        label = _EMOTION_LABEL.get(state.emotion, state.emotion)
        parts.append(f"此刻情绪偏向{label}（{_intensity_band(state.intensity)}）")
    elif mood_active:
        label = _EMOTION_LABEL.get(state.mood_emotion, state.mood_emotion)
        parts.append(f"此刻情绪平平，底色偏向{label}")

    if mood_active and acute_active:
        label = _EMOTION_LABEL.get(state.mood_emotion, state.mood_emotion)
        parts.append(f"心情底色偏向{label}（{_intensity_band(state.mood_intensity)}）")

    try:
        tension = float(state.relationship_tension or 0.0)
    except (TypeError, ValueError):
        tension = 0.0
    if tension >= 0.35:
        parts.append("两人之间有明显未散开的紧张")
    elif tension >= 0.15:
        parts.append("两人之间还残留一点紧张")

    frame = str(interaction_frame or "").upper()
    if frame in _FRAME_LABEL:
        prefix = "本轮气氛刚刚转为" if frame_changed else "本轮对话气氛是"
        parts.append(f"{prefix}{_FRAME_LABEL[frame]}")

    if not parts:
        return ""

    return "；".join(parts) + "。"
