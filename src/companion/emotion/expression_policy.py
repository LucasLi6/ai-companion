"""Bounded emotion-to-text expression influence (V0.4.1).

``SpeechPlanner`` turns affect into *voice delivery* instructions (rate, pitch,
TTS hints). Those must not shape the generated words: a speaking-rate value in a
text prompt nudges the model toward a performance rather than a stance.

This module is the text-side counterpart. It converts the trusted emotion state
into a small, bounded adjustment of stance:

- directness / warmth / playfulness / questioning / seriousness in [-bound, +bound];
- a short natural-language explanation, for diagnostics and prompt rendering;
- **no** action, permission, silence or factual authority.

Low-intensity affect returns all zeros, so a 0.1 curiosity blip cannot rewrite
the persona. The gate is composite since V0.4.3: acute intensity, background
mood and relationship tension are considered separately, so a calm acute layer
does not hide a meaningful mood or lingering tension. Intensity and mood add
gradually; relationship tension raises seriousness/directness but is capped well
below a personality rewrite. The personality compiler applies the base style;
this adjustment is layered on top, never the reverse.

V0.4.3 also exposes :func:`compile_affect_context`, the compact state
description the response prompt actually uses. It replaces numeric
"style commands": the model reads a short description of the character's
simulated state, not five floats to obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

# Category -> (directness, warmth, playfulness, questioning, seriousness)
_STANCE: dict[str, tuple[float, float, float, float, float]] = {
    "JOY": (0.0, 0.6, 0.5, 0.0, -0.1),
    "EXCITEMENT": (0.2, 0.6, 0.7, 0.2, -0.3),
    "RELIEF": (0.0, 0.5, 0.2, -0.1, -0.2),
    "CURIOSITY": (0.1, 0.1, 0.1, 0.7, 0.0),
    # Concern is attention, not a script for comfort. V0.4.2 removes the
    # automatic warmth/question bump: it slightly lowers playfulness and raises
    # seriousness, but it must not make every reply softer or end in a question.
    "CONCERN": (-0.05, 0.15, -0.35, -0.10, 0.35),
    "SADNESS": (-0.2, 0.5, -0.6, -0.2, 0.3),
    "FRUSTRATION": (0.6, -0.6, -0.7, -0.2, 0.5),
    "SERIOUS": (0.3, 0.0, -0.4, 0.0, 0.8),
    "NEUTRAL": (0.0, 0.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class EmotionExpressionInfluence:
    directness_adjustment: float = 0.0
    warmth_adjustment: float = 0.0
    playfulness_adjustment: float = 0.0
    questioning_adjustment: float = 0.0
    seriousness_adjustment: float = 0.0
    explanation: str = ""
    source_emotion: str = "NEUTRAL"
    intensity: float = 0.0
    extra: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        return any(
            abs(value) > 1e-9
            for value in (
                self.directness_adjustment,
                self.warmth_adjustment,
                self.playfulness_adjustment,
                self.questioning_adjustment,
                self.seriousness_adjustment,
            )
        )

    def as_dict(self) -> dict:
        return {
            "directness_adjustment": round(self.directness_adjustment, 4),
            "warmth_adjustment": round(self.warmth_adjustment, 4),
            "playfulness_adjustment": round(self.playfulness_adjustment, 4),
            "questioning_adjustment": round(self.questioning_adjustment, 4),
            "seriousness_adjustment": round(self.seriousness_adjustment, 4),
            "explanation": self.explanation,
            "source_emotion": self.source_emotion,
            "intensity": round(self.intensity, 4),
        }


def compile_expression_influence(
    state: EmotionState,
    *,
    config: EmotionDynamicsConfig | None = None,
    interaction_frame: str | None = None,
) -> EmotionExpressionInfluence:
    """Derive a bounded stance adjustment; never an action or content change.

    The gate is the strongest of the three layers rather than the acute value
    alone: a low acute spike must not mask a lasting mood or tension, and an
    old high acute value must not be treated as current when mood and tension
    have settled.
    """
    cfg = config or DEFAULT_CONFIG
    signal = _expression_signal(state)
    if signal < cfg.text_expression_min_intensity:
        return EmotionExpressionInfluence(
            explanation="情绪强度低，不改变文本立场。",
            source_emotion=state.emotion,
            intensity=state.intensity,
        )

    # When acute is quiet, the mood's own label is the better base.
    base_emotion = state.emotion
    if state.intensity < cfg.text_expression_min_intensity:
        if state.mood_intensity >= cfg.text_expression_min_intensity:
            base_emotion = state.mood_emotion
    base = _STANCE.get(base_emotion, _STANCE["NEUTRAL"])
    # Acute intensity contributes most; mood is a slower, smaller influence.
    acute_weight = state.intensity
    mood_weight = min(cfg.mood_influence, state.mood_intensity) * 0.5
    scale = min(1.0, acute_weight + mood_weight)
    values = [component * scale for component in base]

    # Recent relationship tension raises directness/seriousness and lowers
    # playfulness, but is capped so one rough exchange cannot rewrite warmth.
    tension = min(0.5, max(0.0, state.relationship_tension))
    values[0] += tension * 0.25
    values[2] -= tension * 0.20
    values[4] += tension * 0.20

    bound = cfg.expression_adjustment_bound
    bounded = [max(-bound, min(bound, value)) for value in values]
    if not any(abs(value) > 1e-9 for value in bounded):
        return EmotionExpressionInfluence(
            explanation="情绪强度低，不改变文本立场。",
            source_emotion=state.emotion,
            intensity=state.intensity,
        )

    return EmotionExpressionInfluence(
        directness_adjustment=bounded[0],
        warmth_adjustment=bounded[1],
        playfulness_adjustment=bounded[2],
        questioning_adjustment=bounded[3],
        seriousness_adjustment=bounded[4],
        explanation=_explain(state, bounded),
        source_emotion=state.emotion,
        intensity=state.intensity,
    )


def _explain(state: EmotionState, values: list[float]) -> str:
    """Short, inspectable reason for the adjustment (no psychological claim)."""
    names = {
        0: "更直接",
        1: "更温和",
        2: "更轻松/俏皮",
        3: "更愿意追问",
        4: "更认真",
    }
    parts = [
        names[index]
        for index, value in enumerate(values)
        if value > 0.05
    ]
    parts += [
        names[index].replace("更", "少一些")
        for index, value in enumerate(values)
        if value < -0.05
    ]
    detail = "、".join(parts) if parts else "基本不改变"
    return (
        f"当前{state.emotion}（强度 {state.intensity:.2f}）让表达{detail}；"
        "只影响措辞与立场，不改变是否回复或任何权限。"
    )


def _expression_signal(state: EmotionState) -> float:
    """The strongest meaningful layer, so no one layer hides another."""
    try:
        tension = float(state.relationship_tension or 0.0)
    except (TypeError, ValueError):
        tension = 0.0
    return max(
        float(state.intensity or 0.0),
        float(state.mood_intensity or 0.0) * 0.5,
        max(0.0, tension) * 0.6,
    )


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

    Deliberately not a list of numeric adjustments: the character system should
    sound like a person in a mood, not execute a style vector. Nothing here
    changes whether to reply, what is true, or any permission.
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
    else:
        parts.append("此刻情绪比较平稳")

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

    if len(parts) == 1 and parts[0] == "此刻情绪比较平稳" and frame not in _FRAME_LABEL:
        return ""

    return (
        "；".join(parts)
        + "。这是角色当下的内在状态，只用来把握语气和分寸；"
        "自然带出来即可，不必逐条说明，也不要把它描述成真实的身体感受。"
    )
