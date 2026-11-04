"""Typed, centrally defined affect dynamics parameters (V0.4.1).

Every time constant and bound used by the emotion engine lives here so tests,
CLI diagnostics and production share one definition. The defaults are product
policy for a *simulated* affect model, not claims about human psychology.

The model has three time layers:

- **acute affect** — the immediate reaction to one event (minutes);
- **mood** — a slow, background shift over several events (tens of minutes);
- **relationship tension** — recent interpersonal friction (under an hour).

All values can be overridden with ``EMOTION_*`` environment variables, or by
constructing a different :class:`EmotionDynamicsConfig` explicitly.
"""

from __future__ import annotations

import os
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Per-emotion acute half-lives. These are suggested ranges, deliberately
# differentiated so a spike of curiosity fades sooner than sustained concern.
DEFAULT_EMOTION_HALF_LIVES: dict[str, float] = {
    "CURIOSITY": 180.0,
    "EXCITEMENT": 300.0,
    "FRUSTRATION": 300.0,
    "JOY": 600.0,
    "RELIEF": 600.0,
    "CONCERN": 600.0,
    "SADNESS": 900.0,
    "SERIOUS": 900.0,
    "NEUTRAL": 300.0,
}


class EmotionDynamicsConfig(BaseModel):
    """Validated affect parameters shared by dynamics, service and tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acute_half_life_seconds: float = Field(default=300.0, gt=0, le=86400)
    mood_half_life_seconds: float = Field(default=1800.0, gt=0, le=604800)
    tension_half_life_seconds: float = Field(default=2700.0, gt=0, le=604800)

    acute_intensity_cap: float = Field(default=0.85, gt=0, le=1.0)
    neutral_threshold: float = Field(default=0.02, ge=0.0, lt=0.5)

    # Fallback responsiveness when no profile is available. The profile still
    # wins when present; this keeps pure-function tests deterministic.
    responsiveness: float = Field(default=0.5, ge=0.0, le=1.0)

    repeated_event_window_seconds: float = Field(default=300.0, gt=0)
    repeated_event_gain_cap: float = Field(default=1.5, ge=1.0, le=3.0)

    text_expression_min_intensity: float = Field(default=0.20, ge=0.0, le=1.0)

    # How strongly an acute event leaks into mood, and how quickly tension
    # rises/falls. Small mood influence prevents one turn replacing mood.
    mood_influence: float = Field(default=0.20, ge=0.0, le=1.0)
    mood_intensity_cap: float = Field(default=0.70, gt=0, le=1.0)
    tension_increase: float = Field(default=0.80, ge=0.0, le=1.0)
    tension_decrease: float = Field(default=0.50, ge=0.0, le=1.0)

    # Bounds for text-expression adjustments. Kept here so the response policy
    # cannot silently become a personality rewrite.
    expression_adjustment_bound: float = Field(default=0.30, gt=0, le=1.0)

    # -- V0.4.2 conversation-driven affect -------------------------------
    # Time decay is auxiliary: OFF ignores elapsed time entirely; AUXILIARY
    # keeps the existing decay math *and* enables scene-relevance suppression
    # and ranking override; LEGACY is the V0.4.1 behaviour (decay only).
    time_decay_mode: Literal["OFF", "AUXILIARY", "LEGACY"] = "AUXILIARY"
    # Old, scene-irrelevant activations are scaled by
    # 1 - (1 - relevance) * suppression before a new event is composed.
    scene_relevance_suppression: float = Field(default=0.65, ge=0.0, le=1.0)
    # An additional suppression when the interaction frame itself changed.
    frame_change_suppression: float = Field(default=0.25, ge=0.0, le=1.0)
    # A new event at least this strong may take over the expressed emotion.
    ranking_override_min_impact: float = Field(default=0.35, ge=0.0, le=1.0)
    ranking_override_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    # Frame-change-only transitions (no discrete appraisal event).
    frame_transition_enabled: bool = True
    frame_transition_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    frame_transition_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)

    # -- V0.4.3 conversation transitions are independent of time decay ------
    # Conversation-driven change (scene relevance, frame transitions, same
    # frame progress) runs whenever this is enabled, in every time-decay mode
    # including OFF. Time decay is a separate mechanism.
    conversation_transition_enabled: bool = True
    # A new, evidence-backed development *inside* an unchanged frame (a joke
    # landing, a misunderstanding clearing) may still move affect. Merely
    # repeating the same frame does not.
    frame_same_event_enabled: bool = True
    frame_same_event_scale: float = Field(default=0.4, ge=0.0, le=1.0)
    frame_same_event_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)

    emotion_half_lives: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_EMOTION_HALF_LIVES)
    )

    @field_validator("emotion_half_lives")
    @classmethod
    def _validate_half_lives(cls, value: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, seconds in value.items():
            number = float(seconds)
            if number <= 0:
                raise ValueError(f"half-life for {key!r} must be positive")
            clean[str(key).upper()] = number
        return clean

    def half_life_for(self, emotion: str) -> float:
        """Acute half-life for a category, falling back to the global default."""
        return float(
            self.emotion_half_lives.get(
                str(emotion).upper(), self.acute_half_life_seconds
            )
        )


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_mode(
    env: Mapping[str, str], key: str, default: str
) -> Literal["OFF", "AUXILIARY", "LEGACY"]:
    raw = str(env.get(key) or "").strip().upper()
    return raw if raw in ("OFF", "AUXILIARY", "LEGACY") else default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_emotion_config(
    env: Optional[Mapping[str, str]] = None,
) -> EmotionDynamicsConfig:
    """Build a validated config from environment values (defaults on error)."""
    values = dict(os.environ if env is None else env)
    return EmotionDynamicsConfig(
        acute_half_life_seconds=_env_float(
            values, "EMOTION_ACUTE_HALF_LIFE_SECONDS", 300.0
        ),
        mood_half_life_seconds=_env_float(
            values, "EMOTION_MOOD_HALF_LIFE_SECONDS", 1800.0
        ),
        tension_half_life_seconds=_env_float(
            values, "EMOTION_TENSION_HALF_LIFE_SECONDS", 2700.0
        ),
        acute_intensity_cap=_env_float(
            values, "EMOTION_ACUTE_INTENSITY_CAP", 0.85
        ),
        neutral_threshold=_env_float(
            values, "EMOTION_NEUTRAL_THRESHOLD", 0.02
        ),
        responsiveness=_env_float(values, "EMOTION_RESPONSIVENESS", 0.5),
        repeated_event_window_seconds=_env_float(
            values, "EMOTION_REPEATED_EVENT_WINDOW_SECONDS", 300.0
        ),
        repeated_event_gain_cap=_env_float(
            values, "EMOTION_REPEATED_EVENT_GAIN_CAP", 1.5
        ),
        text_expression_min_intensity=_env_float(
            values, "EMOTION_TEXT_EXPRESSION_MIN_INTENSITY", 0.20
        ),
        mood_influence=_env_float(values, "EMOTION_MOOD_INFLUENCE", 0.20),
        mood_intensity_cap=_env_float(
            values, "EMOTION_MOOD_INTENSITY_CAP", 0.70
        ),
        tension_increase=_env_float(
            values, "EMOTION_TENSION_INCREASE", 0.80
        ),
        tension_decrease=_env_float(
            values, "EMOTION_TENSION_DECREASE", 0.50
        ),
        expression_adjustment_bound=_env_float(
            values, "EMOTION_EXPRESSION_ADJUSTMENT_BOUND", 0.30
        ),
        time_decay_mode=_env_mode(
            values, "EMOTION_TIME_DECAY_MODE", "AUXILIARY"
        ),
        scene_relevance_suppression=_env_float(
            values, "EMOTION_SCENE_RELEVANCE_SUPPRESSION", 0.65
        ),
        frame_change_suppression=_env_float(
            values, "EMOTION_FRAME_CHANGE_SUPPRESSION", 0.25
        ),
        ranking_override_min_impact=_env_float(
            values, "EMOTION_RANKING_OVERRIDE_MIN_IMPACT", 0.35
        ),
        ranking_override_strength=_env_float(
            values, "EMOTION_RANKING_OVERRIDE_STRENGTH", 0.55
        ),
        frame_transition_enabled=_env_bool(
            values, "EMOTION_FRAME_TRANSITION_ENABLED", True
        ),
        frame_transition_scale=_env_float(
            values, "EMOTION_FRAME_TRANSITION_SCALE", 1.0
        ),
        frame_transition_min_confidence=_env_float(
            values, "EMOTION_FRAME_TRANSITION_MIN_CONFIDENCE", 0.55
        ),
        conversation_transition_enabled=_env_bool(
            values, "EMOTION_CONVERSATION_TRANSITION_ENABLED", True
        ),
        frame_same_event_enabled=_env_bool(
            values, "EMOTION_FRAME_SAME_EVENT_ENABLED", True
        ),
        frame_same_event_scale=_env_float(
            values, "EMOTION_FRAME_SAME_EVENT_SCALE", 0.4
        ),
        frame_same_event_min_confidence=_env_float(
            values, "EMOTION_FRAME_SAME_EVENT_MIN_CONFIDENCE", 0.60
        ),
    )


DEFAULT_CONFIG = EmotionDynamicsConfig()
