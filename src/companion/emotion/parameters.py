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
from typing import Mapping, Optional

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
    )


DEFAULT_CONFIG = EmotionDynamicsConfig()
