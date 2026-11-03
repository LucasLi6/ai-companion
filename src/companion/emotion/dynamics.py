"""Deterministic emotion decay.

The temporary emotion magnitude follows an exponential decay:

    I(t) = I0 * 2 ** (-elapsed_seconds / half_life_seconds)

and valence/arousal drift back toward their baseline with the same factor. This
is computed *lazily* on read (there is no background timer writing to the
database): the stored state keeps ``updated_at`` and ``half_life_seconds``, and
:func:`decay` folds the elapsed time in whenever state is read. A decayed state
is optionally checkpointed back to disk in bounded batches by the service, but
decay never deletes the underlying event memory or appraisal rules.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from companion.emotion.models import EmotionState

_NEUTRAL_INTENSITY = 0.02


def _elapsed_seconds(updated_at: datetime | None, now: datetime) -> float:
    if updated_at is None:
        return 0.0
    delta = now - updated_at
    return max(0.0, delta.total_seconds())


def decay_factor(elapsed_seconds: float, half_life_seconds: float) -> float:
    """Return ``2 ** (-elapsed / half_life)``, clamped to [0, 1]."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if elapsed_seconds <= 0:
        return 1.0
    return math.pow(2.0, -elapsed_seconds / half_life_seconds)


def decay(
    state: EmotionState,
    *,
    now: datetime | None = None,
    baseline_valence: float = 0.0,
    baseline_arousal: float = 0.0,
) -> EmotionState:
    """Return ``state`` advanced to ``now`` (default: the current UTC time).

    Valence and arousal move toward their baselines and intensity toward zero
    with the same half-life factor, so after enough time the state returns to
    ``NEUTRAL``. The ``updated_at`` timestamp is preserved so further decay can
    be composed; callers that persist the result may refresh it themselves.
    """
    now = now or datetime.now(timezone.utc)
    elapsed = _elapsed_seconds(state.updated_at, now)
    factor = decay_factor(elapsed, state.half_life_seconds)

    valence = baseline_valence + (state.valence - baseline_valence) * factor
    arousal = baseline_arousal + (state.arousal - baseline_arousal) * factor
    intensity = state.intensity * factor

    if intensity < _NEUTRAL_INTENSITY:
        return EmotionState(
            valence=baseline_valence,
            arousal=baseline_arousal,
            intensity=0.0,
            emotion="NEUTRAL",
            updated_at=now,
            half_life_seconds=state.half_life_seconds,
        )
    return EmotionState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        emotion=state.emotion,
        updated_at=now,
        half_life_seconds=state.half_life_seconds,
    )


def label_for(valence: float, arousal: float, intensity: float) -> str:
    """Derive a categorical label from continuous affect (neutral below a floor)."""
    if intensity < _NEUTRAL_INTENSITY:
        return "NEUTRAL"
    if valence >= 0.25:
        if arousal >= 0.55:
            return "EXCITEMENT"
        if arousal < 0.2:
            return "RELIEF"
        return "JOY"
    if valence <= -0.25:
        if arousal >= 0.55:
            return "FRUSTRATION"
        return "SADNESS"
    if valence < 0.0:
        return "CONCERN"
    if arousal < 0.15:
        return "RELIEF"
    return "NEUTRAL"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
