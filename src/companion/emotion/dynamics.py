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
from typing import Mapping

from companion.emotion.models import (
    AppraisalRule,
    EmotionProfile,
    EmotionState,
)

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


def reaction_gain(
    profile: EmotionProfile | None, traits: Mapping[str, int] | None
) -> tuple[float, float]:
    """Initial V0.3.1 reaction strength from responsiveness and empathy.

    ``responsiveness`` 0.0/0.5/1.0 maps to gain 0.0/1.0/2.0. Empathy applies a
    modest ×0.8–×1.2 factor only to concern/sadness events, so a warm persona
    can soften or deepen a reaction but never flip its direction. This is an
    explicit product policy, not a psychological standard.
    """
    responsiveness = profile.responsiveness if profile is not None else 0.5
    gain = clamp(float(responsiveness) / 0.5, 0.0, 2.0)

    traits = traits or {}
    try:
        empathy = float(traits.get("empathy", 50)) / 100.0
    except (TypeError, ValueError):
        empathy = 0.5
    empathy = clamp(empathy, 0.0, 1.0)
    return clamp(gain, 0.0, 2.0), empathy


def apply_emotional_stimulus(
    current: EmotionState,
    rule: AppraisalRule,
    profile: EmotionProfile | None,
    traits: Mapping[str, int] | None,
    now: datetime,
) -> EmotionState:
    """Pure function: compose one rule's stimulus with the decayed state.

    No database, no clock, no mutation of the inputs. Kept replaceable so a
    future affect model can evolve without touching EmotionService.
    """
    gain, empathy = reaction_gain(profile, traits)
    empathy_factor = (
        0.8 + 0.4 * empathy if rule.emotion in ("CONCERN", "SADNESS") else 1.0
    )
    gain = clamp(gain * empathy_factor, 0.0, 2.0)

    new_valence = clamp(current.valence + rule.valence_delta * gain, -1.0, 1.0)
    new_arousal = clamp(current.arousal + rule.arousal_delta * gain, 0.0, 1.0)
    event_intensity = clamp(rule.intensity * gain, 0.0, 1.0)
    retained_intensity = current.intensity * 0.70
    new_intensity = max(retained_intensity, event_intensity)

    if event_intensity < 0.02:
        emotion = current.emotion
    elif event_intensity >= retained_intensity:
        emotion = rule.emotion
    else:
        emotion = current.emotion

    half_life = (
        float(profile.default_half_life_seconds)
        if profile is not None
        else current.half_life_seconds
    )
    return EmotionState(
        valence=new_valence,
        arousal=new_arousal,
        intensity=new_intensity,
        emotion=emotion,
        updated_at=now,
        half_life_seconds=half_life,
    )
