"""Deterministic, three-layer affect dynamics (V0.4.1).

Layers and decay
----------------
Each layer follows exponential decay with its own half-life:

    value_t = baseline + (value_0 - baseline) * 2 ** (-elapsed / half_life)

- **acute affect** — category activations, minutes;
- **mood** — a slow background shift, tens of minutes;
- **relationship tension** — a scalar in [0, 1], under an hour.

Acute intensity uses saturating accumulation, not ``max()``:

    new = min(cap, decayed + (1 - decayed) * impact * sensitivity)

so repeated same-category stimuli compound (0.30 → 0.51 → 0.657 at impact
0.30) without instantly reaching 1.0. Opposite-valence stimulation reduces the
retained activation instead of being added to it.

Decay is lazy: state stores each layer's last update time and is folded forward
on read. Elapsed time is clamped at zero, so a clock rollback or an old
timestamp cannot amplify an emotion.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping, Optional

from companion.emotion.models import (
    AppraisalRule,
    EmotionProfile,
    EmotionState,
)
from companion.emotion.parameters import (
    DEFAULT_CONFIG,
    EmotionDynamicsConfig,
)

# A small category -> (valence, arousal) target used to turn per-category
# activation into the continuous summary. These are expressive targets, not a
# claim about a human's measured affect.
CATEGORY_AFFECT: dict[str, tuple[float, float]] = {
    "JOY": (0.65, 0.45),
    "EXCITEMENT": (0.80, 0.80),
    "RELIEF": (0.40, 0.15),
    "CURIOSITY": (0.25, 0.45),
    "CONCERN": (-0.35, 0.35),
    "SADNESS": (-0.60, 0.15),
    "FRUSTRATION": (-0.60, 0.70),
    "SERIOUS": (-0.05, 0.35),
    "NEUTRAL": (0.0, 0.0),
}

TENSION_RULES: dict[str, float] = {
    "direct_insult": 0.80,
    "repeated_hostility": 1.00,
    "agent_criticism": 0.45,
}


def _elapsed_seconds(updated_at: datetime | None, now: datetime) -> float:
    """Elapsed seconds, clamped at zero and normalised across naive/aware times."""
    if updated_at is None:
        return 0.0
    left = _as_utc(updated_at)
    right = _as_utc(now)
    return max(0.0, (right - left).total_seconds())


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def decay_factor(elapsed_seconds: float, half_life_seconds: float) -> float:
    """Return ``2 ** (-elapsed / half_life)``, clamped to [0, 1]."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if elapsed_seconds <= 0:
        return 1.0
    return math.pow(2.0, -elapsed_seconds / half_life_seconds)


def derive_acute(
    activations: Mapping[str, float],
    *,
    cap: float = 0.85,
    neutral_threshold: float = 0.02,
) -> tuple[str, float, float, float, dict[str, float]]:
    """Derive (emotion, valence, arousal, intensity, clean activations).

    The categorical label is the strongest activation, never a separate
    threshold rule that could contradict the values. Valence/arousal are the
    activation-weighted category targets, bounded by the cap.
    """
    clean = {
        str(key).upper(): clamp(float(value), 0.0, cap)
        for key, value in (activations or {}).items()
        if float(value) > neutral_threshold
    }
    if not clean:
        return "NEUTRAL", 0.0, 0.0, 0.0, {}
    emotion = max(clean, key=lambda key: clean[key])
    intensity = clean[emotion]
    valence = sum(
        CATEGORY_AFFECT.get(key, (0.0, 0.0))[0] * value
        for key, value in clean.items()
    )
    arousal = sum(
        CATEGORY_AFFECT.get(key, (0.0, 0.0))[1] * value
        for key, value in clean.items()
    )
    return (
        emotion,
        clamp(valence, -1.0, 1.0),
        clamp(arousal, 0.0, 1.0),
        intensity,
        clean,
    )


def decay(
    state: EmotionState,
    *,
    now: datetime | None = None,
    baseline_valence: float = 0.0,
    baseline_arousal: float = 0.0,
    config: EmotionDynamicsConfig | None = None,
) -> EmotionState:
    """Return ``state`` advanced to ``now`` across all three layers.

    The state's stored timestamps are preserved for composition when the caller
    does not persist; a caller that checkpoints writes ``now`` back. Per-category
    half-lives apply to activated emotions, so curiosity fades before concern.
    """
    cfg = config or DEFAULT_CONFIG
    now = now or datetime.now(timezone.utc)

    if str(getattr(cfg, "time_decay_mode", "AUXILIARY")).upper() == "OFF":
        # Time decay is auxiliary in V0.4.2. In OFF mode elapsed time never
        # changes affect; only conversational events do.
        return state

    activations = dict(state.activations or {})
    elapsed = _elapsed_seconds(state.updated_at, now)
    if activations:
        decayed: dict[str, float] = {}
        for emotion_name, value in activations.items():
            factor = decay_factor(elapsed, cfg.half_life_for(emotion_name))
            remaining = float(value) * factor
            if remaining > cfg.neutral_threshold:
                decayed[emotion_name] = remaining
        emotion, valence, arousal, intensity, clean = derive_acute(
            decayed,
            cap=cfg.acute_intensity_cap,
            neutral_threshold=cfg.neutral_threshold,
        )
        if not clean:
            valence = baseline_valence
            arousal = baseline_arousal
    else:
        # Legacy snapshot without category activations: decay its scalar
        # summary, preserving the old value until a new stimulus activates a
        # category. No activation is fabricated.
        factor = decay_factor(elapsed, state.half_life_seconds)
        intensity = state.intensity * factor
        if intensity <= cfg.neutral_threshold:
            emotion = "NEUTRAL"
            valence = baseline_valence
            arousal = baseline_arousal
            intensity = 0.0
        else:
            valence = clamp(
                baseline_valence + (state.valence - baseline_valence) * factor,
                -1.0,
                1.0,
            )
            arousal = clamp(
                baseline_arousal + (state.arousal - baseline_arousal) * factor,
                0.0,
                1.0,
            )
            emotion = label_for(
                valence,
                arousal,
                intensity,
                neutral_threshold=cfg.neutral_threshold,
            )
        clean = {}

    mood_elapsed = _elapsed_seconds(state.mood_updated_at or state.updated_at, now)
    mood_factor = decay_factor(mood_elapsed, cfg.mood_half_life_seconds)
    mood_valence = baseline_valence + (
        state.mood_valence - baseline_valence
    ) * mood_factor
    mood_arousal = baseline_arousal + (
        state.mood_arousal - baseline_arousal
    ) * mood_factor
    mood_intensity = state.mood_intensity * mood_factor
    if mood_intensity <= cfg.neutral_threshold:
        mood_valence = baseline_valence
        mood_arousal = baseline_arousal
        mood_intensity = 0.0
        mood_emotion = "NEUTRAL"
    else:
        mood_emotion = label_for(
            mood_valence,
            mood_arousal,
            mood_intensity,
            neutral_threshold=cfg.neutral_threshold,
        )

    tension_elapsed = _elapsed_seconds(
        state.tension_updated_at or state.updated_at, now
    )
    tension = state.relationship_tension * decay_factor(
        tension_elapsed, cfg.tension_half_life_seconds
    )
    if tension <= cfg.neutral_threshold:
        tension = 0.0

    return EmotionState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        emotion=emotion,
        updated_at=now,
        half_life_seconds=cfg.half_life_for(emotion),
        activations=clean,
        mood_valence=clamp(mood_valence, -1.0, 1.0),
        mood_arousal=clamp(mood_arousal, 0.0, 1.0),
        mood_intensity=clamp(mood_intensity, 0.0, cfg.mood_intensity_cap),
        mood_emotion=mood_emotion,
        mood_updated_at=now,
        mood_half_life_seconds=cfg.mood_half_life_seconds,
        relationship_tension=clamp(tension, 0.0, 1.0),
        tension_updated_at=now,
        tension_half_life_seconds=cfg.tension_half_life_seconds,
        last_event_type=state.last_event_type,
        last_event_at=state.last_event_at,
    )


def label_for(
    valence: float,
    arousal: float,
    intensity: float,
    *,
    neutral_threshold: float = 0.02,
) -> str:
    """Derive a categorical mood label from continuous affect."""
    if intensity < neutral_threshold:
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
    profile: EmotionProfile | None,
    traits: Mapping[str, int] | None,
    config: EmotionDynamicsConfig | None = None,
) -> tuple[float, float]:
    """Reaction strength from responsiveness and empathy.

    ``responsiveness`` 0.0/0.5/1.0 maps to gain 0.0/1.0/2.0. Empathy applies a
    modest ×0.8–×1.2 factor only to concern/sadness events, so a warm persona
    can soften or deepen a reaction but never flip its direction.
    """
    cfg = config or DEFAULT_CONFIG
    responsiveness = (
        profile.responsiveness if profile is not None else cfg.responsiveness
    )
    gain = clamp(float(responsiveness) / 0.5, 0.0, 2.0)

    traits = traits or {}
    try:
        empathy = float(traits.get("empathy", 50)) / 100.0
    except (TypeError, ValueError):
        empathy = 0.5
    empathy = clamp(empathy, 0.0, 1.0)
    return clamp(gain, 0.0, 2.0), empathy


def _is_same_direction(current: str, new: str) -> bool:
    old_v = CATEGORY_AFFECT.get(current, (0.0, 0.0))[0]
    new_v = CATEGORY_AFFECT.get(new, (0.0, 0.0))[0]
    if old_v == 0.0 or new_v == 0.0:
        return True
    return (old_v > 0) == (new_v > 0)


def _apply_tension(
    current: float,
    rule: AppraisalRule,
    impact: float,
    config: EmotionDynamicsConfig,
) -> float:
    name = rule.name
    if name in TENSION_RULES:
        delta = TENSION_RULES[name] * impact
        return clamp(current + delta * (1.0 - current), 0.0, 1.0)
    if name in ("agent_apology", "relationship_repair"):
        # Repair lowers tension but never instantly zeroes it.
        return clamp(
            current - current * config.tension_decrease * impact, 0.0, 1.0
        )
    return current


def apply_emotional_stimulus(
    current: EmotionState,
    rule: AppraisalRule,
    profile: EmotionProfile | None,
    traits: Mapping[str, int] | None,
    now: datetime,
    *,
    config: EmotionDynamicsConfig | None = None,
) -> EmotionState:
    """Pure function: compose one rule's stimulus with the decayed state."""
    cfg = config or DEFAULT_CONFIG
    gain, empathy = reaction_gain(profile, traits, cfg)
    empathy_factor = (
        0.8 + 0.4 * empathy if rule.emotion in ("CONCERN", "SADNESS") else 1.0
    )
    gain = clamp(gain * empathy_factor, 0.0, 2.0)

    impact = clamp(rule.intensity * gain, 0.0, 1.0)
    if impact <= 0.0:
        return current

    # Repeated *hostility* (selected locally after verified repeats) gets a
    # bounded extra weight. Ordinary repeated stimuli use the plain saturating
    # formula so 0.30 → 0.51 → 0.657 remains predictable.
    if (
        rule.name == "repeated_hostility"
        and current.last_event_type in ("direct_insult", "repeated_hostility")
        and current.last_event_at is not None
        and _elapsed_seconds(current.last_event_at, now)
        <= cfg.repeated_event_window_seconds
    ):
        # Keep the documented 1.15 bounded boost, but honour a stricter
        # operator cap if one is configured.
        repeated_gain = max(1.0, min(1.15, cfg.repeated_event_gain_cap))
        impact = min(1.0, impact * repeated_gain)

    activations = dict(current.activations or {})
    if not activations and current.intensity > cfg.neutral_threshold:
        activations = {current.emotion: current.intensity}

    dominant = max(activations, key=activations.get) if activations else "NEUTRAL"
    if dominant != "NEUTRAL" and not _is_same_direction(dominant, rule.emotion):
        # Opposite valence: mitigate the old activation instead of adding to it.
        activations = {
            key: value * max(0.0, 1.0 - impact)
            for key, value in activations.items()
        }

    previous = float(activations.get(rule.emotion, 0.0))
    activations[rule.emotion] = min(
        cfg.acute_intensity_cap, previous + (1.0 - previous) * impact
    )

    emotion, valence, arousal, intensity, clean = derive_acute(
        activations,
        cap=cfg.acute_intensity_cap,
        neutral_threshold=cfg.neutral_threshold,
    )

    target_valence, target_arousal = CATEGORY_AFFECT.get(
        rule.emotion, (0.0, 0.0)
    )
    mood_impact = impact * cfg.mood_influence
    mood_valence = clamp(
        current.mood_valence
        + target_valence * mood_impact * (1.0 - abs(current.mood_valence)),
        -1.0,
        1.0,
    )
    mood_arousal = clamp(
        current.mood_arousal
        + target_arousal * mood_impact * (1.0 - abs(current.mood_arousal)),
        0.0,
        1.0,
    )
    mood_intensity = min(
        cfg.mood_intensity_cap,
        current.mood_intensity
        + (1.0 - current.mood_intensity) * mood_impact * 0.5,
    )
    mood_emotion = label_for(
        mood_valence,
        mood_arousal,
        mood_intensity,
        neutral_threshold=cfg.neutral_threshold,
    )

    tension = _apply_tension(
        current.relationship_tension, rule, impact, cfg
    )

    return EmotionState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        emotion=emotion,
        updated_at=now,
        half_life_seconds=cfg.half_life_for(emotion),
        activations=clean,
        mood_valence=mood_valence,
        mood_arousal=mood_arousal,
        mood_intensity=mood_intensity,
        mood_emotion=mood_emotion,
        mood_updated_at=now,
        mood_half_life_seconds=cfg.mood_half_life_seconds,
        relationship_tension=tension,
        tension_updated_at=now,
        tension_half_life_seconds=cfg.tension_half_life_seconds,
        last_event_type=rule.name,
        last_event_at=now,
    )
