"""Validated personality traits and relationship presets.

Personality is *stylistic*, never probabilistic. Trait values are integers in
``0..100`` and are combined deterministically:

1. the agent's global persona (base);
2. a relationship preset delta for the current person's familiarity tier;
3. absolute per-person overrides;
4. temporary, per-conversation style adjustments;
5. the local Policy configuration, which remains authoritative and is never
   modified by a memory file or by a model.

The existing ``policy.response_conservatism``, ``policy.interruption_threshold``
and ``policy.curiosity_level`` settings stay in ``app_settings`` and are passed
through to the effective context; they are not duplicated as integer persona
fields. The persona's ``curiosity`` trait is conversational style, while
``policy.curiosity_level`` governs whether the runtime may initiate background
work.
"""

from __future__ import annotations

PERSONA_TRAITS: tuple[str, ...] = (
    "humor",
    "warmth",
    "curiosity",
    "formality",
    "verbosity",
    "assertiveness",
    "empathy",
    "playfulness",
    "patience",
)

DEFAULT_PERSONA: dict[str, int] = {
    "humor": 50,
    "warmth": 60,
    "curiosity": 60,
    "formality": 45,
    "verbosity": 45,
    "assertiveness": 45,
    "empathy": 65,
    "playfulness": 50,
    "patience": 65,
}

TRAIT_MIN = 0
TRAIT_MAX = 100

FAMILIARITY_TIERS: tuple[str, ...] = (
    "STRANGER",
    "ACQUAINTANCE",
    "FAMILIAR",
    "CLOSE",
)

DEFAULT_FAMILIARITY = "STRANGER"

# Preset deltas mirror the seeded relationship_presets rows. They are used only
# as a fallback if a database predates the seed; the database row wins.
DEFAULT_PRESET_DELTAS: dict[str, dict[str, int]] = {
    "STRANGER": {"humor": -5, "warmth": -5, "formality": 10, "verbosity": -10},
    "ACQUAINTANCE": {"warmth": 5, "formality": 5},
    "FAMILIAR": {"humor": 10, "warmth": 15, "formality": -10, "verbosity": 5},
    "CLOSE": {
        "humor": 15,
        "warmth": 25,
        "formality": -20,
        "verbosity": 10,
        "playfulness": 10,
    },
}


class PersonalityValidationError(ValueError):
    pass


class RelationshipValidationError(ValueError):
    pass


def validate_trait(name: str, value: object) -> int:
    if name not in PERSONA_TRAITS:
        raise PersonalityValidationError(
            f"unknown personality trait {name!r}; expected one of {PERSONA_TRAITS}"
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersonalityValidationError(f"{name} must be an integer 0..100")
    if not TRAIT_MIN <= value <= TRAIT_MAX:
        raise PersonalityValidationError(
            f"{name} must be between {TRAIT_MIN} and {TRAIT_MAX}, got {value}"
        )
    return value


def validate_tier(tier: str) -> str:
    value = str(tier).strip().upper()
    if value not in FAMILIARITY_TIERS:
        raise RelationshipValidationError(
            f"unknown familiarity tier {tier!r}; expected one of {FAMILIARITY_TIERS}"
        )
    return value


def clamp(value: int) -> int:
    return max(TRAIT_MIN, min(TRAIT_MAX, int(value)))
