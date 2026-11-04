"""Immutable per-turn effective agent context.

The orchestrator builds one of these for every utterance. It is never mutated
after construction, and it is stored on the transient ``CognitiveContext`` so
both the cognitive and response providers see exactly the same identity and
relationship configuration for that turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# Optional, admin-configurable emotional tendencies. They are compiled from
# trusted configuration (never from keyword-searching the character text) and
# only scale the existing numeric affect model; they never override the
# natural-language persona. Missing values fall back to the conservative 0.5
# midpoint.
DEFAULT_EMOTIONAL_DISPOSITIONS: dict[str, float] = {
    "tolerance_for_teasing": 0.5,
    "conflict_sensitivity": 0.5,
    "reconciliation_openness": 0.5,
    "expressiveness": 0.5,
}


@dataclass(frozen=True)
class CharacterSnapshot:
    """One trusted, per-turn view of who the companion is.

    Both the cognitive and the response provider receive the *same* snapshot
    object for a turn, so they can never silently disagree about the persona
    revision. It is built once by the orchestrator from the trusted
    ``AgentService`` and is never accepted from a model.
    """

    agent_id: str
    revision: int
    display_name: str
    description: str
    traits: Mapping[str, int] = field(default_factory=dict)
    persona_core: Mapping[str, Any] = field(default_factory=dict)
    relationship_tier: str = "STRANGER"
    relevant_stances: tuple[Mapping[str, Any], ...] = ()
    emotional_dispositions: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EMOTIONAL_DISPOSITIONS)
    )

    @property
    def character_description(self) -> str:
        return str(self.persona_core.get("character_description") or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "revision": self.revision,
            "display_name": self.display_name,
            "description": self.description,
            "traits": dict(self.traits),
            "persona_core": dict(self.persona_core),
            "character_description": self.character_description,
            "relationship_tier": self.relationship_tier,
            "relevant_stances": [dict(item) for item in self.relevant_stances],
            "emotional_dispositions": dict(self.emotional_dispositions),
        }


@dataclass(frozen=True)
class EffectiveAgentContext:
    agent_id: str
    display_name: str
    pronouns: str | None
    description: str | None
    person_id: str | None
    familiarity: str
    traits: Mapping[str, int] = field(default_factory=dict)
    overrides: Mapping[str, int] = field(default_factory=dict)
    persona_core: Mapping[str, Any] = field(default_factory=dict)
    response_conservatism: str = "balanced"
    interruption_threshold: str = "medium"
    curiosity_level: str = "low"
    revision: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "pronouns": self.pronouns,
            "description": self.description,
            "person_id": self.person_id,
            "familiarity": self.familiarity,
            "traits": dict(self.traits),
            "overrides": dict(self.overrides),
            "persona_core": dict(self.persona_core),
            "response_conservatism": self.response_conservatism,
            "interruption_threshold": self.interruption_threshold,
            "curiosity_level": self.curiosity_level,
            "revision": self.revision,
        }

    def trait_summary(self) -> str:
        return ", ".join(f"{name}={value}" for name, value in self.traits.items())
