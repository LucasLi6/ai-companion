"""Immutable per-turn effective agent context.

The orchestrator builds one of these for every utterance. It is never mutated
after construction, and it is stored on the transient ``CognitiveContext`` so
both the cognitive and response providers see exactly the same identity and
relationship configuration for that turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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
