"""Persistent, local conversation-scene state (V0.4).

The scene is a *short-horizon* description of what a conversation with one
person is currently about: its topic, topic stack, discourse phase and tone, a
small set of unfinished threads, and any environment detail the speaker
literally stated. It is deliberately not a long-term fact store:

- only explicitly evidenced environment details are retained, and they decay
  out of the prompt after a few turns;
- open loops are bounded, must be supported by a local pending-follow-up
  signal, and are dropped rather than accumulated;
- a scene proposal from the model is never trusted on its own; the
  :class:`~companion.cognition.scene_manager.SceneManager` merges it locally.

All state lives in the same CMF ``.memory`` database. The models here are
plain Pydantic objects with a stable JSON representation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from companion.domain.models import DomainModel, SceneProposal, utcnow

SCENE_STATE_VERSION = 1
MAX_OPEN_LOOPS = 5
MAX_TOPIC_STACK = 5
MAX_SOURCE_EVENTS = 30
MAX_EVIDENCE_ITEMS = 12
OPEN_LOOP_TTL_TURNS = 8
ENVIRONMENT_TTL_TURNS = 6

UNKNOWN = "UNKNOWN"

INTERACTION_FRAMES = {
    "CASUAL",
    "PLAYFUL",
    "SERIOUS",
    "CONFLICT",
    "REPAIR",
    "SUPPORT",
    "TASK",
    "UNKNOWN",
}
INTERACTION_PHASES = {"OPENING", "ONGOING", "WRAPPING_UP", "CLOSED", "UNKNOWN"}


class SceneEvidence(BaseModel):
    """A trusted pointer to the utterance that supports a scene field."""

    event_id: str = Field(min_length=1, max_length=128)
    quote: Optional[str] = Field(default=None, max_length=512)
    kind: str = Field(default="UTTERANCE", max_length=32)


class OpenLoop(BaseModel):
    """One unfinished conversational thread.

    ``loop_id`` is generated locally; a model never supplies it. The loop is
    only created when the local nonresponse trace says the turn has a pending
    follow-up, so a routine WAIT cannot manufacture an engagement obligation.
    """

    loop_id: str = Field(min_length=1, max_length=128)
    topic: str = Field(default=UNKNOWN, max_length=120)
    summary: str = Field(min_length=1, max_length=300)
    opened_event_id: str = Field(min_length=1, max_length=128)
    last_event_id: str = Field(min_length=1, max_length=128)
    evidence_span: Optional[str] = Field(default=None, max_length=512)
    opened_at: str
    turns_since_touch: int = Field(default=0, ge=0)


def default_environment() -> dict:
    """An unknown environment. No sensor data exists, so this is the default."""

    return {
        "location": UNKNOWN,
        "other_people_present": UNKNOWN,
        "status": "UNKNOWN",
        "evidence_event_id": None,
        "age_turns": 0,
    }


class InteractionFrameState(BaseModel):
    """The current *interaction* frame, as distinct from the topic scene.

    A topic ("travel") can stay the same while the frame changes ("playful"
    teasing becomes a real conflict). The frame is derived from trusted,
    locally validated pragmatic hints; it is bounded and persisted inside the
    scene snapshot. It informs emotion transitions and response wording, never
    permissions or actions.
    """

    frame: str = Field(default="UNKNOWN", max_length=32)
    phase: str = Field(default="UNKNOWN", max_length=32)
    open_setup: Optional[str] = Field(default=None, max_length=200)
    last_social_act: str = Field(default="UNKNOWN", max_length=32)
    expectation: str = Field(default="", max_length=200)
    revision: int = Field(default=0, ge=0)

    def prompt_dict(self) -> dict:
        return {
            "frame": self.frame,
            "phase": self.phase,
            "open_setup": self.open_setup,
            "last_social_act": self.last_social_act,
            "expectation": self.expectation,
            "revision": self.revision,
        }


class SceneState(DomainModel):
    """The committed scene for one ``(conversation_id, person_id)`` scope."""

    state_version: int = SCENE_STATE_VERSION
    conversation_id: str = Field(min_length=1, max_length=128)
    person_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(default=0, ge=0)
    topic: str = Field(default=UNKNOWN, max_length=200)
    topic_stack: list[str] = Field(default_factory=list)
    mode: str = Field(default="UNKNOWN", max_length=32)
    discourse_phase: str = Field(default="UNCERTAIN", max_length=32)
    tone: str = Field(default="UNCERTAIN", max_length=32)
    interaction: InteractionFrameState = Field(default_factory=InteractionFrameState)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    # Environment values are only ever populated from an explicit, evidenced
    # statement. ``status`` is UNKNOWN unless that evidence existed.
    environment: dict = Field(default_factory=default_environment)
    source_event_ids: list[str] = Field(default_factory=list)
    last_committed_event_id: Optional[str] = Field(default=None, max_length=128)
    evidence: list[SceneEvidence] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)

    def prompt_dict(self) -> dict:
        """A bounded, prompt-safe view. Unknown environment stays unknown."""

        env = self.environment or {}
        location = env.get("location") or UNKNOWN
        present = env.get("other_people_present") or UNKNOWN
        if env.get("status") != "EXPLICIT" or int(env.get("age_turns", 0) or 0) > ENVIRONMENT_TTL_TURNS:
            location = UNKNOWN
            present = UNKNOWN
        return {
            "topic": self.topic,
            "topic_stack": list(self.topic_stack),
            "mode": self.mode,
            "discourse_phase": self.discourse_phase,
            "tone": self.tone,
            "interaction": self.interaction.prompt_dict(),
            "open_loops": [
                {"topic": loop.topic, "summary": loop.summary}
                for loop in self.open_loops[:MAX_OPEN_LOOPS]
            ],
            "environment": {
                "location": location,
                "other_people_present": present,
                "provenance": "EXPLICIT" if env.get("status") == "EXPLICIT" else "UNKNOWN",
            },
            "revision": self.revision,
        }


def normalize_proposal(proposal) -> Optional[SceneProposal]:
    """Return a typed proposal or ``None`` without ever raising."""

    if proposal is None:
        return None
    if isinstance(proposal, SceneProposal):
        return proposal
    if isinstance(proposal, dict):
        try:
            return SceneProposal.model_validate(proposal)
        except (ValueError, TypeError):
            return None
    return None
