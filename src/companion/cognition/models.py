"""In-memory models for the foreground cognition context (Phase 3).

These are transient, never persisted: they exist to carry the *assembled* view
of a conversation — current utterance, recent multi-turn history with
speaker labels, and retrieved memories — into the cognitive and response
providers, so a real DeepSeek provider receives one well-formed object rather
than reconstructing it from a bare ``ConversationContext``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

ROLE_HUMAN = "human"
ROLE_ASSISTANT = "assistant"


class CognitiveTurn(BaseModel):
    """One turn in the recent conversation history (oldest first)."""

    event_id: str = Field(min_length=1, max_length=128)
    role: str  # "human" | "assistant"
    speaker_person_id: Optional[str] = Field(default=None, max_length=128)
    speaker_label: str = Field(default="unknown", max_length=200)
    addressee_hint: Optional[str] = Field(default=None, max_length=32)
    text: str = Field(min_length=0, max_length=8192)


class CognitiveContext(BaseModel):
    """The full, privacy-checked context handed to an external provider."""

    conversation_id: str = Field(min_length=1, max_length=128)
    current_event_id: str = Field(min_length=1, max_length=128)
    current_speaker_id: Optional[str] = Field(default=None, max_length=128)
    current_speaker_label: str = Field(default="unknown", max_length=200)
    current_text: str = Field(min_length=1, max_length=8192)
    current_addressee_hint: str = Field(default="unknown", max_length=32)
    quiet_mode: bool = False
    turns: list[CognitiveTurn] = Field(default_factory=list)
    memories: list[dict] = Field(default_factory=list)
    # Immutable snapshot of the Agent Core for this turn: identity, personality
    # and (for the current speaker) relationship tier. Trusted application data,
    # never written by a model.
    agent_context: Optional[dict] = Field(default=None)
    # Truthful results of the memory write-through for this turn. These are
    # trusted application data (the provider never fills them) so the response
    # can acknowledge only what was actually committed.
    memory_write_outcomes: list[dict] = Field(default_factory=list)
    # Trusted, locally computed emotional state for this turn (valence, arousal,
    # intensity, categorical label). Optional: present only when the emotion
    # framework is wired in. It is the companion's own simulated affect, never
    # an assertion about a human's real feelings.
    emotion_state: Optional[dict] = Field(default=None)
    # Provider-neutral delivery plan (delivery instruction, speaking rate, voice
    # profile) shaping *how* a reply is spoken. It never changes *whether* a
    # reply is sent, and it never authorises anything.
    speech_plan: Optional[dict] = Field(default=None)
