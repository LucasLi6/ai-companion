"""In-memory models for the foreground cognition context (Phase 3).

These are transient, never persisted: they exist to carry the *assembled* view
of a conversation — current utterance, recent multi-turn history with
speaker labels, and retrieved memories — into the cognitive and response
providers, so a real DeepSeek provider receives one well-formed object rather
than reconstructing it from a bare ``ConversationContext``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ROLE_HUMAN = "human"
ROLE_ASSISTANT = "assistant"


class ActionHistoryRecord(BaseModel):
    """A past *executed-but-unspoken* action, separate from real dialogue turns.

    WAIT/IGNORE are not assistant utterances: the companion never said "I chose
    silence". This record lets the model know it previously saw an utterance
    and chose not to reply, with the final cause that trusted code recorded.
    """

    event_id: str = Field(min_length=1, max_length=128)
    final_action: Literal["WAIT", "IGNORE"]
    cause_code: str = Field(min_length=1, max_length=64)
    cause_source: str = Field(min_length=1, max_length=32)
    pending_followup: bool = False
    reentry_condition: str = Field(default="NONE", max_length=64)


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
    # Past WAIT/IGNORE decisions, kept separate from ``turns`` so an internal
    # action is never rendered as something the companion actually said.
    action_history: list[ActionHistoryRecord] = Field(default_factory=list)
    memories: list[dict] = Field(default_factory=list)
    # Immutable snapshot of the Agent Core for this turn: identity, personality
    # and (for the current speaker) relationship tier. Trusted application data,
    # never written by a model.
    agent_context: Optional[dict] = Field(default=None)
    # V0.4.3: one trusted character view compiled once per turn and shared by
    # the cognitive and response providers. It carries the persona revision so
    # the two calls cannot silently disagree about who the character is.
    character_snapshot: Optional[dict] = Field(default=None)
    # V0.4.3: short, human-readable description of the current conversational
    # affect (state, not an instruction). Computed locally from the emotion
    # state and interaction frame.
    affect_context: Optional[str] = Field(default=None)
    # Truthful results of the memory write-through for this turn. These are
    # trusted application data (the provider never fills them) so the response
    # can acknowledge only what was actually committed.
    memory_write_outcomes: list[dict] = Field(default_factory=list)
    # Trusted, locally computed emotional state for this turn (valence, arousal,
    # intensity, categorical label). Optional: present only when the emotion
    # framework is wired in. It is the companion's own simulated affect, never
    # an assertion about a human's real feelings.
    emotion_state: Optional[dict] = Field(default=None)
    # Trusted current scene produced locally by SceneManager. It is a
    # short-horizon description of this conversation with this person, never a
    # long-term fact and never a claim about physical reality without evidence.
    scene_state: Optional[dict] = Field(default=None)
    # Locally compiled delivery style (thresholds and short guidance). It is
    # derived from the existing persona/relationship data, not a new source of
    # identity, and it never changes permissions or reply necessity.
    dialogue_style: Optional[dict] = Field(default=None)
    # Compiled personality section for the response prompt. It only contains
    # stylistic guidance.
    personality: Optional[dict] = Field(default=None)
    # Short pragmatic reminders assembled from scene/action history.
    pragmatic_guidance: list[str] = Field(default_factory=list)
    # V0.6: per-event runtime state (local time, temporal gaps, re-entry state,
    # optional network-egress location). Derived and persisted once per event
    # so a retry sees the same values. Prompt data, never an instruction.
    runtime_snapshot: Optional[dict] = Field(default=None)
    # V0.6.1: outcome of this turn's local memory lookup. ``hits`` means related
    # memories were retrieved, ``empty`` means nothing matched the query (which
    # does NOT mean the database is empty), ``none`` means no lookup ran.
    memory_retrieval_status: str = Field(default="none", max_length=16)
    # Real, access-checked memory records selected for an explicit "show me a
    # memory" request. Trusted local output; the model may cite these, not
    # invent others.
    memory_samples: list[dict] = Field(default_factory=list)
    # Real, access-checked earliest conversation turns returned for an explicit
    # "what did I first say" request.
    history_excerpt: list[dict] = Field(default_factory=list)
    # How many recent turns the local builder actually loaded for this request.
    history_loaded: int = 0
    # Trusted, read-only counts for the current speaker when they explicitly ask
    # how much conversation history exists. ``None`` means no count was
    # requested (never a guess of zero).
    conversation_stats: Optional[dict] = Field(default=None)
    # V0.7: short, trusted summary of the *current* speaker's confirmed
    # understanding profile. It is background data about this person only,
    # never another person's profile and never an instruction.
    person_profile_context: Optional[str] = Field(default=None, max_length=800)
    # V0.8: the validated semantic route for this turn, rendered for the
    # speaker as *model judgment* (not a trusted instruction). Never contains
    # the router's raw output or hidden chain of thought.
    route_context: Optional[dict] = Field(default=None)
    # V0.8: truthful, bounded results of read-only tools requested by the
    # router. External/web content is labelled untrusted by the executor.
    tool_results: list[dict] = Field(default_factory=list)
