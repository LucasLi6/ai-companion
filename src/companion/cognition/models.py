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
    # Provider-neutral delivery plan (delivery instruction, speaking rate, voice
    # profile) shaping *how* a reply is spoken. It never changes *whether* a
    # reply is sent, and it never authorises anything.
    speech_plan: Optional[dict] = Field(default=None)
    # Text-side, bounded stance influence derived from the same emotion state.
    # Unlike ``speech_plan`` it may shape wording; it never changes whether to
    # reply, what is true, or any permission.
    emotion_expression: Optional[dict] = Field(default=None)
    # Prompt-only feedback from the dialogue pattern monitor (repeated openings,
    # trailing questions, name overuse, formula repeats). It is guidance, not a
    # rewrite request and not an extra model call.
    dialogue_pattern_guidance: list[str] = Field(default_factory=list)
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
    # Trusted, admin-authored style references (max 3). These are style
    # standards, not real conversation history and never fixed replies.
    style_examples: list[dict] = Field(default_factory=list)
    # Short pragmatic reminders assembled from scene/action history.
    pragmatic_guidance: list[str] = Field(default_factory=list)
