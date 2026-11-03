"""Emotion domain models for V0.3.

Emotion is modelled as *computational affect and expression* — a deliberate,
deterministic simulation of the companion's own presentation state. It is never
presented as a claim about a human's real emotional state, and it is strictly
separated into:

- **stable personality** (``EmotionProfile``) — how the companion tends to
  respond emotionally, configured per agent and never inferred from a single
  turn;
- **persistent appraisal rules** (``AppraisalRule``) — the durable, evidence-
  backed rules that map a situation to an emotion; changes are versioned and
  audited;
- **temporary state** (``EmotionState``) — the current, decaying affect;
- **the reply's tone** (``SpeechStylePlan``) — provider-neutral delivery
  instructions that affect *how* a reply sounds, never *whether* it is sent.

``VALENCE`` is in [-1, 1] (negative..positive), ``AROUSAL`` in [0, 1] (calm..
energised) and ``INTENSITY`` in [0, 1] (how strongly the emotion is expressed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import Field

from companion.domain.models import DomainModel

# Categorical labels. ``NEUTRAL`` is the baseline; the rest are the expressive
# states the appraisal rules may produce.
EMOTION_LABELS = (
    "NEUTRAL",
    "JOY",
    "EXCITEMENT",
    "RELIEF",
    "CURIOSITY",
    "CONCERN",
    "SADNESS",
    "FRUSTRATION",
    "SERIOUS",
)


class EmotionProfile(DomainModel):
    """Stable per-agent emotional personality. Configured, not inferred."""

    agent_id: str = Field(min_length=1, max_length=128)
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0
    default_half_life_seconds: int = 3600
    responsiveness: float = 0.5
    expression_style: dict = Field(default_factory=dict)
    revision: int = 1
    created_at: datetime
    updated_at: datetime


class AppraisalRule(DomainModel):
    """A persistent, evidence-backed rule that maps a situation to an emotion."""

    rule_id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    trigger_type: str = Field(min_length=1, max_length=32)
    pattern: dict = Field(default_factory=dict)
    emotion: str = Field(min_length=1, max_length=32)
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    intensity: float = 0.5
    priority: int = 0
    status: str = "ACTIVE"
    source: str = "system"
    evidence_ref: Optional[str] = None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class EmotionState:
    """Temporary affect with three separately decaying layers (V0.4.1).

    ``valence``/``arousal``/``intensity``/``emotion`` are the **acute** affect
    summary kept for backward compatibility. ``activations`` records the
    per-category acute activation used to derive that summary, so a strong older
    emotion is not overwritten by a weak new rule and labels stay consistent
    with values.

    ``mood_*`` is a slower background shift; ``relationship_tension`` is recent
    interpersonal friction in [0, 1]. The three layers have independent
    timestamps and half-lives, so one turn cannot replace the others.
    """

    valence: float = 0.0
    arousal: float = 0.0
    intensity: float = 0.0
    emotion: str = "NEUTRAL"
    updated_at: Optional[datetime] = None
    half_life_seconds: float = 300.0
    activations: dict = field(default_factory=dict)

    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    mood_intensity: float = 0.0
    mood_emotion: str = "NEUTRAL"
    mood_updated_at: Optional[datetime] = None
    mood_half_life_seconds: float = 1800.0

    relationship_tension: float = 0.0
    tension_updated_at: Optional[datetime] = None
    tension_half_life_seconds: float = 2700.0

    last_event_type: Optional[str] = None
    last_event_at: Optional[datetime] = None

    def as_dict(self) -> dict:
        return {
            # Legacy acute fields remain at the top level for compatibility.
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "intensity": round(self.intensity, 4),
            "emotion": self.emotion,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "half_life_seconds": self.half_life_seconds,
            "acute": {
                "valence": round(self.valence, 4),
                "arousal": round(self.arousal, 4),
                "intensity": round(self.intensity, 4),
                "emotion": self.emotion,
                "activations": {
                    key: round(float(value), 4)
                    for key, value in (self.activations or {}).items()
                },
            },
            "mood": {
                "valence": round(self.mood_valence, 4),
                "arousal": round(self.mood_arousal, 4),
                "intensity": round(self.mood_intensity, 4),
                "emotion": self.mood_emotion,
            },
            "relationship_tension": round(self.relationship_tension, 4),
        }


@dataclass(frozen=True)
class AppraisalResult:
    """The structured, validated result of appraising one utterance.

    ``emotion``/``valence``/``arousal``/``intensity`` are the resulting acute
    summary. ``trigger_rule_id`` and ``reason`` record *why*, so the decision is
    auditable; ``matched_memory_id`` links the emotion to the durable memory
    that triggered it (the emotion may decay, the reason persists).

    ``diagnostic`` distinguishes the many ways a turn can produce NO_CHANGE
    (no event, low confidence, invalid evidence, unsupported target, disabled
    rule, no matching rule) instead of logging every miss as an emotion change.
    """

    emotion: str
    valence: float
    arousal: float
    intensity: float
    trigger_rule_id: Optional[str] = None
    reason: Optional[str] = None
    matched_memory_id: Optional[str] = None
    diagnostic: str = "NO_CHANGE"
    event_type: Optional[str] = None
    target: Optional[str] = None
    social_intent: Optional[str] = None
    validation: str = "none"
    acute_before: float = 0.0
    acute_after: float = 0.0
    mood_after: float = 0.0
    tension_after: float = 0.0
    # The complete resulting state (all three layers + activations) when the
    # appraisal produced one. ``None`` for callers constructing a legacy
    # AppraisalResult by hand.
    state: Optional[EmotionState] = None

    @property
    def applied(self) -> bool:
        return self.trigger_rule_id is not None


@dataclass(frozen=True)
class SpeechStylePlan:
    """Provider-neutral delivery instructions for an emotion-aware reply.

    These affect *how* the reply is delivered (rhythm, warmth, formality, word
    choice, question style, enthusiasm) — never *whether* a reply is sent, and
    never permission, privacy, silence or factual content.
    """

    text: str = ""
    emotion: str = "NEUTRAL"
    intensity: float = 0.0
    speaking_rate: float = 1.0
    delivery_instruction: str = ""
    voice_profile_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "intensity": round(self.intensity, 4),
            "speaking_rate": round(self.speaking_rate, 4),
            "delivery_instruction": self.delivery_instruction,
            "voice_profile_id": self.voice_profile_id,
            "extra": dict(self.extra),
        }


class VoiceProfile(DomainModel):
    """A provider-neutral voice configuration (no real TTS calls)."""

    voice_profile_id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    provider: str = Field(default="none", min_length=1, max_length=64)
    voice_id: Optional[str] = None
    language: str = "zh-CN"
    speaking_rate: float = 1.0
    pitch: Optional[float] = None
    extra: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
