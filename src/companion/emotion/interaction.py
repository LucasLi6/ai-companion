"""Per-event validation of a model-supplied interaction signal.

The model may describe *what it thinks happened in this utterance*, but it is
never allowed to decide who the event was directed at, whether it was asserted,
or that it happened at all. V0.4.1 used one global gate (``subject == speaker``
and ``factuality == asserted``) for every event type. That gate was both too
weak and too strong: it rejected a question or a request that is naturally not
asserted, while saying nothing precise about who an insult targeted.

This module replaces it with a small, explicit policy table:

- ``target`` is validated **per event type**;
- ``social_intent`` is validated per event type (an insult requires HOSTILE);
- ``factuality`` is validated per event type (interpersonal events must be
  asserted, a question may be uncertain, and a quoted/negated report never
  counts);
- the actor is the *trusted* speaker id supplied by the runtime, never a value
  the model chose.

The result is an immutable :class:`InteractionEvent` or a structured reason.
Nothing here mutates state or calls a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# What each event may target. ``UNKNOWN`` is retained only for the legacy
# speaker-event hints that predate the target field.
EVENT_TARGETS: dict[str, frozenset[str]] = {
    "achievement": frozenset({"SPEAKER", "UNKNOWN"}),
    "setback": frozenset({"SPEAKER", "UNKNOWN"}),
    "gratitude": frozenset({"AGENT", "UNKNOWN"}),
    "urgency": frozenset({"SPEAKER", "UNKNOWN"}),
    "question": frozenset({"AGENT", "SPEAKER", "OTHER", "UNKNOWN"}),
    "direct_insult": frozenset({"AGENT"}),
    "criticism": frozenset({"AGENT"}),
    "playful_teasing": frozenset({"AGENT"}),
    "apology": frozenset({"AGENT"}),
    "reconciliation": frozenset({"AGENT"}),
    "direct_praise": frozenset({"AGENT"}),
    "self_distress": frozenset({"SPEAKER"}),
    "support_request": frozenset({"AGENT", "UNKNOWN"}),
}

# Social intent required per event. In particular, an insult requires explicit
# HOSTILE intent; playful/uncertain wording never becomes hostility by keyword.
EVENT_INTENTS: dict[str, frozenset[str]] = {
    "direct_insult": frozenset({"HOSTILE"}),
    "criticism": frozenset({"HOSTILE", "SINCERE", "NEUTRAL", "UNCERTAIN"}),
    "playful_teasing": frozenset({"PLAYFUL"}),
    "apology": frozenset({"SINCERE"}),
    "reconciliation": frozenset({"SINCERE"}),
    "direct_praise": frozenset({"SINCERE", "PLAYFUL", "NEUTRAL"}),
    "self_distress": frozenset({"SINCERE", "NEUTRAL", "UNCERTAIN"}),
    "support_request": frozenset({"SINCERE", "NEUTRAL", "UNCERTAIN"}),
}

# Factuality allowed per event. A question is not an assertion, so it may be
# ``uncertain``; every other event must be a genuine assertion. Quoted,
# negated and hypothetical reports never move the companion's own affect.
EVENT_FACTUALITIES: dict[str, frozenset[str]] = {
    "achievement": frozenset({"asserted"}),
    "setback": frozenset({"asserted"}),
    "gratitude": frozenset({"asserted"}),
    "urgency": frozenset({"asserted"}),
    "question": frozenset({"asserted", "uncertain"}),
    "direct_insult": frozenset({"asserted"}),
    "criticism": frozenset({"asserted"}),
    "playful_teasing": frozenset({"asserted"}),
    "apology": frozenset({"asserted"}),
    "reconciliation": frozenset({"asserted"}),
    "direct_praise": frozenset({"asserted"}),
    "self_distress": frozenset({"asserted"}),
    "support_request": frozenset({"asserted"}),
}

SUPPORTED_EVENTS = frozenset(EVENT_TARGETS)

# Events whose legacy ``subject == speaker`` really does imply the target.
SPEAKER_EVENTS = frozenset({"achievement", "setback", "urgency", "self_distress"})

# Structured, auditable rejection reasons. They are values, not prose.
NO_EVENT = "no_event"
UNSUPPORTED_EVENT = "unsupported_event"
LOW_CONFIDENCE = "low_confidence"
INVALID_CONFIDENCE = "invalid_confidence"
NOT_ASSERTED = "not_asserted"
UNSUPPORTED_TARGET = "unsupported_target"
UNSUPPORTED_SOCIAL_INTENT = "unsupported_social_intent"
NOT_ADDRESSED_TO_AGENT = "not_addressed_to_agent"
INVALID_EVIDENCE = "evidence_not_in_text"


@dataclass(frozen=True)
class InteractionEvent:
    """A signal after trusted, per-event validation."""

    event_type: str
    target: str
    social_intent: str
    factuality: str
    confidence: float
    evidence_span: Optional[str]
    subject: str = "unknown"
    actor_person_id: Optional[str] = None


@dataclass(frozen=True)
class InteractionValidation:
    """Either a validated event or the exact reason it was rejected."""

    event: Optional[InteractionEvent]
    reason: str

    @property
    def ok(self) -> bool:
        return self.event is not None


def _enum(value, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().upper()


def validate_interaction(
    signal,
    text: str,
    *,
    addressee_hint: Optional[str] = None,
    actor_person_id: Optional[str] = None,
    min_confidence: float = 0.75,
) -> InteractionValidation:
    """Validate one model signal against the per-event policy.

    ``actor_person_id`` is the trusted speaker id captured by the runtime; it
    is carried on the result for audit and never read from the signal.
    """
    event_type = getattr(signal, "event_type", None)
    if event_type is None or event_type == "none":
        return InteractionValidation(None, NO_EVENT)
    if event_type not in SUPPORTED_EVENTS:
        return InteractionValidation(None, UNSUPPORTED_EVENT)

    try:
        confidence = float(getattr(signal, "confidence", 0.0))
    except (TypeError, ValueError):
        return InteractionValidation(None, INVALID_CONFIDENCE)
    if not math.isfinite(confidence):
        return InteractionValidation(None, INVALID_CONFIDENCE)
    if confidence < min_confidence:
        return InteractionValidation(None, LOW_CONFIDENCE)

    factuality = _enum(getattr(signal, "factuality", None), "UNCERTAIN").lower()
    if factuality not in EVENT_FACTUALITIES[event_type]:
        return InteractionValidation(None, NOT_ASSERTED)

    target = _enum(getattr(signal, "target", None), "UNKNOWN")
    subject = str(getattr(signal, "subject", "unknown") or "unknown").strip().lower()
    if target == "UNKNOWN":
        # Older direct callers set only the legacy subject. A third-party or
        # agent subject narrows the target; a speaker subject does so only for
        # the events that are intrinsically about the speaker. An explicit
        # target always wins, and ``subject`` is never a global precondition.
        if subject in {"other", "agent"}:
            target = subject.upper()
        elif subject == "speaker" and event_type in SPEAKER_EVENTS:
            target = "SPEAKER"
    if target not in EVENT_TARGETS[event_type]:
        return InteractionValidation(None, UNSUPPORTED_TARGET)

    intent = _enum(getattr(signal, "social_intent", None), "UNCERTAIN")
    if intent not in EVENT_INTENTS.get(event_type, frozenset({intent})):
        return InteractionValidation(None, UNSUPPORTED_SOCIAL_INTENT)

    # Trusted addressee routing: when the session says this turn is meant for
    # a human/unknown listener, a model claim that it targets the agent is not
    # enough. ``None`` keeps direct callers backward compatible; the
    # orchestrator always supplies the trusted value.
    if (
        target == "AGENT"
        and addressee_hint is not None
        and str(addressee_hint).strip().lower() != "agent"
    ):
        return InteractionValidation(None, NOT_ADDRESSED_TO_AGENT)

    evidence = getattr(signal, "evidence_span", None)
    if not evidence or not isinstance(evidence, str) or evidence not in text:
        return InteractionValidation(None, INVALID_EVIDENCE)

    return InteractionValidation(
        InteractionEvent(
            event_type=event_type,
            target=target,
            social_intent=intent,
            factuality=factuality,
            confidence=confidence,
            evidence_span=evidence,
            subject=subject,
            actor_person_id=actor_person_id,
        ),
        "valid",
    )
