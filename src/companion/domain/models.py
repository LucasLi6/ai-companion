"""Pydantic models that define the wire format of the domain.

Every model is validated by Pydantic on construction:

- ``datetime`` fields must be timezone-aware UTC values (naive or offset
  datetimes are rejected).
- Required identifiers must be non-empty strings and are bounded in length.
- Enum fields reject illegal statuses/action types at construction time.

The storage layer serialises datetimes to ISO-8601 strings and parses them
back on read.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import (
    AddresseeHint,
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
    JobStatus,
    JobType,
    MemoryCandidateStatus,
    MemoryKind,
    MemoryOperation,
    PersonRole,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
    ReplyNecessity,
)


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Generate a stable, unique id with a human-readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


class DomainModel(BaseModel):
    """Base model enforcing timezone-aware UTC on every ``datetime`` field."""

    @model_validator(mode="after")
    def _validate_utc_datetimes(self) -> "DomainModel":
        for name, value in self.__dict__.items():
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{name} must be a timezone-aware UTC datetime")
                if value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be UTC, got offset {value.utcoffset()}")
        return self


class Person(DomainModel):
    """A registered, authorised (or formerly authorised) person."""

    person_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=200)
    role: PersonRole
    registration_status: RegistrationStatus = RegistrationStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class Consent(DomainModel):
    """Explicit, per-person privacy grants.

    ``allow_processing`` gates whether any input may enter the cognitive flow;
    ``allow_persistence`` gates durable storage of utterances;
    ``allow_memory_learning`` gates whether memory candidates may be created;
    and ``allow_external_ai`` gates whether a person's private context may be
    sent to an *external* AI model. These are independent — processing is not
    persistence, persistence is not learning, and local learning is **not**
    permission to upload private data to a remote model. V0.0 never defaults
    these to ``True``.
    """

    person_id: str = Field(min_length=1, max_length=128)
    allow_processing: bool
    allow_persistence: bool
    allow_memory_learning: bool
    allow_external_ai: bool = False
    updated_at: datetime


class UtteranceEvent(DomainModel):
    """A single, fully-confirmed input utterance.

    ``speaker_person_id`` identifies the *speaker*, not the person discussed.
    It is ``None`` for unregistered speakers, who may only carry a transient
    ``speaker_label`` and never enter persistent processing.

    The *identity fingerprint* of an event is the tuple of immutable content
    fields returned by :meth:`identity_fingerprint`: ``conversation_id``,
    ``speaker_person_id``, ``speaker_label``, ``text``, ``source`` and
    ``addressee_hint``. ``event_id`` is the key; ``occurred_at`` and
    ``processing_status`` are *not* part of identity (the former is caller
    timing, the latter is lifecycle state). Re-submitting an id with a
    different fingerprint is a ``CONFLICT``; re-submitting an identical one is
    idempotent.
    """

    event_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    speaker_person_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    speaker_label: Optional[str] = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=8192)
    source: EventSource
    addressee_hint: AddresseeHint
    occurred_at: datetime
    processing_status: ProcessingStatus = ProcessingStatus.PENDING

    def identity_fingerprint(self) -> tuple:
        """Return the immutable content fields that define event identity."""
        return (
            self.conversation_id,
            self.speaker_person_id,
            self.speaker_label,
            self.text,
            self.source,
            self.addressee_hint,
        )


class MemoryCandidate(DomainModel):
    """A candidate memory proposed by a cognitive provider.

    The semantic fields (``assertion_type``, ``certainty_category``,
    ``explicit_remember_request``, ``topic``) are *model suggestions*: the
    trusted consolidation layer still decides whether a durable memory may be
    written. ``candidate_id``, ``source_event_id`` and ``subject_person_id``
    are always filled by the program, never by the model.
    """

    candidate_id: str = Field(min_length=1, max_length=128)
    source_event_id: str = Field(min_length=1, max_length=128)
    subject_person_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=4096)
    operation: MemoryOperation
    status: MemoryCandidateStatus = MemoryCandidateStatus.PENDING
    assertion_type: AssertionType = AssertionType.UNKNOWN
    certainty_category: CertaintyCategory = CertaintyCategory.UNKNOWN
    explicit_remember_request: bool = False
    topic: Optional[str] = Field(default=None, max_length=128)
    kind: Optional[MemoryKind] = None
    source_evidence_span: Optional[str] = Field(default=None, max_length=4096)


class AffectiveSignal(DomainModel):
    """A minimal, locally validated semantic affect hint from the model.

    The model may describe what the *current utterance explicitly asserts* —
    never a diagnosis of the speaker's mental state. Every field is checked by
    trusted code before use:

    - whitelisted ``event_type`` / ``subject`` / ``factuality``;
    - ``confidence`` a finite float in [0, 1];
    - ``evidence_span`` a verbatim substring of the current input.

    A missing or invalid signal is ignored (``None``) rather than failing the
    conversation. This signal can never write a memory or authorise an action.
    """

    event_type: Literal[
        "achievement",
        "setback",
        "gratitude",
        "urgency",
        "question",
        "none",
    ] = "none"
    subject: Literal["speaker", "other", "agent", "unknown"] = "unknown"
    factuality: Literal[
        "asserted",
        "negated",
        "hypothetical",
        "quoted",
        "uncertain",
    ] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = Field(default=None, max_length=512)


_AFT_EVENT_TYPES = {"achievement", "setback", "gratitude", "urgency", "question", "none"}
_AFT_SUBJECTS = {"speaker", "other", "agent", "unknown"}
_AFT_FACTUALITIES = {"asserted", "negated", "hypothetical", "quoted", "uncertain"}


def validate_affective_signal(raw, text: str) -> Optional[AffectiveSignal]:
    """Validate a model-supplied signal locally; return ``None`` when invalid.

    The model's ids, confidence and prose are never trusted on their own. The
    evidence span must be a real substring of the current input, and ``none``
    is allowed without evidence (it is an explicit "no event" answer).
    """
    if raw is None:
        return None
    if isinstance(raw, AffectiveSignal):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        return None

    event_type = data.get("event_type")
    subject = data.get("subject")
    factuality = data.get("factuality")
    if event_type not in _AFT_EVENT_TYPES:
        return None
    if subject not in _AFT_SUBJECTS:
        return None
    if factuality not in _AFT_FACTUALITIES:
        return None

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None

    evidence = data.get("evidence_span")
    if evidence is not None and not isinstance(evidence, str):
        return None
    if event_type == "none":
        return AffectiveSignal(
            event_type="none",
            subject=subject,
            factuality=factuality,
            confidence=confidence,
            evidence_span=None,
        )
    if not evidence or not evidence.strip():
        return None
    if len(evidence) > 512 or evidence not in text:
        return None
    return AffectiveSignal(
        event_type=event_type,
        subject=subject,
        factuality=factuality,
        confidence=confidence,
        evidence_span=evidence,
    )


class CognitiveResult(DomainModel):
    """Structured output of a cognitive provider.

    ``action`` is the provider's *suggestion*; the Policy layer owns the final
    decision on whether that suggestion is allowed to execute. The extra
    structured fields (``intent``, ``addressed_to``, ``silence_requested``,
    ``reply_necessity``, ``conversation_state``) exist so the policy can make
    a deterministic decision rather than guessing from prose.
    """

    event_id: str = Field(min_length=1, max_length=128)
    meaning: str = Field(min_length=1, max_length=2048)
    action: CognitiveAction
    reason_code: ReasonCode
    referenced_memory_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    response_style: str = Field(default="mock_plain", min_length=1, max_length=128)
    provider: str = Field(default="mock", min_length=1, max_length=128)
    intent: str = Field(default="unknown", min_length=1, max_length=128)
    addressed_to: Optional[str] = Field(default=None, max_length=32)
    silence_requested: bool = False
    reply_necessity: ReplyNecessity = ReplyNecessity.OPTIONAL
    conversation_state: str = Field(default="unknown", min_length=1, max_length=128)
    # Optional, locally validated semantic event hint. None for older records,
    # mock providers and invalid model output.
    affective_signal: Optional[AffectiveSignal] = None


class AgentAction(DomainModel):
    """What the agent ultimately did for a single event.

    ``model_suggested_action`` and ``policy_final_action`` are recorded
    separately so a mismatch is inspectable. ``override_reason`` explains a
    local override; it never rewrites the model's original suggestion.
    """

    action_id: str = Field(min_length=1, max_length=128)
    event_id: str = Field(min_length=1, max_length=128)
    action_type: CognitiveAction
    response_text: Optional[str] = Field(default=None, max_length=8192)
    reason_code: ReasonCode
    status: str = Field(default="COMPLETED", min_length=1, max_length=32)
    created_at: datetime
    model_suggested_action: Optional[CognitiveAction] = None
    policy_final_action: Optional[CognitiveAction] = None
    override_reason: Optional[str] = Field(default=None, max_length=128)


class ConversationContext(DomainModel):
    """In-memory snapshot of a conversation's persistent state."""

    conversation_id: str = Field(min_length=1, max_length=128)
    quiet_mode: bool = False
    recent_utterance_ids: list[str] = Field(default_factory=list)
    last_action: Optional[str] = Field(default=None, max_length=32)
    updated_at: Optional[datetime] = None


class ProcessingJob(DomainModel):
    """A durable, recoverable background job."""

    job_id: str = Field(min_length=1, max_length=128)
    job_key: str = Field(min_length=1, max_length=256)
    job_type: JobType
    payload: dict = Field(default_factory=dict)
    status: JobStatus
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Lease/ownership (Phase A): which worker owns an in-flight job, when the
    # lease expires (so a crashed worker's job can be reclaimed), the last
    # heartbeat, and an optional scheduled next-run time.
    owner_id: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None


class ProcessingReport(DomainModel):
    """Result of ``Orchestrator.process_utterance``.

    ``steps`` carries the human-readable ``[STAGE] ...`` lines the CLI prints,
    while ``status``/``action``/``reason_code`` carry the machine-readable
    outcome.
    """

    event_id: str = Field(min_length=1, max_length=128)
    status: str
    action: Optional[CognitiveAction] = None
    response_text: Optional[str] = None
    reason_code: Optional[ReasonCode] = None
    analysis_id: Optional[str] = None
    job_id: Optional[str] = None
    error: Optional[str] = None
    memory_write_outcomes: list[dict] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


def make_utterance_event(
    *,
    conversation_id: str,
    text: str,
    addressee_hint: AddresseeHint | str,
    speaker_person_id: str | None = None,
    speaker_label: str | None = None,
    source: EventSource | str = EventSource.CLI,
    event_id: str | None = None,
    occurred_at: datetime | None = None,
) -> UtteranceEvent:
    """Construct a validated ``UtteranceEvent`` with sensible defaults."""
    return UtteranceEvent(
        event_id=event_id if event_id is not None else new_id("evt"),
        conversation_id=conversation_id,
        speaker_person_id=speaker_person_id,
        speaker_label=speaker_label,
        text=text,
        source=EventSource(source),
        addressee_hint=AddresseeHint(addressee_hint),
        occurred_at=occurred_at or utcnow(),
    )
