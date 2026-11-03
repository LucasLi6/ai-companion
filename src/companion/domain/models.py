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

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .enums import (
    AddresseeHint,
    CognitiveAction,
    EventSource,
    JobStatus,
    JobType,
    MemoryCandidateStatus,
    MemoryOperation,
    PersonRole,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
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
    ``allow_persistence`` gates durable storage of utterances; and
    ``allow_memory_learning`` gates whether memory candidates may be created.
    These are independent — processing is not persistence, and persistence is
    not learning. V0.0 never defaults these to ``True``.
    """

    person_id: str = Field(min_length=1, max_length=128)
    allow_processing: bool
    allow_persistence: bool
    allow_memory_learning: bool
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

    V0.0 only *stores* candidates; it never promotes them into trusted
    long-term facts (that is the V0.2 memory-learning milestone).
    """

    candidate_id: str = Field(min_length=1, max_length=128)
    source_event_id: str = Field(min_length=1, max_length=128)
    subject_person_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=4096)
    operation: MemoryOperation
    status: MemoryCandidateStatus = MemoryCandidateStatus.PENDING


class CognitiveResult(DomainModel):
    """Structured output of a cognitive provider.

    ``action`` is the provider's *suggestion*; the Policy layer owns the final
    decision on whether that suggestion is allowed to execute.
    """

    event_id: str = Field(min_length=1, max_length=128)
    meaning: str = Field(min_length=1, max_length=2048)
    action: CognitiveAction
    reason_code: ReasonCode
    referenced_memory_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    response_style: str = Field(default="mock_plain", min_length=1, max_length=128)
    provider: str = Field(default="mock", min_length=1, max_length=128)


class AgentAction(DomainModel):
    """What the agent ultimately did for a single event."""

    action_id: str = Field(min_length=1, max_length=128)
    event_id: str = Field(min_length=1, max_length=128)
    action_type: CognitiveAction
    response_text: Optional[str] = Field(default=None, max_length=8192)
    reason_code: ReasonCode
    status: str = Field(default="COMPLETED", min_length=1, max_length=32)
    created_at: datetime


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
