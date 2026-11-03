"""Pydantic models that define the wire format of the domain.

Every model is validated by Pydantic on construction. ``datetime`` fields are
timezone-aware UTC values; the storage layer serialises them to ISO-8601
strings and parses them back on read.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

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


class Person(BaseModel):
    """A registered, authorised (or formerly authorised) person."""

    person_id: str
    display_name: str
    role: PersonRole
    registration_status: RegistrationStatus = RegistrationStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class Consent(BaseModel):
    """Explicit, per-person privacy grants.

    ``allow_processing`` gates whether any input may enter the cognitive flow;
    ``allow_persistence`` gates durable storage of utterances; and
    ``allow_memory_learning`` gates whether memory candidates may be created.
    These are independent — processing is not persistence, and persistence is
    not learning. V0.0 never defaults these to ``True``.
    """

    person_id: str
    allow_processing: bool
    allow_persistence: bool
    allow_memory_learning: bool
    updated_at: datetime


class UtteranceEvent(BaseModel):
    """A single, fully-confirmed input utterance.

    ``speaker_person_id`` identifies the *speaker*, not the person discussed.
    It is ``None`` for unregistered speakers, who may only carry a transient
    ``speaker_label`` and never enter persistent processing.
    """

    event_id: str
    conversation_id: str
    speaker_person_id: Optional[str] = None
    speaker_label: Optional[str] = None
    text: str
    source: EventSource
    addressee_hint: AddresseeHint
    occurred_at: datetime
    processing_status: ProcessingStatus = ProcessingStatus.PENDING


class MemoryCandidate(BaseModel):
    """A candidate memory proposed by a cognitive provider.

    V0.0 only *stores* candidates; it never promotes them into trusted
    long-term facts (that is the V0.2 memory-learning milestone).
    """

    candidate_id: str
    source_event_id: str
    subject_person_id: Optional[str] = None
    content: str
    operation: MemoryOperation
    status: MemoryCandidateStatus = MemoryCandidateStatus.PENDING


class CognitiveResult(BaseModel):
    """Structured output of a cognitive provider.

    ``action`` is the provider's *suggestion*; the Policy layer owns the final
    decision on whether that suggestion is allowed to execute.
    """

    event_id: str
    meaning: str
    action: CognitiveAction
    reason_code: ReasonCode
    referenced_memory_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    response_style: str = "mock_plain"
    provider: str = "mock"


class AgentAction(BaseModel):
    """What the agent ultimately did for a single event."""

    action_id: str
    event_id: str
    action_type: CognitiveAction
    response_text: Optional[str] = None
    reason_code: ReasonCode
    status: str = "COMPLETED"
    created_at: datetime


class ConversationContext(BaseModel):
    """In-memory snapshot of a conversation's persistent state."""

    conversation_id: str
    quiet_mode: bool = False
    recent_utterance_ids: list[str] = Field(default_factory=list)
    last_action: Optional[str] = None
    updated_at: Optional[datetime] = None


class ProcessingJob(BaseModel):
    """A durable, recoverable background job."""

    job_id: str
    job_key: str
    job_type: JobType
    payload: dict = Field(default_factory=dict)
    status: JobStatus
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ProcessingReport(BaseModel):
    """Result of ``Orchestrator.process_utterance``.

    ``steps`` carries the human-readable ``[STAGE] ...`` lines the CLI prints,
    while ``status``/``action``/``reason_code`` carry the machine-readable
    outcome.
    """

    event_id: str
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
        event_id=event_id or new_id("evt"),
        conversation_id=conversation_id,
        speaker_person_id=speaker_person_id,
        speaker_label=speaker_label,
        text=text,
        source=EventSource(source),
        addressee_hint=AddresseeHint(addressee_hint),
        occurred_at=occurred_at or utcnow(),
    )
