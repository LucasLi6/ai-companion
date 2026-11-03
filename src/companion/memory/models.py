"""Unified memory domain models.

``MemoryItem`` is the single durable record for *every* kind of memory —
personal facts, episodes, relationships, project state, externally-sourced
world knowledge, AI reflections and hypotheses — with independent provenance
and trust attributes. Subjects, sources and evidence live in their own tables
so one item can involve several people/projects and cite several sources.

Requesters are represented by :class:`AccessContext`, which the *trusted
application* supplies (never a value an LLM returns).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import Field

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    VerificationLevel,
)
from companion.domain.models import DomainModel


@dataclass
class AccessContext:
    """Who is requesting a memory operation, and with what privileges.

    ``actor_id`` is a registered ``person_id`` for cognitive/idle flows, or the
    literal ``"admin"`` for the authenticated local management UI. ``is_admin``
    is set only by the trusted app, never derived from model output or a role
    label.
    """

    actor_id: str
    is_admin: bool = False
    role: str = "app"  # "app" | "cognitive" | "idle" | "ui"


class MemoryItem(DomainModel):
    """The current live version of one memory record."""

    memory_id: str = Field(min_length=1, max_length=128)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=16384)
    status: MemoryStatus = MemoryStatus.CANDIDATE
    visibility: MemoryVisibility = MemoryVisibility.PRIVATE
    created_by: str = Field(min_length=1, max_length=128)
    created_at: datetime
    updated_at: datetime
    version: int = 1
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    # Provenance/trust attributes — kept independent of a single "confidence".
    human_confirmed: bool = False
    source_supported: bool = False
    self_reported: bool = False
    third_party_reported: bool = False
    model_inferred: bool = False


class MemorySource(DomainModel):
    """A citable origin for memory content (utterance, web page, manual, …)."""

    source_id: str = Field(min_length=1, max_length=128)
    source_type: str = Field(min_length=1, max_length=32)
    source_ref: Optional[str] = Field(default=None, max_length=2048)
    title: Optional[str] = Field(default=None, max_length=2048)
    url: Optional[str] = Field(default=None, max_length=2048)
    retrieved_at: Optional[datetime] = None
    published_at: Optional[datetime] = None


class MemoryRevision(DomainModel):
    """An immutable snapshot of one version of a memory item."""

    revision_id: str = Field(min_length=1, max_length=128)
    memory_id: str = Field(min_length=1, max_length=128)
    version: int
    kind: MemoryKind
    content: str
    status: MemoryStatus
    changed_by: str
    changed_at: datetime
    reason: Optional[str] = None


class MemoryEvidence(DomainModel):
    """A link between a memory item and one source, with a verification level.

    ``memory_id`` is filled in by the repository during creation; callers only
    supply ``source_id``, the evidence span and the verification level.
    """

    memory_id: Optional[str] = None
    source_id: str = Field(min_length=1, max_length=128)
    evidence_span: Optional[str] = None
    verification_level: VerificationLevel = VerificationLevel.UNVERIFIED
    retrieved_at: Optional[datetime] = None


class MemoryNotFoundError(Exception):
    pass


class MemoryVersionConflictError(Exception):
    """Raised when an update's ``expected_version`` no longer matches."""


class MemoryPermissionDeniedError(Exception):
    pass


class MemoryConflictError(Exception):
    """Raised when an AI proposal would overwrite a human-confirmed fact."""
