"""Enumerated values shared across the domain.

These enums are string subclasses so that they serialise to their plain
string value both in Pydantic models and in SQLite text columns.
"""

from __future__ import annotations

from enum import Enum


class PersonRole(str, Enum):
    OWNER = "owner"
    FRIEND = "friend"
    GUEST = "guest"


class RegistrationStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class EventSource(str, Enum):
    CLI = "cli"
    TEST = "test"


class AddresseeHint(str, Enum):
    """Who the utterance appears to be addressed to.

    V0.0 treats this as an explicit hint supplied by the caller; a future
    cognitive engine will infer it from semantics and audio context.
    """

    AGENT = "agent"
    HUMAN = "human"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CognitiveAction(str, Enum):
    """The action a cognitive provider suggests / policy finalises."""

    RESPOND = "RESPOND"
    WAIT = "WAIT"
    IGNORE = "IGNORE"


class ReasonCode(str, Enum):
    DIRECT_AGENT_ADDRESS = "direct_agent_address"
    DIRECT_HUMAN_ADDRESS = "direct_human_address"
    UNKNOWN_ADDRESSEE = "unknown_addressee"
    QUIET_MODE = "quiet_mode"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN = "unknown"


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    LINK = "LINK"
    SKIP = "SKIP"


class MemoryCandidateStatus(str, Enum):
    """Lifecycle of a raw memory *candidate* inside a consolidation job.

    This is a separate axis from :class:`MemoryStatus`. ``CONSOLIDATED`` means
    "the candidate passed validation in the job", *not* that a trusted,
    ``ACTIVE`` memory now exists — V0.0 does not promote candidates into the
    unified memory store at all.
    """

    PENDING = "PENDING"
    CONSOLIDATED = "CONSOLIDATED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class JobType(str, Enum):
    MEMORY_CONSOLIDATION = "MEMORY_CONSOLIDATION"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MemoryKind(str, Enum):
    """The unified memory taxonomy.

    ``WORLD_KNOWLEDGE`` (externally-sourced facts) is distinct from
    ``PERSONAL_FACT`` (facts about a person), and ``REFLECTION`` / ``HYPOTHESIS``
    are never treated as verified facts.
    """

    PERSONAL_FACT = "PERSONAL_FACT"
    EPISODE = "EPISODE"
    RELATIONSHIP = "RELATIONSHIP"
    PROJECT_STATE = "PROJECT_STATE"
    WORLD_KNOWLEDGE = "WORLD_KNOWLEDGE"
    REFLECTION = "REFLECTION"
    HYPOTHESIS = "HYPOTHESIS"


class MemoryStatus(str, Enum):
    """Lifecycle/trust state of a unified memory item.

    ``ACTIVE`` means "currently used", *not* "absolutely correct". This is a
    separate axis from :class:`MemoryCandidateStatus` (the job-pipeline state):
    a ``CONSOLIDATED`` candidate is not an ``ACTIVE`` memory.
    """

    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CONFLICTED = "CONFLICTED"
    RETRACTED = "RETRACTED"
    OUTDATED = "OUTDATED"


class MemoryVisibility(str, Enum):
    PRIVATE = "PRIVATE"
    SHARED = "SHARED"
    PUBLIC = "PUBLIC"


class SubjectType(str, Enum):
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    ENTITY = "ENTITY"


class VerificationLevel(str, Enum):
    """Evidence-based verification, *not* a model's self-reported confidence."""

    UNVERIFIED = "UNVERIFIED"
    SINGLE_SOURCE = "SINGLE_SOURCE"
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    OUTDATED = "OUTDATED"


class MemorySourceType(str, Enum):
    UTTERANCE = "utterance"
    WEB = "web"
    MANUAL = "manual"
    REFLECTION = "reflection"
    RESEARCH = "research"


class ReportStatus(str, Enum):
    """Top-level result of ``Orchestrator.process_utterance``."""

    COMPLETED = "COMPLETED"
    DUPLICATE = "DUPLICATE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    FAILED = "FAILED"
