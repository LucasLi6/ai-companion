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


class ReportStatus(str, Enum):
    """Top-level result of ``Orchestrator.process_utterance``."""

    COMPLETED = "COMPLETED"
    DUPLICATE = "DUPLICATE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    FAILED = "FAILED"
