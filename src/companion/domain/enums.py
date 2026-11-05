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
    # ``cli``/``test`` are the historical text inputs. ``text``/``voice`` make
    # the input modality explicit for the response context (V0.6.2): a voice
    # event may carry trusted acoustic features; a text event must never claim
    # to have heard tone.
    CLI = "cli"
    TEST = "test"
    TEXT = "text"
    VOICE = "voice"

    @property
    def modality(self) -> str:
        """The coarse input modality: ``text`` or ``voice``."""
        return "voice" if self is EventSource.VOICE else "text"


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
    FOLLOW_UP_TO_AGENT = "follow_up_to_agent"
    CONVERSATION_BETWEEN_HUMANS = "conversation_between_humans"
    AMBIGUOUS_ADDRESSEE = "ambiguous_addressee"
    SOCIAL_RESPONSE_NOT_NEEDED = "social_response_not_needed"
    EXPLICIT_SILENCE_REQUEST = "explicit_silence_request"
    RESPONSE_NOT_NECESSARY = "response_not_necessary"
    CONSERVATIVE_AMBIGUITY = "conservative_ambiguity"
    QUIET_MODE = "quiet_mode"
    PERMISSION_DENIED = "permission_denied"
    SAFETY_RESPONSE = "safety_response"
    UNKNOWN = "unknown"


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    LINK = "LINK"
    SKIP = "SKIP"


class MemoryCandidateStatus(str, Enum):
    """Lifecycle of a raw memory *candidate* inside a consolidation job.

    This is a separate axis from :class:`MemoryStatus`. ``CONSOLIDATED`` means
    "the candidate was processed"; the truthful durable result is recorded
    separately on the candidate write mapping (:class:`MemoryWriteOutcome`).
    A candidate may be ``CONSOLIDATED`` with ``NO_MEMORY`` when its content is
    hypothetical or needs no durable fact.
    """

    PENDING = "PENDING"
    CONSOLIDATED = "CONSOLIDATED"
    SKIPPED = "SKIPPED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class MemoryWriteOutcome(str, Enum):
    """Truthful result of attempting to turn one candidate into a memory."""

    SAVED = "SAVED"
    UPDATED = "UPDATED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    PENDING_REVIEW = "PENDING_REVIEW"
    DENIED = "DENIED"
    FAILED = "FAILED"
    NO_MEMORY = "NO_MEMORY"


class AssertionType(str, Enum):
    """What kind of statement the candidate is, from the speaker's view."""

    SELF_REPORT = "SELF_REPORT"
    THIRD_PARTY_REPORT = "THIRD_PARTY_REPORT"
    HYPOTHETICAL = "HYPOTHETICAL"
    INFERENCE = "INFERENCE"
    CORRECTION = "CORRECTION"
    UNKNOWN = "UNKNOWN"


class CertaintyCategory(str, Enum):
    """How directly the statement supports a durable fact.

    This is deliberately coarse: it is not a model confidence score and is
    never used on its own to authorise an ``ACTIVE`` write.
    """

    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class ReplyNecessity(str, Enum):
    """How much a reply is needed for the current turn."""

    REQUIRED = "REQUIRED"
    USEFUL = "USEFUL"
    OPTIONAL = "OPTIONAL"
    UNNECESSARY = "UNNECESSARY"


class JobType(str, Enum):
    MEMORY_CONSOLIDATION = "MEMORY_CONSOLIDATION"
    EMBEDDING = "EMBEDDING"
    # V0.6: deferred cognitive/memory extraction for a Fast Path turn. The
    # reply does not wait for it; it is durable, idempotent per event and
    # retryable like any other job.
    TURN_ANALYSIS = "TURN_ANALYSIS"
    # V0.9: unified off-task execution types. Foreground explicit memory writes
    # and background research use the same durable job substrate as the older
    # job types; adding values is backward compatible because the column is
    # plain text.
    MEMORY_SAVE = "MEMORY_SAVE"
    MEMORY_CORRECTION = "MEMORY_CORRECTION"
    RESEARCH_TASK = "RESEARCH_TASK"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    # Cancellation is terminal. A cooperative pause is *not* represented by a
    # new status: the job goes back to PENDING so a restart can recover it.
    CANCELLED = "CANCELLED"


class TaskPriority(str, Enum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class TaskSource(str, Enum):
    USER = "USER"
    ROUTER = "ROUTER"
    SYSTEM = "SYSTEM"


class TaskMode(str, Enum):
    FOREGROUND = "FOREGROUND"
    BACKGROUND = "BACKGROUND"


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
