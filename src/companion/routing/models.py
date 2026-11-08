"""Types for the LLM-driven Adaptive Router.

The router's job is semantic: decide *whether* the current turn calls for a
visible reply, what internal operations it asks for, and which read-only tools
would help answer it. It never writes data, never executes a tool and never
produces the final reply text.

Two representations are deliberately kept apart:

``RawRouteDecision``
    Exactly what the model returned. It is untrusted input.
``ValidatedRouteDecision``
    Produced only by :class:`companion.routing.validator.RoutingValidator`.
    Policy, the tool executor and the adapter may consume it.

Parsing a model payload with ``model_validate`` is *not* validation.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from companion.domain.enums import ReplyNecessity
from companion.domain.models import AffectiveSignal, PragmaticAnalysis, SceneProposal

ROUTER_SCHEMA_VERSION = 1
ROUTER_PROMPT_VERSION = "router-v4"

MAX_TOOL_REQUESTS = 8
MAX_TOOL_CALLS = 4
MAX_EVIDENCE_ITEMS = 8
MAX_REASON_CHARS = 500
MAX_PURPOSE_CHARS = 200


class RouteAction(str, Enum):
    RESPOND = "RESPOND"
    WAIT = "WAIT"
    IGNORE = "IGNORE"


class RoutePhase(str, Enum):
    INITIAL = "INITIAL"
    FOLLOWUP = "FOLLOWUP"


class ProcessingIntent(str, Enum):
    NONE = "NONE"
    MEMORY_SAVE = "MEMORY_SAVE"
    MEMORY_CORRECTION = "MEMORY_CORRECTION"
    PRIVACY_REQUEST = "PRIVACY_REQUEST"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    DENIED = "DENIED"
    DISABLED = "DISABLED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class RouteAddressee(str, Enum):
    AGENT = "AGENT"
    HUMAN = "HUMAN"
    UNKNOWN = "UNKNOWN"


class InputSemanticStatus(str, Enum):
    """Whether this turn is an understandable communicative act *at all*.

    This is independent of whether a reply is wanted: a communicative turn can
    still be a WAIT. It only records the router's semantic reading of the input
    as an act in the current conversation, never a claim about text shape,
    language, symbols or length.
    """

    COMMUNICATIVE = "COMMUNICATIVE"
    NON_COMMUNICATIVE = "NON_COMMUNICATIVE"
    UNCERTAIN = "UNCERTAIN"


class TurnState(str, Enum):
    NEW_TOPIC = "NEW_TOPIC"
    CONTINUING = "CONTINUING"
    FOLLOW_UP = "FOLLOW_UP"
    CLOSING = "CLOSING"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


class ToneDirection(str, Enum):
    """An optional suggestion for the Speaker's delivery, never an order.

    This is deliberately not an emotion state and carries no permission or
    action control. The Speaker may adopt, adjust or ignore it.
    """

    NATURAL = "NATURAL"
    PLAYFUL = "PLAYFUL"
    CURIOUS = "CURIOUS"
    SURPRISED = "SURPRISED"
    CONCERNED = "CONCERNED"
    SKEPTICAL = "SKEPTICAL"
    SERIOUS = "SERIOUS"


def _coerce_tone_direction(value: Any) -> Optional[ToneDirection]:
    """Unknown or malformed suggestions degrade to ``None``.

    A tone hint must never invalidate the whole route decision. Older payloads
    simply omit the field; hostile or sloppy values are dropped here.
    """

    if value is None or isinstance(value, ToneDirection):
        return value if isinstance(value, ToneDirection) else None
    text = str(value).strip().upper()
    if not text:
        return None
    try:
        return ToneDirection(text)
    except ValueError:
        return None


class EvidenceReference(BaseModel):
    """A quoted span from the current input or an authorized recent turn."""

    model_config = ConfigDict(extra="forbid")

    event_id: Optional[str] = Field(default=None, max_length=128)
    quote: str = Field(default="", max_length=512)

    @field_validator("quote", mode="before")
    @classmethod
    def _bounded_quote(cls, value: Any) -> str:
        return str(value or "")[:512]


class DataControlRequest(BaseModel):
    """A structured, non-executable privacy request proposed by the router.

    The router can only *describe* what it understood. ``DataControlService``
    validates the scope against trusted identity, asks for confirmation and
    only then calls an existing trusted data-management API. Free-text reasons
    are never parsed into operations.
    """

    model_config = ConfigDict(extra="forbid")

    operation: str = Field(default="UNKNOWN", max_length=64)
    target_type: str = Field(default="UNKNOWN", max_length=32)
    target_id: Optional[str] = Field(default=None, max_length=128)
    scope: str = Field(default="SELF", max_length=32)
    requires_confirmation: bool = True
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @field_validator("operation", "target_type", "scope", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        return str(value or "UNKNOWN").strip().upper()[:64]

    @field_validator("target_id", mode="before")
    @classmethod
    def _clean_target(cls, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text[:128] or None

    @field_validator("evidence")
    @classmethod
    def _bounded_evidence(
        cls, value: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        return value[:MAX_EVIDENCE_ITEMS]


class ToolRequest(BaseModel):
    """A model's *request*; the executor decides whether it may run."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(default="", max_length=MAX_PURPOSE_CHARS)
    required: bool = False

    @field_validator("tool_name")
    @classmethod
    def _clean_tool_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("arguments")
    @classmethod
    def _json_only(cls, value: dict) -> dict:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("arguments must be finite JSON") from exc
        return value

    @field_validator("purpose", mode="before")
    @classmethod
    def _bounded_purpose(cls, value: Any) -> str:
        return str(value or "")[:MAX_PURPOSE_CHARS]


class RawRouteDecision(BaseModel):
    """The unvalidated shape returned by the router model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = ROUTER_SCHEMA_VERSION
    action: RouteAction
    reply_necessity: ReplyNecessity
    addressee_assessment: RouteAddressee
    # Required: the new protocol must state whether the turn is communicative
    # at all. An absent field is a schema error, never an implicit IGNORE.
    input_semantics: InputSemanticStatus
    turn_state: TurnState = TurnState.UNKNOWN
    # Optional delivery suggestion for the Speaker. It never changes action,
    # tools, memory or any other routing judgement, and an unknown value is
    # dropped rather than rejecting the whole decision.
    tone_direction: Optional[ToneDirection] = None
    processing_intents: list[ProcessingIntent] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    needs_followup_planning: bool = False
    pragmatic: Optional[dict] = None
    affective_signal: Optional[dict] = None
    scene_proposal: Optional[dict] = None
    data_control_request: Optional[DataControlRequest] = None
    reason: str = Field(default="", max_length=MAX_REASON_CHARS)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    # Explicit, semantic "do not answer this turn" signal. It is only used to
    # pick an explainable reason code; the action remains authoritative.
    silence_requested: bool = False

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value != ROUTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value}")
        return value

    @field_validator("tool_requests")
    @classmethod
    def _bounded_tools(cls, value: list[ToolRequest]) -> list[ToolRequest]:
        if len(value) > MAX_TOOL_REQUESTS:
            raise ValueError(f"at most {MAX_TOOL_REQUESTS} tool requests")
        return value

    @field_validator("evidence")
    @classmethod
    def _bounded_evidence(
        cls, value: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        if len(value) > MAX_EVIDENCE_ITEMS:
            raise ValueError(f"at most {MAX_EVIDENCE_ITEMS} evidence items")
        return value

    @field_validator("processing_intents")
    @classmethod
    def _dedupe_intents(
        cls, value: list[ProcessingIntent]
    ) -> list[ProcessingIntent]:
        out: list[ProcessingIntent] = []
        for intent in value:
            if intent != ProcessingIntent.NONE and intent not in out:
                out.append(intent)
        return out[:4]

    @field_validator("tone_direction", mode="before")
    @classmethod
    def _clean_tone_direction(cls, value: Any) -> Optional[ToneDirection]:
        return _coerce_tone_direction(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _clean_reason(cls, value: Any) -> str:
        return " ".join(str(value or "").split())[:MAX_REASON_CHARS]


class ValidatedRouteDecision(BaseModel):
    """A decision that passed :class:`RoutingValidator`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = ROUTER_SCHEMA_VERSION
    action: RouteAction
    reply_necessity: ReplyNecessity
    addressee_assessment: RouteAddressee
    # V0.8.1 persisted decisions predate this field. Reading one must not turn
    # it into NON_COMMUNICATIVE, so the compatible default is UNCERTAIN.
    input_semantics: InputSemanticStatus = InputSemanticStatus.UNCERTAIN
    turn_state: TurnState = TurnState.UNKNOWN
    # Read-only suggestion carried for the Speaker. Older persisted decisions
    # omit it and deserialize with ``None``.
    tone_direction: Optional[ToneDirection] = None
    processing_intents: list[ProcessingIntent] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    needs_followup_planning: bool = False
    pragmatic: Optional[PragmaticAnalysis] = None
    affective_signal: Optional[AffectiveSignal] = None
    scene_proposal: Optional[SceneProposal] = None
    data_control_request: Optional[DataControlRequest] = None
    reason: str = Field(default="", max_length=MAX_REASON_CHARS)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    silence_requested: bool = False
    # Audit metadata filled by the validator / service.
    router_version: str = ROUTER_PROMPT_VERSION
    provider: str = "deepseek"
    model: str = ""

    @field_validator("tone_direction", mode="before")
    @classmethod
    def _clean_tone_direction(cls, value: Any) -> Optional[ToneDirection]:
        return _coerce_tone_direction(value)

    @field_validator("tool_requests")
    @classmethod
    def _bounded_tools(cls, value: list[ToolRequest]) -> list[ToolRequest]:
        if len(value) > MAX_TOOL_CALLS:
            raise ValueError(f"at most {MAX_TOOL_CALLS} tool requests")
        return value

    def audit_dict(self) -> dict:
        return self.model_dump(mode="json")

    def core_fingerprint(self) -> tuple:
        """Fields a follow-up plan may not silently rewrite."""
        control = self.data_control_request
        control_key = (
            (
                control.operation,
                control.target_type,
                control.target_id,
                control.scope,
            )
            if control is not None
            else None
        )
        return (
            self.action,
            self.reply_necessity,
            self.addressee_assessment,
            self.input_semantics,
            tuple(self.processing_intents),
            self.silence_requested,
            control_key,
        )


class RoutingContext(BaseModel):
    """The minimal, already-authorized input given to the router."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    current_event_id: str
    current_speaker_id: Optional[str] = None
    current_speaker_label: str = "unknown"
    current_text: str
    trusted_addressee_hint: str = "unknown"
    input_modality: Literal["text", "voice"] = "text"
    quiet_mode: bool = False
    recent_turns: list[dict] = Field(default_factory=list)
    recent_nonresponse_actions: list[dict] = Field(default_factory=list)
    conversation_state: dict = Field(default_factory=dict)
    compact_scene: Optional[dict] = None
    compact_person_profile: Optional[str] = None
    # The exact profile revision the compact block was compiled from. The
    # speaker reuses that block instead of reading a second, possibly newer
    # revision for the same turn.
    compact_person_profile_revision: Optional[int] = None
    available_tool_specs: list[dict] = Field(default_factory=list)
    # Trusted metadata for the validator (never rendered as instructions).
    visible_event_ids: list[str] = Field(default_factory=list)
    authorized_history_texts: list[str] = Field(default_factory=list)

    def as_prompt_dict(self) -> dict:
        """The exact data sent to the model: no ids of hidden rows."""
        return {
            "current_event": {
                "event_id": self.current_event_id,
                "speaker": self.current_speaker_label,
                "input_modality": self.input_modality,
                "trusted_addressee_hint": self.trusted_addressee_hint,
                "text": self.current_text,
            },
            "conversation": {
                "conversation_id": self.conversation_id,
                "quiet_mode": self.quiet_mode,
                "state": self.conversation_state,
            },
            "recent_turns": self.recent_turns,
            "recent_nonresponse_actions": self.recent_nonresponse_actions,
            "compact_scene": self.compact_scene,
            "compact_person_profile": self.compact_person_profile,
            "available_tools": self.available_tool_specs,
        }


class RoutingOutcome(BaseModel):
    """Everything the orchestrator needs after the routing phase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    route: ValidatedRouteDecision
    tool_results: list[Any] = Field(default_factory=list)
    planning_steps: int = 0
    provider_calls: int = 0
    initial_route: Optional[ValidatedRouteDecision] = None
    # Process-local stage durations (milliseconds). A restored decision reports
    # zero model time because no model call happened for this attempt.
    router_llm_ms: float = 0.0
    router_followup_ms: float = 0.0
    tool_execution_ms: float = 0.0


def finite_json(value: Any) -> bool:
    """Whether ``value`` is finite JSON (no NaN/Infinity)."""
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return not (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    )
