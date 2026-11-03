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
        "direct_insult",
        "criticism",
        "playful_teasing",
        "apology",
        "reconciliation",
        "direct_praise",
        "self_distress",
        "support_request",
        "none",
    ] = "none"
    # ``subject`` is the V0.3 field kept for stored compatibility; ``target``
    # is the V0.4.1 direction of the event relative to this conversation.
    subject: Literal["speaker", "other", "agent", "unknown"] = "unknown"
    target: Literal["AGENT", "SPEAKER", "OTHER", "UNKNOWN"] = "UNKNOWN"
    social_intent: Literal[
        "HOSTILE", "PLAYFUL", "SINCERE", "NEUTRAL", "UNCERTAIN"
    ] = "UNCERTAIN"
    factuality: Literal[
        "asserted",
        "negated",
        "hypothetical",
        "quoted",
        "uncertain",
    ] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = Field(default=None, max_length=512)


_AFT_EVENT_TYPES = {
    "achievement",
    "setback",
    "gratitude",
    "urgency",
    "question",
    "direct_insult",
    "criticism",
    "playful_teasing",
    "apology",
    "reconciliation",
    "direct_praise",
    "self_distress",
    "support_request",
    "none",
}
_AFT_SUBJECTS = {"speaker", "other", "agent", "unknown"}
_AFT_TARGETS = {"AGENT", "SPEAKER", "OTHER", "UNKNOWN"}
_AFT_SOCIAL_INTENTS = {"HOSTILE", "PLAYFUL", "SINCERE", "NEUTRAL", "UNCERTAIN"}
_AFT_FACTUALITIES = {"asserted", "negated", "hypothetical", "quoted", "uncertain"}


class NonresponseMotive(DomainModel):
    """A model's *hint* about why it suggests not replying.

    This is an auditable classification, not a claim that the model observed a
    fact about the room. Every field is locally re-validated before use:
    ``motive``/``reentry_hint`` must be whitelisted, ``confidence`` finite, and
    ``evidence_span`` a verbatim substring of the current input. The local
    Policy layer still owns the final action and final reason.
    """

    motive: Literal[
        "NOT_ADDRESSED",
        "AWAITING_CONTINUATION",
        "AVOID_INTERRUPTION",
        "LISTEN_WITHOUT_REPLY",
        "NO_USEFUL_CONTRIBUTION",
        "UNCERTAIN_CONTEXT",
        "UNKNOWN",
    ] = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = Field(default=None, max_length=512)
    reentry_hint: Literal[
        "NEXT_DIRECT_ADDRESS",
        "NEXT_TOPIC_CONTINUATION",
        "WHEN_CONTEXT_CLEAR",
        "NO_FOLLOWUP",
    ] = "NO_FOLLOWUP"


_NONRESPONSE_MOTIVES = {
    "NOT_ADDRESSED",
    "AWAITING_CONTINUATION",
    "AVOID_INTERRUPTION",
    "LISTEN_WITHOUT_REPLY",
    "NO_USEFUL_CONTRIBUTION",
    "UNCERTAIN_CONTEXT",
    "UNKNOWN",
}
_REENTRY_HINTS = {
    "NEXT_DIRECT_ADDRESS",
    "NEXT_TOPIC_CONTINUATION",
    "WHEN_CONTEXT_CLEAR",
    "NO_FOLLOWUP",
}


def _unknown_motive() -> NonresponseMotive:
    return NonresponseMotive()


def validate_nonresponse_motive(
    raw, current_text: str
) -> Optional[NonresponseMotive]:
    """Validate a model-supplied nonresponse motive locally.

    ``None`` means "the model supplied no motive" and preserves the old
    behaviour. Any malformed or unsupported payload degrades to an explicit
    ``UNKNOWN`` motive instead of leaking an unvalidated classification.

    Only the four whitelisted fields are read. Model-supplied person, event or
    memory ids are ignored, and an evidence span that is missing, too long or
    not a verbatim part of the current input cannot support an actionable
    motive.
    """
    if raw is None:
        return None
    if isinstance(raw, NonresponseMotive):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        return _unknown_motive()

    motive = data.get("motive")
    if motive not in _NONRESPONSE_MOTIVES:
        return _unknown_motive()

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return _unknown_motive()
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return _unknown_motive()

    evidence = data.get("evidence_span")
    if evidence is not None and not isinstance(evidence, str):
        return _unknown_motive()
    if evidence is not None:
        if len(evidence) > 512 or evidence not in current_text:
            return _unknown_motive()

    if motive != "UNKNOWN":
        if not evidence or not evidence.strip():
            return _unknown_motive()

    reentry = data.get("reentry_hint")
    if reentry not in _REENTRY_HINTS:
        reentry = "NO_FOLLOWUP"

    return NonresponseMotive(
        motive=motive,
        confidence=confidence,
        evidence_span=evidence,
        reentry_hint=reentry,
    )


_SCENE_TOPIC_CHANGES = {"CONTINUE", "SHIFT", "RETURN", "UNCERTAIN"}
_SCENE_MODES = {"CONVERSATION", "TASK", "PLAY", "SUPPORT", "UNKNOWN"}
_SCENE_PHASES = {
    "OPENING",
    "EXPLORING",
    "DEEPENING",
    "WRAPPING_UP",
    "CLOSED",
    "UNCERTAIN",
}
_SCENE_TONES = {
    "NEUTRAL",
    "WARM",
    "PLAYFUL",
    "SERIOUS",
    "TENSE",
    "SAD",
    "CELEBRATORY",
    "UNCERTAIN",
}
_SCENE_PRESENCE = {"YES", "NO", "UNCERTAIN"}


class SceneEnvironmentProposal(BaseModel):
    """An environment observation the model claims the input literally supports.

    Only fields backed by a verbatim evidence span are ever retained. There is
    no sensor data in this system, so "location" and "other people present" are
    never inferred; they must be stated explicitly by the speaker.
    """

    location: Optional[str] = Field(default=None, max_length=120)
    other_people_present: Optional[Literal["YES", "NO", "UNCERTAIN"]] = None
    evidence_span: Optional[str] = Field(default=None, max_length=512)


class SceneOpenLoopProposal(BaseModel):
    """A proposed unfinished thread. It is only persisted when the local trace
    also says there is a pending follow-up."""

    topic: Optional[str] = Field(default=None, max_length=120)
    summary: str = Field(min_length=1, max_length=300)
    evidence_span: Optional[str] = Field(default=None, max_length=512)


class SceneProposal(DomainModel):
    """A *proposal* about the conversational scene, not a trusted fact.

    The model may suggest a topic change, discourse phase, tone, a pending
    thread or an explicitly stated environment detail. Trusted code revalidates
    every evidence span and decides what is merged into :class:`SceneState`.
    Model-supplied ids are ignored.
    """

    topic_change: Literal["CONTINUE", "SHIFT", "RETURN", "UNCERTAIN"] = "UNCERTAIN"
    topic: Optional[str] = Field(default=None, max_length=200)
    mode: Literal[
        "CONVERSATION", "TASK", "PLAY", "SUPPORT", "UNKNOWN"
    ] = "UNKNOWN"
    discourse_phase: Literal[
        "OPENING",
        "EXPLORING",
        "DEEPENING",
        "WRAPPING_UP",
        "CLOSED",
        "UNCERTAIN",
    ] = "UNCERTAIN"
    tone: Literal[
        "NEUTRAL",
        "WARM",
        "PLAYFUL",
        "SERIOUS",
        "TENSE",
        "SAD",
        "CELEBRATORY",
        "UNCERTAIN",
    ] = "UNCERTAIN"
    environment: Optional[SceneEnvironmentProposal] = None
    open_loop: Optional[SceneOpenLoopProposal] = None
    resolve_loop_topics: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = Field(default=None, max_length=512)


def _clean_text(value, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text or len(text) > limit:
        return None
    return text


def _valid_evidence(value, allowed_texts) -> Optional[str]:
    if not isinstance(value, str):
        return None
    span = value.strip()
    if not span or len(span) > 512:
        return None
    if not any(span in text for text in allowed_texts if text):
        return None
    return span


def validate_scene_proposal(
    raw, *, current_text: str, authorized_texts=()
) -> Optional[SceneProposal]:
    """Locally validate a model's scene proposal.

    Returns ``None`` when the payload is malformed or contains no usable,
    evidence-supported signal. Invalid individual fields are dropped rather
    than failing the turn. Topic changes, environment claims and open loops all
    require a verbatim evidence span from the current input or an authorized
    recent turn; model ids and confidence never make a claim true.
    """
    if raw is None:
        return None
    if isinstance(raw, SceneProposal):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        return None

    allowed = [current_text, *list(authorized_texts or ())]

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        confidence = 0.0

    evidence = _valid_evidence(data.get("evidence_span"), allowed)

    topic_change = str(data.get("topic_change") or "UNCERTAIN").strip().upper()
    if topic_change not in _SCENE_TOPIC_CHANGES:
        topic_change = "UNCERTAIN"
    topic = _clean_text(data.get("topic"), 200)
    if topic is None or evidence is None:
        topic = None
        if topic_change in ("SHIFT", "RETURN"):
            topic_change = "UNCERTAIN"

    mode = str(data.get("mode") or "UNKNOWN").strip().upper()
    if mode not in _SCENE_MODES:
        mode = "UNKNOWN"
    phase = str(data.get("discourse_phase") or "UNCERTAIN").strip().upper()
    if phase not in _SCENE_PHASES:
        phase = "UNCERTAIN"
    tone = str(data.get("tone") or "UNCERTAIN").strip().upper()
    if tone not in _SCENE_TONES:
        tone = "UNCERTAIN"

    environment = None
    env_raw = data.get("environment")
    if isinstance(env_raw, dict):
        env_evidence = _valid_evidence(env_raw.get("evidence_span"), allowed)
        location = _clean_text(env_raw.get("location"), 120)
        presence = str(
            env_raw.get("other_people_present") or "UNCERTAIN"
        ).strip().upper()
        if presence not in _SCENE_PRESENCE:
            presence = "UNCERTAIN"
        if env_evidence is None:
            location = None
            presence = "UNCERTAIN"
        elif location is not None:
            # A verbatim span must at least contain the claimed place. This
            # prevents pairing a real sentence ("我一个人在家") with an
            # unrelated inferred location ("办公室"). Semantic entailment is
            # still the model's job; absence of a lexical anchor is dropped.
            if location.casefold() not in env_evidence.casefold():
                location = None
        if location is not None or presence != "UNCERTAIN":
            environment = SceneEnvironmentProposal(
                location=location,
                other_people_present=presence,
                evidence_span=env_evidence,
            )

    open_loop = None
    loop_raw = data.get("open_loop")
    if isinstance(loop_raw, dict):
        loop_evidence = _valid_evidence(loop_raw.get("evidence_span"), allowed)
        summary = _clean_text(loop_raw.get("summary"), 300)
        loop_topic = _clean_text(loop_raw.get("topic"), 120)
        if summary is not None and loop_evidence is not None:
            open_loop = SceneOpenLoopProposal(
                topic=loop_topic,
                summary=summary,
                evidence_span=loop_evidence,
            )

    resolve_topics: list[str] = []
    raw_resolve = data.get("resolve_loop_topics")
    if isinstance(raw_resolve, list):
        for item in raw_resolve[:5]:
            cleaned = _clean_text(item, 120)
            if cleaned and cleaned not in resolve_topics:
                resolve_topics.append(cleaned)

    if (
        topic is None
        and environment is None
        and open_loop is None
        and not resolve_topics
        and mode == "UNKNOWN"
        and phase == "UNCERTAIN"
        and tone == "UNCERTAIN"
        and topic_change == "UNCERTAIN"
    ):
        return None

    return SceneProposal(
        topic_change=topic_change,
        topic=topic,
        mode=mode,
        discourse_phase=phase,
        tone=tone,
        environment=environment,
        open_loop=open_loop,
        resolve_loop_topics=resolve_topics,
        confidence=confidence,
        evidence_span=evidence,
    )


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

    target = _normalise_enum(
        data.get("target", _target_from_subject(subject)), _AFT_TARGETS
    )
    if target is None:
        return None
    social_intent = _normalise_enum(
        data.get("social_intent", "UNCERTAIN"), _AFT_SOCIAL_INTENTS
    )
    if social_intent is None:
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
            target=target,
            social_intent=social_intent,
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
        target=target,
        social_intent=social_intent,
        factuality=factuality,
        confidence=confidence,
        evidence_span=evidence,
    )


def _normalise_enum(value, allowed: set[str]) -> Optional[str]:
    """Case-insensitive enum normalisation for model output.

    Missing values stay missing; the caller supplies the legacy default.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in allowed else None


def _target_from_subject(subject: str) -> str:
    return {
        "agent": "AGENT",
        "speaker": "SPEAKER",
        "other": "OTHER",
    }.get(subject, "UNKNOWN")


class DialogueImpulse(DomainModel):
    """A model's compact answer to "what is this turn doing, and where next?"

    It is guidance for the response call, not an action decision: the Policy
    layer still decides whether to reply, and the response model may adapt the
    direction to the persona and the moment. Fields are short natural-language
    observations, never fixed reply templates.
    """

    interpretation: str = Field(default="", max_length=512)
    response_direction: str = Field(default="", max_length=512)
    stance: str = Field(default="", max_length=256)
    unresolved_uncertainty: str = Field(default="", max_length=512)

    def as_prompt_dict(self) -> dict:
        return {
            "interpretation": self.interpretation,
            "response_direction": self.response_direction,
            "stance": self.stance,
            "unresolved_uncertainty": self.unresolved_uncertainty,
        }


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
    # Optional model hint explaining a suggested WAIT/IGNORE. It never decides
    # the final action or final reason; ``derive_nonresponse_trace`` records
    # whether local policy overrode it.
    nonresponse_motive: Optional[NonresponseMotive] = None
    # Optional, locally validated scene proposal. It describes a candidate
    # conversational scene; SceneManager decides what, if anything, is merged
    # into the persisted state. None for older records and mock providers.
    scene_proposal: Optional[SceneProposal] = None
    # Optional interpretation of the turn and a suggested conversational
    # direction. Guidance for the response prompt; never a fixed reply and
    # never an action decision. None for older records and mock providers.
    dialogue_impulse: Optional[DialogueImpulse] = None


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
    # Persisted, explainable record for a final WAIT/IGNORE. None for RESPOND
    # (and for legacy rows written before schema 13).
    nonresponse_trace: Optional[dict] = None


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
