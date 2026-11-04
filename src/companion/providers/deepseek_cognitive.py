"""Real DeepSeek cognitive provider (Phase 5).

Turns the assembled :class:`CognitiveContext` into a strict-JSON request and
maps the model's output onto a trusted :class:`CognitiveResult`.

Trust boundaries enforced here (never delegated to the model):

- ``event_id``, ``source_event_id``, ``subject_person_id`` and candidate ids
  are filled by the program from trusted metadata. Any ids the model emits are
  ignored (``extra="ignore"``).
- ``referenced_memory_ids`` are validated against the retrieved set; fabricated
  ids are dropped.
- An explicit ``addressee_hint == "human"`` can never be overridden into a
  RESPOND, even if the model says so.
- The external-AI privacy gate is checked *before* any network call; a denied
  speaker gets a local ``PERMISSION_DENIED`` result, never a silent mock reply.

One limited repair retry is attempted when the model returns invalid JSON or a
schema-invalid object; a second failure raises :class:`CognitiveValidationError`.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from companion.cognition.models import CognitiveContext
from companion.domain.enums import (
    AddresseeHint,
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    MemoryKind,
    MemoryOperation,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    CognitiveResult,
    DialogueImpulse,
    MemoryCandidate,
    UtteranceEvent,
    new_id,
    validate_affective_signal,
    validate_nonresponse_motive,
    validate_pragmatic_analysis,
    validate_scene_proposal,
)
from companion.memory.store import validate_referenced_memory_ids
from companion.prompts.cognitive import build_cognitive_messages, build_repair_messages

PROVIDER_NAME = "deepseek"


class CognitiveValidationError(Exception):
    """The model returned output that is not valid JSON or fails validation."""


class _RawCandidate(BaseModel):
    # Ignore any ids the model tries to supply; the program fills them.
    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1, max_length=4096)
    operation: str = "ADD"
    assertion_type: str = "UNKNOWN"
    certainty: str = "UNKNOWN"
    topic: Optional[str] = Field(default=None, max_length=128)
    kind: Optional[str] = Field(default=None, max_length=32)
    evidence_span: Optional[str] = Field(default=None, max_length=4096)


class _RawAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meaning: str = Field(min_length=1, max_length=2048)
    topic: Optional[str] = Field(default=None, max_length=512)
    intent: str = Field(default="unknown", max_length=128)
    addressee: Optional[str] = Field(default=None, max_length=32)
    addressed_to: Optional[str] = Field(default=None, max_length=32)
    silence_requested: bool = False
    reply_necessity: str = "OPTIONAL"
    conversation_state: str = Field(default="unknown", max_length=128)
    action: str = Field(min_length=1, max_length=32)
    reason_code: str = Field(min_length=1, max_length=64)
    referenced_memory_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[_RawCandidate] = Field(default_factory=list)
    response_style: str = Field(default="deepseek_plain", max_length=128)
    # Kept as untyped objects on purpose: invalid optional hints must degrade
    # locally, not fail the whole analysis schema.
    affective_signal: Optional[object] = None
    nonresponse_motive: Optional[object] = None
    scene_proposal: Optional[object] = None
    dialogue_impulse: Optional[object] = None
    pragmatic: Optional[object] = None


def _parse_json_object(content: str) -> dict:
    """Parse one JSON object, tolerating a single markdown fence."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CognitiveValidationError(f"invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise CognitiveValidationError("top-level JSON must be an object")
    return data


class DeepSeekCognitiveProvider:
    """``CognitiveProvider`` backed by the real DeepSeek client."""

    name = PROVIDER_NAME

    def __init__(
        self,
        client,
        *,
        privacy_gateway=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.privacy_gateway = privacy_gateway
        self.logger = logger or logging.getLogger("companion.deepseek.cognitive")

    async def analyze(
        self,
        event: UtteranceEvent,
        context,
        memories: list,
    ) -> CognitiveResult:
        ctx = self._as_context(event, context, memories)

        if self.privacy_gateway is not None:
            auth = await self.privacy_gateway.authorize_event(event.speaker_person_id)
            if not auth.allowed:
                self.logger.info(
                    "deepseek_analysis_denied event_id=%s reason=%s",
                    event.event_id,
                    auth.reason,
                )
                return self._denied(event)

        messages = build_cognitive_messages(ctx)
        content = await self._request(event, messages)

        first_error: Optional[CognitiveValidationError] = None
        try:
            return self._to_result(self._parse(content), event, ctx)
        except CognitiveValidationError as exc:
            first_error = exc
            self.logger.warning(
                "deepseek_analysis_invalid event_id=%s error=%s repair=1",
                event.event_id,
                first_error,
            )

        repair_messages = build_repair_messages(messages, content, str(first_error))
        repaired = await self._request(event, repair_messages)
        try:
            return self._to_result(self._parse(repaired), event, ctx)
        except CognitiveValidationError as second_error:
            raise CognitiveValidationError(
                f"model output invalid after one repair retry: {second_error}"
            ) from second_error

    # -- request/parse ------------------------------------------------------

    async def _request(self, event: UtteranceEvent, messages: list[dict]) -> str:
        response = await self.client.complete(
            messages,
            request_type="analysis",
            event_id=event.event_id,
            json_mode=True,
        )
        return response.content

    def _parse(self, content: str) -> _RawAnalysis:
        data = _parse_json_object(content)
        try:
            return _RawAnalysis.model_validate(data)
        except ValidationError as exc:
            # Report field names/types only — never the model's input values.
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}:{err['type']}"
                for err in exc.errors()
            )
            raise CognitiveValidationError(f"schema validation failed ({details})") from exc

    def _to_result(
        self, raw: _RawAnalysis, event: UtteranceEvent, ctx: CognitiveContext
    ) -> CognitiveResult:
        meaning = raw.meaning.strip()
        if not meaning:
            raise CognitiveValidationError("meaning is blank")

        try:
            action = CognitiveAction(raw.action.strip().upper())
        except (ValueError, AttributeError) as exc:
            raise CognitiveValidationError(f"invalid action: {raw.action!r}") from exc
        try:
            reason = ReasonCode(raw.reason_code.strip().lower())
        except (ValueError, AttributeError) as exc:
            raise CognitiveValidationError(
                f"invalid reason_code: {raw.reason_code!r}"
            ) from exc

        # Explicit human addressee is a hard constraint, not a model choice.
        if (
            event.addressee_hint == AddresseeHint.HUMAN
            and action == CognitiveAction.RESPOND
        ):
            self.logger.info(
                "deepseek_respond_downgraded event_id=%s reason=explicit_human_addressee",
                event.event_id,
            )
            action = CognitiveAction.IGNORE
            reason = ReasonCode.DIRECT_HUMAN_ADDRESS

        referenced = validate_referenced_memory_ids(
            raw.referenced_memory_ids, ctx.memories
        )

        candidates = [
            MemoryCandidate(
                candidate_id=new_id("memc"),
                source_event_id=event.event_id,  # trusted
                subject_person_id=event.speaker_person_id,  # trusted
                content=c.content,
                operation=self._operation(c.operation),
                assertion_type=self._assertion(c.assertion_type),
                certainty_category=self._certainty(c.certainty),
                topic=c.topic,
                kind=self._kind(c.kind),
                source_evidence_span=self._evidence(c.evidence_span, event),
            )
            for c in raw.memory_candidates
        ]

        signal = validate_affective_signal(raw.affective_signal, event.text)
        if raw.affective_signal is not None and signal is None:
            self.logger.info(
                "deepseek_affective_signal_rejected event_id=%s", event.event_id
            )

        motive = validate_nonresponse_motive(
            raw.nonresponse_motive, event.text
        )
        if raw.nonresponse_motive is not None and (
            motive is None or motive.motive == "UNKNOWN"
        ):
            self.logger.info(
                "deepseek_nonresponse_motive_rejected event_id=%s",
                event.event_id,
            )
        # A response is not a nonresponse; never persist a motive for it even
        # if the model returned one.
        if action == CognitiveAction.RESPOND:
            motive = None

        # Scene proposals are untrusted. Evidence may come from the current
        # input or this speaker's authorized history; another speaker's words
        # can never support this person's scene.
        authorized_texts = [
            turn.text
            for turn in ctx.turns
            if turn.role == "human"
            and turn.speaker_person_id is not None
            and turn.speaker_person_id == event.speaker_person_id
        ]
        scene_proposal = validate_scene_proposal(
            raw.scene_proposal,
            current_text=event.text,
            authorized_texts=authorized_texts,
        )
        dialogue_impulse = self._dialogue_impulse(raw.dialogue_impulse)
        # Antecedents must point at a turn the model could actually see. The
        # ids are internal, but a fabricated one is still a fabricated history
        # claim, so it is dropped rather than carried into the scene/frame.
        pragmatic = validate_pragmatic_analysis(
            raw.pragmatic,
            event.text,
            allowed_antecedent_ids={turn.event_id for turn in ctx.turns},
        )
        if raw.pragmatic is not None and pragmatic is None:
            self.logger.info(
                "deepseek_pragmatic_rejected event_id=%s", event.event_id
            )

        return CognitiveResult(
            event_id=event.event_id,
            meaning=meaning,
            action=action,
            reason_code=reason,
            referenced_memory_ids=referenced,
            memory_candidates=candidates,
            response_style=raw.response_style.strip() or "deepseek_plain",
            provider=PROVIDER_NAME,
            intent=(raw.intent or "unknown").strip()[:128] or "unknown",
            addressed_to=raw.addressed_to or raw.addressee,
            silence_requested=bool(raw.silence_requested),
            reply_necessity=self._necessity(raw.reply_necessity),
            conversation_state=(raw.conversation_state or "unknown").strip()[:128]
            or "unknown",
            affective_signal=signal,
            nonresponse_motive=motive,
            scene_proposal=scene_proposal,
            dialogue_impulse=dialogue_impulse,
            pragmatic=pragmatic,
        )

    @staticmethod
    def _dialogue_impulse(raw) -> Optional[DialogueImpulse]:
        """Parse an optional guidance hint; malformed input degrades to None.

        The impulse never decides an action. It only gives the single response
        call a sense of what this turn is doing and where it might go, so the
        reply does not have to re-derive it from raw history.
        """
        if raw is None:
            return None
        if isinstance(raw, DialogueImpulse):
            return raw
        if not isinstance(raw, dict):
            return None
        data = {
            key: str(value or "").strip()
            for key, value in raw.items()
            if key
            in {
                "interpretation",
                "response_direction",
                "stance",
                "unresolved_uncertainty",
            }
        }
        if not any(data.values()):
            return None
        try:
            return DialogueImpulse(**data)
        except Exception:
            return None

    @staticmethod
    def _operation(value: str) -> MemoryOperation:
        try:
            return MemoryOperation(value.strip().upper())
        except (ValueError, AttributeError):
            return MemoryOperation.ADD

    @staticmethod
    def _assertion(value: str) -> AssertionType:
        try:
            return AssertionType(value.strip().upper())
        except (ValueError, AttributeError):
            return AssertionType.UNKNOWN

    @staticmethod
    def _certainty(value: str) -> CertaintyCategory:
        try:
            return CertaintyCategory(value.strip().upper())
        except (ValueError, AttributeError):
            return CertaintyCategory.UNKNOWN

    @staticmethod
    def _kind(value: str | None) -> MemoryKind | None:
        if not value:
            return None
        try:
            return MemoryKind(value.strip().upper())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _necessity(value: str) -> ReplyNecessity:
        try:
            return ReplyNecessity(value.strip().upper())
        except (ValueError, AttributeError):
            return ReplyNecessity.OPTIONAL

    @staticmethod
    def _evidence(value: str | None, event: UtteranceEvent) -> str | None:
        """Keep an evidence span only if it is a verbatim part of the input."""
        if not value:
            return None
        span = value.strip()
        if not span or span not in event.text:
            return None
        return span

    def _denied(self, event: UtteranceEvent) -> CognitiveResult:
        return CognitiveResult(
            event_id=event.event_id,
            meaning="external AI is not permitted for this speaker",
            action=CognitiveAction.WAIT,
            reason_code=ReasonCode.PERMISSION_DENIED,
            referenced_memory_ids=[],
            memory_candidates=[],
            response_style="permission_denied",
            provider=PROVIDER_NAME,
        )

    @staticmethod
    def _as_context(event: UtteranceEvent, context, memories: list) -> CognitiveContext:
        """Accept a rich CognitiveContext, or degrade a bare one for safety."""
        if isinstance(context, CognitiveContext):
            return context
        return CognitiveContext(
            conversation_id=event.conversation_id,
            current_event_id=event.event_id,
            current_speaker_id=event.speaker_person_id,
            current_speaker_label=event.speaker_label
            or event.speaker_person_id
            or "unknown",
            current_text=event.text,
            current_addressee_hint=event.addressee_hint.value,
            quiet_mode=getattr(context, "quiet_mode", False),
            turns=[],
            memories=memories or [],
        )
