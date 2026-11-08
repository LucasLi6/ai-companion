"""Turn untrusted model output into a locally validated route decision.

Nothing here is decided by matching user text against phrases: evidence is
checked against the *authorized transcript*, tool names against the live
registry, and optional semantic hints through the existing local validators.
"""

from __future__ import annotations

from typing import Optional

from pydantic import ValidationError

from companion.domain.enums import ReplyNecessity
from companion.domain.models import (
    validate_affective_signal,
    validate_pragmatic_analysis,
    validate_scene_proposal,
)
from companion.routing.errors import RouterValidationError
from companion.routing.models import (
    MAX_TOOL_CALLS,
    InputSemanticStatus,
    RawRouteDecision,
    RoutingContext,
    ToolRequest,
    ValidatedRouteDecision,
)
from companion.tools.registry import ToolRegistry


class RoutingValidator:
    def __init__(self, *, max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        self.max_tool_calls = max(0, int(max_tool_calls))

    def validate(
        self,
        raw: RawRouteDecision,
        context: RoutingContext,
        registry: ToolRegistry,
    ) -> ValidatedRouteDecision:
        if not isinstance(raw, RawRouteDecision):
            raise RouterValidationError("router output must be RawRouteDecision")

        if len(raw.tool_requests) > self.max_tool_calls:
            raise RouterValidationError(
                f"tool request budget exceeded: {len(raw.tool_requests)}"
            )

        self._validate_input_semantics(raw)

        requests: list[ToolRequest] = []
        for request in raw.tool_requests:
            registered = registry.get(request.tool_name)
            if registered is None:
                raise RouterValidationError(
                    f"unregistered tool: {request.tool_name!r}"
                )
            try:
                parsed = registered.spec.argument_model.model_validate(
                    request.arguments
                )
            except ValidationError as exc:
                raise RouterValidationError(
                    f"invalid arguments for {request.tool_name}: {exc}"
                ) from exc
            requests.append(
                request.model_copy(update={"arguments": parsed.model_dump()})
            )

        evidence = self._filter_evidence(raw.evidence, context)
        visible_ids = {eid for eid in context.visible_event_ids if eid}
        allowed_texts = [
            context.current_text,
            *[t for t in context.authorized_history_texts if t],
        ]

        affective = validate_affective_signal(
            raw.affective_signal, context.current_text
        )
        pragmatic = validate_pragmatic_analysis(
            raw.pragmatic,
            context.current_text,
            allowed_antecedent_ids=visible_ids,
        )
        scene = validate_scene_proposal(
            raw.scene_proposal,
            current_text=context.current_text,
            authorized_texts=allowed_texts[1:],
        )
        control = raw.data_control_request
        if control is not None:
            # The model may describe a request but can never waive confirmation
            # or widen the scope. Both are enforced again by DataControlService.
            control = control.model_copy(
                update={
                    "requires_confirmation": True,
                    "scope": "SELF",
                    "evidence": self._filter_evidence(control.evidence, context),
                }
            )

        return ValidatedRouteDecision(
            schema_version=raw.schema_version,
            action=raw.action,
            reply_necessity=raw.reply_necessity,
            addressee_assessment=raw.addressee_assessment,
            input_semantics=raw.input_semantics,
            turn_state=raw.turn_state,
            tone_direction=raw.tone_direction,
            processing_intents=list(raw.processing_intents),
            tool_requests=requests,
            needs_followup_planning=bool(raw.needs_followup_planning),
            pragmatic=pragmatic,
            affective_signal=affective,
            scene_proposal=scene,
            data_control_request=control,
            reason=raw.reason,
            evidence=evidence,
            silence_requested=bool(
                raw.silence_requested and raw.action.value == "IGNORE"
            ),
        )

    def validate_followup(
        self,
        raw: RawRouteDecision,
        context: RoutingContext,
        registry: ToolRegistry,
        previous: ValidatedRouteDecision,
    ) -> ValidatedRouteDecision:
        """Validate a tool-informed follow-up plan.

        A follow-up may add read-only tools and refine optional hints. It may
        not silently reinterpret the event and flip the reply decision.
        """
        followup = self.validate(raw, context, registry)
        if followup.core_fingerprint() != previous.core_fingerprint():
            raise RouterValidationError(
                "follow-up planning may not change the reply decision"
            )
        return followup

    @staticmethod
    def _validate_input_semantics(raw: RawRouteDecision) -> None:
        """Reject structural contradictions, never inspect the user text.

        ``NON_COMMUNICATIVE`` means there is no understandable act to process.
        A payload that also asks for a durable memory operation, a privacy
        operation or a concrete tool task contradicts itself; the local layer
        must not try to guess which half to believe. This reads validated
        enums only — never ``reason``, ``evidence`` or the raw input text.
        """
        if raw.input_semantics != InputSemanticStatus.NON_COMMUNICATIVE:
            return
        if raw.processing_intents:
            raise RouterValidationError(
                "non-communicative input cannot request internal processing"
            )
        if raw.tool_requests:
            raise RouterValidationError(
                "non-communicative input cannot request tools"
            )
        if raw.data_control_request is not None:
            raise RouterValidationError(
                "non-communicative input cannot carry a data control request"
            )
        # A reply cannot be *required* by something the model itself says is
        # not a communicative act. (A RESPOND suggestion is still allowed to
        # reach local policy, which downgrades it to IGNORE.)
        if raw.reply_necessity == ReplyNecessity.REQUIRED:
            raise RouterValidationError(
                "non-communicative input cannot require a reply"
            )

    @staticmethod
    def _filter_evidence(evidence, context: RoutingContext):
        """Drop references that are not in the authorized visible context.

        Invalid references are cleared rather than allowed to create a
        plausible-looking audit trail. The decision itself can still stand on
        the current input.
        """
        texts: dict[Optional[str], list[str]] = {
            context.current_event_id: [context.current_text]
        }
        for index, event_id in enumerate(context.visible_event_ids):
            if index < len(context.authorized_history_texts):
                texts.setdefault(event_id, []).append(
                    context.authorized_history_texts[index]
                )
        kept = []
        for item in evidence:
            if item.event_id is None:
                if not item.quote or item.quote in context.current_text:
                    kept.append(item)
                continue
            sources = texts.get(item.event_id)
            if sources is None:
                continue
            if item.quote and not any(item.quote in source for source in sources):
                continue
            kept.append(item)
        return kept
