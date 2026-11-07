"""Build the minimal, authorized context the router sees.

Deliberately absent: retrieved long-term memories, full persona compilation,
dialogue style, runtime prompt and memory samples. If the router needs more
information it must ask for a registered read-only tool.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.cognition.models import (
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    CognitiveContext,
    CognitiveTurn,
)
from companion.cognition.nonresponse import build_action_history_record
from companion.domain.enums import CognitiveAction
from companion.domain.models import ConversationContext, UtteranceEvent
from companion.people.profile_compiler import compile_profile_context
from companion.people.registry import PersonRegistry
from companion.routing.models import RoutingContext
from companion.storage.repositories import Repositories
from companion.tools.registry import ToolRegistry

DEFAULT_HISTORY_LIMIT = 8
DEFAULT_MAX_CHARS = 1600


def truncate_dialogue_turns(turns: list[dict], max_chars: int) -> list[dict]:
    """Drop the oldest *complete* dialogue units until the budget fits.

    A unit is one human turn plus the assistant reply produced for it. Whole
    units are removed from the front, so the remaining history never starts
    with an orphan assistant reply whose question was cut away. Speaker
    identity and chronological order are untouched, and an assistant reply is
    attached only to the human turn immediately before it. The most recent
    unit is kept even if it alone exceeds the soft budget.
    """
    units: list[list[dict]] = []
    current: list[dict] = []
    for turn in turns:
        role = turn.get("role")
        if role == ROLE_HUMAN:
            if current:
                units.append(current)
            current = [turn]
        elif role == ROLE_ASSISTANT and current:
            current.append(turn)
        # An assistant turn with no preceding human turn is never kept: it has
        # no visible question in this window.
    if current:
        units.append(current)

    total = sum(
        len(str(turn.get("text") or ""))
        for unit in units
        for turn in unit
    )
    while total > max_chars and len(units) > 1:
        dropped = units.pop(0)
        total -= sum(len(str(turn.get("text") or "")) for turn in dropped)

    return [turn for unit in units for turn in unit]


class RoutingContextBuilder:
    def __init__(
        self,
        *,
        repos: Repositories,
        registry: PersonRegistry,
        privacy_gateway=None,
        profile_service=None,
        scene_manager=None,
        tool_registry: Optional[ToolRegistry] = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        max_chars: int = DEFAULT_MAX_CHARS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.privacy_gateway = privacy_gateway
        self.profile_service = profile_service
        self.scene_manager = scene_manager
        self.tool_registry = tool_registry
        self.history_limit = max(1, int(history_limit))
        self.max_chars = max(200, int(max_chars))
        self.logger = logger or logging.getLogger("companion.routing.context")

    async def build(
        self,
        event: UtteranceEvent,
        conversation: ConversationContext,
    ) -> RoutingContext:
        history = await self.repos.utterance.recent_context(
            event.conversation_id,
            exclude_event_id=event.event_id,
            limit=self.history_limit,
        )
        if self.privacy_gateway is not None:
            history = await self.privacy_gateway.filter_utterances(history)

        actions = await self.repos.analysis.get_actions_for_events(
            [item.event_id for item in history]
        )
        labels = await self._labels(history, event)
        turns = self._interleave(history, actions, labels)
        turns = self._truncate(turns, self.max_chars)
        action_history = self._action_history(history, actions, turns)

        scene = None
        if self.scene_manager is not None:
            try:
                record = await self.scene_manager.load(
                    event.conversation_id, event.speaker_person_id
                )
                if record is not None:
                    scene = record.state.prompt_dict()
            except Exception as exc:  # scene is optional context
                self.logger.warning(
                    "routing_scene_failed event_id=%s error=%s",
                    event.event_id,
                    type(exc).__name__,
                )

        profile_context, profile_revision = await self._profile_context(
            event.speaker_person_id, query=event.text
        )
        visible_ids = [turn["event_id"] for turn in turns]
        # One text per visible turn, index-aligned for evidence validation.
        authorized_texts = [str(turn.get("text") or "") for turn in turns]

        specs = self.tool_registry.specs() if self.tool_registry else []
        return RoutingContext(
            conversation_id=event.conversation_id,
            current_event_id=event.event_id,
            current_speaker_id=event.speaker_person_id,
            current_speaker_label=labels.get(
                event.speaker_person_id, event.speaker_label or "unknown"
            ),
            current_text=event.text,
            trusted_addressee_hint=event.addressee_hint.value,
            input_modality=event.source.modality,
            quiet_mode=bool(conversation.quiet_mode),
            recent_turns=turns,
            recent_nonresponse_actions=action_history,
            conversation_state={
                "quiet_mode": bool(conversation.quiet_mode),
                "last_action": conversation.last_action,
            },
            compact_scene=scene,
            compact_person_profile=profile_context,
            compact_person_profile_revision=profile_revision,
            available_tool_specs=specs,
            visible_event_ids=visible_ids,
            authorized_history_texts=authorized_texts,
        )

    async def _labels(self, history, event) -> dict:
        labels: dict = {}
        for item in [*history, event]:
            person_id = item.speaker_person_id
            if person_id is None:
                labels[None] = item.speaker_label or "unknown"
                continue
            if person_id in labels:
                continue
            person = await self.registry.get(person_id)
            labels[person_id] = (
                person.display_name if person is not None else person_id
            )
        return labels

    @staticmethod
    def _interleave(history, actions, labels) -> list[dict]:
        turns: list[dict] = []
        for item in history:
            label = labels.get(
                item.speaker_person_id, item.speaker_label or "unknown"
            )
            turns.append(
                CognitiveTurn(
                    event_id=item.event_id,
                    role=ROLE_HUMAN,
                    speaker_person_id=item.speaker_person_id,
                    speaker_label=label,
                    addressee_hint=item.addressee_hint.value,
                    text=item.text,
                ).model_dump()
            )
            action = actions.get(item.event_id)
            if (
                action is not None
                and action.action_type == CognitiveAction.RESPOND
                and action.status in ("COMPLETED", "TRUNCATED")
                and action.response_text
            ):
                turns.append(
                    CognitiveTurn(
                        event_id=item.event_id,
                        role=ROLE_ASSISTANT,
                        speaker_person_id=None,
                        speaker_label="assistant",
                        addressee_hint=None,
                        text=action.response_text,
                    ).model_dump()
                )
        return turns

    @staticmethod
    def _truncate(turns: list[dict], max_chars: int) -> list[dict]:
        return truncate_dialogue_turns(turns, max_chars)

    @staticmethod
    def _action_history(history, actions, turns) -> list[dict]:
        kept = {
            turn["event_id"]
            for turn in turns
            if turn.get("role") == ROLE_HUMAN
        }
        records = []
        for utterance in history:
            if utterance.event_id not in kept:
                continue
            action = actions.get(utterance.event_id)
            if action is None:
                continue
            record = build_action_history_record(action)
            if record is not None:
                records.append(record.model_dump())
        return records[-8:]

    async def _profile_context(
        self, person_id, *, query: str = ""
    ) -> tuple[Optional[str], Optional[int]]:
        if self.profile_service is None or person_id is None:
            return None, None
        if not await self.registry.is_authorized(person_id):
            return None, None
        consent = await self.repos.person.get_consent(person_id)
        if (
            consent is None
            or not consent.allow_persistence
            or not consent.allow_external_ai
        ):
            return None, None
        profile = await self.profile_service.get(person_id)
        if profile is None:
            return None, None
        # One read per turn: the speaker reuses this same block, so the router
        # and the reply cannot silently see two different profile revisions.
        return (
            compile_profile_context(profile, query=query),
            int(profile.metadata.revision),
        )

    @staticmethod
    def speaker_context(
        routing: RoutingContext,
        *,
        memories: list[dict],
        agent_context: Optional[dict],
        route_context: Optional[dict],
        tool_results: list[dict],
    ) -> CognitiveContext:
        """Build a speaker context from the already-authorized routing view.

        The turns are reused verbatim (no second history query, no new privacy
        bypass): they were filtered before they were placed on the routing
        context.
        """
        return CognitiveContext(
            conversation_id=routing.conversation_id,
            current_event_id=routing.current_event_id,
            current_speaker_id=routing.current_speaker_id,
            current_speaker_label=routing.current_speaker_label,
            current_text=routing.current_text,
            current_addressee_hint=routing.trusted_addressee_hint,
            quiet_mode=routing.quiet_mode,
            turns=[CognitiveTurn.model_validate(t) for t in routing.recent_turns],
            action_history=[
                _action_record(item)
                for item in routing.recent_nonresponse_actions
            ],
            memories=memories,
            agent_context=agent_context,
            person_profile_context=routing.compact_person_profile,
            scene_state=routing.compact_scene,
            route_context=route_context,
            tool_results=tool_results,
        )


def build_speaker_context(
    event: UtteranceEvent,
    routing: RoutingContext,
    route,
    tool_results,
    *,
    base: Optional[CognitiveContext] = None,
    memories: Optional[list[dict]] = None,
    agent_context: Optional[dict] = None,
) -> CognitiveContext:
    """Attach the semantic route and tool results to the speaker context.

    ``base`` is the full, already privacy-checked ``CognitiveContext`` built by
    the regular context builder after routing; when supplied it is copied, so
    history, memories, runtime state and profile context are not rebuilt or
    bypassed. Without ``base`` a minimal context is built from the routing
    view. The route is rendered as *model judgment* — a short summary, never
    the raw payload, hidden reasoning or a new instruction.
    """
    payload = _route_payload(route)
    results = [_tool_result_payload(item) for item in (tool_results or [])]
    if base is not None:
        return base.model_copy(
            update={"route_context": payload, "tool_results": results}
        )
    return RoutingContextBuilder.speaker_context(
        routing,
        memories=list(memories or []),
        agent_context=agent_context,
        route_context=payload,
        tool_results=results,
    )


def _route_payload(route) -> dict:
    """Project the validated route for the Speaker.

    ``reason`` is intentionally absent. It is the router's natural-language
    explanation of its own judgment; it stays on the decision object (and in
    the ``route_decisions`` audit row, via ``audit_dict``) but must not reach
    the Speaker, where it would read as advice about what to say.
    """
    return {
        "source": "semantic_router",
        "action": route.action.value,
        "reply_necessity": route.reply_necessity.value,
        "addressee_assessment": route.addressee_assessment.value,
        "turn_state": route.turn_state.value,
        "processing_intents": [item.value for item in route.processing_intents],
        "tool_requests": [
            {
                "tool_name": item.tool_name,
                "purpose": item.purpose,
                "required": bool(item.required),
            }
            for item in route.tool_requests
        ],
        "evidence": [
            {"event_id": item.event_id, "quote": item.quote[:512]}
            for item in route.evidence
        ],
    }


def _tool_result_payload(result) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return dict(result)


def _action_record(item: dict):
    from companion.cognition.models import ActionHistoryRecord

    return ActionHistoryRecord.model_validate(item)
