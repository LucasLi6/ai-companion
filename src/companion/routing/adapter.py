"""Convert a validated route decision into the existing ``CognitiveResult``.

The rest of the pipeline (policy, scene, emotion, persistence, speaker) already
speaks ``CognitiveResult``. This adapter is the *only* bridge from the semantic
router into that shape, so the meaning of each field stays explicit:

- ``event_id`` always comes from the trusted event, never from model output;
- ``action`` is the router's validated semantic decision;
- ``reason_code`` is a controlled mapping over validated enums, not prose;
- ``meaning`` is a short audit note (the router's reason), never a reply;
- ``memory_candidates`` stays empty — memory writes continue through the
  existing extraction/consolidation path;
- referenced memory ids come only from already-authorized memory tool results.
"""

from __future__ import annotations

from typing import Any, Sequence

from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.routing.models import (
    RouteAction,
    RouteAddressee,
    ToolStatus,
    ValidatedRouteDecision,
)

__all__ = ["route_reason_code", "route_to_cognitive_result"]

_MEMORY_TOOLS = frozenset({"memory.search", "memory.browse"})


def route_reason_code(route: ValidatedRouteDecision) -> ReasonCode:
    """A controlled reason mapping over validated semantic enums.

    The router's free-text ``reason`` is never parsed. This mapping only reads
    enums the validator accepted, so an unknown sentence cannot smuggle a new
    rule into the audit trail.
    """
    addressee = route.addressee_assessment
    if route.action == RouteAction.IGNORE:
        if route.silence_requested:
            return ReasonCode.EXPLICIT_SILENCE_REQUEST
        if addressee == RouteAddressee.HUMAN:
            return ReasonCode.CONVERSATION_BETWEEN_HUMANS
        return ReasonCode.RESPONSE_NOT_NECESSARY

    if route.action == RouteAction.WAIT:
        if addressee == RouteAddressee.HUMAN:
            return ReasonCode.CONVERSATION_BETWEEN_HUMANS
        if addressee == RouteAddressee.AGENT:
            return ReasonCode.CONSERVATIVE_AMBIGUITY
        return ReasonCode.AMBIGUOUS_ADDRESSEE

    # RESPOND
    if addressee == RouteAddressee.HUMAN:
        return ReasonCode.DIRECT_HUMAN_ADDRESS
    if route.turn_state.value == "FOLLOW_UP":
        return ReasonCode.FOLLOW_UP_TO_AGENT
    if addressee == RouteAddressee.AGENT:
        return ReasonCode.DIRECT_AGENT_ADDRESS
    return ReasonCode.UNKNOWN_ADDRESSEE


def route_to_cognitive_result(
    event,
    route: ValidatedRouteDecision,
    tool_results: Sequence[Any] = (),
) -> CognitiveResult:
    """Build the trusted ``CognitiveResult`` for a routed event."""
    meaning = " ".join((route.reason or "").split()).strip()
    if not meaning:
        meaning = "semantic_router_decision"

    intents = [item.value.lower() for item in route.processing_intents]
    turn_state = route.turn_state.value.lower() or "unknown"
    return CognitiveResult(
        event_id=event.event_id,
        meaning=meaning[:2048],
        action=CognitiveAction(route.action.value),
        reason_code=route_reason_code(route),
        referenced_memory_ids=_referenced_memory_ids(tool_results),
        memory_candidates=[],
        response_style="semantic_router",
        provider="semantic_router",
        intent=(intents[0] if intents else "unknown")[:128],
        addressed_to=_addressed_to(route.addressee_assessment),
        silence_requested=bool(
            route.silence_requested and route.action == RouteAction.IGNORE
        ),
        reply_necessity=route.reply_necessity,
        conversation_state=turn_state[:128],
        affective_signal=route.affective_signal,
        scene_proposal=route.scene_proposal,
        pragmatic=route.pragmatic,
    )


def _addressed_to(addressee: RouteAddressee):
    if addressee == RouteAddressee.AGENT:
        return "agent"
    if addressee == RouteAddressee.HUMAN:
        return "human"
    return None


def _referenced_memory_ids(tool_results: Sequence[Any]) -> list[str]:
    """Ids from memory tool results only, i.e. after the store's own scoping."""
    ids: list[str] = []
    for result in tool_results:
        name = getattr(result, "tool_name", None)
        status = getattr(result, "status", None)
        if name not in _MEMORY_TOOLS or status != ToolStatus.SUCCESS:
            continue
        for ref in getattr(result, "source_refs", ()) or ():
            text = str(ref)
            if text and text not in ids:
                ids.append(text)
    return ids[:20]
