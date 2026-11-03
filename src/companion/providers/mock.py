"""Mock cognitive and response providers.

The mock is intentionally *predictable*, driven only by event metadata and a
small set of explicit rules. It does **not** pretend to understand semantics,
and it never fabricates confidence scores. Every result records
``provider = "mock"`` and an explicit ``reason_code``.
"""

from __future__ import annotations

from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    MemoryOperation,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    CognitiveResult,
    ConversationContext,
    MemoryCandidate,
    UtteranceEvent,
    new_id,
)


class MockCognitiveProvider:
    """Deterministic rule-based cognition using event metadata only."""

    async def analyze(
        self,
        event: UtteranceEvent,
        context: ConversationContext,
        memories: list,
    ) -> CognitiveResult:
        if event.addressee_hint.value == "agent":
            action = CognitiveAction.RESPOND
            reason = ReasonCode.DIRECT_AGENT_ADDRESS
            necessity = ReplyNecessity.REQUIRED
        elif event.addressee_hint.value == "human":
            action = CognitiveAction.IGNORE
            reason = ReasonCode.DIRECT_HUMAN_ADDRESS
            necessity = ReplyNecessity.UNNECESSARY
        else:
            action = CognitiveAction.WAIT
            reason = ReasonCode.UNKNOWN_ADDRESSEE
            necessity = ReplyNecessity.OPTIONAL

        # A single fixed candidate, clearly marked as mock-generated, so the
        # database -> memory-job pipeline can be exercised end-to-end without
        # pretending any real memory extraction happened. It is hypothetical,
        # so write-through truthfully records NO_MEMORY for it.
        candidate = MemoryCandidate(
            candidate_id=new_id("memc"),
            source_event_id=event.event_id,
            subject_person_id=event.speaker_person_id,
            content=f"[mock] candidate derived from {event.event_id}",
            operation=MemoryOperation.ADD,
            assertion_type=AssertionType.HYPOTHETICAL,
            certainty_category=CertaintyCategory.UNKNOWN,
        )

        return CognitiveResult(
            event_id=event.event_id,
            meaning=f"[mock] understood utterance as {event.addressee_hint.value}-addressed",
            action=action,
            reason_code=reason,
            referenced_memory_ids=[],
            memory_candidates=[candidate],
            response_style="mock_plain",
            provider="mock",
            intent="statement",
            addressed_to=event.addressee_hint.value,
            silence_requested=False,
            reply_necessity=necessity,
            conversation_state="ongoing",
        )


class MockResponseProvider:
    """Deterministic response generator."""

    async def generate(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context: ConversationContext,
    ) -> str:
        return "模拟回复：我收到你的话了。"
