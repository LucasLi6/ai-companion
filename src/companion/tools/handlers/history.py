"""Read-only conversation-history tools, scoped to one speaker/conversation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult

MAX_ITEM_CHARS = 1000


class RecentHistoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=6, ge=1, le=10)


class EarliestHistoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=3, ge=1, le=5)


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def _filter(context: ToolExecutionContext, turns: list) -> list:
    if context.privacy_gateway is not None:
        return await context.privacy_gateway.filter_utterances(turns)
    return turns


def _compact(turns: list, limit: int) -> list[dict]:
    return [
        {
            "occurred_at": turn.occurred_at.isoformat(),
            "speaker": turn.speaker_label or "unknown",
            "text": str(turn.text or "")[:MAX_ITEM_CHARS],
        }
        for turn in turns[:limit]
    ]


class RecentHistoryHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = RecentHistoryArguments.model_validate(arguments)
        if context.person_id is None:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="no_speaker"
            )
        turns = await context.repos.utterance.recent_context(
            context.conversation_id,
            exclude_event_id=context.event_id,
            limit=args.limit,
        )
        turns = await _filter(context, turns)
        if not turns:
            return ToolHandlerResult(status=ToolStatus.EMPTY, data=[])
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=_compact(turns, args.limit),
            source_refs=[turn.event_id for turn in turns],
        )


class EarliestHistoryHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = EarliestHistoryArguments.model_validate(arguments)
        if context.person_id is None:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="no_speaker"
            )
        turns = await context.repos.utterance.earliest_for_speaker(
            context.person_id,
            context.conversation_id,
            exclude_event_id=context.event_id,
            limit=args.limit,
        )
        turns = await _filter(context, turns)
        if not turns:
            return ToolHandlerResult(status=ToolStatus.EMPTY, data=[])
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=_compact(turns, args.limit),
            source_refs=[turn.event_id for turn in turns],
        )


class ConversationStatsHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        EmptyArguments.model_validate(arguments)
        if context.person_id is None:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="no_speaker"
            )
        stats = await context.repos.utterance.conversation_stats(
            context.person_id
        )
        if not any(stats.values()):
            return ToolHandlerResult(status=ToolStatus.EMPTY, data=stats)
        return ToolHandlerResult(status=ToolStatus.SUCCESS, data=stats)
