"""Read-only memory tools.

Both tools call the existing access-controlled :class:`SqliteMemoryStore`.
They cannot widen visibility: retrieval runs with the speaker's non-admin
access context and browse re-checks every record through the same gateway.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.tools.models import ToolExecutionContext, ToolHandlerResult
from companion.routing.models import ToolStatus

MAX_ITEM_CHARS = 1200


class MemorySearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=6, ge=1, le=10)


class MemoryBrowseArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=3, ge=1, le=5)


def _compact(memory: dict, limit: int) -> list[dict]:
    return [
        {
            "memory_id": item.get("memory_id"),
            "kind": item.get("kind"),
            "content": str(item.get("content") or "")[:MAX_ITEM_CHARS],
            "updated_at": item.get("updated_at"),
        }
        for item in memory[:limit]
    ]


class MemorySearchHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = MemorySearchArguments.model_validate(arguments)
        if context.person_id is None:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="no_speaker"
            )
        rows = await context.memory.retrieve(args.query, context.person_id)
        data = _compact(rows, args.limit)
        if not data:
            return ToolHandlerResult(status=ToolStatus.EMPTY, data=[])
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=data,
            source_refs=[
                str(item["memory_id"]) for item in data if item.get("memory_id")
            ],
        )


class MemoryBrowseHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = MemoryBrowseArguments.model_validate(arguments)
        if context.person_id is None:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="no_speaker"
            )
        rows = await context.memory.browse(context.person_id, limit=args.limit)
        data = _compact(rows, args.limit)
        if not data:
            return ToolHandlerResult(status=ToolStatus.EMPTY, data=[])
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=data,
            source_refs=[
                str(item["memory_id"]) for item in data if item.get("memory_id")
            ],
        )
