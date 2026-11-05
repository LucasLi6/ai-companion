"""Trusted runtime clock tool."""

from __future__ import annotations

from pydantic import BaseModel

from companion.domain.models import utcnow
from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult


class EmptyArguments(BaseModel):
    model_config = {"extra": "forbid"}


class CurrentTimeHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        EmptyArguments.model_validate(arguments)
        snapshot = context.runtime_snapshot
        if snapshot is not None:
            data = {
                "utc": snapshot.received_at_utc.isoformat(),
                "local_time": snapshot.local_time,
                "timezone": snapshot.timezone,
                "timezone_source": snapshot.timezone_source,
                "reentry_state": snapshot.reentry_state,
            }
            return ToolHandlerResult(status=ToolStatus.SUCCESS, data=data)
        now = utcnow()
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data={"utc": now.isoformat(), "local_time": "", "timezone": "UTC"},
        )
