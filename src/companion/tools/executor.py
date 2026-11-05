"""Validate and execute tool requests; never infer intent from user text."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Optional

from pydantic import ValidationError

from companion.routing.models import ToolRequest, ToolStatus
from companion.tools.models import ToolExecutionContext, ToolResult
from companion.tools.registry import ToolRegistry

MAX_TOOL_CALLS = 4


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.registry = registry
        self.logger = logger or logging.getLogger("companion.tools")

    def stable_call_id(
        self, event_id: str, step_index: int, index: int, request: ToolRequest
    ) -> str:
        payload = json.dumps(
            {
                "event_id": event_id,
                "step": step_index,
                "index": index,
                "tool": request.tool_name,
                "arguments": request.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"tc_{digest}"

    @staticmethod
    def arguments_hash(request: ToolRequest) -> str:
        payload = json.dumps(
            request.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def execute_many(
        self,
        requests: list[ToolRequest],
        context: ToolExecutionContext,
        *,
        step_index: int = 0,
        max_calls: int = MAX_TOOL_CALLS,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for index, request in enumerate(requests[: max(0, max_calls)]):
            call_id = self.stable_call_id(
                context.event_id, step_index, index, request
            )
            results.append(
                await self.execute_one(
                    request, context, tool_call_id=call_id
                )
            )
        return results

    async def execute_one(
        self,
        request: ToolRequest,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        registered = self.registry.get(request.tool_name)
        if registered is None:
            return self._result(
                tool_call_id,
                request,
                ToolStatus.DENIED,
                error_code="unknown_tool",
            )

        spec = registered.spec
        if not self._permitted(spec.permission, context):
            return self._result(
                tool_call_id,
                request,
                ToolStatus.DENIED,
                error_code="permission_denied",
            )

        try:
            arguments = spec.argument_model.model_validate(request.arguments)
        except ValidationError:
            return self._result(
                tool_call_id,
                request,
                ToolStatus.ERROR,
                error_code="invalid_arguments",
            )

        try:
            outcome = await asyncio.wait_for(
                registered.handler.execute(arguments, context),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "tool_timeout event_id=%s tool=%s", context.event_id, spec.name
            )
            return self._result(
                tool_call_id,
                request,
                ToolStatus.TIMEOUT,
                error_code="timeout",
            )
        except Exception as exc:  # never turn a failure into a filled success
            self.logger.warning(
                "tool_failed event_id=%s tool=%s error=%s",
                context.event_id,
                spec.name,
                type(exc).__name__,
            )
            return self._result(
                tool_call_id,
                request,
                ToolStatus.ERROR,
                error_code=type(exc).__name__,
            )

        try:
            encoded = json.dumps(
                outcome.data, ensure_ascii=False, allow_nan=False
            )
        except (TypeError, ValueError):
            return self._result(
                tool_call_id,
                request,
                ToolStatus.ERROR,
                error_code="non_json_result",
            )
        if len(encoded) > spec.max_result_chars:
            return self._result(
                tool_call_id,
                request,
                ToolStatus.ERROR,
                error_code="result_too_large",
            )

        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=spec.name,
            status=outcome.status,
            data=outcome.data,
            error_code=outcome.error_code,
            source_refs=list(outcome.source_refs)[:20],
            untrusted=outcome.untrusted,
        )

    @staticmethod
    def _permitted(permission: str, context: ToolExecutionContext) -> bool:
        if permission in ("memory.read", "history.read"):
            return context.person_id is not None
        if permission == "runtime.read":
            return True
        if permission == "web.read":
            return bool(context.allow_external_ai and context.web_search)
        return False

    @staticmethod
    def _result(
        tool_call_id: str,
        request: ToolRequest,
        status: ToolStatus,
        *,
        error_code: Optional[str] = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=request.tool_name,
            status=status,
            error_code=error_code,
        )


__all__ = ["ToolExecutor", "MAX_TOOL_CALLS"]
