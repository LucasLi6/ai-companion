"""Orchestrate routing, bounded tool execution and at most one follow-up plan.

The service is deliberately resumable. A retry of the same event:

- reuses a committed step-0 decision instead of calling the router again;
- reuses a committed step-1 decision instead of asking for a second plan;
- re-reads read-only tools through the current permission checks (the saved
  row never contains private result data and is never trusted as a substitute);
- counts the whole event against one tool budget, not one budget per phase.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from companion.routing.errors import RouterUnavailable
from companion.routing.models import (
    MAX_TOOL_CALLS,
    RoutePhase,
    RoutingContext,
    RoutingOutcome,
    ToolRequest,
    ToolStatus,
    ValidatedRouteDecision,
)
from companion.routing.repository import RoutingRepository, request_signature
from companion.routing.validator import RoutingValidator
from companion.tools.executor import ToolExecutor
from companion.tools.models import ToolExecutionContext, ToolResult
from companion.tools.registry import ToolRegistry

_FAILED_EXECUTION_STATUSES = {
    ToolStatus.DENIED,
    ToolStatus.DISABLED,
    ToolStatus.TIMEOUT,
    ToolStatus.ERROR,
}


class RoutingService:
    def __init__(
        self,
        *,
        provider,
        validator: RoutingValidator,
        registry: ToolRegistry,
        executor: ToolExecutor,
        repository: Optional[RoutingRepository] = None,
        max_router_calls: int = 2,
        max_tool_calls: int = MAX_TOOL_CALLS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.provider = provider
        self.validator = validator
        self.registry = registry
        self.executor = executor
        self.repository = repository
        self.max_router_calls = max(1, min(2, int(max_router_calls)))
        self.max_tool_calls = max(0, int(max_tool_calls))
        self.logger = logger or logging.getLogger("companion.routing")

    async def execute(
        self,
        event,
        context: RoutingContext,
        execution_context: ToolExecutionContext,
        *,
        persist: bool = True,
        allow_tool_reads: bool = True,
    ) -> RoutingOutcome:
        provider_calls = 0
        repo = self.repository if persist else None
        event_id = event.event_id

        route, made = await self._load_or_create_route(
            event_id, 0, context, repo
        )
        provider_calls += made
        initial_route = route

        persisted0 = await self._calls_for_step(repo, event_id, 0)
        planned0 = min(len(route.tool_requests), self.max_tool_calls)
        planned_signatures: set[str] = {
            request_signature(item) for item in route.tool_requests[:planned0]
        }
        results: list[ToolResult] = []
        executed0: list[ToolRequest] = []
        if allow_tool_reads:
            _, executed0, new_results0 = await self._execute_or_restore(
                event_id,
                0,
                route.tool_requests[:planned0],
                persisted0,
                execution_context,
                repo,
            )
            results.extend(new_results0)

        remaining = max(0, self.max_tool_calls - planned0)
        followup, made = await self._load_or_create_followup(
            event_id,
            context,
            route,
            results,
            repo,
            provider_calls,
            remaining=remaining,
            allow_tool_reads=allow_tool_reads,
        )
        provider_calls += made
        planning_steps = 0
        if followup is not None:
            planning_steps = 1
            persisted1 = await self._calls_for_step(repo, event_id, 1)
            seen = set(planned_signatures)
            for item in executed0:
                seen.add(request_signature(item))
            candidates = [
                item
                for item in followup.tool_requests
                if request_signature(item) not in seen
            ][:remaining]
            if allow_tool_reads:
                _, _, new_results1 = await self._execute_or_restore(
                    event_id,
                    1,
                    candidates,
                    persisted1,
                    execution_context,
                    repo,
                )
                results.extend(new_results1)
            route = followup

        return RoutingOutcome(
            route=route,
            tool_results=results,
            planning_steps=planning_steps,
            provider_calls=provider_calls,
            initial_route=initial_route,
        )

    # -- phase loading -----------------------------------------------------

    async def _load_or_create_route(
        self,
        event_id: str,
        step_index: int,
        context: RoutingContext,
        repo: Optional[RoutingRepository],
    ) -> tuple[ValidatedRouteDecision, int]:
        if repo is not None:
            route = await repo.load_validated(event_id, step_index)
            if route is not None:
                return route, 0
        if self.provider is None:
            raise RouterUnavailable("semantic routing has no router provider")
        raw = await self.provider.route(context)
        route = self.validator.validate(raw, context, self.registry)
        route = self._annotate(route)
        if repo is not None:
            await repo.save_decision(
                event_id=event_id,
                phase=RoutePhase.INITIAL,
                step_index=step_index,
                route=route,
            )
        return route, 1

    async def _load_or_create_followup(
        self,
        event_id: str,
        context: RoutingContext,
        route: ValidatedRouteDecision,
        results: Sequence[ToolResult],
        repo: Optional[RoutingRepository],
        provider_calls: int,
        *,
        remaining: int,
        allow_tool_reads: bool,
    ) -> tuple[Optional[ValidatedRouteDecision], int]:
        if repo is not None:
            saved = await repo.load_validated(event_id, 1)
            if saved is not None:
                return saved, 0
        if not allow_tool_reads or remaining <= 0:
            return None, 0
        if provider_calls + 1 > self.max_router_calls:
            return None, 0
        if not self._should_replan(route, route.tool_requests, list(results)):
            return None, 0
        if self.provider is None:
            raise RouterUnavailable("semantic routing has no router provider")
        follow_raw = await self.provider.replan(context, route, list(results))
        followup = self.validator.validate_followup(
            follow_raw, context, self.registry, route
        )
        followup = self._annotate(followup)
        if repo is not None:
            await repo.save_decision(
                event_id=event_id,
                phase=RoutePhase.FOLLOWUP,
                step_index=1,
                route=followup,
            )
        return followup, 1

    # -- tools -------------------------------------------------------------

    async def _execute_or_restore(
        self,
        event_id: str,
        step_index: int,
        requests: Sequence[ToolRequest],
        persisted: list[dict],
        execution_context: ToolExecutionContext,
        repo: Optional[RoutingRepository],
    ) -> tuple[list[ToolRequest], list[ToolRequest], list[ToolResult]]:
        """Run a step, re-reading persisted read-only calls with stable ids.

        A saved row restores only its *identity* (tool name, arguments, id).
        The data is fetched again under the current identity and permissions;
        stale private payloads are never reused.
        """
        restored_requests: list[ToolRequest] = []
        executed_requests: list[ToolRequest] = []
        results: list[ToolResult] = []

        saved_signatures = {request_signature_for_row(row) for row in persisted}
        for row in persisted:
            request = request_from_row(row)
            if request is None:
                continue
            if repo is not None:
                await repo.save_tool_call(
                    tool_call_id=row["tool_call_id"],
                    event_id=event_id,
                    route_step_index=step_index,
                    tool_name=request.tool_name,
                    arguments=request.arguments,
                    arguments_hash=ToolExecutor.arguments_hash(request),
                    execution_status="RUNNING",
                )
            result = await self.executor.execute_one(
                request,
                execution_context,
                tool_call_id=row["tool_call_id"],
                restored=True,
            )
            results.append(result)
            restored_requests.append(request)
            executed_requests.append(request)
            if repo is not None:
                await self._save_result(
                    repo, event_id, step_index, request, result
                )

        for request in requests:
            signature = request_signature(request)
            if signature in saved_signatures:
                continue
            call_id = self.executor.stable_call_id(
                event_id, step_index, len(executed_requests), request
            )
            if repo is not None:
                await repo.save_tool_call(
                    tool_call_id=call_id,
                    event_id=event_id,
                    route_step_index=step_index,
                    tool_name=request.tool_name,
                    arguments=request.arguments,
                    arguments_hash=ToolExecutor.arguments_hash(request),
                    execution_status="RUNNING",
                )
            result = await self.executor.execute_one(
                request, execution_context, tool_call_id=call_id
            )
            results.append(result)
            executed_requests.append(request)
            if repo is not None:
                await self._save_result(
                    repo, event_id, step_index, request, result
                )

        return restored_requests, executed_requests, results

    @staticmethod
    async def _save_result(
        repo: RoutingRepository,
        event_id: str,
        step_index: int,
        request: ToolRequest,
        result: ToolResult,
    ) -> None:
        await repo.save_tool_call(
            tool_call_id=result.tool_call_id,
            event_id=event_id,
            route_step_index=step_index,
            tool_name=request.tool_name,
            arguments=request.arguments,
            arguments_hash=ToolExecutor.arguments_hash(request),
            status=result.status,
            execution_status=(
                "FAILED"
                if result.status in _FAILED_EXECUTION_STATUSES
                else "COMPLETED"
            ),
            error_code=result.error_code,
            source_refs=result.source_refs,
        )

    @staticmethod
    async def _calls_for_step(
        repo: Optional[RoutingRepository], event_id: str, step_index: int
    ) -> list[dict]:
        if repo is None:
            return []
        rows = await repo.list_tool_calls(event_id)
        return [row for row in rows if int(row["route_step_index"]) == step_index]

    def _annotate(
        self, route: ValidatedRouteDecision
    ) -> ValidatedRouteDecision:
        return route.model_copy(
            update={
                "provider": str(
                    getattr(self.provider, "provider_name", "deepseek_router")
                ),
                "model": str(getattr(self.provider, "model", "") or ""),
            }
        )

    @staticmethod
    def _should_replan(
        route: ValidatedRouteDecision,
        requests,
        results: list[ToolResult],
    ) -> bool:
        if route.needs_followup_planning:
            return True
        for request, result in zip(requests, results):
            if request.required and result.status not in (
                ToolStatus.SUCCESS,
                ToolStatus.EMPTY,
            ):
                return True
        return False


def request_from_row(row: dict) -> Optional[ToolRequest]:
    """Rebuild the request identity from an audit row. No result data is used."""
    try:
        return ToolRequest(
            tool_name=str(row.get("tool_name") or ""),
            arguments=dict(row.get("arguments") or {}),
            required=False,
        )
    except (TypeError, ValueError):
        return None


def request_signature_for_row(row: dict) -> str:
    try:
        arguments = dict(row.get("arguments") or {})
    except (TypeError, ValueError):
        arguments = {}
    return request_signature(
        ToolRequest(tool_name=str(row.get("tool_name") or ""), arguments=arguments)
    )
