"""Orchestrate routing, bounded tool execution and at most one follow-up plan."""

from __future__ import annotations

import logging
from typing import Optional

from companion.routing.errors import RouterUnavailable, RouterValidationError
from companion.routing.models import (
    MAX_TOOL_CALLS,
    RoutePhase,
    RoutingContext,
    RoutingOutcome,
    ToolStatus,
    ValidatedRouteDecision,
)
from companion.routing.repository import RoutingRepository, request_signature
from companion.routing.validator import RoutingValidator
from companion.tools.executor import ToolExecutor
from companion.tools.models import ToolExecutionContext, ToolResult
from companion.tools.registry import ToolRegistry


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
    ) -> RoutingOutcome:
        provider_calls = 0
        repo = self.repository if persist else None

        route: Optional[ValidatedRouteDecision] = None
        if repo is not None:
            route = await repo.load_validated(event.event_id, 0)

        if route is None:
            if self.provider is None:
                raise RouterUnavailable("semantic routing has no router provider")
            raw = await self.provider.route(context)
            provider_calls += 1
            route = self.validator.validate(raw, context, self.registry)
            route = self._annotate(route)
            if repo is not None:
                await repo.save_decision(
                    event_id=event.event_id,
                    phase=RoutePhase.INITIAL,
                    step_index=0,
                    route=route,
                )

        initial_route = route
        results: list[ToolResult] = []
        executed_requests = list(route.tool_requests[: self.max_tool_calls])
        results = await self.executor.execute_many(
            executed_requests,
            execution_context,
            step_index=0,
            max_calls=self.max_tool_calls,
        )
        await self._persist_calls(
            event.event_id, 0, executed_requests, results, repo
        )

        planning_steps = 0
        if (
            provider_calls + 1 <= self.max_router_calls
            and self._should_replan(route, executed_requests, results)
        ):
            follow_raw = await self.provider.replan(context, route, results)
            provider_calls += 1
            followup = self.validator.validate_followup(
                follow_raw, context, self.registry, route
            )
            followup = self._annotate(followup)
            if repo is not None:
                await repo.save_decision(
                    event_id=event.event_id,
                    phase=RoutePhase.FOLLOWUP,
                    step_index=1,
                    route=followup,
                )
            seen = {request_signature(item) for item in executed_requests}
            new_requests = [
                item
                for item in followup.tool_requests
                if request_signature(item) not in seen
            ][: self.max_tool_calls]
            new_results = await self.executor.execute_many(
                new_requests,
                execution_context,
                step_index=1,
                max_calls=self.max_tool_calls,
            )
            await self._persist_calls(
                event.event_id, 1, new_requests, new_results, repo
            )
            results.extend(new_results)
            route = followup
            planning_steps = 1

        return RoutingOutcome(
            route=route,
            tool_results=results,
            planning_steps=planning_steps,
            provider_calls=provider_calls,
            initial_route=initial_route,
        )

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

    @staticmethod
    async def _persist_calls(
        event_id: str,
        step_index: int,
        requests,
        results: list[ToolResult],
        repo: Optional[RoutingRepository],
    ) -> None:
        if repo is None:
            return
        for request, result in zip(requests, results):
            await repo.save_tool_call(
                tool_call_id=result.tool_call_id,
                event_id=event_id,
                route_step_index=step_index,
                tool_name=request.tool_name,
                arguments=request.arguments,
                arguments_hash=ToolExecutor.arguments_hash(request),
                status=result.status,
                error_code=result.error_code,
                source_refs=result.source_refs,
            )
