"""DeepSeek-backed semantic router provider.

Uses the same audited transport as the analysis/response paths, with its own
request type (``"router"``) and its own model/token/thinking/timeout settings.
The provider is deliberately thin:

- it builds a compact prompt,
- calls the model in JSON mode,
- parses the payload into :class:`RawRouteDecision`.

It does **not** validate trust constraints (that is
:class:`companion.routing.validator.RoutingValidator`), execute tools or log
message content. A parse failure raises :class:`RouterInvalidOutput`; it never
returns a fabricated decision or falls back to keyword routing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Sequence

from companion.prompts.router import build_router_messages
from companion.providers.deepseek_client import (
    DeepSeekConfig,
    DeepSeekError,
    DeepSeekTimeoutError,
)
from companion.routing.errors import (
    RouterInvalidOutput,
    RouterTimeout,
    RouterUnavailable,
)
from companion.routing.models import RawRouteDecision, RoutingContext

__all__ = ["DeepSeekRouterProvider"]

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*|\s*```$")


class DeepSeekRouterProvider:
    """Adapter around ``DeepSeekClient.complete`` for router requests."""

    provider_name = "deepseek_router"

    def __init__(
        self,
        client,
        *,
        config: Optional[DeepSeekConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.config = config or getattr(client, "config", None)
        self.logger = logger or logging.getLogger("companion.routing.provider")

    @property
    def model(self) -> str:
        if self.config is None:
            return ""
        return str(getattr(self.config, "router_model", "") or self.config.model)

    async def route(self, context: RoutingContext) -> RawRouteDecision:
        return await self._complete(context, previous=None, tool_results=None)

    async def replan(
        self,
        context: RoutingContext,
        previous: Any,
        tool_results: Sequence[Any],
    ) -> RawRouteDecision:
        return await self._complete(
            context, previous=previous, tool_results=list(tool_results)
        )

    async def _complete(
        self,
        context: RoutingContext,
        *,
        previous: Optional[Any],
        tool_results: Optional[list[Any]],
    ) -> RawRouteDecision:
        messages = build_router_messages(
            context, previous=previous, tool_results=tool_results
        )
        try:
            response = await self.client.complete(
                messages,
                request_type="router",
                event_id=context.current_event_id,
                json_mode=True,
                model=self.model or None,
            )
        except DeepSeekTimeoutError as exc:
            raise RouterTimeout(
                "router request timed out (retryable; no decision was made)"
            ) from exc
        except DeepSeekError as exc:
            raise RouterUnavailable(
                f"router request failed: {exc.error_type}"
            ) from exc
        except Exception as exc:  # transport/unknown failure, never silence
            raise RouterUnavailable(
                f"router request failed: {type(exc).__name__}"
            ) from exc

        content = getattr(response, "content", "") or ""
        data = _extract_json_object(content)
        try:
            decision = RawRouteDecision.model_validate(data)
        except Exception as exc:  # schema error is invalid output, not WAIT
            raise RouterInvalidOutput(
                f"router payload failed schema parsing: {type(exc).__name__}"
            ) from exc

        self.logger.info(
            "router_decision event_id=%s provider=%s model=%s action=%s"
            " phase=%s",
            context.current_event_id,
            self.provider_name,
            self.model or "unknown",
            decision.action.value,
            "FOLLOWUP" if previous is not None else "INITIAL",
        )
        return decision


def _extract_json_object(content: str) -> dict:
    text = (content or "").strip()
    if not text:
        raise RouterInvalidOutput("router returned empty content")
    cleaned = _FENCE_RE.sub("", text).strip()
    for candidate in (cleaned, _outermost_object(cleaned)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    raise RouterInvalidOutput("router returned non-JSON content")


def _outermost_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]
