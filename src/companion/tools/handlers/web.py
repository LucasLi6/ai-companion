"""Web search as a *real* opt-in external tool.

There is no mock fallback here: when no enabled real provider is wired in the
tool reports ``DISABLED`` and returns no results. Mock/simulated search output
is never presented as a real result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.providers.websearch import MockWebSearchProvider
from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult


class WebSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=5, ge=1, le=5)


class WebSearchHandler:
    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = WebSearchArguments.model_validate(arguments)
        provider = context.web_search
        if not context.allow_external_ai or provider is None or isinstance(
            provider, MockWebSearchProvider
        ):
            return ToolHandlerResult(
                status=ToolStatus.DISABLED, error_code="web_search_disabled"
            )
        try:
            results = await provider.search(args.query, limit=args.limit)
        except RuntimeError:
            return ToolHandlerResult(
                status=ToolStatus.DISABLED, error_code="web_search_disabled"
            )
        data = [
            {
                "title": str(getattr(item, "title", ""))[:300],
                "url": str(getattr(item, "url", ""))[:1000],
                "snippet": str(getattr(item, "snippet", ""))[:1000],
            }
            for item in results[: args.limit]
        ]
        if not data:
            return ToolHandlerResult(
                status=ToolStatus.EMPTY, data=[], untrusted=True
            )
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=data,
            source_refs=[item["url"] for item in data if item["url"]],
            untrusted=True,
        )
