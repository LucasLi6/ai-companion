"""Web search as a *real* opt-in external tool.

There is no mock fallback here: when no enabled real provider is wired in the
tool reports ``DISABLED`` and returns no results. Mock/simulated search output
is never presented as a real result. Network results are ``untrusted`` data:
the speaker is instructed to treat them as information, never instructions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.providers.websearch import (
    MockWebSearchProvider,
    WebSearchConfigError,
    WebSearchError,
)
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
        # The operator's setting is the authority, read at execution time (not
        # at composition time) so a change takes effect on the next call.
        enabled = await _web_search_enabled(context)
        if (
            not enabled
            or not context.allow_external_ai
            or provider is None
            or isinstance(provider, MockWebSearchProvider)
        ):
            # A mock provider is never presented as a real search result.
            return ToolHandlerResult(
                status=ToolStatus.DISABLED, error_code="web_search_disabled"
            )
        # ``allow_external_ai`` is the trusted snapshot from routing time. The
        # live consent is re-read here so a withdrawal between planning and
        # execution blocks the network call.
        if context.person_id and context.registry is not None:
            try:
                still_allowed = await context.registry.is_external_ai_authorized(
                    context.person_id
                )
            except Exception:
                still_allowed = False
            if not still_allowed:
                return ToolHandlerResult(
                    status=ToolStatus.DISABLED,
                    error_code="external_ai_permission_revoked",
                )
        try:
            results = await provider.search(args.query, limit=args.limit)
        except WebSearchConfigError as exc:
            # Missing credentials/config are explicit errors, never a silent
            # fallback to mock and never an empty success.
            return ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code=f"web_search_config:{exc.code}",
            )
        except WebSearchError as exc:
            if exc.code in ("web_search_disabled",):
                return ToolHandlerResult(
                    status=ToolStatus.DISABLED, error_code=exc.code
                )
            return ToolHandlerResult(
                status=ToolStatus.ERROR, error_code=f"web_search_failed:{exc.code}"
            )
        except RuntimeError:
            return ToolHandlerResult(
                status=ToolStatus.DISABLED, error_code="web_search_disabled"
            )
        except Exception as exc:
            # A network/provider failure is an error, never a fake success
            # and never an empty-but-successful lookup.
            return ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code=f"web_search_failed:{type(exc).__name__}",
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


async def _web_search_enabled(context: ToolExecutionContext) -> bool:
    settings = getattr(getattr(context, "repos", None), "settings", None)
    if settings is None:
        return False
    try:
        return bool(await settings.get_bool("web_search.enabled", False))
    except Exception:
        # A settings read failure must not silently enable the network.
        return False
