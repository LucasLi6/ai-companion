"""Read the body of a page the user found *through this turn's search*.

First version: the only URLs allowed are the ones a successful ``web.search``
returned in the same turn. The executor records them in
``context.extras["web_search_urls"]`` from trusted, executed results — never
from a list the Router supplied. That keeps the tool on the same footing as
search: the user sees where the information comes from, and the model cannot
use it to reach arbitrary hosts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.providers.webfetch import WebFetchError, normalize_web_url
from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult

# A turn is short; two page reads are enough for "search then read the best
# hit", and the cap stops a loop from turning one turn into a crawl.
MAX_PAGES_PER_TURN = 2


class WebFetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=10, max_length=2000)


class WebFetchHandler:
    def __init__(self, provider) -> None:
        self.provider = provider

    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = WebFetchArguments.model_validate(arguments)
        if (
            self.provider is None
            or not context.allow_external_ai
            or not await _web_search_enabled(context)
        ):
            return ToolHandlerResult(
                status=ToolStatus.DISABLED, error_code="web_search_disabled"
            )
        if context.person_id and context.registry is not None:
            try:
                still_allowed = await context.registry.is_external_ai_authorized(
                    context.person_id
                )
            except Exception:
                still_allowed = False
            if not still_allowed:
                return ToolHandlerResult(
                    status=ToolStatus.DENIED,
                    error_code="external_ai_permission_revoked",
                )

        target = normalize_web_url(args.url)
        allowed = context.extras.get("web_search_urls") or set()
        if not target or target not in allowed:
            # Only pages this turn actually found are readable. A URL the
            # Router invented is refused, not fetched.
            return ToolHandlerResult(
                status=ToolStatus.DENIED,
                error_code="web_fetch_url_not_from_search",
            )

        used = int(context.extras.get("web_fetch_count") or 0)
        if used >= MAX_PAGES_PER_TURN:
            return ToolHandlerResult(
                status=ToolStatus.DENIED, error_code="web_fetch_turn_limit"
            )
        context.extras["web_fetch_count"] = used + 1

        try:
            page = await self.provider.fetch(target)
        except WebFetchError as exc:
            if exc.code in (
                "web_fetch_scheme_not_allowed",
                "web_fetch_credentials_not_allowed",
                "web_fetch_private_address",
            ):
                return ToolHandlerResult(
                    status=ToolStatus.DENIED, error_code=exc.code
                )
            return ToolHandlerResult(
                status=ToolStatus.ERROR, error_code=f"web_fetch_failed:{exc.code}"
            )
        except Exception as exc:
            return ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code=f"web_fetch_failed:{type(exc).__name__}",
            )
        if not isinstance(page, dict) or not str(page.get("text") or "").strip():
            return ToolHandlerResult(
                status=ToolStatus.EMPTY, error_code="web_fetch_empty_content"
            )
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=page,
            source_refs=[str(page.get("url") or target)],
            untrusted=True,
        )


async def _web_search_enabled(context: ToolExecutionContext) -> bool:
    settings = getattr(getattr(context, "repos", None), "settings", None)
    if settings is None:
        return False
    try:
        return bool(await settings.get_bool("web_search.enabled", False))
    except Exception:
        return False


__all__ = ["MAX_PAGES_PER_TURN", "WebFetchArguments", "WebFetchHandler"]
