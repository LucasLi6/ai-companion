"""Web search as a *real* opt-in external tool.

There is no mock fallback here: when no enabled real provider is wired in the
tool reports ``DISABLED`` and returns no results. Mock/simulated search output
is never presented as a real result. Network results are ``untrusted`` data:
the speaker is instructed to treat them as information, never instructions.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from companion.providers.websearch import (
    MockWebSearchProvider,
    WebSearchConfigError,
    WebSearchError,
)
from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult

# Bounded result shape. The tool's ``max_result_chars`` (6000) is enforced by
# the executor as a hard error, so the handler budgets *below* it and drops
# whole results instead of handing over an object that would be rejected.
MAX_TITLE_CHARS = 150
MAX_SNIPPET_CHARS = 400
MAX_URL_CHARS = 2000
RESULT_BUDGET_CHARS = 5200


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
        data, truncated, dropped_invalid = _fit_results(results, args.limit)
        meta = {
            "provider": str(getattr(provider, "name", "") or ""),
            "requested_limit": args.limit,
            "returned_count": len(data),
            "truncated_count": truncated,
        }
        if dropped_invalid:
            meta["invalid_count"] = dropped_invalid
        if not data:
            return ToolHandlerResult(
                status=ToolStatus.EMPTY, data=[], untrusted=True, meta=meta
            )
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS,
            data=data,
            source_refs=[item["url"] for item in data if item["url"]],
            untrusted=True,
            meta=meta,
        )


def _fit_results(results, limit: int) -> tuple[list[dict], int, int]:
    """Convert provider results into the bounded payload the Speaker receives.

    Titles and snippets are clipped, but a URL is never shortened: a truncated
    link is worse than no link, so an unusable URL drops the whole result.
    Results are added in order until the byte budget is reached; the rest are
    counted as truncated.
    """
    data: list[dict] = []
    invalid = 0
    for item in results[:limit]:
        title = _clip(getattr(item, "title", ""), MAX_TITLE_CHARS)
        url = str(getattr(item, "url", "") or "").strip()
        snippet = _clip(getattr(item, "snippet", ""), MAX_SNIPPET_CHARS)
        if not title or len(url) > MAX_URL_CHARS or not _is_http_url(url):
            invalid += 1
            continue
        candidate = {"title": title, "url": url, "snippet": snippet}
        if len(json.dumps(data + [candidate], ensure_ascii=False)) > (
            RESULT_BUDGET_CHARS
        ):
            break
        data.append(candidate)
    considered = min(len(results), limit)
    return data, max(0, considered - len(data) - invalid), invalid


def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _is_http_url(url: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


async def _web_search_enabled(context: ToolExecutionContext) -> bool:
    settings = getattr(getattr(context, "repos", None), "settings", None)
    if settings is None:
        return False
    try:
        return bool(await settings.get_bool("web_search.enabled", False))
    except Exception:
        # A settings read failure must not silently enable the network.
        return False
