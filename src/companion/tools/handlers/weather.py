"""Weather as a real, opt-in external tool.

The handler enforces the same gates as ``web.search``: the operator setting is
authoritative, the speaker must hold ``allow_external_ai``, and the live consent
is re-read immediately before the network call. There is no mock fallback: with
search disabled the result is ``DISABLED``, never a fabricated forecast.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from companion.providers.weather import WeatherError
from companion.routing.models import ToolStatus
from companion.tools.models import ToolExecutionContext, ToolHandlerResult


class WeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str = Field(min_length=1, max_length=150)


class WeatherForecastHandler:
    def __init__(self, provider) -> None:
        self.provider = provider

    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolHandlerResult:
        args = WeatherArguments.model_validate(arguments)
        enabled = await _web_search_enabled(context)
        if not enabled or not context.allow_external_ai or self.provider is None:
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
        try:
            data = await self.provider.forecast(args.location)
        except WeatherError as exc:
            if exc.code == "weather_location_not_found":
                # A real lookup that matched nothing: not a success, not a
                # generic error the Speaker could misread as "weather is off".
                return ToolHandlerResult(
                    status=ToolStatus.EMPTY, error_code="weather_location_not_found"
                )
            return ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code=f"weather_failed:{exc.code}",
            )
        except Exception as exc:
            return ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code=f"weather_failed:{type(exc).__name__}",
            )
        if not isinstance(data, dict) or data.get("temperature") is None:
            return ToolHandlerResult(
                status=ToolStatus.ERROR, error_code="weather_failed:empty_result"
            )
        return ToolHandlerResult(
            status=ToolStatus.SUCCESS, data=data, untrusted=True
        )


async def _web_search_enabled(context: ToolExecutionContext) -> bool:
    settings = getattr(getattr(context, "repos", None), "settings", None)
    if settings is None:
        return False
    try:
        return bool(await settings.get_bool("web_search.enabled", False))
    except Exception:
        return False
