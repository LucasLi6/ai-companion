"""Live weather via Open-Meteo (no API key required).

Two requests are made: a geocoding lookup turns a place name into coordinates
and a timezone, and the forecast endpoint returns the current conditions plus
today's high/low and a compact tomorrow block (so a natural follow-up such as
"那明天呢" is answered from data, not from an invented extension). Everything is
normalised into a small, explicit structure so the Speaker never receives a raw
provider payload.

A missing place, an invalid response and a network failure are all distinct,
truthful errors. An empty result set is never presented as successful weather.
"""

from __future__ import annotations

import logging
from typing import Optional

GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
SOURCE_NAME = "Open-Meteo"
SOURCE_URL = "https://open-meteo.com/"

CURRENT_FIELDS = (
    "temperature_2m,"
    "apparent_temperature,"
    "relative_humidity_2m,"
    "weather_code,"
    "wind_speed_10m"
)
DAILY_FIELDS = (
    "weather_code,"
    "temperature_2m_max,"
    "temperature_2m_min,"
    "precipitation_probability_max"
)
# V0.9.6 continuity: "那明天呢" is a natural follow-up to a weather answer.
# Two days are requested so tomorrow's figures are real data instead of an
# invented extension of today's. Only tomorrow is exposed; the rest stays out.
FORECAST_DAYS = 2
TOMORROW_INDEX = 1

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "凇雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


# Open-Meteo's geocoder indexes Chinese names sparsely: ``休斯顿`` matches two
# hamlets in Kentucky, ``纽约`` matches nothing, and ``东京`` matches villages in
# Jiangsu. A small, explicit alias table turns the common Chinese names for
# well-known cities into the Latin name the index actually knows. It is input
# normalisation, not translation and not a second model; anything not listed
# is looked up as typed.
_CITY_ALIASES: dict[str, str] = {
    "休斯顿": "Houston",
    "休斯敦": "Houston",
    "纽约": "New York",
    "洛杉矶": "Los Angeles",
    "旧金山": "San Francisco",
    "三藩市": "San Francisco",
    "西雅图": "Seattle",
    "芝加哥": "Chicago",
    "波士顿": "Boston",
    "华盛顿": "Washington",
    "迈阿密": "Miami",
    "达拉斯": "Dallas",
    "奥斯汀": "Austin",
    "休斯敦市": "Houston",
    "拉斯维加斯": "Las Vegas",
    "费城": "Philadelphia",
    "亚特兰大": "Atlanta",
    "多伦多": "Toronto",
    "温哥华": "Vancouver",
    "伦敦": "London",
    "巴黎": "Paris",
    "柏林": "Berlin",
    "罗马": "Rome",
    "马德里": "Madrid",
    "莫斯科": "Moscow",
    "东京": "Tokyo",
    "大阪": "Osaka",
    "首尔": "Seoul",
    "新加坡": "Singapore",
    "悉尼": "Sydney",
    "墨尔本": "Melbourne",
    "迪拜": "Dubai",
}

# A Chinese query that only matches a tiny hamlet is usually a prefix mismatch
# in the geocoder, not the place the speaker meant.
WEAK_CJK_POPULATION = 1000

# Same-name places below this size are noise in a "did you mean" list.
MEANINGFUL_ALTERNATIVE_POPULATION = 50000


class WeatherError(RuntimeError):
    """A truthful weather failure; never a fabricated reading."""

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message or code)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class OpenMeteoProvider:
    name = "open-meteo"

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        transport=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.timeout = timeout
        self.transport = transport
        self.logger = logger or logging.getLogger("companion.weather")

    async def forecast(self, location: str) -> dict:
        place = str(location or "").strip()
        if not place:
            raise WeatherError("empty_query")
        resolved = await self._geocode(place)
        return await self._forecast(resolved)

    # -- requests ----------------------------------------------------------

    async def _geocode(self, place: str) -> dict:
        query = place
        key = place.strip()
        alias = _CITY_ALIASES.get(key)
        if alias:
            query = alias
        payload = await self._get(
            GEOCODING_ENDPOINT,
            {
                "name": query,
                "count": 8,
                "language": "zh",
                "format": "json",
            },
            label="geocoding",
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            raise WeatherError(
                "weather_location_not_found",
                f"no location matched {place!r}",
            )
        candidates = [item for item in results if isinstance(item, dict)]
        if not candidates:
            raise WeatherError("weather_invalid_response", "geocoding payload")
        # Prefer the largest place: "Houston" must mean Texas, not a village
        # in Missouri with a similar name.
        candidates.sort(key=_population, reverse=True)
        first = candidates[0]
        if (
            has_cjk(key)
            and not alias
            and _population(first) < WEAK_CJK_POPULATION
        ):
            # Only tiny prefix matches exist, e.g. "休斯顿威尔". Reporting
            # that as the user's city would be a confident wrong answer.
            raise WeatherError(
                "weather_location_not_found",
                f"no reliable location matched {place!r}",
            )
        try:
            latitude = float(first["latitude"])
            longitude = float(first["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherError(
                "weather_invalid_response", "geocoding coordinates"
            ) from exc
        matches = []
        for item in candidates[1:4]:
            # Only other substantial places are worth mentioning; a 400-person
            # hamlet with a similar name is noise, not a candidate.
            if _population(item) < MEANINGFUL_ALTERNATIVE_POPULATION:
                continue
            label = self._place_label(item)
            if label and label not in matches:
                matches.append(label)
        return {
            "location": self._place_label(first),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": str(first.get("timezone") or "auto"),
            "alternatives": matches,
        }

    async def _forecast(self, resolved: dict) -> dict:
        payload = await self._get(
            FORECAST_ENDPOINT,
            {
                "latitude": resolved["latitude"],
                "longitude": resolved["longitude"],
                "current": CURRENT_FIELDS,
                "daily": DAILY_FIELDS,
                "timezone": resolved["timezone"],
                "forecast_days": FORECAST_DAYS,
            },
            label="forecast",
        )
        if not isinstance(payload, dict):
            raise WeatherError("weather_invalid_response", "forecast payload")
        current = payload.get("current")
        if not isinstance(current, dict):
            raise WeatherError(
                "weather_invalid_response", "forecast payload has no current block"
            )
        daily = payload.get("daily") if isinstance(payload.get("daily"), dict) else {}
        units = (
            payload.get("current_units")
            if isinstance(payload.get("current_units"), dict)
            else {}
        )

        code = _as_int(current.get("weather_code"))
        result = {
            "location": resolved["location"],
            "timezone": str(payload.get("timezone") or resolved["timezone"]),
            "observed_at": str(current.get("time") or ""),
            "temperature": _as_float(current.get("temperature_2m")),
            "apparent_temperature": _as_float(
                current.get("apparent_temperature")
            ),
            "temperature_unit": str(units.get("temperature_2m") or "°C"),
            "weather_code": code,
            "weather": WEATHER_CODES.get(code, "未知") if code is not None else "未知",
            "humidity": _as_float(current.get("relative_humidity_2m")),
            "wind_speed": _as_float(current.get("wind_speed_10m")),
            "wind_speed_unit": str(units.get("wind_speed_10m") or "km/h"),
            "daily_max": _first_float(daily.get("temperature_2m_max")),
            "daily_min": _first_float(daily.get("temperature_2m_min")),
            "precipitation_probability": _first_float(
                daily.get("precipitation_probability_max")
            ),
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
        }
        tomorrow = _daily_point(daily, TOMORROW_INDEX)
        if tomorrow:
            result["tomorrow"] = tomorrow
        if result["temperature"] is None and result["daily_max"] is None:
            raise WeatherError(
                "weather_invalid_response", "no usable temperature values"
            )
        if resolved.get("alternatives"):
            result["alternatives"] = resolved["alternatives"]
        return result

    async def _get(self, url: str, params: dict, *, label: str) -> dict:
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise WeatherError(
                "weather_timeout"
                if isinstance(exc, httpx.TimeoutException)
                else "weather_network_error",
                retryable=True,
            ) from exc
        status = response.status_code
        if status >= 500:
            raise WeatherError(
                "weather_server_error", status_code=status, retryable=True
            )
        if status != 200:
            raise WeatherError("weather_http_error", status_code=status)
        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherError(
                "weather_invalid_response", f"{label} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise WeatherError("weather_invalid_response", label)
        if payload.get("error"):
            raise WeatherError(
                "weather_api_error", str(payload.get("reason") or label)
            )
        return payload

    @staticmethod
    def _place_label(item: dict) -> str:
        parts = [
            str(item.get(key) or "").strip()
            for key in ("name", "admin1", "country")
        ]
        return ", ".join(part for part in parts if part)


def _population(item: dict) -> int:
    try:
        return int(item.get("population") or 0)
    except (TypeError, ValueError):
        return 0


def has_cjk(value: str) -> bool:
    """Whether the query contains CJK ideographs (the geocoder's weak spot)."""
    for char in value or "":
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _as_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_float(value) -> Optional[float]:
    if isinstance(value, (list, tuple)) and value:
        return _as_float(value[0])
    return _as_float(value)


def _at(values, index: int):
    if isinstance(values, (list, tuple)) and len(values) > index:
        return values[index]
    return None


def _daily_point(daily: dict, index: int) -> Optional[dict]:
    """One day of the daily arrays, with absent values omitted."""
    code = _as_int(_at(daily.get("weather_code"), index))
    point: dict = {}
    if code is not None:
        point["weather_code"] = code
        point["weather"] = WEATHER_CODES.get(code, "未知")
    for key, source in (
        ("daily_max", "temperature_2m_max"),
        ("daily_min", "temperature_2m_min"),
        ("precipitation_probability", "precipitation_probability_max"),
    ):
        value = _as_float(_at(daily.get(source), index))
        if value is not None:
            point[key] = value
    return point or None


__all__ = [
    "GEOCODING_ENDPOINT",
    "FORECAST_ENDPOINT",
    "SOURCE_NAME",
    "WEATHER_CODES",
    "OpenMeteoProvider",
    "WeatherError",
]
