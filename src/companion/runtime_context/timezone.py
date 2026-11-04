"""IANA time-zone resolution from trusted configuration.

Rules (V0.6):

- Times are stored and compared in UTC. A display time is always derived with
  ``zoneinfo.ZoneInfo`` from an IANA name, never with a hard-coded ``UTC-6``
  style offset, so daylight-saving transitions stay correct.
- Priority is ``user_config`` > ``trusted_device`` > ``ip_geo`` >
  ``system_default`` > ``utc``. The first *valid* IANA name wins; an invalid
  higher-priority value is skipped rather than trusted.
- A device-supplied name is only used when the caller marks the device as
  trusted. An IP-derived name is only used when GeoIP is enabled and the
  lookup produced a valid zone.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMEZONE_SOURCES = (
    "user_config",
    "trusted_device",
    "ip_geo",
    "system_default",
    "utc",
)
DEFAULT_TIMEZONE = "UTC"

logger = logging.getLogger("companion.runtime.timezone")


@dataclass(frozen=True)
class ResolvedTimeZone:
    """A validated IANA zone plus the trusted source that supplied it."""

    name: str
    source: str
    offset_seconds: int = 0

    def local(self, at: datetime) -> datetime:
        return at.astimezone(ZoneInfo(self.name))


def valid_zone(name: Optional[str]) -> Optional[str]:
    """Return ``name`` only when it is a real IANA zone, else ``None``."""
    if not name or not isinstance(name, str):
        return None
    candidate = name.strip()
    if not candidate or candidate.upper() in {"UTC", "UNKNOWN", "NONE"}:
        return candidate.upper() if candidate.upper() == "UTC" else None
    try:
        ZoneInfo(candidate)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    return candidate


def _system_zone() -> Optional[str]:
    """Best-effort IANA zone of the host, without inventing a fixed offset."""
    env = valid_zone(os.environ.get("TZ"))
    if env:
        return env
    try:
        target = Path("/etc/localtime").resolve()
    except OSError:
        return None
    marker = "zoneinfo/"
    text = str(target)
    if marker in text:
        return valid_zone(text.split(marker, 1)[1])
    return None


class TimeZoneService:
    """Resolve the display zone for one event."""

    def __init__(self, system_zone: Optional[str] = None) -> None:
        # ``system_zone`` is injectable for deterministic tests; the default
        # reads the host configuration on first use.
        self._system_zone = system_zone

    def resolve(
        self,
        *,
        user_timezone: Optional[str] = None,
        device_timezone: Optional[str] = None,
        device_trusted: bool = False,
        geo_timezone: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> ResolvedTimeZone:
        candidates: list[tuple[str, Optional[str]]] = [
            ("user_config", user_timezone),
            ("trusted_device", device_timezone if device_trusted else None),
            ("ip_geo", geo_timezone),
            ("system_default", self._system_zone or _system_zone()),
        ]
        for source, value in candidates:
            name = valid_zone(value)
            if name:
                return ResolvedTimeZone(
                    name=name,
                    source=source,
                    offset_seconds=self._offset(name, at),
                )
        return ResolvedTimeZone(
            name=DEFAULT_TIMEZONE,
            source="utc",
            offset_seconds=0,
        )

    @staticmethod
    def _offset(name: str, at: Optional[datetime]) -> int:
        moment = (at or datetime.now(timezone.utc)).astimezone(ZoneInfo(name))
        offset = moment.utcoffset()
        return int(offset.total_seconds()) if offset is not None else 0


def format_local_time(at: datetime, zone: ResolvedTimeZone) -> str:
    """Render a compact local time with a stable, prompt-friendly format."""
    local = zone.local(at)
    weekdays = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
    return f"{local:%Y-%m-%d %H:%M}（{weekdays[local.weekday()]}）"


def sanitise_zone_for_prompt(name: str) -> str:
    """A zone name is metadata; keep it short and free of prompt markup."""
    text = " ".join(str(name or "").split())
    return text.replace("【", "[").replace("】", "]")[:64]
