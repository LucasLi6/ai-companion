"""Optional, local-only city-level geolocation.

Design constraints (V0.6):

- Disabled by default. Nothing is looked up unless both a local GeoLite2 City
  ``.mmdb`` file and an explicit enable flag are present.
- The database is read locally with ``maxminddb``; no network call is made.
- Only city/region/country, coarse coordinates and the zone name are kept.
  The raw IP is never persisted, never logged and never sent to a model.
- A result always means "network egress location", not the speaker's physical
  location. The prompt labels it that way.
- City fields are only reported when the database actually provides them;
  nothing is fabricated. A country-only result stays country-only.
- GeoLite2 credentials/download keys are never read, stored or committed. The
  operator places an already-licensed database on disk and points
  ``GEOIP_DB_PATH`` at it.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("companion.runtime.geoip")

SOURCE_IP_GEO = "ip_geo"


@dataclass(frozen=True)
class GeoLocation:
    """A coarse, local lookup result. Never a precise position."""

    city: Optional[str] = None
    subdivision: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    accuracy_radius_km: Optional[int] = None
    source: str = SOURCE_IP_GEO

    def as_dict(self) -> dict:
        data = {
            "city": self.city,
            "subdivision": self.subdivision,
            "country": self.country,
            "country_code": self.country_code,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "accuracy_radius_km": self.accuracy_radius_km,
            "source": self.source,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "GeoLocation":
        if not isinstance(data, dict):
            return cls()
        return cls(
            city=_clean(data.get("city"), 120),
            subdivision=_clean(data.get("subdivision"), 120),
            country=_clean(data.get("country"), 120),
            country_code=_clean(data.get("country_code"), 8),
            latitude=_number(data.get("latitude"), -90.0, 90.0),
            longitude=_number(data.get("longitude"), -180.0, 180.0),
            timezone=_clean(data.get("timezone"), 64),
            accuracy_radius_km=_int_or_none(data.get("accuracy_radius_km")),
            source=_clean(data.get("source"), 32) or SOURCE_IP_GEO,
        )


def _clean(value, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text[:limit] if text else None


def _number(value, low: float, high: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not low <= number <= high:
        return None
    return round(number, 4)


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class GeoLocationService:
    """Read a local MaxMind database, if enabled and available."""

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        enabled: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = db_path or os.environ.get("GEOIP_DB_PATH")
        self.enabled = bool(enabled)
        self.logger = logger or logging.getLogger("companion.runtime.geoip")
        self._reader = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        return bool(
            self.enabled
            and self.db_path
            and not self._load_failed
            and Path(self.db_path).is_file()
        )

    def lookup(
        self, ip: Optional[str], *, enabled: Optional[bool] = None
    ) -> Optional[GeoLocation]:
        """Resolve a public IP to a coarse location, or ``None``.

        ``enabled`` lets the caller apply the persisted ``privacy.geoip.enabled``
        setting; the service itself stays default-off.
        """
        active = self.enabled if enabled is None else bool(enabled)
        if (
            not active
            or not self.db_path
            or self._load_failed
            or not Path(self.db_path).is_file()
            or not ip
        ):
            return None
        try:
            address = ipaddress.ip_address(str(ip).strip())
        except ValueError:
            return None
        # Defence in depth: never query a private/reserved address even if the
        # caller passed one directly.
        if not address.is_global:
            return None

        reader = self._open()
        if reader is None:
            return None
        try:
            record = reader.get(str(address))
        except Exception as exc:  # a corrupt DB entry is not a crash
            self.logger.warning("geoip_lookup_failed error_type=%s", type(exc).__name__)
            return None
        if not isinstance(record, dict):
            return None
        return self._to_location(record)

    def _open(self):
        if self._reader is not None:
            return self._reader
        if self._load_failed:
            return None
        try:
            import maxminddb  # lazy: only needed when GeoIP is enabled
        except ImportError:
            self._load_failed = True
            self.logger.warning("geoip_unavailable reason=maxminddb_not_installed")
            return None
        try:
            self._reader = maxminddb.open_database(self.db_path)
        except Exception as exc:
            self._load_failed = True
            self.logger.warning(
                "geoip_unavailable reason=open_failed error_type=%s",
                type(exc).__name__,
            )
            return None
        return self._reader

    def _to_location(self, record: dict) -> Optional[GeoLocation]:
        city = record.get("city") or {}
        subdivision = (record.get("subdivisions") or [{}])[0] if record.get(
            "subdivisions"
        ) else {}
        country = record.get("country") or {}
        location = record.get("location") or {}
        city_name = _clean(city.get("names", {}).get("en") or city.get("name"), 120)
        region = _clean(
            subdivision.get("names", {}).get("en") or subdivision.get("name"), 120
        )
        country_name = _clean(
            country.get("names", {}).get("en") or country.get("name"), 120
        )
        country_code = _clean(country.get("iso_code"), 8)
        result = GeoLocation(
            city=city_name,
            subdivision=region,
            country=country_name,
            country_code=country_code,
            latitude=_number(location.get("latitude"), -90.0, 90.0),
            longitude=_number(location.get("longitude"), -180.0, 180.0),
            timezone=_clean(location.get("time_zone"), 64),
            accuracy_radius_km=_int_or_none(location.get("accuracy_radius")),
        )
        return result if result.as_dict() else None


__all__ = ["GeoLocation", "GeoLocationService", "SOURCE_IP_GEO"]
