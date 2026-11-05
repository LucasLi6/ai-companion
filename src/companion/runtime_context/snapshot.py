"""One immutable :class:`RuntimeSnapshot` per input event.

The snapshot records *when* the event happened and arrived, in which display
time zone, the three temporal gaps, the companion's current emotion (filled in
later in the same turn), and — only when explicitly enabled and consented —
a coarse network-egress location.

Persistence rules:

- The snapshot is stored once, keyed by ``event_id``. A retry reuses the
  stored values, so ``received_at_utc`` and every gap never refresh on retry.
- A turn without persistence consent keeps the snapshot in memory only; there
  is nothing to reuse by definition.
- The raw client IP is never part of the snapshot. Only the resolved
  city-level fields survive, and only when GeoIP is enabled and the speaker
  allows external AI processing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal, Optional

from companion.domain.models import DomainModel, utcnow

from .connection import ConnectionInfo, resolve_client_ip
from .geolocation import GeoLocation, GeoLocationService
from .temporal import (
    DEFAULT_REENTRY_GAP_SECONDS,
    TemporalSignals,
    build_temporal_signals,
)
from .timezone import (
    ResolvedTimeZone,
    TimeZoneService,
    format_local_time,
    sanitise_zone_for_prompt,
)


class GeoLocationModel(DomainModel):
    """Serialisable form of :class:`GeoLocation` (no datetime fields)."""

    city: Optional[str] = None
    subdivision: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    accuracy_radius_km: Optional[int] = None
    source: str = "ip_geo"


class RuntimeSnapshot(DomainModel):
    """Trusted state for one event. Prompt data, never an instruction."""

    snapshot_id: str
    event_id: str
    conversation_id: str
    speaker_person_id: Optional[str] = None
    speaker_label: Optional[str] = None
    received_at_utc: datetime
    occurred_at_utc: datetime
    timezone: str
    timezone_source: Literal[
        "user_config", "trusted_device", "ip_geo", "system_default", "utc"
    ]
    utc_offset_seconds: int = 0
    local_time: str = ""
    last_heard_at_utc: Optional[datetime] = None
    last_interaction_at_utc: Optional[datetime] = None
    last_meaningful_context_at_utc: Optional[datetime] = None
    last_heard_gap_seconds: Optional[int] = None
    last_interaction_gap_seconds: Optional[int] = None
    last_meaningful_context_gap_seconds: Optional[int] = None
    reentry: bool = False
    reentry_state: Literal["NEW", "ACTIVE", "CONTINUING", "RETURNING"] = "NEW"
    ava_emotion: Optional[str] = None
    # V0.6.2: the input modality this turn actually arrived through, and
    # whether trusted acoustic features exist. Text input has no voice
    # features, so the reply must not claim to have heard a tone.
    source: Literal["text", "voice"] = "text"
    voice_features: str = "not_available"
    location: Optional[GeoLocationModel] = None

    def as_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


class RuntimeSnapshotBuilder:
    """Assemble the snapshot from local tables and trusted configuration."""

    def __init__(
        self,
        repos,
        *,
        timezone_service: Optional[TimeZoneService] = None,
        geo_service: Optional[GeoLocationService] = None,
        logger: Optional[logging.Logger] = None,
        clock=utcnow,
    ) -> None:
        self.repos = repos
        self.timezone_service = timezone_service or TimeZoneService()
        self.geo_service = geo_service
        self.logger = logger or logging.getLogger("companion.runtime.snapshot")
        self.clock = clock

    async def build(
        self,
        event,
        *,
        connection: Optional[ConnectionInfo] = None,
        received_at: Optional[datetime] = None,
        allow_external_ai: bool = False,
        persist: bool = True,
        emotion_state: Optional[str] = None,
    ) -> RuntimeSnapshot:
        stored = await self.repos.runtime_snapshot.get(event.event_id)
        if stored is not None:
            # A retry sees exactly the original timing. A later privacy change
            # may still strip a location that is no longer authorised.
            if not (allow_external_ai and await self._geo_enabled()):
                stored = stored.model_copy(update={"location": None})
            return stored.model_copy(update={"ava_emotion": emotion_state})

        received = received_at or self.clock()
        location, geo_zone = await self._location(connection, allow_external_ai)

        # A speaker may have their own display zone; it falls back to the
        # application-wide setting, then to the trusted device / GeoIP / host.
        user_timezone = None
        if event.speaker_person_id is not None:
            user_timezone = await self.repos.settings.get(
                f"runtime.timezone.person.{event.speaker_person_id}"
            )
        if not user_timezone:
            user_timezone = await self.repos.settings.get("runtime.timezone")
        zone: ResolvedTimeZone = self.timezone_service.resolve(
            user_timezone=user_timezone,
            device_timezone=(
                connection.device_timezone if connection is not None else None
            ),
            device_trusted=bool(connection is not None and connection.device_trusted),
            geo_timezone=geo_zone,
            at=received,
        )

        temporal = await self._temporal(event)
        modality = getattr(
            getattr(event, "source", None), "modality", "text"
        )
        snapshot = RuntimeSnapshot(
            snapshot_id=f"rts_{event.event_id}",
            source=modality if modality in ("text", "voice") else "text",
            voice_features="not_available",
            event_id=event.event_id,
            conversation_id=event.conversation_id,
            speaker_person_id=event.speaker_person_id,
            speaker_label=event.speaker_label,
            received_at_utc=received,
            occurred_at_utc=event.occurred_at,
            timezone=sanitise_zone_for_prompt(zone.name) or "UTC",
            timezone_source=zone.source,
            utc_offset_seconds=zone.offset_seconds,
            local_time=format_local_time(received, zone),
            last_heard_at_utc=temporal.last_heard_at,
            last_interaction_at_utc=temporal.last_interaction_at,
            last_meaningful_context_at_utc=temporal.last_meaningful_context_at,
            last_heard_gap_seconds=temporal.last_heard_gap_seconds,
            last_interaction_gap_seconds=temporal.last_interaction_gap_seconds,
            last_meaningful_context_gap_seconds=(
                temporal.last_meaningful_context_gap_seconds
            ),
            reentry=temporal.reentry,
            reentry_state=temporal.reentry_state,
            ava_emotion=emotion_state,
            location=location,
        )
        if persist:
            await self.repos.runtime_snapshot.save_if_absent(snapshot)
        return snapshot

    # -- internals ------------------------------------------------------

    async def _location(
        self, connection: Optional[ConnectionInfo], allow_external_ai: bool
    ) -> tuple[Optional[GeoLocationModel], Optional[str]]:
        if (
            connection is None
            or not allow_external_ai
            or self.geo_service is None
            or not await self._geo_enabled()
        ):
            return None, None
        trusted_proxies = await self._trusted_proxies()
        ip = resolve_client_ip(connection, trusted_proxies=trusted_proxies)
        resolved: Optional[GeoLocation] = self.geo_service.lookup(ip, enabled=True)
        if resolved is None:
            return None, None
        model = GeoLocationModel(**resolved.as_dict())
        return model, resolved.timezone

    async def _geo_enabled(self) -> bool:
        """GeoIP is opt-in either by process configuration or by the setting."""
        if self.geo_service is None:
            return False
        if getattr(self.geo_service, "enabled", False):
            return True
        try:
            return await self.repos.settings.get_bool("privacy.geoip.enabled", False)
        except Exception:
            return False

    async def _trusted_proxies(self) -> list[str]:
        raw = await self.repos.settings.get("privacy.geoip.trusted_proxies")
        if not raw:
            return []
        return [item.strip() for item in str(raw).split(",") if item.strip()]

    async def _temporal(self, event) -> TemporalSignals:
        if event.speaker_person_id is None:
            return build_temporal_signals(event.occurred_at)
        last_heard = await self.repos.utterance.last_before_speaker(
            event.speaker_person_id,
            before=event.occurred_at,
            exclude_event_id=None,
        )
        # ``last_interaction`` means the last time the companion actually
        # replied. A run of WAIT/IGNORE is not an interaction, and mixing them
        # in made the "how long since we talked" fact wrong.
        last_action = await self.repos.analysis.last_action_for_speaker(
            event.speaker_person_id,
            before=event.occurred_at,
            action_type="RESPOND",
        )
        last_meaningful = last_action
        raw_gap = await self.repos.settings.get("runtime.reentry_gap_seconds")
        try:
            reentry_gap = int(raw_gap) if raw_gap else DEFAULT_REENTRY_GAP_SECONDS
        except (TypeError, ValueError):
            reentry_gap = DEFAULT_REENTRY_GAP_SECONDS
        return build_temporal_signals(
            event.occurred_at,
            last_heard_at=last_heard.occurred_at if last_heard else None,
            last_interaction_at=last_action[1] if last_action else None,
            last_meaningful_context_at=last_meaningful[1] if last_meaningful else None,
            reentry_gap_seconds=reentry_gap,
        )
