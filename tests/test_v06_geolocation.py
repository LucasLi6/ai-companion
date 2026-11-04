"""V0.6 optional GeoIP: local-only, default-off, coarse and unpersisted.

No test downloads a database or registers for one. A tiny fake reader stands in
for the locally licensed MaxMind file so the parsing and privacy rules can be
exercised deterministically.
"""

from __future__ import annotations

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.runtime_context.connection import (
    ConnectionInfo,
    resolve_client_ip,
)
from companion.runtime_context.geolocation import GeoLocationService


class _Reader:
    def __init__(self, record):
        self.record = record

    def get(self, ip):
        return self.record


def _service(tmp_path, *, enabled=True):
    path = tmp_path / "GeoLite2-City.mmdb"
    if not path.exists():
        path.write_bytes(b"placeholder")
    return GeoLocationService(db_path=str(path), enabled=enabled)


# -- IP provenance -----------------------------------------------------------


def test_public_peer_is_used_without_forwarded_headers():
    assert resolve_client_ip(ConnectionInfo(remote_ip="8.8.8.8")) == "8.8.8.8"
    # A header from an untrusted peer is ignored entirely.
    assert (
        resolve_client_ip(
            ConnectionInfo(remote_ip="8.8.8.8", forwarded_for="1.1.1.1")
        )
        == "8.8.8.8"
    )


def test_private_and_loopback_peers_are_never_located():
    for address in ("127.0.0.1", "10.1.2.3", "192.168.1.9", "::1", "fe80::1"):
        assert resolve_client_ip(ConnectionInfo(remote_ip=address)) is None


def test_forwarded_for_is_honoured_only_from_a_trusted_proxy():
    connection = ConnectionInfo(
        remote_ip="127.0.0.1", forwarded_for="203.0.113.7, 8.8.8.8"
    )
    assert resolve_client_ip(connection) is None
    assert (
        resolve_client_ip(connection, trusted_proxies=["127.0.0.0/8"])
        == "8.8.8.8"
    )


def test_private_hop_in_the_chain_yields_no_location():
    connection = ConnectionInfo(
        remote_ip="127.0.0.1", forwarded_for="8.8.8.8, 10.0.0.5"
    )
    assert resolve_client_ip(connection, trusted_proxies=["127.0.0.0/8"]) is None


def test_oversized_forwarded_chain_is_ignored():
    chain = ", ".join(f"203.0.113.{i}" for i in range(12))
    connection = ConnectionInfo(remote_ip="127.0.0.1", forwarded_for=chain)
    assert (
        resolve_client_ip(connection, trusted_proxies=["127.0.0.0/8"]) is None
    )


# -- local lookup ------------------------------------------------------------


def test_geolocation_is_disabled_by_default(tmp_path):
    service = GeoLocationService(db_path=str(tmp_path / "x.mmdb"))
    assert service.enabled is False
    assert service.available is False
    assert service.lookup("8.8.8.8") is None


def test_private_addresses_are_rejected_before_any_lookup(tmp_path):
    service = _service(tmp_path)
    service._reader = _Reader({"city": {"names": {"en": "Hidden"}}})
    assert service.lookup("127.0.0.1") is None
    assert service.lookup("192.168.0.1") is None


def test_city_fields_come_only_from_the_database(tmp_path):
    service = _service(tmp_path)
    service._reader = _Reader(
        {
            "city": {"names": {"en": "San Francisco"}},
            "subdivisions": [{"names": {"en": "California"}}],
            "country": {"names": {"en": "United States"}, "iso_code": "US"},
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "time_zone": "America/Los_Angeles",
                "accuracy_radius": 20,
            },
        }
    )
    result = service.lookup("8.8.8.8")
    assert result is not None
    assert result.city == "San Francisco"
    assert result.subdivision == "California"
    assert result.country_code == "US"
    assert result.timezone == "America/Los_Angeles"
    assert result.accuracy_radius_km == 20
    assert result.source == "ip_geo"


def test_country_only_result_does_not_fabricate_a_city(tmp_path):
    service = _service(tmp_path)
    service._reader = _Reader(
        {"country": {"names": {"en": "Germany"}, "iso_code": "DE"}}
    )
    result = service.lookup("8.8.8.8")
    assert result is not None
    assert result.city is None
    assert result.subdivision is None
    assert result.country == "Germany"
    assert "city" not in result.as_dict()


# -- end-to-end privacy ------------------------------------------------------


class CapturingClient:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, request_type, **kwargs):
        self.calls += 1
        return ChatResponse(
            content="我在。",
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _setup(db):
    rt = build_runtime(db, provider="deepseek", deepseek_client=CapturingClient())
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你好",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    return rt, event


async def _dump_text(db) -> str:
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'"
    )
    chunks: list[str] = []
    for row in await cur.fetchall():
        cur = await db.execute(f'SELECT * FROM "{row["name"]}"')
        for record in await cur.fetchall():
            for value in tuple(record):
                if isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


async def test_raw_ip_is_never_persisted_or_sent_to_the_model(db):
    rt, event = await _setup(db)
    ip = "203.0.113.42"

    report = await rt.orchestrator.process_utterance(
        event, connection=ConnectionInfo(remote_ip=ip)
    )

    assert report.status == "COMPLETED"
    blob = await _dump_text(db)
    assert ip not in blob
    snapshot = await rt.repos.runtime_snapshot.get(event.event_id)
    assert snapshot is not None
    assert snapshot.location is None
    assert ip not in snapshot.model_dump_json()


async def test_geo_lookup_requires_external_ai_consent(db, tmp_path):
    service = _service(tmp_path)
    service._reader = _Reader(
        {
            "country": {"names": {"en": "Germany"}, "iso_code": "DE"},
            "location": {"time_zone": "Europe/Berlin"},
        }
    )
    rt, event = await _setup(db)
    builder = rt.orchestrator.runtime_snapshot_builder
    builder.geo_service = service

    denied = await builder.build(
        event,
        connection=ConnectionInfo(remote_ip="8.8.8.8"),
        allow_external_ai=False,
        persist=False,
    )
    assert denied.location is None
    assert denied.timezone_source != "ip_geo"

    allowed = await builder.build(
        event,
        connection=ConnectionInfo(remote_ip="8.8.8.8"),
        allow_external_ai=True,
        persist=False,
    )
    assert allowed.location is not None
    assert allowed.location.country_code == "DE"
    assert allowed.location.city is None
    assert allowed.timezone_source == "ip_geo"
