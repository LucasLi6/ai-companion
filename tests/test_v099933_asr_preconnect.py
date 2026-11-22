"""V0.9.9.33: warm ASR websocket preconnect, fallback and first-partial data.

All tests are offline. The warm manager receives fake client/connection
factories; a warm ``ASRSession`` still runs the real session state machine,
event parsing and close semantics against a scripted recognition object.
"""

import logging
import threading
import time
from types import SimpleNamespace

import pytest

import test_asr_manager as asr_tests
from companion.voice.bridge import PartialTranscript
from companion.voice.coordinator import VoiceCoordinator
from test_voice_barge_in import _FakeASR, _FakeTTS, settle
from companion.asr.client import (
    ASRClientError,
    ASRSession,
    STATE_TASK_FINISHED,
)
from companion.asr.config import (
    ASRConfigError,
    DEFAULT_PRECONNECT_ENABLED,
    DEFAULT_WARM_IDLE_TTL_SECONDS,
    load_asr_config,
)
from companion.asr.manager import ASRManager, _Sentence
from companion.asr.microphone import AudioChunk
from companion.asr.warm_connection import (
    ASRWarmConnectionManager,
    WarmConnectionState,
)


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class WarmScript:
    def __init__(
        self,
        partials=None,
        finals=None,
        *,
        fail_start: bool = False,
    ) -> None:
        self.partials = list(partials or [])
        self.finals = list(finals or [])
        self.fail_start = fail_start


class WarmRecognition:
    """Duplex recognition stub that ends only after ``stop()``."""

    def __init__(self, script: WarmScript) -> None:
        self.script = script
        self.started = False
        self.sent: list[bytes] = []
        self.stop_calls = 0
        self.close_calls = 0
        self.request_id = "warm-request"
        self._stopped = threading.Event()

    def start(self):
        if self.script.fail_start:
            raise ASRClientError("模拟 warm run-task 失败")
        self.started = True
        return self

    def send_audio_frame(self, pcm: bytes) -> None:
        self.sent.append(bytes(pcm))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()

    def events(self):
        for text, sid in self.script.partials:
            yield asr_tests._sentence(text, sid=sid)
        self._stopped.wait(timeout=3.0)
        for text, sid in self.script.finals:
            yield asr_tests._sentence(text, sid=sid, final=True)

    def close(self) -> None:
        self.close_calls += 1
        self._stopped.set()


class WarmApi:
    def __init__(self, scripts: list[WarmScript]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict] = []

    def recognize(self, **kwargs):
        self.calls.append(kwargs)
        script = (
            self.scripts.pop(0) if self.scripts else WarmScript()
        )
        return WarmRecognition(script)


class WarmClient:
    def __init__(self, api: WarmApi) -> None:
        self.audio = SimpleNamespace(asr=api)
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class WarmFixture:
    def __init__(
        self,
        cfg,
        *,
        api: WarmApi,
        connection_factory=None,
    ) -> None:
        self.api = api
        self.clients: list[WarmClient] = []
        self.connections: list[FakeConnection] = []
        self._connection_factory = connection_factory
        self.manager = ASRWarmConnectionManager(
            cfg,
            client_factory=self._client_factory,
            connection_factory=self._connection_factory
            or self._default_connection,
        )

    def _client_factory(self, _cfg) -> WarmClient:
        client = WarmClient(self.api)
        self.clients.append(client)
        return client

    def _default_connection(self, _client) -> FakeConnection:
        connection = FakeConnection()
        self.connections.append(connection)
        return connection

    def wait_ready(self) -> bool:
        return asr_tests.wait_for(
            lambda: self.manager.state is WarmConnectionState.READY
        )

    def shutdown(self) -> None:
        self.manager.shutdown()


def _warm(cfg, *, api: WarmApi, connection_factory=None) -> WarmFixture:
    return WarmFixture(
        cfg, api=api, connection_factory=connection_factory
    )


def _no_live_warm_threads() -> bool:
    return not any(
        thread.name.startswith("asr-warm-connect-") and thread.is_alive()
        for thread in threading.enumerate()
    )


# --------------------------------------------------------------- config switch
def test_preconnect_config_defaults_and_switches() -> None:
    cfg = load_asr_config(asr_tests._credentials())
    assert cfg.preconnect_enabled is DEFAULT_PRECONNECT_ENABLED
    assert cfg.warm_idle_ttl_seconds == DEFAULT_WARM_IDLE_TTL_SECONDS

    off = load_asr_config(
        asr_tests._credentials(AVA_ASR_PRECONNECT_ENABLED="false")
    )
    assert off.preconnect_enabled is False
    assert (
        load_asr_config(
            asr_tests._credentials(AVA_ASR_PRECONNECT_ENABLED="TRUE")
        ).preconnect_enabled
        is True
    )

    ttl = load_asr_config(
        asr_tests._credentials(AVA_ASR_WARM_IDLE_TTL_SECONDS="12.5")
    )
    assert ttl.warm_idle_ttl_seconds == 12.5


@pytest.mark.parametrize("raw", ["maybe", "", "2", "on!"])
def test_invalid_preconnect_switch_is_rejected(raw: str) -> None:
    with pytest.raises(
        ASRConfigError, match="AVA_ASR_PRECONNECT_ENABLED"
    ):
        load_asr_config(
            asr_tests._credentials(AVA_ASR_PRECONNECT_ENABLED=raw)
        )


@pytest.mark.parametrize("raw", ["0", "-3", "abc", ""])
def test_invalid_warm_idle_ttl_is_rejected(raw: str) -> None:
    with pytest.raises(
        ASRConfigError, match="AVA_ASR_WARM_IDLE_TTL_SECONDS"
    ):
        load_asr_config(
            asr_tests._credentials(AVA_ASR_WARM_IDLE_TTL_SECONDS=raw)
        )


def test_disabled_switch_leaves_manager_without_warm_transport() -> None:
    renderer, _out, _status = asr_tests.make_renderer()
    disabled = ASRManager(
        asr_tests.config(preconnect_enabled=False),
        microphone=asr_tests.FakeMic(),
        vad_segmenter=asr_tests.ScriptedSegmenter([]),
        renderer=renderer,
    )
    assert disabled._warm_connections is None

    enabled = ASRManager(
        asr_tests.config(preconnect_enabled=True),
        microphone=asr_tests.FakeMic(),
        vad_segmenter=asr_tests.ScriptedSegmenter([]),
        renderer=renderer,
    )
    assert isinstance(enabled._warm_connections, ASRWarmConnectionManager)
    enabled._warm_connections.shutdown()


def test_injected_session_factory_disables_implicit_preconnect() -> None:
    renderer, _out, _status = asr_tests.make_renderer()
    manager = ASRManager(
        asr_tests.config(preconnect_enabled=True),
        microphone=asr_tests.FakeMic(),
        vad_segmenter=asr_tests.ScriptedSegmenter([]),
        session_factory=lambda: None,
        renderer=renderer,
    )
    assert manager._warm_connections is None


# ---------------------------------------------------------- warm manager units
def test_preconnect_leases_exclusively_and_shutdown_cleans_up() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager

    assert warm.start_preconnect() is True
    assert warm.start_preconnect() is False  # already CONNECTING
    assert fixture.wait_ready()
    assert len(fixture.clients) == 1

    first = warm.try_acquire()
    assert first is not None
    assert warm.stats()["in_use"] is True
    # A second PCM uploader must never receive the same socket.
    assert warm.try_acquire() is None

    first.release(healthy=True)
    assert warm.state is WarmConnectionState.READY
    second = warm.try_acquire()
    assert second is not None
    assert second.connection is first.connection
    second.release(healthy=True)

    warm.shutdown()
    assert warm.state is WarmConnectionState.DISCONNECTED
    assert fixture.connections[0].closed is True
    assert fixture.clients[0].close_calls == 1
    assert warm.start_preconnect() is False
    assert _no_live_warm_threads()


def test_unhealthy_release_discards_the_transport() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()

    lease = warm.try_acquire()
    assert lease is not None
    lease.release(healthy=False)

    assert fixture.connections[0].closed is True
    assert warm.state is not WarmConnectionState.IN_USE
    assert asr_tests.wait_for(
        lambda: warm.state is WarmConnectionState.READY
        and len(fixture.connections) == 2
    )
    warm.shutdown()


def test_invalidate_defers_close_until_the_active_lease_releases() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()

    lease = warm.try_acquire()
    assert lease is not None
    warm.invalidate("test")
    # The socket is still owned by the recognition task.
    assert fixture.connections[0].closed is False
    assert warm.try_acquire() is None
    # A background rebuild must never replace a connection still in use.
    assert warm.start_preconnect() is False
    assert fixture.connections[0].closed is False

    lease.release(healthy=True)
    assert fixture.connections[0].closed is True
    assert asr_tests.wait_for(
        lambda: warm.state is WarmConnectionState.READY
        and len(fixture.connections) == 2
    )
    assert fixture.connections[1].closed is False
    warm.shutdown()


def test_idle_ttl_rejects_and_rebuilds_a_ready_connection() -> None:
    fixture = _warm(
        asr_tests.config(warm_idle_ttl_seconds=1.0), api=WarmApi([])
    )
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()
    with warm._lock:  # deterministic: simulate an old idle timestamp
        warm._idle_since = time.monotonic() - 10.0

    assert warm.try_acquire() is None
    assert fixture.connections[0].closed is True
    assert asr_tests.wait_for(
        lambda: len(fixture.connections) == 2
        and warm.state is WarmConnectionState.READY
    )
    warm.shutdown()


def test_preconnect_failure_is_reported_and_backed_off() -> None:
    attempts: list[int] = []

    def failing_factory(_cfg):
        attempts.append(1)
        raise RuntimeError("offline")

    warm = ASRWarmConnectionManager(
        asr_tests.config(),
        client_factory=failing_factory,
        connection_factory=lambda _client: FakeConnection(),
    )
    assert warm.start_preconnect() is True
    assert asr_tests.wait_for(lambda: warm.stats()["failures"] >= 1)
    assert len(attempts) == 1
    assert warm.try_acquire() is None
    # Inside the 10 s backoff: no tight retry loop.
    assert warm.start_preconnect() is False
    time.sleep(0.05)
    assert len(attempts) == 1
    warm.shutdown()


def test_missing_shared_connection_api_disables_preconnect() -> None:
    attempts: list[int] = []

    def factory(_cfg):
        attempts.append(1)
        return WarmClient(WarmApi([]))  # no ``ws_connection`` method

    warm = ASRWarmConnectionManager(
        asr_tests.config(), client_factory=factory
    )
    assert warm.start_preconnect() is True
    assert asr_tests.wait_for(
        lambda: warm.stats()["disabled"] is True
        and warm.stats()["state"] == "DISCONNECTED"
    )
    assert warm.try_acquire() is None
    assert warm.start_preconnect() is False
    time.sleep(0.05)
    assert len(attempts) == 1
    warm.shutdown()


def test_shutdown_closes_an_leased_socket_exactly_once() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()
    lease = warm.try_acquire()
    assert lease is not None

    warm.shutdown()
    lease.release(healthy=True)  # a late release must not double-close
    assert fixture.connections[0].close_calls == 1
    assert fixture.clients[0].close_calls == 1


def test_shutdown_during_connect_never_publishes_a_socket() -> None:
    release = threading.Event()
    connections: list[FakeConnection] = []

    def slow_connection(_client):
        connection = FakeConnection()
        connections.append(connection)
        release.wait(timeout=1.0)
        return connection

    fixture = _warm(
        asr_tests.config(),
        api=WarmApi([]),
        connection_factory=slow_connection,
    )
    fixture.manager.start_preconnect()
    fixture.manager.shutdown()

    assert fixture.manager.state is WarmConnectionState.DISCONNECTED
    assert len(connections) == 1
    assert connections[0].closed is True
    assert _no_live_warm_threads()


# ------------------------------------------------------- ASRSession warm mode
def test_warm_session_passes_connection_and_only_detaches_on_close() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()
    lease = warm.try_acquire()
    assert lease is not None
    lease.client.audio.asr.scripts.append(
        WarmScript(partials=[("你好", "s1")], finals=[("你好。", "s1")])
    )

    session = ASRSession(asr_tests.config(), warm_lease=lease)
    session.open()
    call = lease.client.audio.asr.calls[0]
    assert call["connection"] is lease.connection
    session.send_audio(b"\x01\x02")
    session.stop()
    events = list(session.events())
    assert [(item.text, item.is_final) for item in events] == [
        ("你好", False),
        ("你好。", True),
    ]
    assert session.state == STATE_TASK_FINISHED
    session.close()

    # Detached only: the socket and client stay owned by the manager.
    assert lease.connection.closed is False
    assert lease.client.close_calls == 0
    assert warm.state is WarmConnectionState.READY
    reuse = warm.try_acquire()
    assert reuse is not None
    reuse.release(healthy=True)
    warm.shutdown()


def test_warm_session_closed_mid_stream_invalidates_the_socket() -> None:
    fixture = _warm(asr_tests.config(), api=WarmApi([]))
    warm = fixture.manager
    warm.start_preconnect()
    assert fixture.wait_ready()
    lease = warm.try_acquire()
    assert lease is not None

    session = ASRSession(asr_tests.config(), warm_lease=lease)
    session.open()
    session.close()  # no task-finished: never reuse this transport

    assert lease.released is True
    assert fixture.connections[0].closed is True
    # A replacement is prepared in the background, but the discarded socket
    # can never be handed out again.
    assert asr_tests.wait_for(
        lambda: len(fixture.connections) == 2
        and warm.state is WarmConnectionState.READY
    )
    replacement = warm.try_acquire()
    assert replacement is not None
    assert replacement.connection is fixture.connections[1]
    replacement.release(healthy=True)
    warm.shutdown()


# --------------------------------------------------------- manager integration
def _running_manager(cfg, groups, cold_factory, warm):
    renderer, _out, status = asr_tests.make_renderer()
    finals = []
    manager = ASRManager(
        cfg,
        microphone=asr_tests.FakeMic(),
        vad_segmenter=asr_tests.ScriptedSegmenter(groups),
        session_factory=cold_factory,
        renderer=renderer,
        on_final=finals.append,
        warm_connection_manager=warm,
    )
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    assert asr_tests.wait_for(lambda: "正在监听" in status.getvalue())
    return manager, thread, finals


def _feed(manager: ASRManager, count: int = 1) -> None:
    for _ in range(count):
        manager.audio_queue.put(
            AudioChunk(b"\x01\x00" * 512, 16000, 16000, time.monotonic()),
            timeout=2.0,
        )


def _stop_manager(manager, thread) -> None:
    manager.stop()
    thread.join(timeout=4.0)
    manager.shutdown()
    assert not thread.is_alive()


def test_warm_transport_is_used_and_reused_for_consecutive_turns() -> None:
    cfg = asr_tests.config(warm_idle_ttl_seconds=30.0)
    api = WarmApi(
        [
            WarmScript(partials=[("你好", "s1")], finals=[("你好。", "s1")]),
            WarmScript(partials=[("再见", "s1")], finals=[("再见。", "s1")]),
        ]
    )
    fixture = _warm(cfg, api=api)
    cold = asr_tests.SessionFactory([])
    groups = [
        asr_tests.start_group(),
        asr_tests.audio_group(),
        asr_tests.end_group(),
    ] * 2
    manager, thread, finals = _running_manager(
        cfg, groups, cold, fixture.manager
    )
    try:
        assert fixture.wait_ready()
        _feed(manager, 3)
        assert asr_tests.wait_for(lambda: len(finals) == 1)
        assert asr_tests.wait_for(
            lambda: fixture.manager.state is WarmConnectionState.READY
        )
        _feed(manager, 3)
        assert asr_tests.wait_for(lambda: len(finals) == 2)

        assert [item.text for item in finals] == ["你好。", "再见。"]
        assert len(api.calls) == 2
        assert all(
            call["connection"] is fixture.connections[0]
            for call in api.calls
        )
        assert cold.created == []
        assert fixture.clients[0].close_calls == 0
        assert len(fixture.connections) == 1
        assert fixture.connections[0].closed is False
    finally:
        _stop_manager(manager, thread)

    assert fixture.connections[0].closed is True
    assert fixture.clients[0].close_calls == 1
    assert _no_live_warm_threads()


def test_warm_open_failure_retries_cold_once_without_losing_audio() -> None:
    cfg = asr_tests.config()
    api = WarmApi([WarmScript(fail_start=True)])
    fixture = _warm(cfg, api=api)
    cold = asr_tests.SessionFactory(
        [
            asr_tests.SessionScript(
                partials=[("你好", "s1")], finals=[("你好。", "s1")]
            )
        ]
    )
    groups = [
        asr_tests.start_group(),
        asr_tests.audio_group(),
        asr_tests.end_group(),
    ]
    manager, thread, finals = _running_manager(
        cfg, groups, cold, fixture.manager
    )
    modes: list[str] = []
    original_open = manager._open_session

    def tracked_open(utterance):
        session = original_open(utterance)
        modes.append(utterance.connection_mode)
        return session

    manager._open_session = tracked_open
    try:
        assert fixture.wait_ready()
        _feed(manager, 3)
        assert asr_tests.wait_for(lambda: len(finals) == 1)

        assert [item.text for item in finals] == ["你好。"]
        assert modes == ["FALLBACK"]
        assert len(cold.created) == 1
        assert cold.created[0].sent_bytes > 0
        # The failed warm transport was discarded, not retried in place.
        assert len(api.calls) == 1
        assert fixture.connections[0].closed is True
    finally:
        _stop_manager(manager, thread)
    assert _no_live_warm_threads()


# ------------------------------------------------------------ first-partial data
def test_note_partial_displayed_keeps_only_the_first_observation() -> None:
    manager = asr_tests._timing_manager()
    utterance = asr_tests._track(manager, 7)
    manager.note_partial_displayed(7, 12.0)
    manager.note_partial_displayed(7, 99.0)
    assert utterance.first_partial_to_ui_ms == 12.0
    # Unknown and finished turns are ignored.
    manager.note_partial_displayed(999, 5.0)
    utterance.finalized = True
    manager.note_partial_displayed(7, 1.0)
    assert utterance.first_partial_to_ui_ms == 12.0


def test_first_partial_line_reports_warm_vs_cold_path(caplog) -> None:
    manager = asr_tests._timing_manager()
    utterance = asr_tests._track(manager, 18)
    utterance.connection_mode = "FALLBACK"
    utterance.preconnect_ms = 630.0
    utterance.open_latency_ms = 110.0
    utterance.speech_started_at = 1.0
    utterance.task_ready_at = 1.2
    utterance.first_audio_sent_at = 1.214
    utterance.first_partial_at = 1.35
    utterance.first_partial_to_ui_ms = 5.0
    utterance.max_queued_audio_ms = 80.0
    utterance.vad_end_at = 2.0
    utterance.final_emitted_at = 2.25
    with caplog.at_level(logging.DEBUG, logger="companion.asr"):
        manager._log_first_partial(utterance)
    line = next(
        record.getMessage()
        for record in caplog.records
        if "[asr first]" in record.getMessage()
    )
    assert "uid=18" in line
    assert "mode=FALLBACK" in line
    assert "preconnect_ms=630ms" in line
    assert "task_start_ms=110ms" in line
    assert "speech_to_first_partial_ms=350ms" in line
    assert "first_audio_ms=214ms" in line
    assert "task_start_to_first_audio_ms=14ms" in line
    assert "first_audio_to_first_partial_ms=136ms" in line
    assert "first_partial_to_ui_ms=5ms" in line
    assert "max_queue_ms=80ms" in line
    assert "vad_to_final_ms=250ms" in line


def test_first_partial_line_uses_na_for_unknown_values(caplog) -> None:
    manager = asr_tests._timing_manager()
    utterance = asr_tests._track(manager, 7)
    with caplog.at_level(logging.DEBUG, logger="companion.asr"):
        manager._log_first_partial(utterance)
    line = next(
        record.getMessage()
        for record in caplog.records
        if "[asr first]" in record.getMessage()
    )
    assert "mode=COLD" in line
    for name in (
        "preconnect_ms",
        "task_start_ms",
        "speech_to_first_partial_ms",
        "first_audio_ms",
        "task_start_to_first_audio_ms",
        "first_audio_to_first_partial_ms",
        "first_partial_to_ui_ms",
        "max_queue_ms",
        "vad_to_final_ms",
    ):
        assert f"{name}=N/A" in line


def test_first_partial_line_is_debug_only(caplog) -> None:
    manager = asr_tests._timing_manager(debug=False)
    utterance = asr_tests._track(manager, 7)
    with caplog.at_level(logging.DEBUG, logger="companion.asr"):
        manager._log_first_partial(utterance)
    assert not [
        record
        for record in caplog.records
        if "[asr first]" in record.getMessage()
    ]


def test_partial_event_carries_receive_time_for_ui_latency() -> None:
    manager = asr_tests._timing_manager()
    utterance = asr_tests._track(manager, 3)
    received: list = []
    manager._on_partial = received.append
    manager._handle_sentence(
        _Sentence(3, "你好", "s1", False, 100.0)
    )
    assert received
    assert received[0].at == 100.0
    manager.note_partial_displayed(3, 4.0)
    assert utterance.first_partial_to_ui_ms == 4.0


async def test_coordinator_reports_first_partial_display_latency() -> None:
    fake = _FakeASR()
    notes: list[tuple[int, float]] = []
    fake.note_partial_displayed = (
        lambda uid, ms: notes.append((uid, ms))
    )
    coordinator = VoiceCoordinator(
        manager_factory=lambda: fake,
        tts_manager=_FakeTTS(),
        cooldown_seconds=0.0,
    )
    ok, _ = await coordinator.enable()
    assert ok
    coordinator.set_partial_listener(lambda _uid, _text: None)
    try:
        fake.speech("started", 7)
        await settle()
        coordinator._bridge.publish_partial(
            PartialTranscript(
                7, "我想问", sequence=1, at=time.monotonic() - 0.05
            )
        )
        await settle()
        assert len(notes) == 1
        assert notes[0][0] == 7
        assert notes[0][1] >= 0.0

        # A superseded older revision is never displayed, so it is not
        # counted as a first-partial display.
        coordinator._bridge.publish_partial(
            PartialTranscript(
                7, "我", sequence=0, at=time.monotonic()
            )
        )
        await settle()
        assert len(notes) == 1
    finally:
        await coordinator.shutdown()
