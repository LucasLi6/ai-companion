"""Phase 0/1: opt-in REPL speaker wiring, nonblocking and identity-neutral."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from companion.asr.config import ASRConfig
from companion.asr.manager import ASRManager
from companion.speaker_id.config import load_speaker_config
from companion.speaker_id.models import ERROR, SpeakerResult
from companion.voice.speaker_shadow import (
    MODE_ACTIVE,
    MODE_OFF,
    MODE_SHADOW,
    ShadowASRManager,
    ShadowSpeakerObserver,
    build_repl_asr_manager,
    log_shadow_final,
    repl_speaker_mode,
)


def config():
    return ASRConfig(api_key="test-key", workspace_id="test-workspace")


class FakeObserver:
    run_id = "run-test"
    status = "LOADING"

    def __init__(self, events=None):
        self.events = events if events is not None else []

    def start_loading(self):
        self.events.append("speaker_start")

    def shutdown(self):
        self.events.append("speaker_shutdown")

    def diagnostics(self):
        return {"run_id": self.run_id, "state": self.status, "updates": 0}


def sample_result():
    return SpeakerResult(
        utterance_id=3,
        speaker_id="S-0003",
        person_id=None,
        status="KNOWN",
        reid_quality="HIGH",
        similarity=0.85,
        margin=0.2,
        voiced_seconds=2.4,
        provisional=False,
        model_id="fake",
        inference_ms=5.2,
        completed_at=1.0,
    )


def test_default_mode_is_active_and_unknown_values_fail_closed(caplog):
    # Formal voice identity is the default; off/shadow are explicit rollbacks.
    assert repl_speaker_mode({}) == MODE_ACTIVE
    assert repl_speaker_mode({"AVA_REPL_SPEAKER_MODE": "off"}) == MODE_OFF
    assert repl_speaker_mode({"AVA_REPL_SPEAKER_MODE": " SHADOW "}) == MODE_SHADOW
    assert repl_speaker_mode({"AVA_REPL_SPEAKER_MODE": "active"}) == MODE_ACTIVE
    with caplog.at_level(logging.WARNING):
        assert repl_speaker_mode({"AVA_REPL_SPEAKER_MODE": "typo"}) == MODE_OFF
    assert "speaker_shadow_invalid_mode" in caplog.text


def test_off_constructs_original_asr_without_speaker_config(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("OFF must not touch speaker config")
    monkeypatch.setattr(
        "companion.voice.speaker_shadow.load_speaker_config", forbidden
    )
    manager = build_repl_asr_manager(config(), shadow=False)
    assert type(manager) is ASRManager
    assert manager._speaker_observer is None


def test_shadow_observer_is_attached_without_person_identity(tmp_path):
    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = FakeObserver()
    manager = build_repl_asr_manager(
        config(), shadow=True, speaker_config=speaker_config,
        observer_factory=lambda _: observer,
    )
    assert isinstance(manager, ShadowASRManager)
    assert manager._speaker_observer is observer
    assert manager.shadow_run_id == "run-test"
    assert manager.speaker_shadow_status()["state"] == "LOADING"
    assert not hasattr(manager, "person_id")


def test_speaker_config_error_falls_back_to_original_asr(monkeypatch, caplog):
    def fail(**kwargs):
        raise RuntimeError("optional model config failed")
    monkeypatch.setattr(
        "companion.voice.speaker_shadow.load_speaker_config", fail
    )
    with caplog.at_level(logging.WARNING):
        manager = build_repl_asr_manager(config(), shadow=True)
    assert type(manager) is ASRManager
    assert manager._speaker_observer is None
    assert "fallback=asr_only" in caplog.text


def test_model_is_loaded_only_after_asr_start_and_closed_once(monkeypatch):
    events = []
    monkeypatch.setattr(
        ASRManager, "start", lambda self: events.append("asr_start")
    )
    monkeypatch.setattr(
        ASRManager, "shutdown", lambda self: events.append("asr_shutdown")
    )
    observer = FakeObserver(events)
    manager = ShadowASRManager(config(), speaker_observer=observer)
    manager.start()
    manager.start()  # Coordinator + manager.run()
    assert events == ["asr_start", "speaker_start", "asr_start"]
    manager.shutdown()
    manager.shutdown()
    assert events.count("speaker_shutdown") == 1


def test_background_model_error_yields_safe_unknown(tmp_path):
    class BrokenModel:
        def load(self):
            raise RuntimeError("simulated loading failure")
    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = ShadowSpeakerObserver(
        speaker_config, embedding_model=BrokenModel()
    )
    observer.start_loading()
    try:
        until = time.monotonic() + 3.0
        while observer.status != "UNAVAILABLE" and time.monotonic() < until:
            time.sleep(0.005)
        assert observer.status == "UNAVAILABLE"
        assert observer.snapshot(3).status == ERROR
        assert observer.diagnostics()["failed"]
    finally:
        observer.shutdown()


def test_diagnostics_never_log_transcripts(tmp_path, caplog):
    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = ShadowSpeakerObserver(speaker_config, run_id="test-session")
    with caplog.at_level(logging.INFO):
        observer._on_result_shadow(sample_result())
        log_shadow_final(
            SimpleNamespace(
                utterance_id=3, text="SECRET TRANSCRIPT",
                speaker=sample_result(),
            ),
            run_id="test-session",
        )
    assert "speaker_shadow_update" in caplog.text
    assert "speaker_shadow_final" in caplog.text
    assert "run_id=test-session" in caplog.text
    assert "speaker_id=S-0003" in caplog.text
    assert "SECRET TRANSCRIPT" not in caplog.text
    assert observer.diagnostics()["updates"] == 1
    observer.shutdown()


def test_pending_final_then_late_result_are_logged_separately(tmp_path, caplog):
    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = ShadowSpeakerObserver(speaker_config, run_id="late")
    with caplog.at_level(logging.INFO):
        log_shadow_final(
            SimpleNamespace(
                utterance_id=3, text="sensitive",
                speaker=SimpleNamespace(
                    status="PENDING", speaker_id=None,
                    reid_quality="UNKNOWN", provisional=True,
                ),
            ),
            run_id=observer.run_id,
        )
        observer._on_result_shadow(sample_result())
    lines = caplog.text.splitlines()
    assert any("speaker_shadow_final" in x and "PENDING" in x for x in lines)
    assert any("speaker_shadow_update" in x and "KNOWN" in x for x in lines)
    observer.shutdown()


# ------------------------------------------------ V0.9.9.31 formal activation
def test_active_mode_default_observer_publishes_run_scoped_updates(tmp_path):
    from companion.voice.bridge import SpeakerUpdate
    from companion.voice.speaker_shadow import ReplSpeakerObserver

    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = ReplSpeakerObserver(speaker_config, run_id="run-formal")
    received = []
    observer.set_speaker_update_callback(received.append)
    try:
        observer._on_result_shadow(sample_result())
        observer._on_result_shadow(sample_result())

        assert [update.revision for update in received] == [1, 2]
        assert all(isinstance(update, SpeakerUpdate) for update in received)
        assert all(update.run_id == "run-formal" for update in received)
        assert observer.diagnostics()["updates"] == 2
    finally:
        observer.shutdown()


def test_update_callback_failure_never_escapes_the_speaker_thread(tmp_path):
    from companion.voice.speaker_shadow import ReplSpeakerObserver

    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = ReplSpeakerObserver(speaker_config, run_id="run-broken")
    observer.set_speaker_update_callback(
        lambda _update: (_ for _ in ()).throw(RuntimeError("sink died"))
    )
    try:
        observer._on_result_shadow(sample_result())
        assert observer.diagnostics()["updates"] == 1
    finally:
        observer.shutdown()


def test_manager_forwards_the_formal_callback_and_status(tmp_path):
    speaker_config = load_speaker_config(
        env={}, db_path=tmp_path / "speaker.sqlite3"
    )
    observer = FakeObserver()
    observer.run_id = "run-forward"
    manager = ShadowASRManager(config(), speaker_observer=observer)
    received = []

    class FormalObserver:
        run_id = "run-forward"

        def start_loading(self):
            observer.start_loading()

        def shutdown(self):
            observer.shutdown()

        def set_speaker_update_callback(self, callback):
            received.append(callback)

        def diagnostics(self):
            return {"run_id": self.run_id, "state": "READY", "updates": 3}

    formal = FormalObserver()
    manager = ShadowASRManager(config(), speaker_observer=formal)
    callback = lambda _update: None
    manager.set_speaker_update_callback(callback)
    assert received == [callback]
    assert manager.repl_run_id == "run-forward"
    assert manager.speaker_status()["updates"] == 3
    # Historical alias still works for older callers.
    assert manager.speaker_shadow_status()["updates"] == 3


def test_disabled_formal_manager_has_no_callback_surface():
    manager = build_repl_asr_manager(config(), speaker_enabled=False)
    assert type(manager) is ASRManager
    assert getattr(manager, "set_speaker_update_callback", None) is None
