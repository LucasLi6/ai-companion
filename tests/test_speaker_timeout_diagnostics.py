"""The REPL distinguishes a slow worker from missing final publication."""

from __future__ import annotations

from types import SimpleNamespace

from companion.voice.speaker_shadow import ReplASRManager


def diagnostic(*, ready=True, finished=False, provisional=True, status="KNOWN"):
    latest = SimpleNamespace(status=status, provisional=provisional)
    engine = SimpleNamespace(
        snapshot=lambda uid: latest,
        is_finished=lambda uid: finished,
    )
    observer = SimpleNamespace(
        ready=ready, status="READY" if ready else "LOADING", engine=engine
    )
    manager = SimpleNamespace(_repl_speaker_observer=observer)
    return ReplASRManager.speaker_utterance_diagnostic(manager, 42)


def test_loading_model_has_distinct_timeout_cause():
    assert diagnostic(ready=False)["reason"] == "model_not_ready"


def test_worker_still_running_is_an_inference_timeout():
    assert diagnostic(finished=False)["reason"] == "inference_timeout"


def test_worker_finished_but_snapshot_is_provisional():
    assert diagnostic(finished=True)["reason"] == "final_result_not_published"


def test_confirmed_final_is_not_reported_as_a_timeout():
    assert diagnostic(finished=True, provisional=False)["reason"] == (
        "final_published"
    )


def test_worker_error_is_reported_distinctly():
    assert diagnostic(status="ERROR")["reason"] == "model_error"
