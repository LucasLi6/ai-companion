"""Observation-only speaker identification for the formal voice REPL.

Phase 0/1 contract:
- OFF is the default and constructs the original ASRManager unchanged.
- SHADOW attaches the already tested DeferredSpeakerObserver to that ASR.
- Diagnostic output contains no transcripts or raw audio.
- Identity never reaches Person, Router, Profile or Memory in shadow mode.
- Each mic session owns its observer and closes it on disable, restart or exit.

The engine can still store pseudonymous embeddings in its separate local
SQLite database; enabling shadow is therefore an explicit opt-in.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Callable, Mapping, Optional

from companion.asr.manager import ASRManager, FinalUtterance
from companion.speaker_id.config import SpeakerConfig, load_speaker_config
from companion.speaker_id.runtime import DeferredSpeakerObserver

REPL_SPEAKER_MODE_ENV = "AVA_REPL_SPEAKER_MODE"
MODE_OFF = "off"
MODE_SHADOW = "shadow"

_LOG = logging.getLogger("companion.speaker.shadow")


def repl_speaker_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Opt in explicitly; unknown settings fail closed to OFF."""
    source = os.environ if env is None else env
    mode = str(source.get(REPL_SPEAKER_MODE_ENV, MODE_OFF) or MODE_OFF)
    mode = mode.strip().lower()
    if mode in (MODE_OFF, MODE_SHADOW):
        return mode
    _LOG.warning("speaker_shadow_invalid_mode effective=off")
    return MODE_OFF


class ShadowSpeakerObserver(DeferredSpeakerObserver):
    """Nonblocking model owner with metadata-only worker diagnostics."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        self.run_id = run_id or secrets.token_hex(8)
        self._diagnostics = logger or _LOG
        self._result_updates = 0
        super().__init__(config, on_result=self._on_result_shadow, **kwargs)

    def _on_result_shadow(self, result: Any) -> None:
        # This callback runs on the speaker worker, not the ASR/VAD path.
        self._result_updates += 1
        self._diagnostics.info(
            "speaker_shadow_update run_id=%s uid=%s status=%s "
            "speaker_id=%s quality=%s provisional=%s similarity=%s "
            "margin=%s voiced_seconds=%s inference_ms=%s",
            self.run_id,
            getattr(result, "utterance_id", None),
            getattr(result, "status", None),
            getattr(result, "speaker_id", None),
            getattr(result, "reid_quality", None),
            getattr(result, "provisional", None),
            getattr(result, "similarity", None),
            getattr(result, "margin", None),
            getattr(result, "voiced_seconds", None),
            getattr(result, "inference_ms", None),
        )

    def diagnostics(self) -> dict[str, Any]:
        timings = self.timings
        return {
            "run_id": self.run_id,
            "state": self.status,
            "ready": self.ready,
            "updates": self._result_updates,
            "model_load_ms": timings["model_load_ms"],
            "warmup_ms": timings["warmup_ms"],
            "failed": bool(timings["error"]),
        }


class ShadowASRManager(ASRManager):
    """Own exactly one deferred speaker observer per ASR mic session."""

    def __init__(
        self,
        asr_config: Any,
        *,
        speaker_observer: Any,
        **asr_kwargs: Any,
    ) -> None:
        super().__init__(
            asr_config, speaker_observer=speaker_observer, **asr_kwargs
        )
        self._repl_shadow_observer = speaker_observer
        self._shadow_started = False
        self._shadow_closed = False

    @property
    def shadow_run_id(self) -> Optional[str]:
        return getattr(self._repl_shadow_observer, "run_id", None)

    def speaker_shadow_status(self) -> dict[str, Any]:
        observer = self._repl_shadow_observer
        diagnostics = getattr(observer, "diagnostics", None)
        if callable(diagnostics):
            return diagnostics()
        return {
            "run_id": self.shadow_run_id,
            "state": getattr(observer, "status", "UNKNOWN"),
            "updates": 0,
        }

    def start(self) -> None:
        # Coordinator.enable() and manager.run() both call start(). Load
        # the model only once, after the microphone starts successfully.
        # Loading itself is a background operation and never awaited.
        super().start()
        if not self._shadow_started:
            self._shadow_started = True
            try:
                self._repl_shadow_observer.start_loading()
            except Exception as exc:  # noqa: BLE001 - optional sidecar
                _LOG.warning(
                    "speaker_shadow_start_failed run_id=%s error_type=%s",
                    self.shadow_run_id,
                    type(exc).__name__,
                )

    def shutdown(self) -> None:
        # Also runs after a failed start. No observer can outlive its ASR.
        try:
            super().shutdown()
        finally:
            if not self._shadow_closed:
                self._shadow_closed = True
                try:
                    self._repl_shadow_observer.shutdown()
                except Exception as exc:  # noqa: BLE001 - optional sidecar
                    _LOG.warning(
                        "speaker_shadow_shutdown_failed run_id=%s error_type=%s",
                        self.shadow_run_id,
                        type(exc).__name__,
                    )


def build_repl_asr_manager(
    asr_config: Any,
    *,
    shadow: bool = False,
    speaker_debug: bool = False,
    speaker_config: Optional[SpeakerConfig] = None,
    observer_factory: Optional[Callable[[SpeakerConfig], Any]] = None,
    **asr_kwargs: Any,
) -> ASRManager:
    """Build the original ASR by default; fail open to ASR on shadow errors."""
    if not shadow:
        return ASRManager(asr_config, **asr_kwargs)
    try:
        config = (
            speaker_config
            if speaker_config is not None
            else load_speaker_config(debug=speaker_debug)
        )
        observer = (
            observer_factory(config)
            if observer_factory is not None
            else ShadowSpeakerObserver(config)
        )
    except Exception as exc:  # noqa: BLE001 - bad optional setup
        _LOG.warning(
            "speaker_shadow_setup_failed error_type=%s fallback=asr_only",
            type(exc).__name__,
        )
        return ASRManager(asr_config, **asr_kwargs)

    manager = ShadowASRManager(
        asr_config, speaker_observer=observer, **asr_kwargs
    )
    _LOG.info(
        "speaker_shadow_created run_id=%s profile_mode=%s "
        "model_key=%s effects=diagnostics_only",
        manager.shadow_run_id,
        config.profile_mode,
        config.model_key,
    )
    return manager


def log_shadow_final(
    final: FinalUtterance,
    *,
    run_id: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log the final's *snapshot* on the REPL loop, never the ASR thread."""
    speaker = getattr(final, "speaker", None)
    (logger or _LOG).info(
        "speaker_shadow_final run_id=%s uid=%s status=%s "
        "speaker_id=%s quality=%s provisional=%s",
        run_id,
        getattr(final, "utterance_id", None),
        getattr(speaker, "status", "NO_OBSERVER"),
        getattr(speaker, "speaker_id", None),
        getattr(speaker, "reid_quality", None),
        getattr(speaker, "provisional", None),
    )


__all__ = [
    "MODE_OFF",
    "MODE_SHADOW",
    "REPL_SPEAKER_MODE_ENV",
    "ShadowASRManager",
    "ShadowSpeakerObserver",
    "build_repl_asr_manager",
    "log_shadow_final",
    "repl_speaker_mode",
]
