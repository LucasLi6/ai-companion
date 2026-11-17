"""Voice-identity sidecar for the formal voice REPL.

The default mode is ``active``: ``repl --voice`` attaches the reviewed,
asynchronous speaker observer to the real ASR manager, and publishes identity
updates to the conversation loop. The observer never runs on the ASR/VAD path,
never makes a person decision itself, and never writes to the main database.

Modes are kept as an explicit rollback mechanism:

* ``active`` (default) -- load the model and publish identity updates;
* ``shadow``            -- load and log diagnostics, but the REPL ignores
  identity for person resolution;
* ``off``               -- construct the original ASR manager unchanged.

Storage consent is a separate decision. :class:`EnrollmentPolicy` gates every
durable voiceprint write; the observer receives it from the REPL.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from collections import deque
from typing import Any, Callable, Mapping, Optional

from companion.asr.manager import ASRManager, FinalUtterance
from companion.speaker_id.config import SpeakerConfig, load_speaker_config
from companion.speaker_id.consent import EnrollmentPolicy
from companion.speaker_id.models import SpeakerResult
from companion.speaker_id.runtime import DeferredSpeakerObserver
from companion.voice.bridge import SpeakerUpdate

REPL_SPEAKER_MODE_ENV = "AVA_REPL_SPEAKER_MODE"
MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
DEFAULT_MODE = MODE_ACTIVE
_MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

_MAX_TRACKED_REVISIONS = 256

_LOG = logging.getLogger("companion.speaker.shadow")


def repl_speaker_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the effective REPL speaker mode.

    Voice identity is the formal default now. An explicit ``off``/``shadow``
    remains available for rollback or diagnostics; an unrecognised value fails
    closed to ``off`` so a typo cannot silently change the privacy posture.
    """
    source = os.environ if env is None else env
    raw = str(source.get(REPL_SPEAKER_MODE_ENV, "") or "").strip().lower()
    if not raw:
        return DEFAULT_MODE
    if raw in _MODES:
        return raw
    _LOG.warning("speaker_shadow_invalid_mode effective=off")
    return MODE_OFF


class ReplSpeakerObserver(DeferredSpeakerObserver):
    """Nonblocking model owner that publishes thread-safe identity updates."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        enrollment_policy: Optional[EnrollmentPolicy] = None,
        **kwargs: Any,
    ) -> None:
        self.run_id = run_id or secrets.token_hex(8)
        self._diagnostics = logger or _LOG
        self._result_updates = 0
        self._update_lock = threading.Lock()
        self._update_callback: Optional[Callable[[SpeakerUpdate], None]] = None
        self._revisions: dict[int, int] = {}
        self._revision_order: deque[int] = deque(maxlen=_MAX_TRACKED_REVISIONS)
        super().__init__(
            config,
            enrollment_policy=enrollment_policy,
            **kwargs,
        )

    # ------------------------------------------------------------- callbacks
    def set_speaker_update_callback(
        self, callback: Optional[Callable[[SpeakerUpdate], None]]
    ) -> None:
        """Attach a thread-safe receiver for identity updates.

        The callback runs on the speaker worker thread. It must only hand the
        update to the asyncio loop (for example ``VoiceBridge`` does with
        ``call_soon_threadsafe``); it must not read profiles, query memory,
        call an LLM or write SQLite.
        """
        with self._update_lock:
            self._update_callback = callback

    def _next_revision(self, utterance_id: int) -> int:
        uid = int(utterance_id)
        with self._update_lock:
            revision = self._revisions.get(uid, 0) + 1
            if uid not in self._revisions:
                if len(self._revision_order) == self._revision_order.maxlen:
                    oldest = self._revision_order[0]
                    self._revisions.pop(oldest, None)
                self._revision_order.append(uid)
            self._revisions[uid] = revision
            return revision

    def _on_result_shadow(self, result: Any) -> None:
        # Kept under its historical name: the observer is constructed with
        # this bound method, and existing diagnostics tests call it directly.
        self._result_updates += 1
        update = SpeakerUpdate(
            run_id=self.run_id,
            utterance_id=int(getattr(result, "utterance_id", 0) or 0),
            revision=self._next_revision(
                int(getattr(result, "utterance_id", 0) or 0)
            ),
            result=result,
        )
        callbacks = None
        with self._update_lock:
            callbacks = self._update_callback
        if callbacks is not None:
            try:
                callbacks(update)
            except Exception:  # noqa: BLE001 - identity must not stop ASR
                self._diagnostics.debug("speaker_update_callback_failed")
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


# Compatibility alias for code and tests written during the shadow phase.
ShadowSpeakerObserver = ReplSpeakerObserver


class ReplASRManager(ASRManager):
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
        self._repl_speaker_observer = speaker_observer
        self._shadow_started = False
        self._shadow_closed = False

    @property
    def repl_run_id(self) -> Optional[str]:
        return getattr(self._repl_speaker_observer, "run_id", None)

    @property
    def shadow_run_id(self) -> Optional[str]:
        # Historical name used by the shadow-phase coordinator/tests.
        return self.repl_run_id

    def set_speaker_update_callback(
        self, callback: Optional[Callable[[SpeakerUpdate], None]]
    ) -> None:
        setter = getattr(
            self._repl_speaker_observer,
            "set_speaker_update_callback",
            None,
        )
        if callable(setter):
            setter(callback)

    def speaker_status(self) -> dict[str, Any]:
        observer = self._repl_speaker_observer
        diagnostics = getattr(observer, "diagnostics", None)
        if callable(diagnostics):
            return diagnostics()
        return {
            "run_id": self.repl_run_id,
            "state": getattr(observer, "status", "UNKNOWN"),
            "updates": 0,
        }

    # Historical name kept for callers written before formal activation.
    speaker_shadow_status = speaker_status

    def start(self) -> None:
        # Coordinator.enable() and manager.run() both call start(). Load the
        # model only once, after the microphone starts successfully. Loading
        # itself is a background operation and never awaited.
        super().start()
        if not self._shadow_started:
            self._shadow_started = True
            try:
                self._repl_speaker_observer.start_loading()
            except Exception as exc:  # noqa: BLE001 - optional sidecar
                _LOG.warning(
                    "speaker_shadow_start_failed run_id=%s error_type=%s",
                    self.repl_run_id,
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
                    self._repl_speaker_observer.shutdown()
                except Exception as exc:  # noqa: BLE001 - optional sidecar
                    _LOG.warning(
                        "speaker_shadow_shutdown_failed run_id=%s error_type=%s",
                        self.repl_run_id,
                        type(exc).__name__,
                    )


ShadowASRManager = ReplASRManager


def build_repl_asr_manager(
    asr_config: Any,
    *,
    speaker_enabled: Optional[bool] = None,
    shadow: bool = False,
    speaker_debug: bool = False,
    speaker_config: Optional[SpeakerConfig] = None,
    observer_factory: Optional[Callable[[SpeakerConfig], Any]] = None,
    enrollment_policy: Optional[EnrollmentPolicy] = None,
    **asr_kwargs: Any,
) -> ASRManager:
    """Build the original ASR when disabled; fail open to ASR on setup errors.

    ``shadow`` is retained as a deprecated spelling of ``speaker_enabled`` so
    existing callers keep working; explicit ``speaker_enabled`` wins.
    """
    if speaker_enabled is None:
        speaker_enabled = bool(shadow)
    if not speaker_enabled:
        return ASRManager(asr_config, **asr_kwargs)
    try:
        config = (
            speaker_config
            if speaker_config is not None
            else load_speaker_config(debug=speaker_debug)
        )
        if observer_factory is not None:
            observer = observer_factory(config)
        else:
            observer = ReplSpeakerObserver(
                config, enrollment_policy=enrollment_policy
            )
    except Exception as exc:  # noqa: BLE001 - bad optional setup
        _LOG.warning(
            "speaker_shadow_setup_failed error_type=%s fallback=asr_only",
            type(exc).__name__,
        )
        return ASRManager(asr_config, **asr_kwargs)

    manager = ReplASRManager(
        asr_config, speaker_observer=observer, **asr_kwargs
    )
    _LOG.info(
        "speaker_identity_created run_id=%s profile_mode=%s "
        "model_key=%s effects=identity_updates",
        manager.repl_run_id,
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
    "DEFAULT_MODE",
    "MODE_ACTIVE",
    "MODE_OFF",
    "MODE_SHADOW",
    "REPL_SPEAKER_MODE_ENV",
    "ReplASRManager",
    "ReplSpeakerObserver",
    "ShadowASRManager",
    "ShadowSpeakerObserver",
    "build_repl_asr_manager",
    "log_shadow_final",
    "repl_speaker_mode",
]
