"""Configuration for optional real-time barge-in (WebRTC AEC3).

The feature is off by default. It only ever adds a detection channel next to
the existing half-duplex voice stack: when it is disabled, or its optional
dependency is missing, the ASR manager keeps pausing input while Ava speaks
exactly as before.

No secret or audio content is stored here; every value is a bounded number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

ENABLED_ENV = "AVA_BARGE_IN_ENABLED"
VAD_THRESHOLD_ENV = "AVA_BARGE_IN_VAD_THRESHOLD"
MIN_SPEECH_MS_ENV = "AVA_BARGE_IN_MIN_SPEECH_MS"
PRE_ROLL_MS_ENV = "AVA_BARGE_IN_PREROLL_MS"
# AVA_BARGE_IN_AEC_DELAY_MS is only the WebRTC AEC3 ``stream_delay_ms`` hint
# (the device buffer latency the model uses internally). It is never applied
# to the reference/mic timeline as well.
AEC_DELAY_MS_ENV = "AVA_BARGE_IN_AEC_DELAY_MS"
# Optional, independent calibration of the reference timeline. Use this (not
# AEC_DELAY_MS) if measured hardware latency requires shifting which playback
# frame is paired with a microphone block.
TIMELINE_OFFSET_MS_ENV = "AVA_BARGE_IN_TIMELINE_OFFSET_MS"

DEFAULT_ENABLED = False
DEFAULT_VAD_THRESHOLD = 0.65
DEFAULT_MIN_SPEECH_MS = 192
DEFAULT_PRE_ROLL_MS = 400
DEFAULT_AEC_DELAY_MS = 0
DEFAULT_TIMELINE_OFFSET_MS = 0.0
# While the user's interrupting utterance is being recognized, keep the
# echo-cancelled path for the decaying playback tail before switching the ASR
# input back to its unchanged normal path.
DEFAULT_ECHO_TAIL_MS = 300

# Silero operates on 32 ms windows; min speech must cover at least one.
VAD_WINDOW_MS = 32

_FALSE = {"0", "false", "off", "no", "disabled", ""}


class BargeInConfigError(ValueError):
    """One barge-in environment value is invalid (display-safe)."""


@dataclass(frozen=True)
class BargeInConfig:
    enabled: bool = DEFAULT_ENABLED
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS
    aec_delay_ms: int = DEFAULT_AEC_DELAY_MS
    # Measured alignment calibration; separate from the AEC3 device hint so a
    # single latency value is never compensated twice.
    timeline_offset_ms: float = DEFAULT_TIMELINE_OFFSET_MS
    echo_tail_ms: int = DEFAULT_ECHO_TAIL_MS

    @property
    def echo_tail_seconds(self) -> float:
        return max(0.0, self.echo_tail_ms / 1000.0)

    @property
    def aec_delay_seconds(self) -> float:
        return max(0, self.aec_delay_ms) / 1000.0


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return str(env.get(name, default) or "").strip()


def _int_env(
    env: Mapping[str, str], name: str, default: int, *, minimum: int
) -> int:
    raw = _env(env, name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BargeInConfigError(f"{name} 必须是整数。") from exc
    if value < minimum:
        raise BargeInConfigError(f"{name} 必须大于等于 {minimum}。")
    return value


def _float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = _env(env, name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BargeInConfigError(f"{name} 必须是数字。") from exc
    if not minimum <= value <= maximum:
        raise BargeInConfigError(
            f"{name} 必须在 {minimum} 和 {maximum} 之间。"
        )
    return value


def load_barge_in_config(
    env: Optional[Mapping[str, str]] = None,
) -> BargeInConfig:
    """Read barge-in settings; invalid values raise :class:`BargeInConfigError`.

    The caller (the REPL) treats a config error as "barge-in unavailable" and
    keeps the existing half-duplex voice path running.
    """
    if env is None:
        env = os.environ

    enabled_raw = _env(env, ENABLED_ENV, "false").lower()
    enabled = enabled_raw not in _FALSE

    threshold_raw = _env(
        env, VAD_THRESHOLD_ENV, str(DEFAULT_VAD_THRESHOLD)
    )
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise BargeInConfigError(
            f"{VAD_THRESHOLD_ENV} 必须是数字。"
        ) from exc
    if not 0.0 < threshold < 1.0:
        raise BargeInConfigError(
            f"{VAD_THRESHOLD_ENV} 必须在 0 和 1 之间（不含端点）。"
        )

    return BargeInConfig(
        enabled=enabled,
        vad_threshold=threshold,
        min_speech_ms=_int_env(
            env, MIN_SPEECH_MS_ENV, DEFAULT_MIN_SPEECH_MS,
            minimum=VAD_WINDOW_MS,
        ),
        pre_roll_ms=_int_env(
            env, PRE_ROLL_MS_ENV, DEFAULT_PRE_ROLL_MS, minimum=0
        ),
        aec_delay_ms=_int_env(
            env, AEC_DELAY_MS_ENV, DEFAULT_AEC_DELAY_MS, minimum=0
        ),
        timeline_offset_ms=_float_env(
            env,
            TIMELINE_OFFSET_MS_ENV,
            DEFAULT_TIMELINE_OFFSET_MS,
            minimum=-500.0,
            maximum=500.0,
        ),
        echo_tail_ms=_int_env(
            env, "AVA_BARGE_IN_ECHO_TAIL_MS", DEFAULT_ECHO_TAIL_MS,
            minimum=0,
        ),
    )


__all__ = [
    "AEC_DELAY_MS_ENV",
    "BargeInConfig",
    "BargeInConfigError",
    "DEFAULT_AEC_DELAY_MS",
    "DEFAULT_ECHO_TAIL_MS",
    "DEFAULT_ENABLED",
    "DEFAULT_MIN_SPEECH_MS",
    "DEFAULT_PRE_ROLL_MS",
    "DEFAULT_TIMELINE_OFFSET_MS",
    "DEFAULT_VAD_THRESHOLD",
    "ENABLED_ENV",
    "MIN_SPEECH_MS_ENV",
    "PRE_ROLL_MS_ENV",
    "TIMELINE_OFFSET_MS_ENV",
    "VAD_THRESHOLD_ENV",
    "load_barge_in_config",
]
