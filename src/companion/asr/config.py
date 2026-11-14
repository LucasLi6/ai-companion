"""Configuration for the independent real-time ASR demo.

This package is intentionally separate from Ava's Speaker, Router, Memory,
TTS and Orchestrator code. It only turns microphone audio into terminal text;
nothing here imports or is imported by those systems.

The DashScope credential is read from the environment (or a git-ignored
``.env`` file) and is never printed or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DASHSCOPE_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_WORKSPACE_ENV = "DASHSCOPE_WORKSPACE_ID"

DEFAULT_ENV_FILE = ".env"
ENV_FILE_OVERRIDE = "AVA_ENV_FILE"

# V0.9.9.18 specifies this streaming model. The installed DashScope 2.0.4
# resource API (client.audio.asr.recognize) is the verified streaming entry
# point; the model name can be overridden with AVA_ASR_MODEL without touching
# code if the account exposes a different streaming model.
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-message"
DEFAULT_REGION = "cn-beijing"

# Qwen streaming ASR receives 16 kHz, mono, signed 16-bit little-endian PCM.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Silero VAD runs on fixed 512-sample windows (32 ms at 16 kHz).
VAD_WINDOW_SAMPLES = 512
VAD_WINDOW_MS = 32

DEFAULT_VAD_THRESHOLD = 0.50
DEFAULT_MIN_SPEECH_MS = 160
DEFAULT_END_SILENCE_MS = 700
DEFAULT_PRE_ROLL_MS = 350

# Connection/handshake bound for the provider client (seconds).
DEFAULT_TIMEOUT_SECONDS = 30.0
# After all audio was sent and stop() was called, how long to wait for the
# final result.
DEFAULT_FINAL_TIMEOUT_SECONDS = 10.0
# While audio is being uploaded, how long the server may stay silent (no
# event of any kind) before the stream is declared dead. Kept below the
# DashScope SDK's internal 23 s no-audio timeout so our failure is explicit.
DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_CONCURRENT_SESSIONS = 2

# Bounded queues keep a slow network or terminal from blocking capture.
DEFAULT_MIC_QUEUE_WINDOWS = 128
DEFAULT_ASR_SEND_QUEUE_FRAMES = 512
DEFAULT_RENDER_QUEUE_EVENTS = 1024

# One handshake plus a long utterance should never grow memory without bound.
DEFAULT_MAX_UTTERANCE_SECONDS = 120


class ASRError(Exception):
    """Base failure for the independent ASR demo."""


class ASRConfigError(ASRError):
    """Missing or invalid configuration (message is safe to display)."""


@dataclass(frozen=True)
class ASRConfig:
    api_key: str
    workspace_id: str
    model: str = DEFAULT_MODEL
    region: str = DEFAULT_REGION
    device: str = ""
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    end_silence_ms: int = DEFAULT_END_SILENCE_MS
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS
    vad_model_path: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    final_timeout_seconds: float = DEFAULT_FINAL_TIMEOUT_SECONDS
    stream_idle_timeout_seconds: float = DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS
    max_concurrent_sessions: int = DEFAULT_MAX_CONCURRENT_SESSIONS
    mic_queue_windows: int = DEFAULT_MIC_QUEUE_WINDOWS
    asr_send_queue_frames: int = DEFAULT_ASR_SEND_QUEUE_FRAMES
    render_queue_events: int = DEFAULT_RENDER_QUEUE_EVENTS
    max_utterance_seconds: int = DEFAULT_MAX_UTTERANCE_SECONDS
    debug: bool = False

    @property
    def sample_rate(self) -> int:
        # Fixed by the ASR service. Kept as a property so the dataclass cannot
        # be constructed with a sample rate the provider did not accept.
        return SAMPLE_RATE

    @property
    def channels(self) -> int:
        return CHANNELS

    @property
    def sample_width(self) -> int:
        return SAMPLE_WIDTH

    @property
    def window_bytes(self) -> int:
        return VAD_WINDOW_SAMPLES * SAMPLE_WIDTH


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default) or "").strip()


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read simple ``KEY=VALUE`` lines; values are never logged."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _env_file_candidates() -> list[Path]:
    override = os.environ.get(ENV_FILE_OVERRIDE, "").strip()
    if override:
        return [Path(override).expanduser()]
    repo_root = Path(__file__).resolve().parents[3]
    return [Path.cwd() / DEFAULT_ENV_FILE, repo_root / DEFAULT_ENV_FILE]


def load_local_env(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Populate missing process variables from a local ``.env`` file."""
    if env is not None:
        return []
    for path in _env_file_candidates():
        if not path.is_file():
            continue
        loaded: list[str] = []
        for key, value in _parse_env_file(path).items():
            if os.environ.get(key):
                continue
            os.environ[key] = value
            loaded.append(key)
        return loaded
    return []


def _int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = _env(env, name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ASRConfigError(f"{name} 必须是整数。") from exc
    if value < minimum:
        raise ASRConfigError(f"{name} 必须大于等于 {minimum}。")
    return value


def _float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw = _env(env, name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ASRConfigError(f"{name} 必须是数字。") from exc
    if value <= minimum:
        raise ASRConfigError(f"{name} 必须大于 {minimum}。")
    return value


def load_asr_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    device: Optional[str] = None,
    debug: bool = False,
    model: Optional[str] = None,
) -> ASRConfig:
    """Return validated settings or raise :class:`ASRConfigError`.

    Only the *names* of missing variables are reported; values are never
    printed or logged.
    """
    if env is None:
        load_local_env()
        env = os.environ

    missing = [
        name
        for name in (DASHSCOPE_KEY_ENV, DASHSCOPE_WORKSPACE_ENV)
        if not _env(env, name)
    ]
    if missing:
        raise ASRConfigError(
            "ASR 配置缺失: " + ", ".join(missing)
            + "。请先设置环境变量后再运行实时识别 Demo。"
        )

    raw_threshold = _env(
        env, "AVA_ASR_THRESHOLD", str(DEFAULT_VAD_THRESHOLD)
    )
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ASRConfigError("AVA_ASR_THRESHOLD 必须是数字。") from exc
    if not 0.0 < threshold < 1.0:
        raise ASRConfigError("AVA_ASR_THRESHOLD 必须在 0 和 1 之间（不含端点）。")

    min_speech_ms = _int_env(
        env, "AVA_ASR_MIN_SPEECH_MS", DEFAULT_MIN_SPEECH_MS, minimum=VAD_WINDOW_MS
    )
    end_silence_ms = _int_env(
        env, "AVA_ASR_END_SILENCE_MS",
        DEFAULT_END_SILENCE_MS, minimum=VAD_WINDOW_MS,
    )
    pre_roll_ms = _int_env(
        env, "AVA_ASR_PRE_ROLL_MS", DEFAULT_PRE_ROLL_MS, minimum=0
    )

    timeout_seconds = _float_env(
        env, "AVA_ASR_TIMEOUT", DEFAULT_TIMEOUT_SECONDS, minimum=0.0
    )
    final_timeout_seconds = _float_env(
        env, "AVA_ASR_FINAL_TIMEOUT",
        DEFAULT_FINAL_TIMEOUT_SECONDS, minimum=0.0,
    )
    stream_idle_timeout_seconds = _float_env(
        env, "AVA_ASR_STREAM_IDLE_TIMEOUT",
        DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS, minimum=0.0,
    )
    max_concurrent = _int_env(
        env, "AVA_ASR_MAX_SESSIONS",
        DEFAULT_MAX_CONCURRENT_SESSIONS, minimum=1,
    )

    chosen_model = (model or "").strip() or _env(
        env, "AVA_ASR_MODEL", DEFAULT_MODEL
    ) or DEFAULT_MODEL
    chosen_device = (
        device if device is not None else _env(env, "AVA_ASR_DEVICE", "")
    ).strip()

    return ASRConfig(
        api_key=_env(env, DASHSCOPE_KEY_ENV),
        workspace_id=_env(env, DASHSCOPE_WORKSPACE_ENV),
        model=chosen_model,
        region=_env(env, "AVA_ASR_REGION", DEFAULT_REGION) or DEFAULT_REGION,
        device=chosen_device,
        vad_threshold=threshold,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
        pre_roll_ms=pre_roll_ms,
        vad_model_path=_env(env, "AVA_ASR_VAD_MODEL", ""),
        timeout_seconds=timeout_seconds,
        final_timeout_seconds=final_timeout_seconds,
        stream_idle_timeout_seconds=stream_idle_timeout_seconds,
        max_concurrent_sessions=max_concurrent,
        debug=debug,
    )


__all__ = [
    "ASRConfig",
    "ASRConfigError",
    "ASRError",
    "CHANNELS",
    "DASHSCOPE_KEY_ENV",
    "DASHSCOPE_WORKSPACE_ENV",
    "DEFAULT_END_SILENCE_MS",
    "DEFAULT_ENV_FILE",
    "DEFAULT_FINAL_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONCURRENT_SESSIONS",
    "DEFAULT_MIN_SPEECH_MS",
    "DEFAULT_MODEL",
    "DEFAULT_PRE_ROLL_MS",
    "DEFAULT_REGION",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_VAD_THRESHOLD",
    "ENV_FILE_OVERRIDE",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "VAD_WINDOW_MS",
    "VAD_WINDOW_SAMPLES",
    "load_asr_config",
    "load_local_env",
]
