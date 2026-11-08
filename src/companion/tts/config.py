"""TTS configuration for the independent Qwen speech output.

Phase A keeps text generation and speech synthesis completely separate: this
module only reads environment settings, it never imports the Router, Speaker,
Memory or Emotion code. No credential, audio or user text is logged here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from companion.tts.errors import TTSError

DASHSCOPE_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_WORKSPACE_ENV = "DASHSCOPE_WORKSPACE_ID"

DEFAULT_MODEL = "qwen-audio-3.1-tts-flash"
DEFAULT_VOICE = "xuyanchu_v3.1"
DEFAULT_REGION = "cn-beijing"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FORMAT = "pcm"

# Phase B splits the display stream into speakable chunks. These are initial
# values; adjust them from real latency/quality measurements, not in the
# segmentation logic itself.
DEFAULT_SEGMENT_MIN_CHARS = 10
DEFAULT_SEGMENT_MAX_CHARS = 60

# Phase B: speech output is on by default for the REPL. ``AVA_TTS_ENABLED``
# only controls the session default, not a persisted setting.
DEFAULT_ENABLED = True

# Phase A submits one complete reply per request. A reply longer than this is
# refused explicitly (never silently truncated).
MAX_TEXT_CHARS = 2000


class TTSConfigError(TTSError):
    """Missing or invalid TTS configuration (message is safe to display)."""


@dataclass(frozen=True)
class TTSConfig:
    api_key: str
    workspace_id: str
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    region: str = DEFAULT_REGION
    sample_rate: int = DEFAULT_SAMPLE_RATE
    audio_format: str = DEFAULT_FORMAT
    max_text_chars: int = MAX_TEXT_CHARS
    segment_min_chars: int = DEFAULT_SEGMENT_MIN_CHARS
    segment_max_chars: int = DEFAULT_SEGMENT_MAX_CHARS

    @property
    def channels(self) -> int:
        return 1

    @property
    def sample_width(self) -> int:
        return 2


def _env(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default) or "").strip()


def describe_tts_config(
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Non-secret settings for ``/tts status`` (never includes the key)."""
    env = os.environ if env is None else env
    model = _env(env, "AVA_TTS_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    voice = _env(env, "AVA_TTS_VOICE", DEFAULT_VOICE) or DEFAULT_VOICE
    region = _env(env, "AVA_TTS_REGION", DEFAULT_REGION) or DEFAULT_REGION
    raw_rate = _env(env, "AVA_TTS_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE))
    try:
        sample_rate = int(raw_rate)
    except (TypeError, ValueError):
        sample_rate = DEFAULT_SAMPLE_RATE
    return {
        "model": model,
        "voice": voice,
        "region": region,
        "sample_rate": sample_rate,
        "audio_format": DEFAULT_FORMAT,
    }


def load_tts_config(
    env: Optional[Mapping[str, str]] = None,
) -> TTSConfig:
    """Return validated settings or raise :class:`TTSConfigError`.

    Only the *names* of missing variables are reported; values are never
    printed or logged.
    """
    env = os.environ if env is None else env
    missing = [
        name
        for name in (DASHSCOPE_KEY_ENV, DASHSCOPE_WORKSPACE_ENV)
        if not _env(env, name)
    ]
    if missing:
        raise TTSConfigError(
            "Qwen TTS 配置缺失: " + ", ".join(missing)
            + "。请先设置环境变量后再执行 /tts on。"
        )

    described = describe_tts_config(env)
    raw_rate = _env(env, "AVA_TTS_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE))
    try:
        sample_rate = int(raw_rate)
    except (TypeError, ValueError) as exc:
        raise TTSConfigError("AVA_TTS_SAMPLE_RATE 必须是整数。") from exc
    if sample_rate <= 0:
        raise TTSConfigError("AVA_TTS_SAMPLE_RATE 必须是正整数。")

    raw_min = _env(
        env, "AVA_TTS_SEGMENT_MIN_CHARS", str(DEFAULT_SEGMENT_MIN_CHARS)
    )
    raw_max = _env(
        env, "AVA_TTS_SEGMENT_MAX_CHARS", str(DEFAULT_SEGMENT_MAX_CHARS)
    )
    try:
        segment_min = int(raw_min)
        segment_max = int(raw_max)
    except (TypeError, ValueError) as exc:
        raise TTSConfigError(
            "AVA_TTS_SEGMENT_MIN_CHARS / AVA_TTS_SEGMENT_MAX_CHARS 必须是整数。"
        ) from exc
    if segment_min < 1 or segment_max < segment_min:
        raise TTSConfigError(
            "分段长度无效：需要 1 <= AVA_TTS_SEGMENT_MIN_CHARS"
            " <= AVA_TTS_SEGMENT_MAX_CHARS。"
        )

    return TTSConfig(
        api_key=_env(env, DASHSCOPE_KEY_ENV),
        workspace_id=_env(env, DASHSCOPE_WORKSPACE_ENV),
        model=described["model"],
        voice=described["voice"],
        region=described["region"],
        sample_rate=sample_rate,
        audio_format=DEFAULT_FORMAT,
        segment_min_chars=segment_min,
        segment_max_chars=segment_max,
    )


def tts_enabled_by_default(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Phase B default: ON unless ``AVA_TTS_ENABLED`` is explicitly false."""
    env = os.environ if env is None else env
    raw = _env(env, "AVA_TTS_ENABLED", "true").lower()
    return raw not in ("0", "false", "off", "no", "disabled")


__all__ = [
    "DASHSCOPE_KEY_ENV",
    "DASHSCOPE_WORKSPACE_ENV",
    "DEFAULT_FORMAT",
    "DEFAULT_MODEL",
    "DEFAULT_REGION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SEGMENT_MAX_CHARS",
    "DEFAULT_SEGMENT_MIN_CHARS",
    "DEFAULT_VOICE",
    "MAX_TEXT_CHARS",
    "TTSConfig",
    "TTSConfigError",
    "describe_tts_config",
    "load_tts_config",
    "tts_enabled_by_default",
]
