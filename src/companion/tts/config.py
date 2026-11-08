"""TTS configuration for the independent Qwen speech output.

Phase A keeps text generation and speech synthesis completely separate: this
module only reads environment settings, it never imports the Router, Speaker,
Memory or Emotion code. No credential, audio or user text is logged here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from companion.tts.errors import TTSError

DASHSCOPE_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_WORKSPACE_ENV = "DASHSCOPE_WORKSPACE_ID"

# Optional local file loaded at startup so the REPL works without exporting
# variables in the terminal. The file itself is git-ignored and never printed.
DEFAULT_ENV_FILE = ".env"
ENV_FILE_OVERRIDE = "AVA_ENV_FILE"

DEFAULT_MODEL = "qwen-audio-3.1-tts-flash"
DEFAULT_VOICE = "xuyanchu_v3.1"
DEFAULT_REGION = "cn-beijing"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FORMAT = "pcm"

# DashScope accepts 0.5~2.0 for the speech rate (1.0 = normal speed).
DEFAULT_SPEECH_RATE = 1.07
MIN_SPEECH_RATE = 0.5
MAX_SPEECH_RATE = 2.0

# Phase B splits the display stream into speakable chunks. These are legacy
# values; segmentation is now sentence-first and no longer force-cuts at
# ``segment_max_chars``. The active long-fallback threshold is below.
DEFAULT_SEGMENT_MIN_CHARS = 6
DEFAULT_SEGMENT_MAX_CHARS = 35

# One sentence must sound like one continuous, natural utterance. This
# instruction is sent to the provider once per session, so the voice does not
# re-read every fragment or re-tune after each comma.
DEFAULT_TTS_INSTRUCTION = (
    "请以自然、真实、轻松的日常交谈方式说话，而不是朗读文本。"
    "保持柔和、略偏低沉、自然亲近的女性声线。"
    "同一句话应作为完整表达，保持连贯的语气、情绪意图、基础语速和声音质感。"
    "根据整句话的语义自然安排重音、停顿、音高和节奏，使语调平滑发展和收束。"
    "允许自然的抑扬顿挫，但不要在句子中途突然改变语速、音调、音量或说话方式，"
    "也不要在逗号后重新起调。"
    "表达应当生动、细腻且自然，避免机械朗读、夸张表演、播音腔和客服腔。"
    "遵循文本语言的发音与韵律规律，尊重明确的情绪标签，"
    "不要无故切换情绪，也不要改写原文。"
)

# Phase B flush strategy. ``auto`` sends a complete sentence and lets the
# server start synthesis on its own; ``sentence_flush`` flushes after every
# sentence for A/B comparison. Neither ever flushes after a plain comma.
DEFAULT_FLUSH_STRATEGY = "auto"
FLUSH_STRATEGIES = ("auto", "sentence_flush")

# Long fallback: only when this many visible characters accumulate without a
# sentence end does the segmenter cut at the nearest safe clause boundary.
# This is a safety valve, not a normal per-sentence threshold.
DEFAULT_LONG_FALLBACK_CHARS = 180

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
    speech_rate: float = DEFAULT_SPEECH_RATE
    instruction: str = DEFAULT_TTS_INSTRUCTION
    flush_strategy: str = DEFAULT_FLUSH_STRATEGY
    long_fallback_chars: int = DEFAULT_LONG_FALLBACK_CHARS
    # Empty = follow the system default output device (including Bluetooth).
    # Otherwise a PortAudio output index ("1") or a name substring
    # ("WH-1000XM6", "蓝牙") selects a specific device.
    output_device: str = ""

    @property
    def channels(self) -> int:
        return 1

    @property
    def sample_width(self) -> int:
        return 2


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
    # cwd first (running from the project), then this checkout's root.
    repo_root = Path(__file__).resolve().parents[3]
    return [Path.cwd() / DEFAULT_ENV_FILE, repo_root / DEFAULT_ENV_FILE]


def load_local_env(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Populate missing process variables from a local ``.env`` file.

    Called only when the caller did not supply an explicit mapping, so tests
    and library callers stay deterministic. An explicitly exported variable
    always wins; the file path can be overridden with ``AVA_ENV_FILE``.
    Returns the names that were loaded (never their values).
    """
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


def describe_tts_config(
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Non-secret settings for ``/tts status`` (never includes the key)."""
    if env is None:
        load_local_env()
        env = os.environ
    model = _env(env, "AVA_TTS_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    voice = _env(env, "AVA_TTS_VOICE", DEFAULT_VOICE) or DEFAULT_VOICE
    region = _env(env, "AVA_TTS_REGION", DEFAULT_REGION) or DEFAULT_REGION
    raw_rate = _env(env, "AVA_TTS_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE))
    try:
        sample_rate = int(raw_rate)
    except (TypeError, ValueError):
        sample_rate = DEFAULT_SAMPLE_RATE
    raw_speech_rate = _env(env, "AVA_TTS_RATE", str(DEFAULT_SPEECH_RATE))
    try:
        speech_rate = float(raw_speech_rate)
    except (TypeError, ValueError):
        speech_rate = DEFAULT_SPEECH_RATE
    flush_strategy = _env(
        env, "AVA_TTS_FLUSH_STRATEGY", DEFAULT_FLUSH_STRATEGY
    )
    return {
        "model": model,
        "voice": voice,
        "region": region,
        "sample_rate": sample_rate,
        "audio_format": DEFAULT_FORMAT,
        "rate": speech_rate,
        "flush_strategy": flush_strategy,
        "output_device": _env(env, "AVA_TTS_OUTPUT_DEVICE", ""),
    }


def load_tts_config(
    env: Optional[Mapping[str, str]] = None,
) -> TTSConfig:
    """Return validated settings or raise :class:`TTSConfigError`.

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

    raw_speech_rate = _env(env, "AVA_TTS_RATE", str(DEFAULT_SPEECH_RATE))
    try:
        speech_rate = float(raw_speech_rate)
    except (TypeError, ValueError) as exc:
        raise TTSConfigError("AVA_TTS_RATE 必须是数字。") from exc
    if not MIN_SPEECH_RATE <= speech_rate <= MAX_SPEECH_RATE:
        raise TTSConfigError(
            f"AVA_TTS_RATE 必须在 {MIN_SPEECH_RATE}~{MAX_SPEECH_RATE} 之间。"
        )

    # Absent -> default instruction; explicitly empty -> no instruction, so
    # the A/B listening test can compare "with vs without instruction" without
    # editing code.
    raw_instruction = env.get("AVA_TTS_INSTRUCTION")
    if raw_instruction is None:
        instruction = DEFAULT_TTS_INSTRUCTION
    else:
        instruction = str(raw_instruction).strip()

    flush_strategy = _env(
        env, "AVA_TTS_FLUSH_STRATEGY", DEFAULT_FLUSH_STRATEGY
    ) or DEFAULT_FLUSH_STRATEGY
    if flush_strategy not in FLUSH_STRATEGIES:
        raise TTSConfigError(
            "AVA_TTS_FLUSH_STRATEGY 必须是 "
            + " / ".join(FLUSH_STRATEGIES)
            + " 之一。"
        )

    raw_fallback = _env(
        env, "AVA_TTS_LONG_FALLBACK_CHARS", str(DEFAULT_LONG_FALLBACK_CHARS)
    )
    try:
        long_fallback_chars = int(raw_fallback)
    except (TypeError, ValueError) as exc:
        raise TTSConfigError(
            "AVA_TTS_LONG_FALLBACK_CHARS 必须是整数。"
        ) from exc
    if long_fallback_chars < 1:
        raise TTSConfigError("AVA_TTS_LONG_FALLBACK_CHARS 必须是正整数。")

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
        speech_rate=speech_rate,
        instruction=instruction,
        flush_strategy=flush_strategy,
        long_fallback_chars=long_fallback_chars,
        output_device=_env(env, "AVA_TTS_OUTPUT_DEVICE", ""),
    )


def tts_enabled_by_default(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Phase B default: ON unless ``AVA_TTS_ENABLED`` is explicitly false."""
    if env is None:
        load_local_env()
        env = os.environ
    raw = _env(env, "AVA_TTS_ENABLED", "true").lower()
    return raw not in ("0", "false", "off", "no", "disabled")


__all__ = [
    "DASHSCOPE_KEY_ENV",
    "DASHSCOPE_WORKSPACE_ENV",
    "DEFAULT_ENV_FILE",
    "DEFAULT_FLUSH_STRATEGY",
    "DEFAULT_FORMAT",
    "DEFAULT_LONG_FALLBACK_CHARS",
    "DEFAULT_MODEL",
    "DEFAULT_REGION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SEGMENT_MAX_CHARS",
    "DEFAULT_SEGMENT_MIN_CHARS",
    "DEFAULT_SPEECH_RATE",
    "DEFAULT_TTS_INSTRUCTION",
    "DEFAULT_VOICE",
    "ENV_FILE_OVERRIDE",
    "FLUSH_STRATEGIES",
    "MAX_SPEECH_RATE",
    "MAX_TEXT_CHARS",
    "MIN_SPEECH_RATE",
    "TTSConfig",    "TTSConfigError",
    "describe_tts_config",
    "load_local_env",
    "load_tts_config",
    "tts_enabled_by_default",
]
