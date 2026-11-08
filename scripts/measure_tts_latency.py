"""Measure real DashScope TTS handshake and first-audio latency.

This script only exercises the TTS transport: it never touches Router,
Speaker, Memory, Emotion or the companion database, and it sends synthetic
text only (no private conversation). It is deliberately NOT part of pytest:

    .venv/bin/python scripts/measure_tts_latency.py --samples 10

Credentials are read from the environment or the git-ignored local ``.env``;
the key is never printed. Each sample opens a fresh Phase B streaming session
(the same per-reply path as the REPL) and records:

  handshake_ms        connect() + start() until the task-started handshake
  first_pcm_ms        send_text()+flush() until the first PCM chunk arrives
  connect_to_pcm_ms   handshake + first PCM

For the full user-facing pipeline (Enter -> first text -> first audio), use the
REPL: every completed reply prints

    [timing] first_text=... | text_done=... | first_audio=...

and ``/tts status`` prints the detailed phase breakdown for the last reply.
"""

from __future__ import annotations

import argparse
import math
import time

from companion.tts.client import TTSClient
from companion.tts.config import load_tts_config

# Varying lengths, all synthetic and free of personal data.
TEXTS = (
    "你好呀。",
    "今天天气不错，我们出去走走吧。",
    "我刚看了一本很有意思的书，讲的是城市里的小人物，读起来很温暖。",
    (
        "如果要把这件事说明白，可能需要先讲清楚背景："
        "这是在一个很普通的下午发生的，当时没有人在意，"
        "但后来它变成了一段很重要的记忆。"
    ),
    (
        "今天想和你聊一个有点长的话题。我们总是急着把事情做完，"
        "却很少停下来想想为什么做。也许慢一点并不是浪费时间，"
        "而是给心里留出一点空间。你觉得呢？"
    ),
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")

    config = load_tts_config()
    client = TTSClient(config)
    handshakes: list[float] = []
    first_pcms: list[float] = []
    totals: list[float] = []
    failures = 0

    try:
        for index in range(args.samples):
            text = TEXTS[index % len(TEXTS)]
            session = None
            chunks = None
            try:
                started = time.monotonic()
                session = client.open_stream()
                handshake = (time.monotonic() - started) * 1000

                sent = time.monotonic()
                session.send_text(text)
                session.flush()
                chunks = session.audio_chunks()
                first = next(chunks)
                first_pcm = (time.monotonic() - sent) * 1000
                byte_count = len(first) if isinstance(first, bytes) else 0

                handshakes.append(handshake)
                first_pcms.append(first_pcm)
                totals.append(handshake + first_pcm)
                print(
                    f"sample {index + 1:02d}: handshake={handshake:7.1f}ms"
                    f" first_pcm={first_pcm:7.1f}ms"
                    f" total={handshake + first_pcm:7.1f}ms"
                    f" bytes={byte_count} chars={len(text)}"
                )
            except Exception as exc:  # noqa: BLE001 - one sample may fail
                failures += 1
                print(f"sample {index + 1:02d}: FAILED {type(exc).__name__}")
            finally:
                if chunks is not None:
                    close = getattr(chunks, "close", None)
                    if callable(close):
                        close()
                if session is not None:
                    for name in ("stop", "close"):
                        method = getattr(session, name, None)
                        if callable(method):
                            try:
                                method()
                            except Exception:  # noqa: BLE001 - cleanup
                                pass
    finally:
        client.close()

    if not handshakes:
        print("no successful samples")
        return 1
    for label, values in (
        ("handshake_ms", handshakes),
        ("first_pcm_ms", first_pcms),
        ("connect_to_pcm_ms", totals),
    ):
        mean = sum(values) / len(values)
        print(
            f"{label}: n={len(values)} mean={mean:.1f}"
            f" p95={_percentile(values, 0.95):.1f}"
            f" min={min(values):.1f} max={max(values):.1f}"
        )
    if failures:
        print(f"failures={failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
