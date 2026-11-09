"""Compare TTS sentence modes with the same model, voice and text.

This runs the real Phase B pipeline (``TTSManager`` -> ``TextSegmenter`` ->
``TTSClient``) with a capture player, so it exercises the same sentence-first
segmentation and flush strategy as the REPL. It writes one WAV per mode so
intonation continuity has to be judged by listening, not by mock tests. Only
the TTS transport is touched (never Router/Speaker/Memory), the text is
synthetic, and the API key is never printed. It is deliberately NOT part of
pytest:

    .venv/bin/python scripts/compare_tts_modes.py \
        --text "今天天气不错，我们出去走走吧，顺便喝杯咖啡。" \
        --out /tmp/tts_compare

Modes (same model qwen-audio-3.1-tts-flash, same voice anruorou_v3.1,
same rate 1.03):

  B_no_instruction    sentence-first, auto flush, no instruction
  C_instruction       sentence-first, auto flush, Chinese instruction
  C_sentence_flush    sentence-first, flush-per-sentence, instruction

Mode A (legacy comma segmentation + per-fragment flush) no longer exists in
the codebase and is intentionally not reproduced here.

Reported timings are separate on purpose:

  handshake_ms                 open_stream() until the session handshake ends
  first_pcm_after_sentence_ms  last sentence sent until the first PCM arrives
  user_to_first_audio_ms       submission until the first PCM is written
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import wave
from pathlib import Path

from companion.tts.client import TTSClient
from companion.tts.config import load_tts_config
from companion.tts.manager import TTSManager


class CapturePlayer:
    """Collects PCM in play order instead of using an audio device."""

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def write(self, data: bytes) -> None:
        self.pcm.extend(data)

    def interrupt(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _synthesize(config, text: str) -> tuple[bytes, dict]:
    player = CapturePlayer()
    manager = TTSManager(
        config_loader=lambda: config,
        client_factory=lambda cfg: TTSClient(cfg),
        player_factory=lambda cfg: player,
        join_timeout=5.0,
    )
    manager.start()
    try:
        submitted = time.monotonic()
        stream = manager.start_stream(
            "compare", "compare", submitted_at=submitted
        )
        if stream is None:
            raise RuntimeError("TTS 未启动")
        stream.feed(text)
        stream.finish()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if manager.status().state == "error":
                raise RuntimeError(manager.last_error or "TTS error")
            metrics = manager.metrics(stream.generation)
            if metrics and metrics.get("playback_finished"):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("playback did not finish")
        return bytes(player.pcm), metrics or {}
    finally:
        manager.shutdown()


def _ms(value) -> str:
    return "-" if value is None else f"{value:.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default="今天天气不错，我们出去走走吧，顺便喝杯咖啡。",
        help="synthetic sentence(s), no private data",
    )
    parser.add_argument("--out", default="/tmp/tts_compare", help="output dir")
    parser.add_argument(
        "--play",
        action="store_true",
        help="play each WAV with macOS afplay after writing it",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_tts_config()
    modes = [
        (
            "B_no_instruction",
            dataclasses.replace(base, instruction="", flush_strategy="auto"),
        ),
        (
            "C_instruction",
            dataclasses.replace(base, flush_strategy="auto"),
        ),
        (
            "C_sentence_flush",
            dataclasses.replace(base, flush_strategy="sentence_flush"),
        ),
    ]

    for name, config in modes:
        try:
            pcm, metrics = _synthesize(config, args.text)
        except Exception as exc:  # noqa: BLE001 - one mode may fail
            print(f"{name}: FAILED {type(exc).__name__}")
            continue
        wav_path = out_dir / f"{name}.wav"
        _write_wav(wav_path, pcm, config.sample_rate)
        duration = len(pcm) / (config.sample_rate * 2)
        print(
            f"{name}: handshake={_ms(metrics.get('tts_connect_ms'))}"
            f"  first_pcm_after_sentence="
            f"{_ms(metrics.get('first_segment_to_first_pcm_ms'))}"
            f"  user_to_first_audio={_ms(metrics.get('user_to_first_audio_ms'))}"
            f"  audio={duration:.2f}s  pcm_bytes={len(pcm)}"
            f"  -> {wav_path}"
        )
        if args.play:
            import shutil
            import subprocess

            if shutil.which("afplay"):
                subprocess.run(
                    ["afplay", str(wav_path)],
                    check=False,
                )
            else:
                print("  (afplay not found; open the WAV manually)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
