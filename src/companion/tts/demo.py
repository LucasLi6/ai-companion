"""Standalone TTS demo: keyboard text -> Qwen TTS -> Mac speaker.

This entry point exercises the TTS package without Router, Speaker, Memory,
Emotion or the companion database. It reads credentials only from the
environment and never logs the API key or reply contents.

Usage
-----
    export DASHSCOPE_API_KEY=...
    export DASHSCOPE_WORKSPACE_ID=...
    .venv/bin/python -m companion.tts.demo
    .venv/bin/python -m companion.tts.demo "你好，我是 Ava。"
"""

from __future__ import annotations

import sys
import time

from companion.tts.manager import SpeechJob, TTSManager


def _wait_for_playback(
    manager: TTSManager, generation: int, timeout: float = 60.0
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status()
        if status.state == "error":
            return False
        metrics = manager.metrics()
        if (
            metrics is not None
            and metrics["generation"] >= generation
            and metrics["playback_finished"]
        ):
            return True
        time.sleep(0.05)
    return False


def main(argv: list[str]) -> int:
    manager = TTSManager()
    ok, message = manager.enable()
    print(message)
    if not ok:
        return 1

    try:
        if argv:
            texts = [" ".join(argv)]
        else:
            print("输入要朗读的文字，空行退出。")
            texts = []
            while True:
                try:
                    line = input("tts> ")
                except EOFError:
                    break
                if not line.strip():
                    break
                texts.append(line.strip())

        for index, text in enumerate(texts, start=1):
            accepted = manager.submit(
                SpeechJob(
                    response_id=f"demo-{index}",
                    conversation_id="demo",
                    text=text,
                )
            )
            if not accepted:
                print(f"未提交: {manager.last_error}")
                continue
            generation = manager.generation
            finished = _wait_for_playback(manager, generation)
            metrics = manager.metrics()
            print(
                f"finished={finished} state={manager.status().state} "
                f"metrics={metrics}"
            )
    finally:
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
