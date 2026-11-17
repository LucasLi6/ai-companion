"""Standalone TTS demo: keyboard text -> Qwen TTS -> Mac speaker.

This entry point exercises the TTS package without Router, Speaker, Memory,
Emotion or the companion database. Credentials come from the environment or
from the git-ignored local ``.env`` file (loaded automatically); the API key
and reply contents are never logged.

Usage
-----
    # either export the variables, or put them in ./.env
    .venv/bin/python -m companion.tts.demo
    .venv/bin/python -m companion.tts.demo "你好，我是 Ava。"
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from companion.tts.manager import SpeechJob, TTSManager


def _wait_for_playback(
    manager: TTSManager, generation: int, timeout: float = 60.0
) -> Optional[str]:
    """Wait for the terminal outcome of exactly this generation.

    Reading the manager-wide ``state`` would misreport every reply after the
    first failure: ``state`` stays ``"error"`` until some later reply
    finishes, so a healthy new generation would look failed the moment it was
    submitted. The per-generation outcome answers the only question that
    matters here: did *this* reply play to the end?
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = manager.playback_outcome(generation)
        if outcome is not None:
            return outcome
        time.sleep(0.05)
    return None


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
            outcome = _wait_for_playback(manager, generation)
            finished = outcome == "finished"
            metrics = manager.metrics(generation)
            print(
                f"finished={finished} outcome={outcome} "
                f"state={manager.status().state} metrics={metrics}"
            )
            if outcome == "failed":
                print("last_error:", manager.last_error)
            elif outcome is None:
                print("timed out waiting for this reply to finish")
    finally:
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
