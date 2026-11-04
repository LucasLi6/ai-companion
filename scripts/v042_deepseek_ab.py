#!/usr/bin/env python
"""Optional live DeepSeek A/B for the V0.4.2 affect policy.

This script is **not** part of the test suite and never runs automatically. It
needs a real ``DEEPSEEK_API_KEY`` and makes real network calls. It compares the
current runtime under the two V0.4.2 time-decay modes:

- ``AUXILIARY`` (V0.4.2 default): decay is auxiliary, scene relevance can
  suppress stale affect and a clear new event can take over;
- ``LEGACY``: V0.4.1 behaviour — decay only, no relevance/ranking override.

Everything else (schema 16, response-v7, persona-first prompt, stance
continuity, local safety floor) is identical in both arms. This is therefore an
A/B of the affect policy, not of the whole V0.4.1 vs V0.4.2 codebase.

Usage:
    DEEPSEEK_API_KEY=... .venv/bin/python scripts/v042_deepseek_ab.py \
        --out /tmp/v042_ab.json

The output is a JSON document with, for each arm and turn: the final action,
reply text, persisted emotion state and the captured request types. It is raw
material for human reading; it is not a blind evaluation and the script makes
no quality claim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.runtime import build_runtime
from companion.storage.database import Database

TURNS = [
    "你就是个废物",
    "刚才是我说得太重了，对不起",
    "算了，换个话题，我最近在学吉他",
    "你觉不觉得我这样很没出息",
    "哈哈，你安慰人的方式真够笨的",
    "那你说说，笨在哪儿",
]

ARMS = {"AUXILIARY": "AUXILIARY", "LEGACY": "LEGACY"}


class _RecordingClient:
    """Delegates to the real client and records every request type."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    async def complete(self, messages, *, request_type, **kwargs):
        response = await self.inner.complete(
            messages, request_type=request_type, **kwargs
        )
        self.calls.append(
            {
                "request_type": request_type,
                "event_id": kwargs.get("event_id"),
                "messages": messages,
                "response_preview": response.content[:400],
            }
        )
        return response


async def _run_arm(name: str, mode: str, key: str) -> dict:
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["EMOTION_TIME_DECAY_MODE"] = mode

    from companion.providers.deepseek_client import build_deepseek_client

    with tempfile.TemporaryDirectory(prefix=f"v042_ab_{name.lower()}_") as tmp:
        db = await Database(Path(tmp) / "ab.db").connect()
        await db.initialize()
        client = _RecordingClient(build_deepseek_client(_config()))
        rt = build_runtime(db, provider="deepseek", deepseek_client=client)
        person = await rt.registry.register(
            "ABSpeaker",
            "owner",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=True,
        )
        manifest = await rt.agent_service.repo.get_manifest()
        agent_id = manifest["agent_id"]

        turns = []
        for index, text in enumerate(TURNS, 1):
            event = make_utterance_event(
                conversation_id=f"ab-{name}",
                text=text,
                addressee_hint="agent",
                speaker_person_id=person.person_id,
                source=EventSource.TEST,
            )
            report = await rt.orchestrator.process_utterance(event)
            state = await rt.emotion_service.get_state(agent_id, person.person_id)
            turns.append(
                {
                    "turn": index,
                    "text": text,
                    "action": report.action.value if report.action else None,
                    "reason_code": report.reason_code.value
                    if report.reason_code
                    else None,
                    "response_text": report.response_text,
                    "emotion": {
                        "emotion": state.emotion,
                        "intensity": state.intensity,
                        "activations": state.activations,
                        "relationship_tension": state.relationship_tension,
                    },
                }
            )
        await db.close()
        return {"mode": mode, "turns": turns, "calls": client.calls}


def _config():
    from companion.providers.deepseek_client import load_deepseek_config

    return load_deepseek_config()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/v042_ab.json")
    parser.add_argument(
        "--arms",
        default="AUXILIARY,LEGACY",
        help="comma-separated: AUXILIARY,LEGACY",
    )
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print(
            "DEEPSEEK_API_KEY is not set — live A/B was NOT executed. "
            "No claim about live model behaviour is made."
        )
        return 2

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    results = {}
    for arm in arms:
        if arm not in ARMS:
            print(f"unknown arm {arm!r}")
            return 2
        results[arm] = await _run_arm(arm, ARMS[arm], key)

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
