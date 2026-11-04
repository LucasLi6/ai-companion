#!/usr/bin/env python
"""Offline full-chain request capture for V0.4.2.

Runs a three-turn scene through the *real* DeepSeek provider classes over a
scripted transport (no network, no model), and writes every request message to
JSON. This is the artifact used to inspect what the model would actually
receive: cognitive-v8 analysis request, then response-v7 generation request,
in the fixed order and with the persona/scene/pragmatic blocks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.storage.database import Database


def _pragmatic(text, *, act, frame, transition="CONTINUE", role="NONE"):
    return {
        "primary_act": act,
        "secondary_acts": [],
        "interaction_frame": frame,
        "refers_to_previous_turn": False,
        "antecedent_event_id": None,
        "setup_payoff_role": role,
        "literal_interpretation": "",
        "implied_interpretation": "",
        "topic_transition": transition,
        "confidence": 0.9,
        "evidence_span": text,
    }


TURNS = [
    (
        "你就是个废物",
        _pragmatic("你就是个废物", act="CRITICISM", frame="CONFLICT"),
        "这句话我听到了。你可以不同意我，但不用这样说话。",
    ),
    (
        "刚才是我说得太重了，对不起",
        _pragmatic("刚才是我说得太重了，对不起", act="APOLOGY", frame="REPAIR"),
        "好，刚才那句我也接得有点硬。",
    ),
    (
        "其实我不喜欢被反复追问，音乐我更喜欢老摇滚",
        _pragmatic(
            "其实我不喜欢被反复追问，音乐我更喜欢老摇滚",
            act="SELF_DISCLOSURE",
            frame="CASUAL",
            transition="SHIFT",
        ),
        "记住了，那我换个问法。老摇滚里你更喜欢哪一类？",
    ),
]


class _ScriptedClient:
    def __init__(self, analyses, responses):
        self.analyses = list(analyses)
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, request_type, event_id=None, **kwargs):
        content = (
            json.dumps(
                {
                    "meaning": "理解这一轮的交流行为",
                    "action": "RESPOND",
                    "reason_code": "direct_agent_address",
                    "referenced_memory_ids": [],
                    "memory_candidates": [],
                    "pragmatic": self.analyses.pop(0),
                },
                ensure_ascii=False,
            )
            if request_type == "analysis"
            else self.responses.pop(0)
        )
        response = ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        self.calls.append(
            {
                "request_type": request_type,
                "event_id": event_id,
                "messages": messages,
                "scripted_response": content,
            }
        )
        return response


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/V042_REQUEST_CAPTURE.json")
    args = parser.parse_args()

    client = _ScriptedClient([turn[1] for turn in TURNS], [turn[2] for turn in TURNS])

    with __import__("tempfile").TemporaryDirectory(prefix="v042_capture_") as tmp:
        db = await Database(Path(tmp) / "capture.db").connect()
        await db.initialize()
        rt = build_runtime(db, provider="deepseek", deepseek_client=client)
        person = await rt.registry.register(
            "Ada",
            "owner",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=True,
        )
        reports = []
        for index, (text, _pragmatic_payload, _reply) in enumerate(TURNS, 1):
            event = make_utterance_event(
                conversation_id="capture",
                text=text,
                addressee_hint="agent",
                speaker_person_id=person.person_id,
                source=EventSource.TEST,
            )
            report = await rt.orchestrator.process_utterance(event)
            reports.append(
                {
                    "turn": index,
                    "text": text,
                    "action": report.action.value if report.action else None,
                    "reason_code": report.reason_code.value
                    if report.reason_code
                    else None,
                    "response_text": report.response_text,
                }
            )
        await db.close()

    payload = {
        "note": (
            "Captured offline with the real DeepSeek provider classes over a "
            "scripted transport. No network call and no live model output."
        ),
        "reports": reports,
        "calls": client.calls,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(client.calls)} requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
