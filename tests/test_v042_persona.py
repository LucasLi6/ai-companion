"""V0.4.2 Step 4: persona-first character and stance continuity.

Persona Core is the stable "who" layer. These tests pin that it reaches the
real response request ahead of style/emotion, that a full description can be
imported through a trusted admin path, and that continuity cites only stances
the companion actually emitted (or an admin configured) — never a model's
private reasoning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from companion.agent.stance import (
    AgentStanceRepository,
    extract_stance_candidates,
)
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


class _FakeClient:
    def __init__(self, *, responses=None):
        self.analysis = json.dumps(
            {
                "meaning": "用户在聊天",
                "action": "RESPOND",
                "reason_code": "direct_agent_address",
                "referenced_memory_ids": [],
                "memory_candidates": [],
            },
            ensure_ascii=False,
        )
        self.responses = list(responses or ["嗯。", "我在。"])
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "analysis":
            content = self.analysis
        else:
            content = self.responses.pop(0) if self.responses else "我在。"
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def response_requests(self) -> list[str]:
        return [
            "\n".join(m["content"] for m in call["messages"])
            for call in self.calls
            if call["request_type"] == "response"
        ]


async def _person(rt):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text, *, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


# -- persona-first compilation ---------------------------------------------


async def test_persona_core_leads_the_response_request(db):
    client = _FakeClient(responses=["好。"])
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    await rt.agent_service.set_persona_core(
        {
            "core_values": ["宁可直说，也不假装同意"],
            "preferences": {"音乐": "偏爱安静的现场演出"},
            "character_description": "她说话不急，喜欢先听完再给判断。",
        },
        actor="admin",
    )

    await rt.orchestrator.process_utterance(_event(person.person_id, "你好"))

    request = client.response_requests()[-1]
    assert "【角色核心】" in request
    assert "宁可直说，也不假装同意" in request
    assert "偏爱安静的现场演出" in request
    assert "她说话不急，喜欢先听完再给判断。" in request
    assert "表达优先级" in request
    # Safety/facts come first, temporary style last.
    precedence_line = next(
        line for line in request.splitlines() if line.startswith("表达优先级")
    )
    assert precedence_line.index("安全与事实") < precedence_line.index("临时风格")


async def test_full_description_import_is_admin_revisioned(db):
    rt = build_runtime(db, provider="mock")
    before = await rt.agent_service.show_persona_core()
    updated = await rt.agent_service.import_persona_description(
        "对熟悉的人话多，对陌生人先观察。不喜欢被敷衍。",
        reason="test import",
    )
    assert updated.character_description.startswith("对熟悉的人话多")
    assert updated.revision == before.revision + 1

    again = await rt.agent_service.show_persona_core()
    assert again.character_description == updated.character_description

    with pytest.raises(Exception):
        await rt.agent_service.set_persona_core(
            {"character_description": "x" * 4001}
        )


# -- stance extraction is conservative -------------------------------------


def test_stance_extraction_keeps_explicit_preferences_only():
    text = (
        "我不喜欢临时改计划。"
        "你觉得呢？"
        "我觉得这件事可以。"
        "他认为应该早点走。"
        "我更喜欢先把事情做完再休息。"
    )
    stances = extract_stance_candidates(text)
    assert stances == [
        "我不喜欢临时改计划。",
        "我更喜欢先把事情做完再休息。",
    ]
    # At most two per response, and a question is never a stance.
    assert extract_stance_candidates("我受不了反复改口。") == [
        "我受不了反复改口。"
    ]


async def test_stance_source_rejects_private_thoughts(rt):
    repo = AgentStanceRepository(rt.repos.db)
    with pytest.raises(ValueError):
        await repo.insert(
            agent_id="agent_1",
            person_id=None,
            conversation_id=None,
            topic="旅行",
            stance="我不喜欢海边",
            source="THOUGHT",
        )


# -- continuity through the real runtime -----------------------------------


async def test_emitted_stance_reaches_the_next_response_request(db):
    client = _FakeClient(
        responses=["我不喜欢临时改计划。", "那我们换个时间。"]
    )
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)

    await rt.orchestrator.process_utterance(
        _event(person.person_id, "周末去爬山？")
    )
    assert "我不喜欢临时改计划。" not in client.response_requests()[-1]

    await rt.orchestrator.process_utterance(
        _event(person.person_id, "那改到下周？")
    )
    request = client.response_requests()[-1]
    assert "【既往立场】" in request
    assert "我不喜欢临时改计划。" in request
    assert "不要因为对方换了个语气就无声地站到反面" in request

    stances = await rt.agent_service.list_stances(
        await _agent_id(rt), person_id=person.person_id
    )
    assert [s.stance for s in stances] == ["我不喜欢临时改计划。"]
    assert stances[0].source == "RESPONSE"
    assert stances[0].source_event_id is not None


async def test_admin_stance_is_available_before_any_response(db):
    client = _FakeClient(responses=["好。"])
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    await rt.agent_service.add_stance("我更喜欢先听完再评价。", topic="交流")

    await rt.orchestrator.process_utterance(_event(person.person_id, "你觉得呢"))

    request = client.response_requests()[-1]
    assert "我更喜欢先听完再评价。" in request
    assert "[交流]" in request


async def test_retry_does_not_duplicate_stance_rows(db):
    client = _FakeClient(responses=["我不喜欢反复改口。"])
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    event = _event(person.person_id, "行吧")

    await rt.orchestrator.process_utterance(event)
    await rt.orchestrator.process_utterance(event)

    agent_id = await _agent_id(rt)
    stances = await rt.agent_service.list_stances(agent_id)
    assert len(stances) == 1
