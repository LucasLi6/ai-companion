"""Speaker prompt delivery: tool results are projected, labelled and bounded.

The repaired contract is checked on the *actual* messages handed to the
response provider: every phase-1 and phase-2 result reaches the speaker, a
large payload is projected field by field (never sliced mid-JSON), the omission
is stated explicitly, and each section has its own budget.
"""

from __future__ import annotations

import json

from companion.cognition.models import CognitiveContext
from companion.prompts.response import (
    MAX_BACKGROUND_CHARS,
    MAX_MEMORY_CONTEXT_CHARS,
    MAX_ROUTE_CONTEXT_CHARS,
    MAX_SCENE_CONTEXT_CHARS,
    MAX_TOOL_CONTEXT_CHARS,
    MAX_TOOL_FIELD_CHARS,
    MAX_TOOL_ITEMS,
    MAX_MEMORY_CHARS,
    build_response_messages,
)

_BACKGROUND_START = '{"schema":"companion.background.v1"'


def _context(**updates):
    payload = dict(
        conversation_id="c1",
        current_event_id="evt_now",
        current_speaker_id="p1",
        current_speaker_label="Alice",
        current_text="what did we say?",
        current_addressee_hint="agent",
    )
    payload.update(updates)
    return CognitiveContext(**payload)


def _background(messages):
    for message in messages:
        content = message.get("content") or ""
        marker = content.find('"schema": "companion.background.v1"')
        if marker < 0:
            marker = content.find('"schema":"companion.background.v1"')
        if marker >= 0:
            start = content.rfind("{", 0, marker)
            return json.loads(content[start:])
    raise AssertionError("no background message in the response request")


def _tool(name, data, *, untrusted=False, status="SUCCESS", **extra):
    result = {
        "tool_name": name,
        "tool_call_id": f"tc_{name}",
        "status": status,
        "untrusted": untrusted,
        "data": data,
    }
    result.update(extra)
    return result


def test_phase_one_and_phase_two_tool_results_both_reach_the_speaker():
    context = _context(
        tool_results=[
            _tool("memory.search", [{"memory_id": "m1", "content": "likes tea"}]),
            _tool("history.recent", [{"text": "hello", "speaker": "Alice"}]),
            _tool("runtime.current_time", {"local_time": "2026-11-04 10:00"}),
            _tool("history.stats", {"total_messages": 12}),
            _tool("memory.browse", [{"memory_id": "m2", "content": "likes cats"}]),
            _tool("history.earliest", [{"text": "first words", "speaker": "Alice"}]),
        ]
    )
    background = _background(build_response_messages(context, None))

    tools = background["tool_results"]
    names = [item["tool"] for item in tools]
    assert names == [
        "memory.search",
        "history.recent",
        "runtime.current_time",
        "history.stats",
        "memory.browse",
        "history.earliest",
    ]
    assert all(item["status"] == "SUCCESS" for item in tools)
    assert all(item["trust"] == "LOCAL_READ" for item in tools)
    assert tools[4]["data"][0]["content"] == "likes cats"
    assert tools[5]["data"][0]["text"] == "first words"


def test_memory_results_are_projected_not_sliced():
    huge = "x" * 5000
    context = _context(
        tool_results=[
            _tool(
                "memory.search",
                [
                    {
                        "memory_id": f"m{i}",
                        "kind": "EPISODE",
                        "content": huge,
                        "visibility": "PRIVATE",
                        "status": "ACTIVE",
                        "subjects": ["PERSON:p1"],
                        "sources": ["event:e1"],
                        "secret_extra_field": huge,
                    }
                    for i in range(2)
                ],
            )
        ]
    )
    messages = build_response_messages(context, None)
    background = _background(messages)  # parses, so it is valid JSON

    memory_tools = [
        item for item in background["tool_results"] if item["tool"] == "memory.search"
    ]
    assert len(memory_tools) == 1
    item = memory_tools[0]
    assert item["trust"] == "LOCAL_READ"
    data = item["data"]
    assert isinstance(data, list)
    for record in data:
        assert len(record["content"]) <= MAX_TOOL_FIELD_CHARS
        assert "secret_extra_field" not in record
        assert record["visibility"] == "PRIVATE"
        assert record["status"] == "ACTIVE"


def test_web_results_are_projected_and_labelled_untrusted():
    context = _context(
        tool_results=[
            _tool(
                "web.search",
                [
                    {
                        "title": "t" * 1000,
                        "url": "https://example.invalid/" + "u" * 2000,
                        "snippet": "s" * 3000,
                        "huge_extra": "e" * 5000,
                    }
                    for _ in range(1)
                ],
                untrusted=True,
            )
        ]
    )
    background = _background(build_response_messages(context, None))
    tools = [
        item for item in background["tool_results"] if item["tool"] == "web.search"
    ]
    assert tools
    item = tools[0]
    assert item["trust"] == "UNTRUSTED_EXTERNAL"
    assert isinstance(item["data"], list)
    assert len(item["data"]) <= 5
    for record in item["data"]:
        assert len(record["title"]) <= 200
        assert len(record["url"]) <= 500
        assert len(record["snippet"]) <= 400
        assert "huge_extra" not in record


def test_budget_overflow_is_an_explicit_marker_not_a_truncated_string():
    context = _context(
        tool_results=[
            _tool(
                "history.recent",
                [{"text": "y" * MAX_TOOL_FIELD_CHARS, "speaker": "Alice"}]
                * MAX_TOOL_ITEMS,
            )
            for _ in range(10)
        ]
    )
    messages = build_response_messages(context, None)
    background = _background(messages)

    tools = background["tool_results"]
    budget = [item for item in tools if item["tool"] == "_budget"]
    assert len(budget) == 1
    assert budget[0]["status"] == "TRUNCATED"
    assert budget[0]["omitted_results"] > 0
    assert budget[0]["note"]
    # Every delivered result keeps its identity and trust label intact.
    delivered = [item for item in tools if item["tool"] != "_budget"]
    assert delivered
    assert all(item["tool_call_id"] for item in delivered)
    assert all(item["trust"] in {"LOCAL_READ", "UNTRUSTED_EXTERNAL"} for item in delivered)
    # Nothing was cut mid-string: re-serializing every payload is lossless.
    assert json.loads(json.dumps(tools, ensure_ascii=False)) == tools


def test_a_single_oversized_result_is_bounded_and_marked_truncated():
    context = _context(
        tool_results=[
            _tool(
                "memory.search",
                [
                    {
                        "memory_id": f"m{i}",
                        "content": "z" * 500,
                        "visibility": "PRIVATE",
                        "status": "ACTIVE",
                        "subjects": ["PERSON:p1"],
                        "sources": ["event:e1"],
                    }
                    for i in range(5)
                ],
            )
        ]
    )
    background = _background(build_response_messages(context, None))
    item = background["tool_results"][0]
    assert item["tool"] == "memory.search"
    assert item["truncated"] is True
    assert item["data"] is not None


def test_each_section_uses_its_own_budget():
    context = _context(
        memories=[
            {"kind": "EPISODE", "content": "m" * 2000} for _ in range(10)
        ],
        scene_state={
            "topic": "t" * 1000,
            "open_loops": [
                {"topic": f"loop{i}", "summary": "s" * 1000} for i in range(10)
            ],
        },
        route_context={
            "action": "RESPOND",
            "reply_necessity": "REQUIRED",
            "addressee_assessment": "AGENT",
            "turn_state": "CONTINUING",
            "processing_intents": [f"INTENT_{i}" for i in range(10)],
            "reason": "r" * 2000,
            "tool_requests": [{"tool_name": f"tool{i}"} for i in range(10)],
        },
        tool_results=[
            _tool("history.recent", [{"text": "h" * 600} for _ in range(5)])
        ],
    )
    messages = build_response_messages(context, None)
    background = _background(messages)

    assert len(background["memories"]) <= 3
    for memory in background["memories"]:
        assert len(memory["content"]) <= MAX_MEMORY_CHARS
    assert len(json.dumps(background["memories"], ensure_ascii=False)) <= (
        MAX_MEMORY_CONTEXT_CHARS
    )

    scene = json.dumps(background["scene"], ensure_ascii=False)
    assert len(scene) <= MAX_SCENE_CONTEXT_CHARS

    route = background["route"]
    assert len(route["reason"]) <= 240
    assert len(route["processing_intents"]) <= 4
    assert len(route["tools_requested"]) <= 4
    assert len(json.dumps(route, ensure_ascii=False)) <= MAX_ROUTE_CONTEXT_CHARS

    tools = background["tool_results"]
    assert len(json.dumps(tools, ensure_ascii=False)) <= MAX_TOOL_CONTEXT_CHARS

    background_json = json.dumps(background, ensure_ascii=False)
    assert len(background_json) <= MAX_BACKGROUND_CHARS


def test_memory_write_outcomes_are_stated_truthfully():
    context = _context(
        memory_write_outcomes=[
            {"outcome": "SAVED", "memory_id": "m1"},
            {"outcome": "DENIED", "detail": "memory_learning_not_granted"},
            {"outcome": "NO_MEMORY"},
        ]
    )
    messages = build_response_messages(context, None)
    system = messages[0]["content"]

    assert "本轮记忆写入结果" in system
    assert "已保存" in system
    assert "未保存（无权限）" in system
    assert "无需保存" not in system
    assert "memory_learning_not_granted" in system


def test_route_data_is_labelled_as_model_judgment_not_instruction():
    context = _context(
        route_context={
            "action": "RESPOND",
            "reply_necessity": "REQUIRED",
            "addressee_assessment": "AGENT",
            "reason": "semantic reason",
        }
    )
    background = _background(build_response_messages(context, None))
    assert background["route"]["kind"] == "MODEL_JUDGMENT"
    assert background["trust"] == "UNTRUSTED_REFERENCE_ONLY"
