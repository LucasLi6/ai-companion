"""V1.2.1: raw-token streaming order, protocol tolerance and history safety.

These tests exercise the full orchestrator streaming path (not just the parser):
a mock provider emits a VOICE/TEXT reply one character at a time, and the test
verifies that prose is published before generation ends, that the canonical and
display delta concatenations equal the completed views, and that neither the
header nor a tag reaches the database, history, memory or profile.

The policy is offline throughout: no real provider and no network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from companion.core.response_events import RESPONSE_COMPLETED, RESPONSE_DELTA
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.prompts.speech_expression import SPEECH_EXPRESSION_PROMPT
from companion.runtime import build_runtime
from companion.speech_expression.catalog import ALL_TAGS
from companion.speech_expression.parser import (
    VoiceResponseParser,
    parse_voice_response,
)


# -- helpers -----------------------------------------------------------------


def _event(text="今天怎么样？", *, conversation_id="c1", event_id=None, person_id="p1"):
    return make_utterance_event(
        conversation_id=conversation_id,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


async def _person(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    return person


class _StreamingResponse:
    def __init__(self, chunks, *, recovery_text=None, finish_reason="stop"):
        self.chunks = list(chunks)
        self.recovery_text = recovery_text
        self.recovery_calls = 0
        self.finish_reason = finish_reason

    async def generate_stream(self, event, analysis, context):
        for chunk in self.chunks:
            yield StreamDelta(content=chunk)
        yield StreamDelta(finish_reason=self.finish_reason)

    async def generate_plain_recovery(self, event, analysis, context, *, mode=None):
        self.recovery_calls += 1
        return self.recovery_text


class _GatedStreamingResponse:
    """Yield the first body token, then stop until the test releases a gate."""

    def __init__(self, prefix, first_body, rest):
        self.prefix = list(prefix)
        self.first_body = first_body
        self.rest = list(rest)
        self.gate = asyncio.Event()
        self.finished = False

    async def generate_stream(self, event, analysis, context):
        for chunk in self.prefix:
            yield StreamDelta(content=chunk)
        yield StreamDelta(content=self.first_body)
        await self.gate.wait()
        for chunk in self.rest:
            yield StreamDelta(content=chunk)
        self.finished = True
        yield StreamDelta(finish_reason="stop")

    async def generate_plain_recovery(self, event, analysis, context, *, mode=None):
        return "恢复。"


class _BlockingResponse:
    def __init__(self, text, *, recovery_text=None):
        self.text = text
        self.recovery_text = recovery_text
        self.recovery_calls = 0

    async def generate(self, event, analysis, context):
        return self.text

    async def generate_plain_recovery(self, event, analysis, context, *, mode=None):
        self.recovery_calls += 1
        return self.recovery_text


async def _stream_events(rt, event):
    return [item async for item in rt.orchestrator.process_utterance_stream(event)]


def _joined(events, name):
    return "".join(getattr(item, name) or "" for item in events)


def _deltas(events):
    return [item for item in events if item.type == RESPONSE_DELTA]


async def _db_cells_with(db, needles):
    found = []
    rows = await (
        await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ).fetchall()
    for row in rows:
        table = row["name"]
        if table.startswith("sqlite_"):
            continue
        columns = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        for column in columns:
            name = column["name"]
            values = await (
                await db.execute(f"SELECT {name} FROM {table} WHERE {name} IS NOT NULL")
            ).fetchall()
            for value in values:
                cell = value[name]
                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8", "ignore")
                if not isinstance(cell, str):
                    continue
                for needle in needles:
                    if needle in cell:
                        found.append((table, name, needle, cell[:80]))
    return found


def _check_invariants(events):
    completed = events[-1]
    assert completed.type == RESPONSE_COMPLETED
    assert _joined(events, "delta") == completed.text
    assert _joined(events, "display_delta") == completed.display_text
    return completed


# -- 1. raw streaming order --------------------------------------------------


async def test_body_delta_is_published_before_generation_finishes(db):
    # The header arrives in one piece, then the first body token; the provider
    # blocks until the surface has received that token.
    response = _GatedStreamingResponse(
        ["VOICE=amazed\n", "TEXT="], "真", ["的", "吗", "？"]
    )
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)

    events = []
    first_body_seen = False

    async def consume():
        nonlocal first_body_seen
        async for item in rt.orchestrator.process_utterance_stream(event):
            events.append(item)
            if item.type == RESPONSE_DELTA and item.delta and not first_body_seen:
                first_body_seen = True
                # Receiving prose while the provider is still blocked proves
                # the orchestration streams instead of buffering the reply.
                assert response.finished is False
                response.gate.set()

    await asyncio.wait_for(consume(), 5)

    completed = _check_invariants(events)
    assert completed.text == "真的吗？"
    assert completed.display_text == "[amazed]真的吗？"
    first_delta = _deltas(events)[0]
    assert first_delta.delta == "真"
    assert first_delta.display_delta == "[amazed]真"
    assert not any("VOICE=" in (item.delta or "") for item in events)
    assert not any("VOICE=" in (item.display_delta or "") for item in events)


async def test_character_by_character_header_is_buffered_then_streamed(db):
    response = _StreamingResponse(list("VOICE=sarcastic\nTEXT=你来了。"))
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)

    events = await _stream_events(rt, event)
    completed = _check_invariants(events)
    assert completed.text == "你来了。"
    assert completed.display_text == "[sarcastic]你来了。"
    # The header never reaches the surface as ordinary text.
    assert "VOICE" not in _joined(events, "delta")
    assert "TEXT=" not in _joined(events, "delta")
    assert "VOICE" not in _joined(events, "display_delta")


async def test_crlf_header_is_accepted(db):
    response = _StreamingResponse(["VOICE=tired\r\n", "TEXT=有点累。"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "有点累。"
    assert completed.display_text == "[tired]有点累。"


# -- 2. protocol tolerance ---------------------------------------------------


async def test_missing_text_keyword_keeps_voice_and_body(db):
    response = _StreamingResponse(["VOICE=sarcastic\n", "你还真会说话。"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "你还真会说话。"
    assert completed.display_text == "[sarcastic]你还真会说话。"
    assert response.recovery_calls == 0


async def test_missing_header_preserves_plain_body(db):
    response = _StreamingResponse(["你好，", "今天怎么样？"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "你好，今天怎么样？"
    assert completed.display_text == "你好，今天怎么样？"
    assert response.recovery_calls == 0


async def test_invalid_voice_yields_no_tag_and_no_empty_brackets(db):
    response = _StreamingResponse(["VOICE=happy\n", "TEXT=[] 你好！"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "你好！"
    assert completed.display_text == "你好！"
    assert "[]" not in _joined(events, "display_delta")
    assert response.recovery_calls == 0


async def test_natural_voice_with_body_never_triggers_recovery(db):
    response = _StreamingResponse(["VOICE=natural\n", "TEXT=嗯，明白了。"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "嗯，明白了。"
    assert completed.display_text == "嗯，明白了。"
    assert response.recovery_calls == 0


@pytest.mark.parametrize("header", ["VOICE=amazed\nTEXT=", "VOICE=\nTEXT=", "VOICE=amazed\n"])
async def test_empty_body_is_recovered_once(db, header):
    response = _StreamingResponse([header], recovery_text="我在这儿。")
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "我在这儿。"
    assert completed.display_text == "我在这儿。"
    assert response.recovery_calls == 1
    assert rt.orchestrator.speech_expression_enabled is True


# -- 3. bracket preservation across token boundaries -------------------------


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (["[happy]", "(https://example.com)"], "[happy](https://example.com)"),
        (["[]", "(https://example.com)"], "[](https://example.com)"),
        (["a[", "happy]", " 正常"], "a[happy] 正常"),
        (["`[sad]`"], "`[sad]`"),
    ],
)
async def test_follower_dependent_brackets_survive_split_chunks(db, chunks, expected):
    response = _StreamingResponse(chunks)
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == expected
    assert completed.display_text == expected


async def test_invalid_voice_markup_at_end_is_still_removed(db):
    response = _StreamingResponse(["VOICE=natural\nTEXT=你好", "[happy]"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "你好"
    assert completed.display_text == "你好"


@pytest.mark.parametrize(
    "body",
    [
        "改成 return [] 之类就行。",
        "先 items = [] 再判断。",
        "把 f(x, []) 调一下。",
    ],
)
def test_empty_array_in_code_prose_survives(body):
    result = parse_voice_response("VOICE=natural\nTEXT=" + body)
    assert "[]" in result.text
    assert "[]" in result.display_text


async def test_empty_array_in_code_prose_survives_streaming_split(db):
    response = _StreamingResponse(
        ["VOICE=natural\nTEXT=改成 ", "return ", "[] 之类就行。"]
    )
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = _check_invariants(events)
    assert completed.text == "改成 return [] 之类就行。"


def test_single_shot_and_every_split_agree_for_tricky_text():
    raw = (
        "VOICE=natural\nTEXT=看 [happy](https://example.com)，"
        "`[sad]`、a[i]、`[]` 都保留；[] 这个清掉。"
    )
    expected = parse_voice_response(raw)
    for size in (1, 2, 3, 5, 8, 13, 21, 200):
        parser = VoiceResponseParser()
        texts, displays = [], []
        for index in range(0, len(raw), size):
            chunk = parser.feed(raw[index : index + size])
            texts.append(chunk.text)
            displays.append(chunk.display_text)
        tail = parser.finish()
        texts.append(tail.text)
        displays.append(tail.display_text)
        result = parser.result()
        assert "".join(texts) == result.text == expected.text
        assert "".join(displays) == result.display_text == expected.display_text
    assert "[happy](https://example.com)" in expected.text
    assert "`[sad]`" in expected.text
    assert "a[i]" in expected.text
    assert "[] 这个清掉" not in expected.text
    assert "这个清掉" in expected.text


# -- 4. persistence and history ---------------------------------------------


async def test_streamed_reply_persists_canonical_text_only(db):
    response = _StreamingResponse(
        ["VOICE=mischievously\n", "TEXT=[giggles] 你猜？"]
    )
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)
    events = await _stream_events(rt, event)
    _check_invariants(events)

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "你猜？"
    assert await _db_cells_with(db, ["VOICE=", "TEXT=", *ALL_TAGS]) == []


async def test_next_speaker_call_sees_no_voice_header_or_tag(db):
    class _Client:
        def __init__(self):
            self.response_calls = []

        async def complete(
            self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
        ):
            if request_type == "analysis":
                content = json.dumps(
                    {
                        "meaning": "对方在说话",
                        "action": "RESPOND",
                        "reason_code": "direct_agent_address",
                        "referenced_memory_ids": [],
                        "memory_candidates": [],
                    },
                    ensure_ascii=False,
                )
            else:
                self.response_calls.append(messages)
                content = "VOICE=curious\nTEXT=说说看？"
            return ChatResponse(
                content=content,
                model="fake",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _Client()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    await rt.orchestrator.process_utterance(
        _event(person_id=person.person_id, event_id="evt_1")
    )
    await rt.orchestrator.process_utterance(
        _event("再聊聊", person_id=person.person_id, event_id="evt_2")
    )

    assistant_turns = [
        message["content"]
        for message in client.response_calls[-1]
        if message.get("role") == "assistant"
    ]
    assert assistant_turns == ["说说看？"]
    assert all("VOICE=" not in text for text in assistant_turns)
    assert all("TEXT=" not in text for text in assistant_turns)


# -- 5. proactive ------------------------------------------------------------


class _ProactiveClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    async def complete(self, messages, *, request_type, event_id=None, **kwargs):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "response":
            return ChatResponse(
                content=self.response_text,
                model="fake",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        return ChatResponse(
            content=json.dumps({"should_initiate": False, "reason": "不需要"}),
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


@pytest.mark.parametrize("mode", ["IMMEDIATE", "LONG_IDLE"])
async def test_proactive_modes_use_the_same_protocol_split(db, mode):
    client = _ProactiveClient("VOICE=excited\nTEXT=[giggles] 太好了！")
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    source = _event("今天有好消息", person_id=person.person_id)
    await rt.orchestrator.process_utterance(source)

    result = await rt.orchestrator.generate_proactive_reply(
        source_event=source, person_id=person.person_id, mode=mode
    )
    assert result["text"] == "太好了！"
    assert result["display_text"] == "[excited][giggles] 太好了！"
    assert await _db_cells_with(db, ["VOICE=", "TEXT="]) == []


# -- 6. concurrency ----------------------------------------------------------


class _SplitVoiceResponse:
    def __init__(self):
        self.recovery_calls = 0

    async def generate_stream(self, event, analysis, context):
        if event.text == "空":
            yield StreamDelta(content="VOICE=sad\nTEXT=")
        else:
            yield StreamDelta(content="VOICE=excited\nTEXT=太好了！")
        yield StreamDelta(finish_reason="stop")

    async def generate_plain_recovery(self, event, analysis, context, *, mode=None):
        self.recovery_calls += 1
        return "我还在。"


async def test_recovery_in_one_conversation_does_not_change_the_other(db):
    response = _SplitVoiceResponse()
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c2")
    empty = _event(
        "空", conversation_id="c1", event_id="evt_empty",
        person_id=person.person_id,
    )
    normal = _event(
        "正常", conversation_id="c2", event_id="evt_normal",
        person_id=person.person_id,
    )

    empty_items, normal_items = await asyncio.gather(
        _stream_events(rt, empty), _stream_events(rt, normal)
    )

    empty_completed = _check_invariants(empty_items)
    normal_completed = _check_invariants(normal_items)
    assert empty_completed.text == "我还在。"
    assert normal_completed.text == "太好了！"
    assert normal_completed.display_text == "[excited]太好了！"
    assert response.recovery_calls == 1
    assert rt.orchestrator.speech_expression_enabled is True


# -- 7. non-streaming consistency -------------------------------------------


async def test_non_streaming_and_streaming_normalise_identically(db):
    raw = "VOICE=serious\nTEXT=这件事\n\n 得认真说。"
    non_stream = _BlockingResponse(raw)
    rt = build_runtime(
        db,
        provider="mock",
        response=non_stream,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    events = await _stream_events(rt, _event(person_id=person.person_id))
    completed = events[-1]
    assert completed.text == "这件事 得认真说。"
    assert completed.display_text == "[serious]这件事 得认真说。"

    stream_response = _StreamingResponse(list(raw))
    rt2 = build_runtime(
        db,
        provider="mock",
        response=stream_response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    await rt2.conversations.get_or_create("c1")
    events2 = await _stream_events(
        rt2, _event("另一个问题", event_id="evt_2", person_id=person.person_id)
    )
    completed2 = _check_invariants(events2)
    assert completed2.text == completed.text
    assert completed2.display_text == completed.display_text
