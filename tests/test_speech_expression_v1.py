"""Ava Speech Expression V1 — labels never enter stored conversation state.

The feature is local and offline: one whitelist, one streaming parser, one
optional prompt block. These tests pin the whitelist, the parser's chunking and
edge cases, the delta/completed invariants, canonical-only persistence across
the normal, streaming, failure/retry and proactive paths, and the prompt switch.
No tag may reach the database, history, memory, profile, stance or scene.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from companion.core.response_events import (
    RESPONSE_COMPLETED,
    RESPONSE_DELTA,
    ResponseEvent,
    ResponseEventSink,
)
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.proactive.judge import ProactiveJudge
from companion.prompts.proactive import (
    PROACTIVE_MODE_BLOCK,
    PROACTIVE_MODE_BLOCK_LONG_IDLE,
)
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    build_proactive_response_messages,
    build_response_messages,
)
from companion.prompts.speech_expression import (
    SPEECH_EXPRESSION_PROMPT,
    SPEECH_EXPRESSION_PROMPT_VERSION,
)
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.deepseek_proactive import ScriptedProactiveJudgeProvider
from companion.runtime import build_runtime
from companion.speech_expression.catalog import (
    ALL_TAGS,
    CONTROL_TAGS,
    INVALID_SPEECH_TAGS,
    KIND_CONTROL,
    KIND_NONVERBAL,
    NONVERBAL_TAGS,
    TAG_KINDS,
    VOICE_NAMES,
    render_tag_catalog,
)
from companion.speech_expression.parser import (
    SpeechMarkupParser,
    parse_speech_expressions,
)


EXPECTED_CONTROL = (
    "[sad]",
    "[amazed]",
    "[deep and loud shouting]",
    "[trembling]",
    "[angry]",
    "[excited]",
    "[sarcastic]",
    "[curious]",
    "[like dracula]",
    "[bored]",
    "[tired]",
    "[scornful]",
    "[shouting]",
    "[asmr]",
    "[panicked]",
    "[mischievously]",
    "[empathetic]",
    "[whispers]",
    "[reluctantly]",
    "[crying]",
    "[serious]",
    "[very slowly]",
    "[very fast]",
)

EXPECTED_NONVERBAL = (
    "[gasp]",
    "[sighing]",
    "[clears throat]",
    "[giggles]",
    "[laughing]",
    "[cough]",
    "[snorts]",
)


# -- whitelist ---------------------------------------------------------------


def test_catalog_is_exactly_the_official_control_and_nonverbal_tags():
    assert CONTROL_TAGS == EXPECTED_CONTROL
    assert NONVERBAL_TAGS == EXPECTED_NONVERBAL
    assert len(CONTROL_TAGS) == 23
    assert len(NONVERBAL_TAGS) == 7
    assert len(set(ALL_TAGS)) == 30
    assert ALL_TAGS == EXPECTED_CONTROL + EXPECTED_NONVERBAL
    for tag in ALL_TAGS:
        assert tag.startswith("[") and tag.endswith("]")
        assert tag == tag.lower()


def test_catalog_has_no_aliases_and_keeps_two_kinds_apart():
    for alias in (
        "[happy]",
        "[neutral]",
        "[playful]",
        "[confused]",
        "[thinking]",
    ):
        assert alias not in TAG_KINDS
    assert all(TAG_KINDS[tag] == KIND_CONTROL for tag in CONTROL_TAGS)
    assert all(TAG_KINDS[tag] == KIND_NONVERBAL for tag in NONVERBAL_TAGS)


def test_invalid_vocabulary_never_overlaps_the_official_whitelist():
    official_inner = {tag[1:-1] for tag in ALL_TAGS}
    assert official_inner.isdisjoint(INVALID_SPEECH_TAGS)


def test_prompt_tag_list_is_generated_from_the_same_whitelist():
    rendered = render_tag_catalog()
    for tag in ALL_TAGS:
        assert tag in rendered
    system = build_response_messages(
        _context(), None, speech_expression=True
    )[0]["content"]
    assert all(tag in system for tag in ALL_TAGS)
    assert "23 个" in system and "7 个" in system


# -- parser ------------------------------------------------------------------


def test_parser_separates_canonical_text_from_display_tags():
    parsed = parse_speech_expressions(
        "[sad]今天有点累。[giggles]不过还好。[whispers]小声说。"
    )
    assert parsed.text == "今天有点累。不过还好。小声说。"
    assert parsed.display_text == (
        "[sad]今天有点累。[giggles]不过还好。[whispers]小声说。"
    )
    assert [segment.kind for segment in parsed.segments] == [
        KIND_CONTROL,
        "text",
        KIND_NONVERBAL,
        "text",
        KIND_CONTROL,
        "text",
    ]
    assert parsed.segments[0].tag == "[sad]"
    assert parsed.segments[2].tag == "[giggles]"
    assert parsed.has_tags is True


def test_control_tag_persists_over_several_sentences_in_the_segments():
    parsed = parse_speech_expressions("[serious]先说结论。再说原因。[sad]最后。")
    assert parsed.text == "先说结论。再说原因。最后。"
    assert [segment.kind for segment in parsed.segments] == [
        KIND_CONTROL,
        "text",
        KIND_CONTROL,
        "text",
    ]
    # One text segment spans everything governed by [serious]; this is what a
    # future TTS consumer needs to apply the control until the next boundary.
    assert parsed.segments[1].text == "先说结论。再说原因。"


def test_parser_is_correct_for_every_single_chunk_split():
    text = "[mischievously]你好呀[giggles]，今天[very slowly]慢慢说。"
    expected = parse_speech_expressions(text)
    assert expected.text == "你好呀，今天慢慢说。"
    for split in range(len(text) + 1):
        parser = SpeechMarkupParser()
        chunks = [
            parser.feed(text[:split]),
            parser.feed(text[split:]),
            parser.finish(),
        ]
        assert "".join(chunk.text for chunk in chunks) == expected.text
        assert (
            "".join(chunk.display_text for chunk in chunks)
            == expected.display_text
        )
        assert parser.result() == expected


@pytest.mark.parametrize(
    "text",
    [
        "[sad]",
        "[gasp]",
        "a[giggles]b[sighing]c",
        "[clears throat] [very fast]快说完啦",
        "no tags at all",
        "",
    ],
)
def test_parser_never_leaks_a_whitelist_tag_into_canonical_text(text):
    parsed = parse_speech_expressions(text)
    for tag in ALL_TAGS:
        assert tag not in parsed.text


@pytest.mark.parametrize(
    "text",
    [
        "[happy]",
        "[playful][neutral]",
        "a[softly]b",
        "[giggling]",
        "[happy]你好[playful]再见",
    ],
)
def test_parser_never_leaks_invalid_voice_markup(text):
    parsed = parse_speech_expressions(text)
    for sample in parsed.invalid_tags:
        assert sample not in parsed.text
        assert sample not in parsed.display_text


def test_parser_handles_partial_tags_across_chunks():
    parser = SpeechMarkupParser()
    assert parser.feed("[mis").text == ""
    assert parser.feed("chievously]你").text == "你"
    assert parser.feed("[gig").text == ""
    tail = parser.feed("gles]好")
    assert tail.text == "好"
    assert tail.display_text == "[giggles]好"
    parser.finish()
    result = parser.result()
    assert result.text == "你好"
    assert result.display_text == "[mischievously]你[giggles]好"


def test_unknown_tags_and_ordinary_brackets_are_protected_literally():
    for text in (
        "数组 a[i] 和 b[0]",
        "看看 [草稿] 这段",
        "[1] 第一项",
        "Markdown [link](https://example.com)",
        "标识符 a[happy] 不是标签",
        "Markdown [happy](https://example.com)",
    ):
        parsed = parse_speech_expressions(text)
        assert parsed.text == text
        assert parsed.display_text == text
        assert parsed.has_tags is False
        assert parsed.invalid_tags == ()


@pytest.mark.parametrize(
    "text,expected_text,expected_display",
    [
        ("[happy]你好", "你好", "你好"),
        ("[playful]你怎么又来了", "你怎么又来了", "你怎么又来了"),
        ("[neutral]我知道了", "我知道了", "我知道了"),
        ("[softly]嗯", "嗯", "嗯"),
        ("[giggling]哈哈", "哈哈", "哈哈"),
        ("你好[neutral]", "你好", "你好"),
        ("我 [happy] 好", "我 好", "我 好"),
        ("[happy] [playful] 你好", "你好", "你好"),
    ],
)
def test_invalid_voice_markup_is_removed_from_both_channels(
    text, expected_text, expected_display
):
    parsed = parse_speech_expressions(text)
    assert parsed.text == expected_text
    assert parsed.display_text == expected_display
    assert parsed.invalid_tags
    assert parsed.has_tags is False


def test_invalid_voice_markup_streaming_is_correct_for_every_split():
    text = "[playful]你怎么又来了[neutral]"
    expected = parse_speech_expressions(text)
    assert expected.text == "你怎么又来了"
    assert expected.display_text == "你怎么又来了"
    for split in range(len(text) + 1):
        parser = SpeechMarkupParser()
        chunks = [
            parser.feed(text[:split]),
            parser.feed(text[split:]),
            parser.finish(),
        ]
        assert "".join(chunk.text for chunk in chunks) == expected.text
        assert (
            "".join(chunk.display_text for chunk in chunks)
            == expected.display_text
        )
        assert parser.result() == expected
    assert parser.invalid_tag_count == 2
    assert parser.invalid_tag_samples == ("[playful]", "[neutral]")


def test_invalid_tags_do_not_become_segments():
    parsed = parse_speech_expressions("[happy]你好[giggles]呀")
    assert [segment.kind for segment in parsed.segments] == [
        "text",
        KIND_NONVERBAL,
        "text",
    ]
    assert all(segment.tag != "[happy]" for segment in parsed.segments)
    assert parsed.invalid_tags == ("[happy]",)


def test_code_and_markdown_brackets_survive_verbatim():
    text = (
        "```python\nvalue = array[i][0]\n```\n"
        "以及 `[草稿]` 与 [link](https://example.com) 原样出现。"
    )
    parsed = parse_speech_expressions(text)
    assert parsed.text == text
    assert parsed.display_text == text


def test_incomplete_whitelist_tag_is_dropped_but_incomplete_ordinary_bracket_is_kept():
    dropped = parse_speech_expressions("你好[gig")
    assert dropped.text == "你好"
    assert dropped.display_text == "你好"

    kept = parse_speech_expressions("你好[abc")
    assert kept.text == "你好[abc"
    assert kept.display_text == "你好[abc"


def test_tag_only_output_has_empty_canonical_text():
    parsed = parse_speech_expressions("[sad][giggles]")
    assert parsed.text == ""
    assert parsed.display_text == "[sad][giggles]"
    assert parsed.has_tags is True


def test_whitespace_seam_left_by_a_removed_tag_is_collapsed():
    parsed = parse_speech_expressions("我 [giggles] 好")
    assert parsed.text == "我 好"
    assert parsed.display_text == "我 [giggles] 好"


def test_parser_is_deterministic_and_finish_is_idempotent():
    text = "[sad]你好[gasp]呀"
    first = parse_speech_expressions(text)
    second = parse_speech_expressions(text)
    assert first == second

    parser = SpeechMarkupParser()
    parser.feed(text)
    parser.finish()
    assert parser.finish().text == ""
    with pytest.raises(RuntimeError):
        parser.feed("more")


# -- event serialisation -----------------------------------------------------


def test_response_event_serialises_display_fields():
    event = ResponseEvent(
        type=RESPONSE_COMPLETED,
        event_id="e",
        conversation_id="c",
        response_id="r",
        sequence=1,
        status="COMPLETED",
        text="你好",
        display_text="[giggles]你好",
    )
    payload = event.as_dict()
    assert payload["text"] == "你好"
    assert payload["display_text"] == "[giggles]你好"
    assert "display_delta" not in payload

    sink = ResponseEventSink("e", "c", lambda item: None)
    delta = sink.delta("r", "", display_delta="[gasp]")
    assert delta.delta == ""
    assert delta.display_delta == "[gasp]"


# -- helpers -----------------------------------------------------------------


def _context(**updates):
    from companion.cognition.models import CognitiveContext

    payload = dict(
        conversation_id="c1",
        current_event_id="evt_now",
        current_speaker_id="p1",
        current_speaker_label="Alice",
        current_text="今天怎么样？",
        current_addressee_hint="agent",
    )
    payload.update(updates)
    return CognitiveContext(**payload)


def _event(text="今天怎么样？", *, person_id="person_1", event_id=None):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


async def _person(rt, name="Alice"):
    person = await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    return person


class _StreamingResponse:
    def __init__(
        self,
        chunks,
        *,
        fail_after=None,
        recovery_text=None,
        recovery_error=None,
        finish_reason="stop",
    ):
        self.chunks = list(chunks)
        self.fail_after = fail_after
        self.contexts = []
        self.recovery_text = recovery_text
        self.recovery_error = recovery_error
        self.recovery_calls = 0
        self.finish_reason = finish_reason

    async def generate_stream(self, event, analysis, context):
        self.contexts.append(context)
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise RuntimeError("stream interrupted")
            yield StreamDelta(content=chunk)
        yield StreamDelta(finish_reason=self.finish_reason)

    async def generate_plain_recovery(
        self, event, analysis, context, *, mode=None
    ):
        self.recovery_calls += 1
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovery_text


class _NoRecoveryStreamingResponse(_StreamingResponse):
    generate_plain_recovery = None


class _BlockingResponse:
    def __init__(self, text, *, recovery_text=None, recovery_error=None):
        self.text = text
        self.contexts = []
        self.recovery_text = recovery_text
        self.recovery_error = recovery_error
        self.recovery_calls = 0

    async def generate(self, event, analysis, context):
        self.contexts.append(context)
        return self.text

    async def generate_plain_recovery(
        self, event, analysis, context, *, mode=None
    ):
        self.recovery_calls += 1
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovery_text


class _SplitStreamingResponse:
    """One tag-only reply and one normal reply, for the concurrency test."""

    def __init__(self):
        self.recovery_calls = 0

    async def generate_stream(self, event, analysis, context):
        if event.text == "只有标签":
            yield StreamDelta(content="[sad]")
        else:
            yield StreamDelta(content="[sad]正常回复")
        yield StreamDelta(finish_reason="stop")

    async def generate_plain_recovery(
        self, event, analysis, context, *, mode=None
    ):
        self.recovery_calls += 1
        return "补上一句。"


async def _stream_events(rt, event):
    return [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]


def _joined(events, name):
    return "".join(getattr(item, name) or "" for item in events)


async def _db_text_with_tags(db, needles=None):
    """Every text cell in every user table, for the isolation assertion."""
    needles = tuple(ALL_TAGS) if needles is None else tuple(needles)
    rows = await (
        await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ).fetchall()
    found = []
    for row in rows:
        table = row["name"]
        if table.startswith("sqlite_"):
            continue
        columns = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        for column in columns:
            name = column["name"]
            values = await (
                await db.execute(
                    f"SELECT {name} FROM {table} WHERE {name} IS NOT NULL"
                )
            ).fetchall()
            for value in values:
                cell = value[name]
                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8", "ignore")
                if not isinstance(cell, str):
                    continue
                for tag in needles:
                    if tag in cell:
                        found.append((table, name, tag, cell[:80]))
    return found


# -- streaming pipeline ------------------------------------------------------


async def test_streaming_emits_both_views_and_persists_canonical_only(db):
    response = _StreamingResponse(
        ["[sad", "]今天", "有点累。[gig", "gles]不过还好。"]
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

    assert [item.type for item in events] == [
        "response_started",
        "response_delta",
        "response_delta",
        "response_delta",
        "response_completed",
    ]
    completed = events[-1]
    assert completed.text == "今天有点累。不过还好。"
    assert completed.display_text == "[sad]今天有点累。[giggles]不过还好。"
    # Acceptance: the concatenation of canonical deltas is exactly the final
    # canonical text, and the display deltas are exactly the final display text.
    assert _joined(events, "delta") == completed.text
    assert _joined(events, "display_delta") == completed.display_text
    for item in events:
        if item.type == RESPONSE_DELTA:
            for tag in ALL_TAGS:
                assert tag not in (item.delta or "")

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.response_text == "今天有点累。不过还好。"
    assert await _db_text_with_tags(db) == []


async def test_history_given_to_the_next_speaker_call_is_canonical(db):
    class _DeepSeekFake:
        def __init__(self, response_text):
            self.response_text = response_text
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
                content = self.response_text
                self.response_calls.append(messages)
            return ChatResponse(
                content=content,
                model="fake",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _DeepSeekFake("[giggles]你好呀")
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    first = _event(person_id=person.person_id, event_id="evt_first")
    report = await rt.orchestrator.process_utterance(first)
    assert report.status == "COMPLETED"

    second = _event("再说一次", person_id=person.person_id, event_id="evt_2")
    report = await rt.orchestrator.process_utterance(second)
    assert report.status == "COMPLETED"

    assistant_turns = [
        message["content"]
        for message in client.response_calls[-1]
        if message.get("role") == "assistant"
    ]
    assert assistant_turns == ["你好呀"]
    # Only the trusted protocol block names the tags; no history turn carries one.
    history = [
        message["content"]
        for message in client.response_calls[-1]
        if message.get("role") in ("user", "assistant")
    ]
    assert all("[giggles]" not in text for text in history)


@pytest.mark.parametrize(
    "chunks",
    [
        ["[giggles] 你好"],
        ["你好 ", "[giggles]"],
        ["你好 [gig", "gles]呀"],
        ["  [sad]你好"],
        ["你好 [sighing] 呀"],
    ],
)
async def test_streaming_invariants_hold_at_tag_whitespace_edges(db, chunks):
    response = _StreamingResponse(chunks)
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
    completed = events[-1]
    assert _joined(events, "delta") == completed.text
    assert _joined(events, "display_delta") == completed.display_text
    for tag in ALL_TAGS:
        assert tag not in completed.text


async def test_disabled_speech_expression_keeps_the_original_stream(db):
    response = _StreamingResponse(["你好", "[giggles]呀"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=False,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)

    events = await _stream_events(rt, event)
    completed = events[-1]
    assert completed.text == "你好[giggles]呀"
    assert completed.display_text is None
    assert all(item.display_delta is None for item in events)


async def test_structured_layout_never_strips_brackets(db):
    response = _StreamingResponse(["示例代码：a[i] 与 [giggles]。"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event("帮我写一个代码示例", person_id=person.person_id)

    events = await _stream_events(rt, event)
    completed = events[-1]
    assert completed.text == "示例代码：a[i] 与 [giggles]。"
    assert completed.display_text is None


async def test_non_streaming_generate_is_split_by_the_same_parser(db):
    response = _BlockingResponse("[sad]有点累。[giggles]不过还好。")
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
    completed = events[-1]
    assert completed.type == "response_completed"
    assert completed.text == "有点累。不过还好。"
    assert completed.display_text == "[sad]有点累。[giggles]不过还好。"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "有点累。不过还好。"
    assert await _db_text_with_tags(db) == []


async def test_non_streaming_empty_reply_recovers_once(db):
    response = _BlockingResponse("[sad]", recovery_text="我没事。")
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
    completed = events[-1]
    assert completed.type == "response_completed"
    assert completed.text == "我没事。"
    assert response.recovery_calls == 1
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.status == "COMPLETED"
    assert action.response_text == "我没事。"


async def test_non_streaming_empty_recovery_failure_is_failed(db):
    response = _BlockingResponse("[happy]", recovery_text="[neutral]")
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
    assert events[-1].type == "response_failed"
    assert response.recovery_calls == 1
    assert await rt.repos.analysis.get_action(event.event_id) is None


async def test_tag_only_streaming_reply_is_not_completed(db):
    response = _StreamingResponse(["[sad]", "[giggles]"])
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
    assert events[-1].type == "response_failed"
    assert events[-1].text in (None, "")
    assert "EMPTY_SPEECH_OUTPUT" in (events[-1].error or "")
    # No empty bubble: leading labels are never published without prose.
    assert [item for item in events if item.type == RESPONSE_DELTA] == []
    assert response.recovery_calls == 1
    assert await rt.repos.analysis.get_action(event.event_id) is None
    assert await _db_text_with_tags(db) == []


async def test_invalid_tag_only_streaming_reply_is_not_completed(db):
    response = _StreamingResponse(["[happy]"])
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
    assert events[-1].type == "response_failed"
    assert response.recovery_calls == 1
    assert await _db_text_with_tags(db) == []


async def test_empty_streaming_reply_recovers_once_in_plain_text(db):
    response = _StreamingResponse(
        ["[sad]"], recovery_text="其实今天还不错，只是有点累。"
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
    completed = events[-1]
    assert completed.type == "response_completed"
    assert completed.text == "其实今天还不错，只是有点累。"
    assert completed.display_text == "其实今天还不错，只是有点累。"
    assert _joined(events, "delta") == completed.text
    assert _joined(events, "display_delta") == completed.display_text
    assert all(item.delta for item in events if item.type == RESPONSE_DELTA)
    assert response.recovery_calls == 1
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.status == "COMPLETED"
    assert action.response_text == completed.text
    assert await _db_text_with_tags(db) == []


async def test_empty_recovery_that_is_also_empty_fails(db):
    response = _StreamingResponse(["[gasp]"], recovery_text="[happy]")
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
    assert events[-1].type == "response_failed"
    assert response.recovery_calls == 1
    assert await rt.repos.analysis.get_action(event.event_id) is None


async def test_provider_without_recovery_fails_cleanly(db):
    response = _NoRecoveryStreamingResponse(["[sad]"])
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
    assert events[-1].type == "response_failed"
    assert "EMPTY_SPEECH_OUTPUT" in (events[-1].error or "")
    assert await rt.repos.analysis.get_action(event.event_id) is None


async def test_recovery_exception_fails_without_double_terminal_event(db):
    response = _StreamingResponse(
        ["[sad]"], recovery_error=RuntimeError("recovery boom")
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
    assert [item.type for item in events].count("response_failed") == 1
    assert response.recovery_calls == 1


async def test_plain_reply_without_tags_is_accepted(db):
    response = _StreamingResponse(["你好"])
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
    assert events[-1].type == "response_completed"
    assert events[-1].text == "你好"
    assert response.recovery_calls == 0


async def test_leading_labels_are_held_until_the_first_prose(db):
    response = _StreamingResponse(["[sad]", "[giggles]", "你好呀"])
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
    deltas = [item for item in events if item.type == RESPONSE_DELTA]
    assert len(deltas) == 1
    assert deltas[0].delta == "你好呀"
    assert deltas[0].display_delta == "[sad][giggles]你好呀"
    assert events[-1].display_text == "[sad][giggles]你好呀"
    assert _joined(events, "delta") == events[-1].text
    assert _joined(events, "display_delta") == events[-1].display_text


async def test_recovery_does_not_change_the_shared_speech_switch(db):
    response = _StreamingResponse(["[sad]"], recovery_text="好。")
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)

    await _stream_events(rt, event)
    assert rt.orchestrator.speech_expression_enabled is True
    if hasattr(response, "speech_expression_enabled"):
        assert response.speech_expression_enabled is True

    # The next turn still gets the protocol and the display/canonical split.
    response.chunks = ["[sad]第二次"]
    response.recovery_text = None
    second = _event("再聊聊", person_id=person.person_id)
    events = await _stream_events(rt, second)
    assert events[-1].type == "response_completed"
    assert events[-1].text == "第二次"
    assert events[-1].display_text == "[sad]第二次"
    assert response.recovery_calls == 1


async def test_concurrent_conversations_keep_recovery_isolated(db):
    response = _SplitStreamingResponse()
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c2")
    empty_event = make_utterance_event(
        conversation_id="c1",
        text="只有标签",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        event_id="evt_empty",
    )
    normal_event = make_utterance_event(
        conversation_id="c2",
        text="正常一轮",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        event_id="evt_normal",
    )

    empty_items, normal_items = await asyncio.gather(
        _stream_events(rt, empty_event),
        _stream_events(rt, normal_event),
    )

    assert empty_items[-1].type == RESPONSE_COMPLETED
    assert empty_items[-1].text == "补上一句。"
    assert normal_items[-1].type == RESPONSE_COMPLETED
    assert normal_items[-1].text == "正常回复"
    assert normal_items[-1].display_text == "[sad]正常回复"
    assert response.recovery_calls == 1
    assert rt.orchestrator.speech_expression_enabled is True
    assert _joined(empty_items, "delta") == empty_items[-1].text
    assert _joined(normal_items, "display_delta") == normal_items[-1].display_text


async def test_tag_only_truncated_stream_is_truncated_not_recovered(db):
    response = _StreamingResponse(["[sad]"], finish_reason="length")
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
    completed = events[-1]
    assert completed.type == "response_completed"
    assert completed.status == "TRUNCATED"
    assert completed.text == ""
    assert completed.display_text is None
    assert response.recovery_calls == 0
    assert [item for item in events if item.type == RESPONSE_DELTA] == []


async def test_truncated_stream_with_text_stays_truncated(db):
    response = _StreamingResponse(["说到一半"], finish_reason="length")
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
    assert events[-1].status == "TRUNCATED"
    assert events[-1].text == "说到一半"
    assert response.recovery_calls == 0


async def test_failed_partial_is_persisted_canonically(db):
    response = _StreamingResponse(["你好[gig", "never"], fail_after=1)
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
    assert events[-1].type == "response_failed"
    # The failure event carries canonical partial text only.
    assert events[-1].text == "你好"

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.status == "FAILED"
    assert action.response_text == "你好"
    assert await _db_text_with_tags(db) == []


async def test_retry_replaces_a_tagged_partial_with_the_canonical_reply(db):
    response = _StreamingResponse(["你好[gig", "never"], fail_after=1)
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event(person_id=person.person_id)

    failed = await _stream_events(rt, event)
    assert failed[-1].type == "response_failed"

    response.chunks = ["重来[gasp]一次"]
    response.fail_after = None
    events = await _stream_events(rt, event)
    assert events[-1].type == "response_completed"
    assert events[-1].text == "重来一次"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.status == "COMPLETED"
    assert action.response_text == "重来一次"
    assert await _db_text_with_tags(db) == []


# -- prompt switch -----------------------------------------------------------


def test_prompt_block_is_off_by_default_and_preserves_the_persona():
    without = build_response_messages(_context(), None)
    assert "【语音表达】" not in without[0]["content"]
    assert BASE_RESPONSE_PROMPT in without[0]["content"]

    with_block = build_response_messages(
        _context(), None, speech_expression=True
    )
    system = with_block[0]["content"]
    assert BASE_RESPONSE_PROMPT in system
    assert SPEECH_EXPRESSION_PROMPT in system
    assert "你是 Ava。" in system
    for tag in ALL_TAGS:
        assert tag in system
    for voice in VOICE_NAMES:
        assert voice in system
    for rule in (
        "VOICE=<初始语气>",
        "TEXT=<真正的回复内容>",
        "VOICE=natural",
        "不要输出 ``[]``",
        "疑问句不一定用 curious",
        "技术或严肃话题不一定用 serious",
        "开心不一定用 excited",
        "tone_direction",
        "你可以采纳、调整或忽略它",
        "代码、表格、Markdown",
        "任何正常回复都必须包含实际文字",
    ):
        assert rule in system
    assert SPEECH_EXPRESSION_PROMPT_VERSION == "speech-expression-v1.2.1"


def test_proactive_prompt_uses_the_same_optional_block():
    off = build_proactive_response_messages(_context(), None, mode="IMMEDIATE")
    assert "【语音表达】" not in off[0]["content"]
    on = build_proactive_response_messages(
        _context(), None, mode="IMMEDIATE", speech_expression=True
    )
    system = on[0]["content"]
    assert SPEECH_EXPRESSION_PROMPT in system
    assert "你是 Ava。" in system


class _CaptureClient:
    def __init__(self, content="[sad]好。"):
        self.content = content
        self.messages = None

    async def complete(self, messages, *, request_type, **kwargs):
        self.messages = messages
        return ChatResponse(
            content=self.content,
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        self.messages = messages
        yield StreamDelta(content=self.content)
        yield StreamDelta(finish_reason="stop")


async def test_provider_injects_the_block_only_when_enabled_and_for_chat():
    client = _CaptureClient()
    provider = DeepSeekResponseProvider(client)
    event = _event()

    await provider.generate(event, None, None)
    assert "【语音表达】" not in client.messages[0]["content"]

    provider.speech_expression_enabled = True
    await provider.generate(event, None, None)
    assert SPEECH_EXPRESSION_PROMPT in client.messages[0]["content"]

    structured = _event("帮我写一个代码示例")
    await provider.generate(structured, None, None)
    assert "【语音表达】" not in client.messages[0]["content"]


async def test_provider_stream_injects_the_block_for_chat_only():
    client = _CaptureClient()
    provider = DeepSeekResponseProvider(client)
    provider.speech_expression_enabled = True

    async for _ in provider.generate_stream(_event(), None, None):
        pass
    assert SPEECH_EXPRESSION_PROMPT in client.messages[0]["content"]

    async for _ in provider.generate_stream(
        _event("输出一段代码"), None, None
    ):
        pass
    assert "【语音表达】" not in client.messages[0]["content"]


# -- proactive ---------------------------------------------------------------


class _ProactiveClient:
    def __init__(self):
        self.response_text = "正常回复。"
        self.recovery_text = None
        self.calls = []

    async def complete(self, messages, *, request_type, event_id=None, **kwargs):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "response":
            system = messages[0]["content"] if messages else ""
            if SPEECH_EXPRESSION_PROMPT in system:
                content = self.response_text
            elif self.recovery_text is not None:
                content = self.recovery_text
            else:
                content = self.response_text
        else:
            content = json.dumps(
                {"should_initiate": False, "reason": "不需要"},
                ensure_ascii=False,
            )
        return ChatResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def response_calls(self):
        return [
            call for call in self.calls if call["request_type"] == "response"
        ]


async def test_proactive_message_record_and_event_use_both_views(db):
    client = _ProactiveClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)

    first = _event("我在研究机器人控制", person_id=person.person_id)
    report = await rt.orchestrator.process_utterance(first)
    assert report.status == "COMPLETED"

    client.response_text = "[sad]有点累。[giggles]不过还好。"
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    await manager.shutdown()

    rows = await rt.repos.proactive.list_for_conversation("c1")
    assert len(rows) == 1
    assert rows[0].text == "有点累。不过还好。"
    assert delivered.text == "有点累。不过还好。"
    assert delivered.display_text == "[sad]有点累。[giggles]不过还好。"
    assert await _db_text_with_tags(db) == []


async def test_disabled_proactive_path_stores_and_shows_the_same_text(db):
    client = _ProactiveClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=False,
    )
    person = await _person(rt)
    first = _event("我在研究机器人控制", person_id=person.person_id)
    await rt.orchestrator.process_utterance(first)

    client.response_text = "[sad]原样保留。"
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    await manager.shutdown()

    rows = await rt.repos.proactive.list_for_conversation("c1")
    assert rows[0].text == "[sad]原样保留。"
    assert delivered.display_text is None


async def test_empty_proactive_reply_recovers_once_and_keeps_mode(db):
    client = _ProactiveClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    first = _event("我在研究机器人控制", person_id=person.person_id)
    await rt.orchestrator.process_utterance(first)

    client.response_text = "[sad]"
    client.recovery_text = "我正好想到你刚才说的控制问题。"
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    await manager.shutdown()

    assert delivered.text == "我正好想到你刚才说的控制问题。"
    assert delivered.display_text == "我正好想到你刚才说的控制问题。"
    rows = await rt.repos.proactive.list_for_conversation("c1")
    assert len(rows) == 1
    assert rows[0].text == delivered.text

    response_calls = client.response_calls()
    # Initial normal turn + proactive attempt + one plain-text recovery.
    proactive_calls = response_calls[-2:]
    assert len(proactive_calls) == 2
    assert SPEECH_EXPRESSION_PROMPT in proactive_calls[0]["messages"][0]["content"]
    assert SPEECH_EXPRESSION_PROMPT not in proactive_calls[1]["messages"][0]["content"]
    # Recovery preserved IMMEDIATE rather than switching modes.
    assert PROACTIVE_MODE_BLOCK in proactive_calls[1]["messages"][0]["content"]
    assert (
        PROACTIVE_MODE_BLOCK_LONG_IDLE
        not in proactive_calls[1]["messages"][0]["content"]
    )


async def test_empty_proactive_reply_that_cannot_recover_is_discarded(db):
    client = _ProactiveClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    first = _event("我在研究机器人控制", person_id=person.person_id)
    await rt.orchestrator.process_utterance(first)

    client.response_text = "[sad]"
    client.recovery_text = "[happy]"
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(manager.next_event(), 0.3)
    await manager.shutdown()

    assert await rt.repos.proactive.list_for_conversation("c1") == []


# -- CLI display -------------------------------------------------------------


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


class _FakeOrchestrator:
    def __init__(self, events):
        self._events = events

    def process_utterance_stream(self, event):
        return _FakeStream(self._events)


def _stream_item(kind, **fields):
    return ResponseEvent(
        type=kind,
        event_id="evt_cli",
        conversation_id="c1",
        response_id="resp_cli",
        sequence=0,
        status=fields.pop("status", "STREAMING"),
        **fields,
    )


async def test_cli_prints_the_display_channel_inline(capsys):
    from types import SimpleNamespace

    from companion.cli import _run_streaming_turn

    runtime = SimpleNamespace(
        orchestrator=_FakeOrchestrator(
            [
                _stream_item("response_started"),
                _stream_item("response_delta", delta="", display_delta="[giggles]"),
                _stream_item("response_delta", delta="你好", display_delta=" 你好"),
                _stream_item(
                    "response_completed",
                    text="你好",
                    display_text="[giggles] 你好",
                    status="COMPLETED",
                    action="RESPOND",
                ),
            ]
        )
    )
    await _run_streaming_turn(runtime, _event())
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "assistant> [giggles] 你好"
    assert lines[1].startswith("[timing] first_text=")
    assert "first_audio=off" in lines[1]
    assert out.count("你好") == 1


async def test_cli_uses_display_text_for_a_non_streaming_completion(capsys):
    from types import SimpleNamespace

    from companion.cli import _run_streaming_turn

    runtime = SimpleNamespace(
        orchestrator=_FakeOrchestrator(
            [
                _stream_item("response_started"),
                _stream_item(
                    "response_completed",
                    text="你好",
                    display_text="[sad]你好",
                    status="COMPLETED",
                    action="RESPOND",
                ),
            ]
        )
    )
    await _run_streaming_turn(runtime, _event())
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "assistant> [sad]你好"
    assert lines[1].startswith("[timing] first_text=")
    assert "first_audio=off" in lines[1]


# -- emotion isolation -------------------------------------------------------


async def test_invalid_voice_markup_never_reaches_any_stored_state(db):
    response = _StreamingResponse(["[happy]你好[playful]呀"])
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
    assert events[-1].type == "response_completed"
    assert events[-1].text == "你好呀"
    assert events[-1].display_text == "你好呀"
    invalid_forms = tuple(f"[{word}]" for word in INVALID_SPEECH_TAGS)
    assert await _db_text_with_tags(db, invalid_forms) == []
    assert await _db_text_with_tags(db) == []


async def test_emotion_tables_never_receive_speech_labels(db):
    """Emotion is appraised from the user's turn, never from reply labels."""
    response = _StreamingResponse(["[excited]好呀！[laughing]"])
    rt = build_runtime(
        db,
        provider="mock",
        response=response,
        with_worker=False,
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    event = _event("我好开心！", person_id=person.person_id)

    await _stream_events(rt, event)

    for table in (
        "emotion_events",
        "emotion_state_snapshots",
        "emotion_turn_receipts",
    ):
        rows = await (await db.execute(f"SELECT * FROM {table}")).fetchall()
        rendered = json.dumps(rows, ensure_ascii=False, default=str)
        for tag in ALL_TAGS:
            assert tag not in rendered
    # The reply itself was split, so no downstream emotion/memory reader can
    # observe the label even if it inspects the action.
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "好呀！"
    assert await _db_text_with_tags(db) == []


# -- V1.2 VOICE/TEXT protocol through the real paths -------------------------


async def test_v12_streaming_voice_header_persists_canonical_only(db):
    response = _StreamingResponse(
        ["VOICE=sarc", "astic\nTE", "XT=你今天", "倒是挺忙", "的。"]
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

    completed = events[-1]
    assert completed.type == RESPONSE_COMPLETED
    assert completed.text == "你今天倒是挺忙的。"
    assert completed.display_text == "[sarcastic]你今天倒是挺忙的。"
    assert _joined(events, "delta") == completed.text
    assert _joined(events, "display_delta") == completed.display_text
    for item in events:
        if item.type == RESPONSE_DELTA:
            assert "VOICE=" not in (item.display_delta or "")
            assert "[sarcastic]" not in (item.delta or "")

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "你今天倒是挺忙的。"


async def test_v12_natural_voice_has_identical_channels(db):
    response = _StreamingResponse(["VOICE=natural\nTEXT=今天天气不错。"])
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
    completed = events[-1]
    assert completed.text == "今天天气不错。"
    assert completed.display_text == "今天天气不错。"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "今天天气不错。"


async def test_v12_non_streaming_voice_protocol_is_parsed(db):
    response = _BlockingResponse(
        "VOICE=curious\nTEXT=[giggles] 你在做什么？"
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

    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "你在做什么？"
    assert await _db_text_with_tags(db) == []


async def test_v12_invalid_voice_never_reaches_stored_or_display(db):
    response = _StreamingResponse(
        ["VOICE=happy\nTEXT=[] 好呀。"]
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
    completed = events[-1]
    assert completed.text == "好呀。"
    assert completed.display_text == "好呀。"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == "好呀。"
    assert "[]" not in action.response_text
    assert "[]" not in completed.display_text


async def test_v12_proactive_stores_canonical_and_shows_voice(db):
    client = _ProactiveClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)
    first = _event("我在研究机器人控制", person_id=person.person_id)
    await rt.orchestrator.process_utterance(first)

    client.response_text = (
        "VOICE=empathetic\nTEXT=[] 听起来你今天很累。"
    )
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    await manager.shutdown()

    rows = await rt.repos.proactive.list_for_conversation("c1")
    assert rows[0].text == "听起来你今天很累。"
    assert delivered.display_text == "[empathetic]听起来你今天很累。"
    assert await _db_text_with_tags(db) == []
