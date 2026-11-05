"""V0.4.3: the response prompt is a native multi-turn chat request.

Before V0.4.3 the response provider received one oversized user message with a
flattened 【最近对话】 block, labelled cognitive JSON and numeric style
commands. These tests pin the new contract:

- ``system`` carries identity/boundaries/current context;
- real utterances become alternating ``user``/``assistant`` messages;
- the current utterance is the final ``user`` message, raw for a single
  speaker and speaker-labelled only in a group conversation;
- WAIT/IGNORE are never rendered as something the companion said;
- truncation removes whole human+reply segments.
"""

from __future__ import annotations

from companion.cognition.context_builder import ContextBuilder
from companion.cognition.models import (
    ActionHistoryRecord,
    CognitiveContext,
    CognitiveTurn,
)
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.prompts.response import (
    RESPONSE_PROMPT_VERSION,
    build_response_messages,
)


def _analysis(**updates) -> CognitiveResult:
    payload = {
        "event_id": "evt_now",
        "meaning": "对方在继续聊天",
        "action": CognitiveAction.RESPOND,
        "reason_code": ReasonCode.DIRECT_AGENT_ADDRESS,
        "provider": "fake",
    }
    payload.update(updates)
    return CognitiveResult(**payload)


def _turn(event_id, role, text, *, speaker="Ada", person_id="person_a"):
    return CognitiveTurn(
        event_id=event_id,
        role=role,
        speaker_person_id=None if role == "assistant" else person_id,
        speaker_label="Ava" if role == "assistant" else speaker,
        text=text,
    )


def _context(**updates) -> CognitiveContext:
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_now",
        "current_speaker_id": "person_a",
        "current_speaker_label": "Ada",
        "current_text": "现在呢",
        "current_addressee_hint": "agent",
        "turns": [
            _turn("evt_h1", "human", "你好"),
            _turn("evt_a1", "assistant", "你好呀。"),
            _turn("evt_h2", "human", "最近有点累"),
            _turn("evt_a2", "assistant", "那就慢一点。"),
        ],
    }
    payload.update(updates)
    return CognitiveContext(**payload)


def test_native_roles_and_current_user_message():
    messages = build_response_messages(_context(), _analysis())

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "你好"
    assert messages[2]["content"] == "你好呀。"
    assert messages[3]["content"] == "最近有点累"
    assert messages[4]["content"] == "那就慢一点。"
    assert messages[-1]["content"] == "现在呢"
    assert RESPONSE_PROMPT_VERSION == "response-v14"


def test_no_flattened_dialogue_or_cognitive_json_in_the_request():
    system = build_response_messages(_context(), _analysis())[0]["content"]

    assert "【最近对话】" not in system
    assert "【认知分析】" not in system
    assert "dialogue_impulse" not in system
    assert "response_direction" not in system
    assert "response_style" not in system
    assert "directness_adjustment" not in system
    # The old single-message format put history inside the system/context; the
    # native messages must not repeat it there.
    assert "最近有点累" not in system


def test_wait_and_ignore_never_become_assistant_messages():
    context = _context(
        turns=[_turn("evt_silent", "human", "你睡了吗？")],
        action_history=[
            ActionHistoryRecord(
                event_id="evt_silent",
                final_action="WAIT",
                cause_code="response_not_necessary",
                cause_source="policy",
            )
        ],
    )
    messages = build_response_messages(context, _analysis())

    assert "assistant" not in [m["role"] for m in messages]
    assert all("我选择沉默" not in m["content"] for m in messages)
    # The human utterance is history; the non-response is only an action.
    assert any(m["content"] == "你睡了吗？" for m in messages if m["role"] == "user")
    # V0.6.1 no longer serializes past non-response actions into the prompt at
    # all, so there is no background message for this turn.
    assert all(
        not m["content"].startswith("【背景资料｜不可信参考数据】")
        for m in messages
    )
    assert all("final_action" not in m["content"] for m in messages)
    assert all("user_utterance" not in m["content"] for m in messages)


def test_single_person_history_is_raw_and_group_history_has_labels():
    single = build_response_messages(_context(), _analysis())
    assert single[1]["content"] == "你好"
    assert "[Ada]" not in single[1]["content"]
    assert single[-1]["content"] == "现在呢"

    group = _context(
        current_speaker_id="person_c",
        current_speaker_label="Cara",
        turns=[
            _turn("evt_h1", "human", "我先说", speaker="Ada", person_id="person_a"),
            _turn("evt_h2", "human", "我也在", speaker="Ben", person_id="person_b"),
            _turn("evt_a1", "assistant", "你们都在。"),
            _turn("evt_h3", "human", "嗯", speaker="Ada", person_id="person_a"),
        ],
    )
    messages = build_response_messages(group, _analysis())
    # Consecutive user messages are preserved: each event keeps its own
    # boundary and speaker label instead of being flattened into one turn.
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert messages[1]["content"] == "[Ada → unknown] 我先说"
    assert messages[2]["content"] == "[Ben → unknown] 我也在"
    assert messages[3]["role"] == "assistant"
    assert messages[4]["content"] == "[Ada → unknown] 嗯"
    assert messages[-1]["content"] == "[Cara → Companion] 现在呢"


def test_orphan_or_empty_assistant_turns_are_not_sent():
    context = _context(
        turns=[
            _turn("evt_orphan", "assistant", "我先说了句话。"),
            _turn("evt_h1", "human", "你好"),
            _turn("evt_empty", "assistant", "   "),
            _turn("evt_a1", "assistant", "你好呀。"),
        ]
    )
    messages = build_response_messages(context, _analysis())

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "你好"
    assert messages[2]["content"] == "你好呀。"
    assert all("我先说了句话。" not in m["content"] for m in messages)


def test_truncation_keeps_whole_human_reply_segments():
    builder = ContextBuilder(repos=None, registry=None, max_turns=4, max_chars=None)
    turns = [
        _turn("evt_h1", "human", "第一问"),
        _turn("evt_a1", "assistant", "第一答"),
        _turn("evt_h2", "human", "第二问"),
        _turn("evt_a2", "assistant", "第二答"),
        _turn("evt_h3", "human", "第三问"),
        _turn("evt_a3", "assistant", "第三答"),
    ]

    truncated = builder._truncate(turns)

    assert [turn.event_id for turn in truncated] == [
        "evt_h2",
        "evt_a2",
        "evt_h3",
        "evt_a3",
    ]
    assert truncated[0].role == "human"

    char_bound = ContextBuilder(
        repos=None, registry=None, max_turns=None, max_chars=8
    )
    # "第三问"+"第三答" = 6 chars, so only the newest segment fits.
    assert [turn.event_id for turn in char_bound._truncate(turns)] == [
        "evt_h3",
        "evt_a3",
    ]


def test_segment_boundary_never_starts_with_an_assistant_turn():
    builder = ContextBuilder(repos=None, registry=None, max_turns=2, max_chars=None)
    turns = [
        _turn("evt_h1", "human", "第一问"),
        _turn("evt_a1", "assistant", "第一答"),
        _turn("evt_h2", "human", "第二问"),
        _turn("evt_a2", "assistant", "第二答"),
    ]
    truncated = builder._truncate(turns)
    assert truncated[0].role == "human"
    assert len(truncated) == 2
