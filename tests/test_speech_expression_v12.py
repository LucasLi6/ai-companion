"""V1.2: Router tone suggestion and the VOICE/TEXT Speaker protocol."""

from __future__ import annotations

import pytest

from companion.prompts.response import build_response_messages
from companion.routing.models import (
    RawRouteDecision,
    ToneDirection,
    ValidatedRouteDecision,
)
from companion.speech_expression.catalog import (
    CONTROL_TAGS,
    NATURAL_VOICE,
    VOICE_NAMES,
    is_valid_voice_name,
)
from companion.speech_expression.history import strip_leading_empty_bracket_markup
from companion.speech_expression.parser import (
    SpeechMarkupParser,
    VoiceResponseParser,
    parse_voice_response,
)


BASE_ROUTE = {
    "action": "RESPOND",
    "reply_necessity": "REQUIRED",
    "addressee_assessment": "AGENT",
    "input_semantics": "COMMUNICATIVE",
}


def _raw(**extra):
    return RawRouteDecision(**BASE_ROUTE, **extra)


def _stream(parser, text, size=1):
    texts, displays = [], []
    for index in range(0, len(text), size):
        chunk = parser.feed(text[index : index + size])
        texts.append(chunk.text)
        displays.append(chunk.display_text)
    tail = parser.finish()
    texts.append(tail.text)
    displays.append(tail.display_text)
    result = parser.result()
    assert "".join(texts) == result.text
    assert "".join(displays) == result.display_text
    return result


# -- Router -----------------------------------------------------------------


def test_router_tone_direction_accepts_official_values():
    decision = _raw(tone_direction="playful")
    assert decision.tone_direction is ToneDirection.PLAYFUL


@pytest.mark.parametrize("value", ["bogus", "[]", "", "HAPPY", 42, ["PLAYFUL"]])
def test_router_unknown_tone_degrades_to_none(value):
    decision = _raw(tone_direction=value)
    assert decision.tone_direction is None


def test_router_old_payload_without_tone_is_compatible():
    payload = dict(BASE_ROUTE)
    decision = RawRouteDecision(**payload)
    assert decision.tone_direction is None


def test_validated_decision_reads_old_row_and_accepts_tone():
    old = ValidatedRouteDecision(**BASE_ROUTE)
    assert old.tone_direction is None
    new = ValidatedRouteDecision(**BASE_ROUTE, tone_direction="CURIOUS")
    assert new.tone_direction is ToneDirection.CURIOUS
    unknown = ValidatedRouteDecision(**BASE_ROUTE, tone_direction="HAPPY")
    assert unknown.tone_direction is None


def test_route_tone_reaches_speaker_as_a_marked_suggestion():
    from companion.cognition.models import CognitiveContext

    route = ValidatedRouteDecision(
        **BASE_ROUTE, tone_direction=ToneDirection.PLAYFUL, reason="hidden"
    )
    context = CognitiveContext(
        conversation_id="c",
        current_event_id="e",
        current_speaker_label="user",
        current_text="hello",
        route_context=route.audit_dict(),
        tool_results=[],
    )
    messages = build_response_messages(context, None)
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    assert "PLAYFUL" in rendered
    assert "tone_direction_is_suggestion" in rendered
    assert "hidden" not in rendered


# -- VOICE/TEXT parser -------------------------------------------------------


def test_voice_names_come_from_the_control_catalog():
    assert VOICE_NAMES == tuple(tag[1:-1] for tag in CONTROL_TAGS)
    assert len(VOICE_NAMES) == 23
    assert is_valid_voice_name("very slowly")
    assert not is_valid_voice_name("happy")


def test_voice_line_emits_prefix_and_body_streams():
    result = _stream(
        VoiceResponseParser(),
        "VOICE=sarcastic\nTEXT=你这是我干活也说，不干活也说呗。",
    )
    assert result.voice == "sarcastic"
    assert result.protocol_valid is True
    assert result.text == "你这是我干活也说，不干活也说呗。"
    assert result.display_text == "[sarcastic]你这是我干活也说，不干活也说呗。"


def test_natural_and_empty_voice_are_legal_and_untagged():
    for header in ("VOICE=natural", "VOICE="):
        result = parse_voice_response(f"{header}\nTEXT=今天天气不错。")
        assert result.voice is None
        assert result.protocol_valid is True
        assert result.text == "今天天气不错。"
        assert result.display_text == "今天天气不错。"


def test_invalid_voice_never_becomes_a_tag_or_empty_brackets():
    result = parse_voice_response("VOICE=happy\nTEXT=[] 好呀。")
    assert result.voice is None
    assert result.invalid_voice == "happy"
    assert result.protocol_error == "INVALID_VOICE"
    assert result.text == "好呀。"
    assert result.display_text == "好呀。"
    assert "[]" not in result.display_text


def test_tone_direction_is_not_enforced():
    # The parser has no router input at all: the Speaker's actual VOICE wins.
    result = parse_voice_response("VOICE=serious\nTEXT=行。")
    assert result.voice == "serious"


def test_empty_bracket_markup_is_removed_from_both_channels():
    for raw in ("[] 你好", "VOICE=natural\nTEXT=[] 你好", "你好 [   ] 世界"):
        result = parse_voice_response(raw)
        assert "[]" not in result.text
        assert "[]" not in result.display_text


def test_legitimate_code_and_markdown_brackets_survive():
    cases = [
        "调用 `a[]` 和 `[]`。",
        "```python\nitems = []\n```",
        "见 [链接](https://example.com)，值 a[] 不变。",
        "arr[0] 和 obj[\"key\"] 都不变。",
    ]
    for raw in cases:
        result = parse_voice_response("VOICE=natural\nTEXT=" + raw)
        assert "[]" in result.text or "a[]" in result.text or "[0]" in result.text


def test_arbitrary_stream_splits_match_complete_parse():
    raw = "VOICE=mischievously\nTEXT=[sighing] 真的吗？[empathetic] 我明白。"
    expected = parse_voice_response(raw)
    for size in (1, 2, 3, 5, 7, 13, 100):
        result = _stream(VoiceResponseParser(), raw, size=size)
        assert result.text == expected.text
        assert result.display_text == expected.display_text
        assert result.voice == expected.voice


@pytest.mark.parametrize(
    "raw,voice",
    [
        ("VOICE=very slowly\nTEXT=慢一点。", "very slowly"),
        ("VOICE=deep and loud shouting\nTEXT=喂！", "deep and loud shouting"),
        ("VOICE=asmr\nTEXT=嘘。", "asmr"),
    ],
)
def test_multiword_voice_names(raw, voice):
    assert parse_voice_response(raw).voice == voice


def test_voice_only_or_header_only_recovers_as_empty_not_error_text():
    for raw in ("VOICE=sarcastic\nTEXT=", "VOICE=sarcastic\n", ""):
        result = parse_voice_response(raw)
        assert result.has_valid_text is False
        assert "[]" not in result.display_text
        assert result.text.strip() == ""


def test_missing_voice_header_preserves_normal_body():
    result = parse_voice_response("[] 你好，今天怎么样？")
    assert result.protocol_valid is False
    assert result.protocol_error == "MISSING_HEADER"
    assert result.text == "你好，今天怎么样？"


def test_legacy_v11_tags_remain_parseable():
    result = parse_voice_response("[sarcastic]你来了。")
    assert result.voice is None
    assert result.text == "你来了。"
    assert result.display_text == "[sarcastic]你来了。"
    assert "[sarcastic]" not in result.text


def test_text_may_switch_local_control_tag():
    result = parse_voice_response(
        "VOICE=natural\nTEXT=我说真的。[serious] 这次不行。"
    )
    assert result.voice is None
    assert result.text == "我说真的。 这次不行。"
    assert "[serious]" in result.display_text


def test_display_body_tag_without_protocol_does_not_leak_to_canonical():
    result = parse_voice_response("VOICE=curious\nTEXT=[giggles] 你说呢？")
    assert result.text == "你说呢？"
    assert result.display_text == "[curious][giggles] 你说呢？"


def test_header_split_before_text_keyword_is_held():
    parser = VoiceResponseParser()
    assert parser.feed("VOICE=sarc").text == ""
    assert parser.feed("astic\nTE").text == ""
    chunk = parser.feed("XT=来了。")
    assert chunk.text == "来了。"
    assert "[sarcastic]" in chunk.display_text


# -- history cleanup ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[] 你好", "你好"),
        ("[   ]\n你好", "你好"),
        ("[] [] 你好", "你好"),
        ("a[] 你好", "a[] 你好"),
        ("[] (x)", "[] (x)"),
        ("正常回复", "正常回复"),
    ],
)
def test_history_prefix_cleanup_is_narrow(raw, expected):
    assert strip_leading_empty_bracket_markup(raw) == expected


async def test_repository_history_load_cleans_legacy_empty_bracket(db):
    from companion.storage.repositories import make_repositories

    await db.execute("PRAGMA foreign_keys = OFF")
    await db.execute(
        "INSERT INTO agent_actions(action_id, event_id, action_type,"
        " response_text, reason_code, status, created_at)"
        " VALUES ('a1', 'e1', 'RESPOND', ?, 'direct_agent_address',"
        " 'COMPLETED', '2026-01-01T00:00:00+00:00')",
        ("[] 你好呀。",),
    )
    action = await make_repositories(db).analysis.get_action("e1")
    assert action.response_text == "你好呀。"


def test_followup_tone_may_change_without_changing_action_rules():
    first = ValidatedRouteDecision(
        **BASE_ROUTE, tone_direction=ToneDirection.PLAYFUL
    )
    second = ValidatedRouteDecision(
        **BASE_ROUTE, tone_direction=ToneDirection.SERIOUS
    )
    assert first.tone_direction is not second.tone_direction
    assert first.core_fingerprint() == second.core_fingerprint()
