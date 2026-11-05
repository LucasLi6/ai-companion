"""V0.6.2 — separated UI, warm persona, single-paragraph chat, truthful context.

All tests are offline: fake streaming providers, a local SQLite database and a
fake logging root. No real DeepSeek call is made. The tests pin the spec's
acceptance list without claiming anything about live-model naturalness.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from companion.agent.persona_core import MAX_DESCRIPTION
from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReasonCode,
)
from companion.domain.models import CognitiveResult, make_utterance_event
from companion.emotion.models import AppraisalRule
from companion.prompts.layout import (
    RESPONSE_LAYOUT_CHAT,
    RESPONSE_LAYOUT_STRUCTURED,
    normalize_response_text,
    response_layout_for,
)
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    build_response_messages,
)
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.runtime import build_runtime
from companion.runtime_context.assembler import compile_runtime_context
from companion.runtime_context.snapshot import RuntimeSnapshot

ASSET = (
    Path(__file__).resolve().parents[1] / "assets" / "ava_persona_core_v1.json"
)


def _analysis() -> CognitiveResult:
    return CognitiveResult(
        event_id="evt_current",
        meaning="对方在继续聊天",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.FOLLOW_UP_TO_AGENT,
        response_style="warm",
        provider="fake",
    )


# -- 1. logging isolation ----------------------------------------------------


def test_logging_goes_to_a_rotating_file_not_the_console(tmp_path, monkeypatch):
    import companion.config as config

    root = logging.Logger("v062-root")
    named: dict[str, logging.Logger] = {}

    def fake_get_logger(name=None):
        if name is None:
            return root
        return named.setdefault(name, logging.Logger(name))

    monkeypatch.setattr(config.logging, "getLogger", fake_get_logger)

    path = tmp_path / "logs" / "companion.log"
    config.configure_logging(debug=False, log_file=path)

    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == 1_000_000 and handler.backupCount == 3
    # A normal run has no console handler, so INFO cannot interrupt a stream.
    assert not any(
        type(item) is logging.StreamHandler for item in root.handlers
    )

    root.info("reply stream must not be interrupted")
    handler.flush()
    assert "reply stream must not be interrupted" in path.read_text("utf-8")

    # Transport chatter is suppressed at the logger level.
    assert named["httpx"].level == logging.WARNING
    assert named["httpcore"].level == logging.WARNING
    assert named["aiosqlite"].level == logging.WARNING


def test_debug_logging_adds_a_console_handler_explicitly(tmp_path, monkeypatch):
    import companion.config as config

    root = logging.Logger("v062-debug-root")

    def fake_get_logger(name=None):
        return root if name is None else logging.Logger(name)

    monkeypatch.setattr(config.logging, "getLogger", fake_get_logger)
    config.configure_logging(debug=True, log_file=tmp_path / "debug.log")

    assert root.level == logging.DEBUG
    assert any(
        type(item) is logging.StreamHandler for item in root.handlers
    ), "debug mode is the explicit opt-in for console diagnostics"
    assert any(
        isinstance(item, RotatingFileHandler) for item in root.handlers
    )


def test_file_logging_does_not_echo_to_stdout_or_stderr(tmp_path, capsys):
    import logging as real_logging

    path = tmp_path / "quiet.log"
    old_handlers = list(real_logging.getLogger().handlers)
    for handler in old_handlers:
        real_logging.getLogger().removeHandler(handler)
    try:
        from companion.config import configure_logging

        configure_logging(debug=False, log_file=path)
        real_logging.getLogger("companion.test").info("hidden from terminal")
    finally:
        for handler in list(real_logging.getLogger().handlers):
            real_logging.getLogger().removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            real_logging.getLogger().addHandler(handler)

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert "hidden from terminal" in path.read_text("utf-8")


# -- 2. monitor masking ------------------------------------------------------


def test_monitor_shows_masked_ids_and_no_raw_content():
    pytest.importorskip("nicegui")
    from companion.ui.app import _ACTIVITY, _mask

    assert _mask("evt_abcdef123456") == "…3456"
    assert _mask("evt_1") == "…vt_1"
    assert _mask("abc") == "…"
    assert _mask("") == "—"
    assert _mask("person_abcdef") == "…cdef"

    _ACTIVITY.begin(conversation_id="c1", person_id="person_abcdef")
    assert _ACTIVITY.current["speaker"] == "person_abcdef"
    assert "text" not in _ACTIVITY.current
    _ACTIVITY.end()
    assert _ACTIVITY.current is None


# -- 3-4. persona / prompt ---------------------------------------------------


def test_persona_asset_carries_warmth_and_the_new_tone():
    data = json.loads(ASSET.read_text(encoding="utf-8"))
    assert data["traits"] == {
        "humor": 68,
        "warmth": 65,
        "curiosity": 80,
        "formality": 25,
        "verbosity": 43,
        "assertiveness": 72,
        "empathy": 52,
        "playfulness": 76,
        "patience": 55,
    }
    core = data["persona_core"]
    assert (
        core["voice_identity"]["tone"]
        == "亲切、灵动、略带俏皮；需要严谨判断时保持冷静"
    )
    description = core["character_description"]
    assert "热情与兴趣" in description
    assert "Ava 对人有真实的温度，而不只是礼貌。" in description
    assert "面对熟悉的人，她会自然地表示欢迎" in description
    assert len(description) <= MAX_DESCRIPTION


def test_base_prompt_requires_a_single_paragraph_for_chat():
    assert "普通聊天和简短问答，整个回复必须只有一个自然段，不使用换行或空行。" in BASE_RESPONSE_PROMPT
    assert "不要使用固定的“先回答、再解释、最后追问”模式。" in BASE_RESPONSE_PROMPT
    # Text input must never invent acoustic information.
    assert "文字输入不包含声音语气" in BASE_RESPONSE_PROMPT


def test_stable_warmth_coexists_with_neutral_short_term_emotion():
    ctx = CognitiveContext(
        conversation_id="c1",
        current_event_id="e1",
        current_text="今天有点累。",
        affect_context="此刻情绪中性（不明显）。",
        history_loaded=1,
    )
    system = build_response_messages(ctx, _analysis())[0]["content"]
    assert "你对熟悉的人亲切、有兴致，也有自己的想法。" in system
    assert "【此刻的状态】此刻情绪中性（不明显）。" in system
    # NEUTRAL is a subtle shift, not an instruction to become cold.
    assert "冷淡" not in system.split("【此刻的状态】", 1)[1]


# -- 5. layout: chat vs structured ------------------------------------------


def test_layout_decision_uses_the_current_explicit_request():
    assert response_layout_for("我回来啦") == RESPONSE_LAYOUT_CHAT
    assert response_layout_for("现在几点") == RESPONSE_LAYOUT_CHAT
    assert response_layout_for("帮我写一个 Python 教程") == RESPONSE_LAYOUT_STRUCTURED
    assert response_layout_for("解释一下原理") == RESPONSE_LAYOUT_STRUCTURED
    assert response_layout_for("写一段代码") == RESPONSE_LAYOUT_STRUCTURED


def test_chat_layout_collapses_newlines_across_chunk_boundaries():
    # The newline is split across two chunks; only whole-text normalisation
    # sees and handles it.
    chunks = ["第一段。\n", "\n第二段。"]
    assert normalize_response_text("".join(chunks), RESPONSE_LAYOUT_CHAT) == (
        "第一段。 第二段。"
    )
    assert normalize_response_text("啊？\n\n这样。\n", RESPONSE_LAYOUT_CHAT) == "啊？ 这样。"


def test_structured_layout_preserves_code_and_paragraphs_verbatim():
    text = "第一步：\n\n```python\nprint('hi')\n```\n\n第二步：运行。"
    assert normalize_response_text(text, RESPONSE_LAYOUT_STRUCTURED) == text


# -- 5/8. streaming: displayed text equals persisted text -------------------


class _StreamingClient:
    def __init__(self, chunks, *, finish_reason="stop"):
        self.chunks = list(chunks)
        self.finish_reason = finish_reason

    async def complete(self, messages, *, request_type, **kwargs):
        return ChatResponse(
            content="".join(self.chunks),
            model="fake",
            finish_reason=self.finish_reason,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        for chunk in self.chunks:
            yield StreamDelta(content=chunk)
        yield StreamDelta(
            finish_reason=self.finish_reason,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            usage_reported=True,
        )


async def _stream_turn(db, text, chunks):
    client = _StreamingClient(chunks)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEXT,
    )
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]
    action = await rt.repos.analysis.get_action(event.event_id)
    return events, action


async def test_chat_stream_persists_exactly_what_the_completed_event_shows(db):
    events, action = await _stream_turn(
        db, "你好", ["你好呀。\n", "\n今天怎么样？"]
    )
    assert [e.type for e in events][-1] == "response_completed"
    completed = events[-1]
    assert "\n" not in completed.text
    # The Chat surface renders ``completed.text``; history stores the same
    # string, so a collapsed display can never re-teach two paragraphs.
    assert action is not None
    assert action.response_text == completed.text


async def test_structured_stream_keeps_the_model_layout_and_persists_it(db):
    text = "帮我写一个 Python 教程"
    chunks = ["第一步：\n\n", "```python\nprint('hi')\n```\n\n", "第二步：运行。"]
    events, action = await _stream_turn(db, text, chunks)
    completed = events[-1]
    expected = "".join(chunks).strip()
    assert completed.text == expected
    assert "\n\n" in completed.text and "```python" in completed.text
    assert action is not None and action.response_text == expected


# -- 6. source / voice features --------------------------------------------


def _snapshot(**updates) -> RuntimeSnapshot:
    payload = dict(
        snapshot_id="rts_e1",
        event_id="evt_1",
        conversation_id="c1",
        speaker_person_id="person_1",
        speaker_label="Alice",
        received_at_utc=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
        occurred_at_utc=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
        timezone="Asia/Shanghai",
        timezone_source="user_config",
        utc_offset_seconds=8 * 3600,
        local_time="2026-11-04 18:00",
    )
    payload.update(updates)
    return RuntimeSnapshot(**payload)


def test_text_turn_says_it_has_no_voice_features():
    block = compile_runtime_context(_snapshot())
    assert "source=text" in block
    assert "voice_features=not_available" in block


def test_voice_turn_is_marked_as_voice():
    block = compile_runtime_context(_snapshot(source="voice"))
    assert "source=voice" in block
    assert "source=text" not in block


async def test_snapshot_builder_records_the_real_modality(rt):
    from companion.runtime_context.snapshot import RuntimeSnapshotBuilder

    person = await rt.registry.register(
        "Alice", "owner", allow_processing=True, allow_persistence=True
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你好",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.VOICE,
    )
    snapshot = await RuntimeSnapshotBuilder(rt.repos).build(event, persist=False)
    assert snapshot.source == "voice"
    assert snapshot.voice_features == "not_available"


# -- 6/12. history attribution ---------------------------------------------


def test_assistant_history_is_never_attributed_to_the_speaker():
    ctx = CognitiveContext(
        conversation_id="c1",
        current_event_id="e2",
        current_speaker_id="p1",
        current_speaker_label="Alice",
        current_text="我几点告诉你的？",
        turns=[
            CognitiveTurn(
                event_id="e1",
                role="human",
                speaker_person_id="p1",
                speaker_label="Alice",
                text="现在几点",
            ),
            CognitiveTurn(
                event_id="e1",
                role="assistant",
                speaker_label="assistant",
                text="18:43。",
            ),
        ],
        history_loaded=2,
    )
    messages = build_response_messages(ctx, _analysis())
    # No untrusted background block is emitted for this context, so the native
    # conversation starts right after the system message.
    assert messages[1] == {"role": "user", "content": "现在几点"}
    assert messages[2] == {"role": "assistant", "content": "18:43。"}
    assert messages[-1] == {"role": "user", "content": "我几点告诉你的？"}


# -- 6/13. memories=0 is not "no records" -----------------------------------


def test_empty_retrieval_is_not_reported_as_an_empty_database():
    empty = CognitiveContext(
        conversation_id="c1",
        current_event_id="e1",
        current_text="你知道我平常做什么吗？",
        memory_retrieval_status="empty",
        history_loaded=1,
    )
    system = build_response_messages(empty, _analysis())[0]["content"]
    assert "本次没有检索到相关记忆；这不代表长期记忆数据库为空。" in system
    assert "没有任何长期记忆记录" not in system

    none = CognitiveContext(
        conversation_id="c1",
        current_event_id="e2",
        current_text="你知道我平常做什么吗？",
        memory_retrieval_status="no_memories",
        history_loaded=1,
    )
    other = build_response_messages(none, _analysis())[0]["content"]
    assert "此人目前没有任何长期记忆记录。" in other


# -- 4/10. emotion: questions no longer imply curiosity --------------------


async def _agent_id(rt) -> str:
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


def _legacy_question_rule(agent_id: str, **updates) -> AppraisalRule:
    now = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    payload = dict(
        rule_id="emrule_legacy_question",
        agent_id=agent_id,
        name="curiosity_question",
        trigger_type="utterance",
        pattern={"question": True},
        emotion="CURIOSITY",
        valence_delta=0.05,
        arousal_delta=0.10,
        intensity=0.40,
        priority=10,
        status="ACTIVE",
        source="system",
        created_at=now,
        updated_at=now,
    )
    payload.update(updates)
    return AppraisalRule(**payload)


async def test_default_rules_never_seed_question_curiosity(rt):
    from companion.emotion.appraisal import DEFAULT_RULES

    names = {spec["name"] for spec in DEFAULT_RULES}
    assert "curiosity_question" not in names

    agent_id = await _agent_id(rt)
    await rt.emotion_service.ensure_defaults(agent_id)
    rules = await rt.emotion_service.repo.list_rules(agent_id)
    assert "curiosity_question" not in {rule.name for rule in rules}


async def test_unmodified_legacy_question_rule_is_disabled_and_audited(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.repo.insert_rule(
        _legacy_question_rule(agent_id)
    )

    await rt.emotion_service.ensure_defaults(agent_id)

    rules = {r.name: r for r in await rt.emotion_service.repo.list_rules(agent_id)}
    assert rules["curiosity_question"].status == "DISABLED"
    audits = await rt.repos.audit.list(limit=50, entity_type="emotion")
    assert any(a["action"] == "emotion_rule_retired" for a in audits)
    # Reversible: an operator can re-enable it explicitly.
    rule_id = rules["curiosity_question"].rule_id
    assert await rt.emotion_service.set_rule_status(
        agent_id, rule_id, "ACTIVE"
    )


async def test_user_edited_question_rule_is_never_overwritten(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.repo.insert_rule(
        _legacy_question_rule(
            agent_id,
            rule_id="emrule_user_question",
            pattern={"question": True, "topic": "engineering_design"},
            source="system",
        )
    )

    await rt.emotion_service.ensure_defaults(agent_id)

    rules = {r.name: r for r in await rt.emotion_service.repo.list_rules(agent_id)}
    assert rules["curiosity_question"].status == "ACTIVE"
    assert rules["curiosity_question"].pattern == {
        "question": True,
        "topic": "engineering_design",
    }


async def test_user_authored_question_rule_is_left_alone(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.repo.insert_rule(
        _legacy_question_rule(
            agent_id,
            rule_id="emrule_custom_question",
            source="user",
        )
    )

    await rt.emotion_service.ensure_defaults(agent_id)

    rules = {r.name: r for r in await rt.emotion_service.repo.list_rules(agent_id)}
    assert rules["curiosity_question"].status == "ACTIVE"
