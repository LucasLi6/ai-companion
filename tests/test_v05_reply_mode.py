"""V0.5 — local SOCIAL/TASK reply-mode routing.

The route is a generation strategy, not a model call and not an action
decision. These tests pin the four rules from the specification:

1. an explicit task request is TASK;
2. an explicit social act is SOCIAL;
3. a short continuation inherits the visible thread;
4. uncertain or legacy input falls back to SOCIAL.

A question mark, a disease name or an isolated technical noun must never be
enough on its own.
"""

from __future__ import annotations

from companion.agent.reply_mode import (
    ReplyMode,
    has_task_signal,
    resolve_reply_mode,
)
from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult, PragmaticAnalysis


def _context(*turns, text="你好"):
    return CognitiveContext(
        conversation_id="c1",
        current_event_id="evt_now",
        current_speaker_id="person_a",
        current_speaker_label="Alice",
        current_text=text,
        current_addressee_hint="agent",
        turns=list(turns),
    )


def _result(pragmatic=None):
    return CognitiveResult(
        event_id="evt_now",
        meaning="fake analysis",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        provider="fake",
        pragmatic=pragmatic,
    )


def _resolve(text, *, pragmatic=None, context=None):
    return resolve_reply_mode(
        text=text,
        analysis=_result(pragmatic),
        context=context or _context(text=text),
    )


def _human(event_id, text):
    return CognitiveTurn(
        event_id=event_id,
        role="human",
        speaker_person_id="person_a",
        speaker_label="Alice",
        text=text,
    )


# -- 1. explicit task request ------------------------------------------------


def test_explicit_request_phrase_is_task():
    assert _resolve("帮我解释一下 PID 控制") is ReplyMode.TASK
    assert _resolve("麻烦你详细讲一下这个算法") is ReplyMode.TASK
    assert _resolve("帮我写封道歉邮件吧。") is ReplyMode.TASK


def test_task_verb_plus_concrete_subject_is_task():
    assert _resolve("解释一下 UART 为什么收不到数据") is ReplyMode.TASK
    assert _resolve("把这段 Python 代码优化一下") is ReplyMode.TASK


def test_confident_pragmatic_task_frame_is_task():
    pragmatic = PragmaticAnalysis(
        primary_act="INFORMATION",
        interaction_frame="TASK",
        confidence=0.8,
    )
    assert _resolve("这东西怎么用", pragmatic=pragmatic) is ReplyMode.TASK


def test_low_confidence_task_frame_is_not_enough():
    pragmatic = PragmaticAnalysis(
        primary_act="INFORMATION",
        interaction_frame="TASK",
        confidence=0.4,
    )
    assert _resolve("这东西怎么用", pragmatic=pragmatic) is ReplyMode.SOCIAL


# -- 2. explicit social acts -------------------------------------------------


def test_explicit_social_act_is_social():
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )
    assert _resolve("我今天真的累坏了", pragmatic=pragmatic) is ReplyMode.SOCIAL


def test_social_act_beats_a_merely_technical_context():
    # The turn is self-disclosure about a hard debugging day, not a request.
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.8,
    )
    assert _resolve("调了一整天的 bug，烦死了", pragmatic=pragmatic) is ReplyMode.SOCIAL


def test_setup_payoff_role_is_social():
    pragmatic = PragmaticAnalysis(
        primary_act="OTHER",
        interaction_frame="PLAYFUL",
        setup_payoff_role="PAYOFF",
        confidence=0.7,
    )
    assert _resolve("原来你也会这么说", pragmatic=pragmatic) is ReplyMode.SOCIAL


# -- 3. continuations inherit the visible thread ----------------------------


def test_continuation_inherits_a_recent_task_thread():
    context = _context(
        _human("evt_h1", "帮我看看这段 STM32 UART 代码"),
        text="继续",
    )
    assert _resolve("继续", context=context) is ReplyMode.TASK


def test_continuation_without_task_context_is_social():
    context = _context(
        _human("evt_h1", "今天天气真好"),
        text="继续",
    )
    assert _resolve("继续", context=context) is ReplyMode.SOCIAL


def test_scene_task_frame_carries_a_continuation():
    context = _context(text="然后呢")
    context = context.model_copy(
        update={"scene_state": {"interaction": {"frame": "TASK"}}}
    )
    assert _resolve("然后呢", context=context) is ReplyMode.TASK


# -- 4. uncertain input falls back to SOCIAL --------------------------------


def test_question_mark_alone_is_not_a_task():
    assert _resolve("你还好吗？") is ReplyMode.SOCIAL
    assert _resolve("真的？") is ReplyMode.SOCIAL


def test_disease_name_alone_is_social():
    assert _resolve("我感冒了") is ReplyMode.SOCIAL
    assert _resolve("昨天有点发烧，今天好多了") is ReplyMode.SOCIAL


def test_health_question_with_advice_request_is_task():
    assert _resolve("我感冒了，怎么办？") is ReplyMode.TASK
    assert _resolve("头疼得厉害，应该怎么处理？") is ReplyMode.TASK


def test_technical_noun_alone_is_not_a_task():
    assert _resolve("这个 PID 好难啊") is ReplyMode.SOCIAL
    assert _resolve("又是 UART。") is ReplyMode.SOCIAL


def test_missing_pragmatic_result_defaults_to_social():
    assert _resolve("随便聊聊") is ReplyMode.SOCIAL


def test_empty_text_falls_back_to_context_text():
    context = _context(text="帮我修一下这个 bug")
    assert _resolve("", context=context) is ReplyMode.TASK


# -- signal helper -----------------------------------------------------------


def test_has_task_signal_requires_more_than_a_noun():
    assert has_task_signal("帮我解释 PID") is True
    assert has_task_signal("PID") is False
    assert has_task_signal("我感冒了") is False
    assert has_task_signal("我感冒了，吃什么药") is True
    assert has_task_signal("？") is False


def test_reply_mode_values_are_stable():
    assert ReplyMode.SOCIAL.value == "SOCIAL"
    assert ReplyMode.TASK.value == "TASK"
    assert str(ReplyMode.SOCIAL) == "SOCIAL"
