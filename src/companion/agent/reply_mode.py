"""Local SOCIAL/TASK reply-mode routing (V0.5).

The mode is a *generation strategy* for the response call. It is not a new
cognitive action, never changes whether a reply is sent, never changes policy,
and never calls a model. Routing uses the already-validated cognitive result
and context; when the signals are weak or absent it falls back to SOCIAL.

The route is deliberately local and deterministic. A question mark, a disease
name or a technical noun is never enough on its own to select TASK.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Mapping

class ReplyMode(StrEnum):
    SOCIAL = "SOCIAL"
    TASK = "TASK"

# Acts that describe social business rather than a task request. A REQUEST can
# still be social ("陪我聊聊"), so primary_act=REQUEST alone is not TASK.
_SOCIAL_ACTS = frozenset(
    {
        "TEASING",
        "SMALL_TALK",
        "GREETING",
        "GRATITUDE",
        "SELF_DISCLOSURE",
        "COMPLAINT",
        "APOLOGY",
        "INVITATION",
        "REPAIR",
    }
)
_SOCIAL_FRAMES = frozenset({"CASUAL", "PLAYFUL", "SUPPORT", "REPAIR", "CONFLICT"})
_SOCIAL_ROLES = frozenset({"SETUP", "PAYOFF"})

# Strong, locally recognisable task requests. These are phrases, not isolated
# technical nouns, and each one carries an explicit request or instruction.
_TASK_PHRASES = (
    "帮我",
    "帮忙",
    "请帮",
    "麻烦你",
    "给我写",
    "写一封",
    "写一份",
    "写一个",
    "写段",
    "生成一个",
    "生成一段",
    "实现一个",
    "实现一下",
    "详细解释",
    "详细讲解",
    "从头教",
    "从头讲",
    "一步一步",
    "step by step",
    "怎么解决",
    "如何解决",
    "怎么实现",
    "如何实现",
    "怎么排查",
    "怎么调试",
    "如何处理",
    "怎么处理",
    "怎么修复",
    "如何修复",
    "分析一下",
    "分析这份",
    "分析这个",
    "列一下",
    "列出步骤",
    "设计方案",
    "翻译一下",
    "总结一下",
)

# Verb + concrete subject is a task even without "帮我". This keeps the route
# from becoming a keyword counter for a single noun.
_TASK_VERBS = (
    "解释",
    "讲解",
    "说明",
    "介绍",
    "分析",
    "排查",
    "调试",
    "修复",
    "优化",
    "设计",
    "实现",
    "写",
    "推导",
    "计算",
    "证明",
    "翻译",
    "总结",
)
_TASK_SUBJECTS = (
    "pid",
    "uart",
    "dma",
    "stm32",
    "python",
    "c++",
    "java",
    "javascript",
    "sql",
    "代码",
    "程序",
    "算法",
    "电路",
    "机器人",
    "imu",
    "mosfet",
    "寄存器",
    "中断",
    "协议",
    "原理",
    "机制",
    "报错",
    "bug",
    "数据",
    "邮件",
    "论文",
    "报告",
    "作业",
    "题目",
    "方案",
    "步骤",
)

# Health questions are TASK only when they explicitly ask for advice or an
# action. "我感冒了" on its own is self-disclosure and stays SOCIAL.
_HEALTH_MARKERS = (
    "感冒",
    "发烧",
    "咳嗽",
    "头疼",
    "头痛",
    "肚子疼",
    "胃疼",
    "过敏",
    "受伤",
    "流血",
    "疼",
    "症状",
)
_ADVICE_MARKERS = (
    "怎么办",
    "怎么处理",
    "如何处理",
    "应该怎么",
    "要怎么",
    "吃什么药",
    "要不要去医院",
    "严重吗",
    "如何治疗",
    "怎么治",
)

_CONTINUATION_MARKERS = (
    "继续",
    "接着说",
    "接着讲",
    "然后呢",
    "还有呢",
    "往下说",
    "再说说",
    "后来呢",
    "go on",
    "continue",
)

_PUNCTUATION = re.compile(r"[\s。！？!?，,、；;：:\"'“”‘’…\.]+")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def has_task_signal(text: str) -> bool:
    """Whether this text contains a local, explicit task request.

    A lone question mark, technical noun or disease name is intentionally not
    enough. Health questions need an explicit advice/action phrase.
    """
    value = _text(text).lower()
    if not value:
        return False
    if _has_any(value, _TASK_PHRASES):
        return True
    if _has_any(value, _TASK_VERBS) and _has_any(value, _TASK_SUBJECTS):
        return True
    if _has_any(value, _HEALTH_MARKERS) and _has_any(value, _ADVICE_MARKERS):
        return True
    return False


def _is_continuation(text: str) -> bool:
    value = _PUNCTUATION.sub("", _text(text).lower())
    if not value or len(value) > 16:
        return False
    return any(value == marker.replace(" ", "") or value.startswith(marker.replace(" ", "")) for marker in _CONTINUATION_MARKERS)


def _recent_task_context(context: Any) -> bool:
    """Look for a technical/task thread in the visible recent history.

    No new long-lived mode state is introduced. The decision is derived from
    the same context each turn; a later task request naturally takes over.
    """
    scene = _field(context, "scene_state", None)
    if isinstance(scene, Mapping):
        interaction = scene.get("interaction")
        if isinstance(interaction, Mapping) and str(interaction.get("frame") or "").upper() == "TASK":
            return True

    human_turns = [
        turn
        for turn in list(_field(context, "turns", []) or [])
        if str(_field(turn, "role", "")).lower() != "assistant"
    ]
    for turn in human_turns[-2:]:
        if has_task_signal(_field(turn, "text", "")):
            return True
    return False


def resolve_reply_mode(
    *,
    text: str,
    analysis: Any,
    context: Any,
) -> ReplyMode:
    """Choose SOCIAL or TASK from local signals, never from a model call.

    Priority:
    1. an explicit task request (local phrase/verb+subject, or a confident
       pragmatic TASK frame) -> TASK;
    2. an explicit social act/frame/setup-payoff -> SOCIAL;
    3. a short continuation inherits a recent task thread -> TASK, else SOCIAL;
    4. everything uncertain -> SOCIAL.
    """
    value = _text(text) or _text(_field(context, "current_text", ""))
    pragmatic = _field(analysis, "pragmatic", None)
    confidence = _confidence(_field(pragmatic, "confidence", 0.0))
    primary_act = str(_field(pragmatic, "primary_act", "OTHER") or "OTHER").upper()
    frame = str(_field(pragmatic, "interaction_frame", "UNKNOWN") or "UNKNOWN").upper()
    role = str(_field(pragmatic, "setup_payoff_role", "NONE") or "NONE").upper()

    # 1. Explicit task request. A confident TASK frame is supporting evidence,
    # but it is not used when the local text shows an unmistakable social act.
    if has_task_signal(value):
        return ReplyMode.TASK
    if frame == "TASK" and confidence >= 0.6 and primary_act in {
        "REQUEST",
        "INFORMATION",
        "QUESTION",
    }:
        return ReplyMode.TASK

    # 2. Explicit social behaviour. Low-confidence pragmatic output is not
    # allowed to override a clear local task signal above, and is not trusted
    # on its own here either.
    if confidence >= 0.35:
        if primary_act in _SOCIAL_ACTS:
            return ReplyMode.SOCIAL
        if role in _SOCIAL_ROLES:
            return ReplyMode.SOCIAL
        if frame in _SOCIAL_FRAMES:
            return ReplyMode.SOCIAL

    # 3. "继续" and similar short continuations inherit the visible thread.
    if _is_continuation(value):
        return ReplyMode.TASK if _recent_task_context(context) else ReplyMode.SOCIAL

    # 4. Ordinary conversation, and every uncertain/legacy analysis, is SOCIAL.
    return ReplyMode.SOCIAL
