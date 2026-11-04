"""Local detector: is this candidate a passing emotional state?

V0.4.2 separates long-term memory from the emotion layer. A sentence such as
"我现在很焦虑" describes a moment; storing it as a durable personal fact would
let one bad afternoon become part of the person's profile and be retrieved as
context weeks later. This module is a conservative, rule-based filter:

- it requires an explicit *momentary* time anchor (now, today, this moment,
  just now) **and** a first-person emotion word;
- it does not fire for durative or preference statements ("我一直很焦虑",
  "我喜欢安静", "我这个人容易紧张");
- it is deliberately narrow, so it can miss paraphrases — that is preferable
  to dropping a real preference.

It is used only to refuse a *durable* write. The transient state still flows
through the scene/emotion layers for the current conversation.
"""

from __future__ import annotations

import re

_MOMENTARY = re.compile(
    r"(现在|目前|此刻|这会儿|今天|今晚|刚刚|刚才|这一会儿|"
    r"最近|这两天|这几天|这段时间|"
    r"\bright\s+now\b|\bat\s+the\s+moment\b|\btoday\b|\bjust\s+now\b|"
    r"\blately\b|\bthese\s+days\b)",
    re.IGNORECASE,
)

_FEELING = re.compile(
    r"(难过|伤心|焦虑|紧张|害怕|生气|愤怒|烦躁|郁闷|沮丧|低落|委屈|"
    r"孤单|孤独|崩溃|心情不好|心情很差|压力(?:很|特别|好)?大|"
    r"很累|好累|累坏了|疲(?:惫|劳)|"
    r"\bsad\b|\banxious\b|\bangry\b|\bupset\b|\blonely\b|\bexhausted\b|"
    r"\bstressed\b|\boverwhelmed\b|\bdepressed\b)",
    re.IGNORECASE,
)

_DURATIVE = re.compile(
    r"(一直|总是|经常|经常性|通常|平常|平时|一般来说|性格|我这个人|"
    r"对我来说|我喜欢|我不喜欢|我讨厌|我偏好|我习惯|我很看重|"
    r"\balways\b|\busually\b|\bgenerally\b|\bprefer\b|\blike\b)",
    re.IGNORECASE,
)

_FIRST_PERSON = re.compile(r"(我|\bI\b|\bme\b|\bmy\b)", re.IGNORECASE)


def is_temporary_emotion_statement(content: str) -> bool:
    """Return ``True`` for a first-person, momentary emotional state."""
    value = " ".join(str(content or "").split()).strip()
    if not value:
        return False
    if _DURATIVE.search(value):
        return False
    if not _FIRST_PERSON.search(value):
        return False
    return bool(_MOMENTARY.search(value) and _FEELING.search(value))
