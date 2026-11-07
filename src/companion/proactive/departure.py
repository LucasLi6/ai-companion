"""Structurally validated explicit-departure signal (V0.9.9).

This is a conservative *floor*, not a router and not the decision itself. It
recognises only a clear, first-person departure statement or an unambiguous
farewell. Question forms ("真的不聊了？"), hypotheticals ("假如我走了"),
quotations ("他说他走了") and negations ("我不走") are refused by structure,
not by a bag of keywords. ``TurnState.CLOSING`` alone is never enough.

The LONG_IDLE judge still reads the conversation; this signal exists so that a
clear "我要下线了" can never be overridden by a model that missed it.
"""

from __future__ import annotations

import re

_QUESTION_END = re.compile(r"[?？]\s*$")
_QUESTION_PARTICLE = re.compile(r"(吗|呢|么|嘛)\s*[。.!！~～]*\s*$")
_HYPOTHETICAL = re.compile(r"(假如|如果|假设|万一|要是|比如说|比如|若是|倘若)")
_QUOTED = re.compile(
    r"(他说|她说|他们说|别人说|有人说|据说|听说|转述说)"
    r"|说[：:]|说[“\"']"
)
_NEGATED_DEPARTURE = re.compile(
    r"(不走了|不想走|不走|还没走|别走|不用走|没打算走|不离开|不下了|先不下)"
)
_FIRST_PERSON_DEPARTURE = re.compile(
    r"我(?:也|先|真的|确实|得|要|该|马上|这就|可能|准备)*"
    r"(?:走|撤|下线|离开|去睡|睡了|去休息|休息|去忙|忙去了|不聊了)"
)
_FAREWELL = re.compile(r"(晚安|再见|拜拜|回头见|改天聊|下次聊)")
# An explicit refusal to keep talking is as final as leaving, and must not be
# left to the model to notice.
_REFUSAL = re.compile(
    r"(?:不想|不愿意|不要再?|别)(?:再)?(?:跟|和|同)?(?:你|我)?"
    r"(?:聊|说话|说下去)"
    r"|(?:别|不要)(?:再)?(?:烦我|打扰我|管我)"
    r"|(?:安静(?:点|一下)?|闭嘴|别吵|别说了|先别说话)"
)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split())


def _is_structural_stop(value: str) -> bool:
    if not value:
        return False
    if _QUESTION_END.search(value) or _QUESTION_PARTICLE.search(value):
        return False
    if _HYPOTHETICAL.search(value):
        return False
    if _QUOTED.search(value):
        return False
    if _NEGATED_DEPARTURE.search(value):
        return False
    return True


def detect_explicit_departure(text: str) -> bool:
    """Return True only for a structurally clear departure statement.

    The function is intentionally high precision / low recall: a false
    negative is handled by the semantic judge, while a false positive would
    wrongly silence Ava.
    """
    value = _normalise(text)
    if not _is_structural_stop(value):
        return False
    return bool(
        _FIRST_PERSON_DEPARTURE.search(value) or _FAREWELL.search(value)
    )


def detect_explicit_stop(text: str) -> bool:
    """True for a clear departure *or* an explicit refusal to keep talking.

    This is the local hard floor used by the proactive manager: once the
    person leaves or asks Ava to stop, no proactive utterance may be
    generated, regardless of what the semantic judge would answer.
    """
    value = _normalise(text)
    if not _is_structural_stop(value):
        return False
    if _FIRST_PERSON_DEPARTURE.search(value) or _FAREWELL.search(value):
        return True
    return bool(_REFUSAL.search(value))


__all__ = ["detect_explicit_departure", "detect_explicit_stop"]
