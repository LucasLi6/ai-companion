"""Deterministic detection of explicit first-person identity statements.

This module is intentionally narrow. It only recognises direct statements
such as "我叫李约" / "我的英文名是 Lucas" / "以后叫我 Lucas". It never infers
anything, and it refuses questions, hypotheticals, quotations and third-person
reports. Both the trusted direct-write path and the turn hook use exactly this
one detector, so the detection result is the evidence a write is based on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_CJK = r"一-鿿"
_NAME_TOKEN = (
    rf"(?:[A-Za-z][A-Za-z0-9.'’\-]*(?:\s+[A-Za-z][A-Za-z0-9.'’\-]*){{0,2}}"
    rf"|[{_CJK}·]{{1,12}})"
)

_QUESTION_END = re.compile(r"[?？]\s*$")
_QUESTION_PARTICLE = re.compile(r"(吗|呢|么|嘛)\s*[。.!！~～]*\s*$")
_HYPOTHETICAL = re.compile(r"(假如|如果|假设|万一|要是|比如说|比如|若是|倘若)")
_QUOTED = re.compile(
    r"(他说|她说|他们说|她们说|别人说|有人说|据说|听说|转述说|"
    r"朋友说|同事说|同学说|老师说)|说[：:]|说[“\"']"
)
_THIRD_PERSON = re.compile(
    r"(他|她|他们|她们|朋友|同事|同学|老师|别人|有人)\s*(?:说|叫|让|希望|想)"
)
_THIRD_PERSON_SUBJECT = re.compile(
    r"(?:^|[^你您])(?:他|她|他们|她们|朋友|同事|同学|老师|别人|有人)\s*$"
)
_TRAILING_CUT = re.compile(
    r"(?:以后|之后|今后|平时|现在|就好|就行|就可以|然后|吧|呀|啊|哦|嘛|呢|你|您)"
)


@dataclass(frozen=True)
class DetectedSelfReport:
    field_key: str
    value: str
    quote: str


_SELF_REPORT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # A direct introduction using "我是李约". Kept narrower than an unrestricted
    # "我是..." so occupations and roles ("我是学生") do not become names.
    (
        "identity.name",
        re.compile(
            r"(?<![\w])我是\s*(?P<value>"
            r"[李王张刘陈杨黄赵周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文]"
            r"[一-鿿]{1,2})(?=$|[，。.!！?？\s])"
        ),
    ),
    # identity.name: "我叫李约" / "我的名字是李约"
    (
        "identity.name",
        re.compile(
            rf"我(?:的)?(?:中文)?(?:名字|名)(?:是|叫|为)\s*(?P<value>{_NAME_TOKEN})"
        ),
    ),
    ("identity.name", re.compile(rf"我(?:叫|名叫)\s*(?P<value>{_NAME_TOKEN})")),
    # preferred name: English name, or an explicit future form of address
    (
        "identity.preferred_name",
        re.compile(rf"(?:我的)?英文名(?:是|叫|为)?\s*(?P<value>{_NAME_TOKEN})"),
    ),
    (
        "identity.preferred_name",
        re.compile(
            rf"(?:以后|之后|今后|平时|你可以|你可以直接)(?:叫我|喊我)\s*"
            rf"(?P<value>{_NAME_TOKEN})"
        ),
    ),
    # nicknames
    (
        "identity.nicknames",
        re.compile(
            rf"我的(?:昵称|外号|小名)(?:是|叫|为)\s*(?P<value>{_NAME_TOKEN})"
        ),
    ),
    # explicit address preference
    (
        "communication.address_preference",
        re.compile(rf"(?:请)?(?:称呼我|称我)(?:为|作)?\s*(?P<value>{_NAME_TOKEN})"),
    ),
    (
        "communication.address_preference",
        re.compile(
            rf"(?:以后)?(?:叫我|喊我)\s*(?P<value>{_NAME_TOKEN})\s*(?:就行|就好|吧)"
        ),
    ),
    # a bare "叫我李约" is a preferred name, not an address syntax
    (
        "identity.preferred_name",
        re.compile(rf"(?:叫我|喊我)\s*(?P<value>{_NAME_TOKEN})"),
    ),
)


def normalise(text: str) -> str:
    return " ".join(str(text or "").split())


def is_question(text: str) -> bool:
    return bool(_QUESTION_END.search(text) or _QUESTION_PARTICLE.search(text))


def clean_name(raw: str) -> Optional[str]:
    text = str(raw or "").strip()
    text = text.strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    # "李约以后你就叫我约约" must not become the name "李约以后".
    text = _TRAILING_CUT.split(text, maxsplit=1)[0].strip()
    text = text.strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    if not text:
        return None
    if any(ch.isdigit() for ch in text):
        return None
    if len(text) > 40:
        return None
    return text


def detect_self_reports(text: str) -> tuple[DetectedSelfReport, ...]:
    """Return explicit first-person identity statements in one message."""
    value = normalise(text)
    if not value:
        return ()
    if is_question(value) or _HYPOTHETICAL.search(value) or _QUOTED.search(value):
        return ()

    found: dict[str, DetectedSelfReport] = {}
    for field_key, pattern in _SELF_REPORT_PATTERNS:
        if field_key in found:
            continue
        for match in pattern.finditer(value):
            prefix = value[max(0, match.start() - 8) : match.start()]
            if _THIRD_PERSON.search(prefix) or _THIRD_PERSON_SUBJECT.search(prefix):
                continue
            cleaned = clean_name(match.group("value"))
            if not cleaned:
                continue
            # Guard role descriptions resembling a Chinese name in a bare
            # "我是..." introduction (e.g. "我是高三生").
            if match.group(0).startswith("我是") and cleaned in {
                "高三生", "高中生", "大学生", "研究生", "初中生", "小学生",
                "王老师", "李老师", "程序员", "中国人", "巴西人",
            }:
                continue
            found[field_key] = DetectedSelfReport(
                field_key=field_key, value=cleaned, quote=value[:200]
            )
            break
    return tuple(found.values())


def has_explicit_self_report(
    text: str, field_key: str, value: str
) -> bool:
    """Whether ``text`` explicitly states exactly this field/value pair."""
    wanted = normalise(value)
    return any(
        item.field_key == field_key and normalise(item.value) == wanted
        for item in detect_self_reports(text)
    )


__all__ = [
    "DetectedSelfReport",
    "clean_name",
    "detect_self_reports",
    "has_explicit_self_report",
    "is_question",
    "normalise",
]
