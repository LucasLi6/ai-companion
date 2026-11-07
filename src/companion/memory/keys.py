"""Entity/attribute claim keys and evidence grounding.

Two memories may only replace each other when they describe the *same
attribute of the same entity*. A shared broad topic is not enough: "I like
jazz" and "I like programming" are both ``preference`` facts but distinct
entities, and "I am building a quadruped robot" and "I am building a milk-tea
machine" are both ``project_state`` facts but distinct projects. Treating the
topic alone as the identity is what let unrelated statements overwrite each
other.

This module extracts a conservative claim key from *content text* only. The
model's ``topic`` hint is still used for retrieval/identity questions, but
conflict detection relies on these keys. When no key can be extracted, callers
must not guess: the claim is stored without a conflict key and can never
silently overwrite an unrelated memory.

It also provides :func:`claim_is_grounded`, the evidence gate for authoritative
self-reports. A model-supplied, verbatim quote is *not* sufficient evidence by
itself — the quoted text must actually support the proposed claim. Otherwise a
model could quote "I wrote some code today" and propose "I especially love
programming" as a directly supported fact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Trailing sentence punctuation is stripped before matching so "我喜欢爵士乐。"
# and "我喜欢爵士乐" normalise identically.
_PUNCT = re.compile(r"[\s　，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】《》<>]+")
_LEAD_SUBJECT = re.compile(
    r"^(?:用户|说话人|该用户|这个人|对方|user|speaker|they)\s*",
    re.IGNORECASE,
)
_TRAILING_PARTICLES = re.compile(r"(?:了|啦|呀|啊|呢|吧|哦|嘛|而已|的)+$")

_NAME = re.compile(
    r"^(?:(?:我的?|本人的?)(?:名字|姓名)(?:是|叫|为)|我叫|称呼我为|称呼我叫|自称(?:名字|姓名)(?:是|叫|为))"
    r"\s*(.+)$"
)
_AGE = re.compile(r"^(?:我)?(?:(?:今年|现在|已经)\s*)?(\d{1,3})\s*岁")
_BIRTHDAY = re.compile(r"^(?:(?:我的?)(?:生日|出生日期)(?:是|在|为)?)\s*(.+)$")
_LOCATION = re.compile(
    r"^(?:我)?(?:现在|目前|已经)?(?:住在|居住在|来自|搬到了|搬到|家在)\s*(.+)$"
)
_OCCUPATION = re.compile(
    r"^(?:我)?(?:现在|目前)?(?:是|是一名|是一个|是个|做|当|担任)(?:一名|一个|位)?\s*"
    r"(.+?(?:工程师|程序员|设计师|老师|教师|医生|学生|律师|会计|研究员|经理|作家|画家|歌手|演员))$"
)
_RELATION = re.compile(
    r"^(?:(?:我的?)(朋友|妈妈|爸爸|母亲|父亲|哥哥|姐姐|弟弟|妹妹|同事|伴侣|"
    r"女朋友|男朋友|妻子|丈夫|儿子|女儿|老师|老板))(?:是|叫|名叫)\s*(.+)$"
)
_GOAL = re.compile(r"^(?:(?:我的?)(?:目标|计划|打算)(?:是|为|就是)?)\s*(.+)$")

_PREFERENCE_VERBS = (
    "不喜欢",
    "不怎么喜欢",
    "不太喜欢",
    "喜欢",
    "偏爱",
    "热爱",
    "爱",
    "讨厌",
    "厌恶",
    "享受",
    "享受",
    "like",
    "love",
    "hate",
    "enjoy",
    "prefer",
)
_PREFERENCE = re.compile(
    r"^(?:我|本人)?(?:很|非常|特别|最|也|还|一直|现在|其实|真的|挺|比较|有点|十分|超级|"
    r"尤其|更|只|就|都)*\s*("
    + "|".join(re.escape(v) for v in _PREFERENCE_VERBS)
    + r")\s*(.+)$",
    re.IGNORECASE,
)
_PROJECT_VERBS = (
    "开发",
    "研发",
    "研究",
    "编写",
    "写",
    "构建",
    "搭建",
    "制作",
    "做",
    "设计",
    "building",
    "developing",
    "designing",
    "writing",
    "working on",
)
_PROJECT = re.compile(
    r"^(?:我)?(?:正在|正|在|目前|现在|最近|一直|已)?\s*(?:"
    + "|".join(re.escape(v) for v in _PROJECT_VERBS)
    + r")\s*(?:着|了|过)?\s*(.+)$",
    re.IGNORECASE,
)
_OBJECT_LEAD = re.compile(r"^(?:一个|一款|一台|一套|一项|一件|一支|一本|个|款|台|套|项|件|支|本)\s*")
_PRONOUN_OBJECT = frozenset(
    {
        "它",
        "这个",
        "那个",
        "这些",
        "那些",
        "这件事",
        "此事",
        "这样",
        "那样",
        "it",
        "this",
        "that",
        "them",
    }
)

_PREFERENCE_CUES = ("喜欢", "爱", "讨厌", "偏爱", "享受", "like", "love", "hate", "enjoy", "prefer")
_PROJECT_CUES = (
    "开发",
    "研发",
    "研究",
    "编写",
    "构建",
    "搭建",
    "制作",
    "做",
    "设计",
    "building",
    "developing",
    "working on",
)
_IDENTITY_CUES = {
    "identity:name": ("叫", "名字", "姓名", "name"),
    "identity:age": ("岁", "年龄", "age"),
    "identity:birthday": ("生日", "出生", "birthday"),
    "identity:location": ("住在", "来自", "地址", "住址", "live", "from", "address"),
    "identity:occupation": ("工作", "职业", "工程师", "老师", "医生", "job", "work"),
}


@dataclass(frozen=True)
class ClaimKey:
    """A conservative identity for one factual attribute of one entity."""

    entity_key: str
    attribute_key: str
    value: str
    core_phrase: str


def normalize_text(text: str) -> str:
    """Case-fold and strip whitespace/punctuation, preserving wording order."""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return _PUNCT.sub("", folded).strip()


def _strip_lead(text: str) -> str:
    value = text.strip()
    value = _LEAD_SUBJECT.sub("", value)
    return value


def _clean_value(value: str) -> str:
    value = _strip_lead(value)
    value = _TRAILING_PARTICLES.sub("", value)
    value = _OBJECT_LEAD.sub("", value)
    value = value.strip(" 　的了着过")
    return value


def _usable_entity(value: str) -> str | None:
    entity = normalize_text(_clean_value(value))
    if not entity or entity in _PRONOUN_OBJECT or len(entity) < 1:
        return None
    return entity


def extract_claim_key(content: str) -> ClaimKey | None:
    """Extract ``(entity, attribute)`` from a self-descriptive statement.

    Returns ``None`` when the statement does not describe a recognisable
    attribute. Callers must treat ``None`` as "do not conflict-match".
    """
    raw = _strip_lead(str(content).strip())
    if not raw:
        return None
    core = normalize_text(raw)
    if not core:
        return None

    name = _NAME.match(raw)
    if name:
        value = _clean_value(name.group(1))
        if value:
            return ClaimKey("self", "identity:name", value, core)

    age = _AGE.match(raw)
    if age:
        return ClaimKey("self", "identity:age", age.group(1), core)

    birthday = _BIRTHDAY.match(raw)
    if birthday:
        value = _clean_value(birthday.group(1))
        if value:
            return ClaimKey("self", "identity:birthday", value, core)

    location = _LOCATION.match(raw)
    if location:
        value = _usable_entity(location.group(1))
        if value:
            return ClaimKey("self", "identity:location", value, core)

    occupation = _OCCUPATION.match(raw)
    if occupation:
        value = _clean_value(occupation.group(1))
        if value:
            return ClaimKey("self", "identity:occupation", value, core)

    relation = _RELATION.match(raw)
    if relation:
        role = relation.group(1).strip()
        value = _clean_value(relation.group(2))
        if value:
            return ClaimKey(
                normalize_text(role), f"relationship:{normalize_text(role)}", value, core
            )

    goal = _GOAL.match(raw)
    if goal:
        value = _clean_value(goal.group(1))
        if value:
            return ClaimKey("self", "goal", value, core)

    preference = _PREFERENCE.match(raw)
    if preference:
        verb = preference.group(1).casefold()
        entity = _usable_entity(preference.group(2))
        if entity:
            return ClaimKey(entity, "preference", verb, core)

    project = _PROJECT.match(raw)
    if project:
        entity = _usable_entity(project.group(1))
        if entity:
            return ClaimKey(entity, "project:status", core, core)

    return None


def claim_is_grounded(
    content: str,
    utterance_text: str,
    *,
    evidence_span: str | None = None,
) -> bool:
    """Whether ``utterance_text`` directly supports the proposed ``content``.

    Direct support means the claim's own wording (or, for preference/project
    claims, the same entity together with the matching verb class) appears in
    the utterance. A verbatim model-supplied span is never sufficient on its
    own: it is only consulted as one more text the claim must be grounded in.
    """
    if not utterance_text:
        return False
    # Compact forms make punctuation/spacing differences irrelevant.
    content_n = normalize_text(_strip_lead(content))
    utterance_n = normalize_text(utterance_text)
    if not content_n or not utterance_n:
        return False
    if content_n in utterance_n:
        return True

    span_n = normalize_text(evidence_span or "")
    if span_n and content_n in span_n:
        return True

    key = extract_claim_key(content)
    if key is None:
        return False
    if key.core_phrase and key.core_phrase in utterance_n:
        return True

    entity = key.entity_key
    attribute = key.attribute_key
    if attribute == "preference" and entity and entity != "self":
        return entity in utterance_n and any(
            cue in utterance_n for cue in _PREFERENCE_CUES
        )
    if attribute == "project:status" and entity and entity != "self":
        return entity in utterance_n and any(
            cue in utterance_n for cue in _PROJECT_CUES
        )
    # A factual extraction may paraphrase "我是李约" as "用户自称名字为李约".
    # Require the same *entire* direct self-introduction, not a coincidental
    # occurrence of the name somewhere in third-party speech. This preserves
    # the evidence gate without turning an unquoted inference into ACTIVE.
    if attribute == "identity:name" and key.value:
        value = normalize_text(key.value)
        if value and utterance_n in {
            f"我是{value}", f"我叫{value}", f"我的名字是{value}",
            f"我的名字叫{value}", f"我名叫{value}",
        }:
            return True
    cues = _IDENTITY_CUES.get(attribute)
    if cues and key.value:
        value = normalize_text(key.value)
        return bool(value) and value in utterance_n and any(
            cue in utterance_n for cue in cues
        )
    return False
