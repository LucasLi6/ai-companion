"""Structured person-fact extraction and classification (V0.9.9.23).

The review pipeline used to accept free-form ``operation + field_key +
proposed_content`` proposals. That made two different questions collapse into
one:

* *what did the person actually say* (a fact with an assertion type, a time
  scope and evidence), and
* *how should the stored profile change* (a relation to an existing entry).

This module keeps those questions separate. The model fills an
:class:`ExtractedPersonFact` for every candidate fact and a
:class:`FactRelationDecision` for every proposed write. Trusted local code
classifies the fact and decides whether it belongs in the long-term Profile, a
short-lived Observation, or Scene/Memory. Local code never trusts the model's
classification when it would admit a one-off event or a third-party claim into
the Profile.

Nothing in this module writes to a database or calls a model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from companion.people.profile_models import profile_field_spec

MAX_FACT_VALUE_CHARS = 1200
MAX_FACT_ENTITY_CHARS = 120
MAX_FACT_QUOTES = 8
MAX_FACT_QUOTE_CHARS = 500
MAX_FACT_EVIDENCE_IDS = 16


class AssertionType(str):
    DIRECT = "DIRECT"          # 本人明确自述
    CORRECTION = "CORRECTION"  # 本人明确纠正/否定旧信息
    OBSERVED = "OBSERVED"      # 从行为中观察，本人未直接陈述
    UNCERTAIN = "UNCERTAIN"    # 语气含糊、无法确定
    HYPOTHETICAL = "HYPOTHETICAL"  # 假设、玩笑
    THIRD_PARTY = "THIRD_PARTY"    # 转述他人


class TemporalScope(str):
    STABLE = "STABLE"        # 长期稳定的属性
    ONGOING = "ONGOING"      # 当前持续进行
    TEMPORARY = "TEMPORARY"  # 临时状态
    EVENT = "EVENT"          # 一次性事件
    FUTURE = "FUTURE"        # 未来计划


class FactClass(str):
    PROFILE = "PROFILE"
    OBSERVATION = "OBSERVATION"
    SCENE_MEMORY = "SCENE_MEMORY"


class FactRelation(str):
    SAME = "SAME"
    EXTENDS = "EXTENDS"
    STATE_CHANGE = "STATE_CHANGE"
    CORRECTS = "CORRECTS"
    DISTINCT = "DISTINCT"
    UNCERTAIN = "UNCERTAIN"


ASSERTION_TYPES = (
    AssertionType.DIRECT,
    AssertionType.CORRECTION,
    AssertionType.OBSERVED,
    AssertionType.UNCERTAIN,
    AssertionType.HYPOTHETICAL,
    AssertionType.THIRD_PARTY,
)
TEMPORAL_SCOPES = (
    TemporalScope.STABLE,
    TemporalScope.ONGOING,
    TemporalScope.TEMPORARY,
    TemporalScope.EVENT,
    TemporalScope.FUTURE,
)
FACT_CLASSES = (
    FactClass.PROFILE,
    FactClass.OBSERVATION,
    FactClass.SCENE_MEMORY,
)
FACT_RELATIONS = (
    FactRelation.SAME,
    FactRelation.EXTENDS,
    FactRelation.STATE_CHANGE,
    FactRelation.CORRECTS,
    FactRelation.DISTINCT,
    FactRelation.UNCERTAIN,
)

# A relation is the semantic decision; the operation is its mechanical form.
RELATION_OPERATIONS: dict[str, str] = {
    FactRelation.SAME: "SAME",
    FactRelation.EXTENDS: "UPDATE",
    FactRelation.STATE_CHANGE: "UPDATE",
    FactRelation.CORRECTS: "CORRECT",
    FactRelation.DISTINCT: "ADD",
    FactRelation.UNCERTAIN: "SAME",  # rejected locally; never written
}

# Attributes for which an observed, repeated behaviour may become a Profile
# item. This is deliberately far narrower than "any observed act": a model may
# not turn repeated questions about robots into an interest.
OBSERVABLE_PROFILE_ATTRIBUTES = frozenset(
    {"behavior_pattern", "habit", "interaction_pattern"}
)

_FUTURE_PROFILE_ATTRIBUTES = frozenset(
    {"goal", "project", "plan", "short_term_goal", "long_term_goal"}
)

_ENTITY_ALIASES: dict[str, str] = {
    "我": "self",
    "自己": "self",
    "本人": "self",
    "用户": "self",
    "这个人": "self",
    "他": "self",
    "她": "self",
    "me": "self",
    "myself": "self",
    "self": "self",
    "兴趣": "interest",
    "爱好": "interest",
    "喜好": "interest",
    "技能": "skill",
    "能力": "skill",
    "项目": "project",
    "目标": "goal",
    "习惯": "habit",
    "性格": "personality",
    "价值观": "value",
    "职业": "occupation",
    "背景": "background",
    "交流偏好": "communication_preference",
    "沟通偏好": "communication_preference",
    "关系": "relationship",
    "偏好": "preference",
    "行为模式": "behavior_pattern",
    "互动模式": "interaction_pattern",
}

_ATTRIBUTE_ALIASES: dict[str, str] = {
    "兴趣": "interest",
    "爱好": "interest",
    "喜好": "like",
    "不喜欢": "dislike",
    "讨厌": "dislike",
    "技能": "skill",
    "擅长": "skill",
    "能力": "skill",
    "项目": "project",
    "正在做的事": "project",
    "目标": "goal",
    "计划": "plan",
    "习惯": "habit",
    "长期习惯": "habit",
    "性格": "personality_trait",
    "性格倾向": "personality_trait",
    "行为模式": "behavior_pattern",
    "互动模式": "interaction_pattern",
    "价值观": "value",
    "职业": "occupation",
    "教育": "education",
    "背景": "background",
    "交流偏好": "communication_preference",
    "沟通偏好": "communication_preference",
    "共同经历": "shared_context",
    "约定": "shared_context",
}

_PUNCT = re.compile(r"[\s，。！？、,.!?;；:：·\-—~～\"'“”‘’()（）\[\]【】]+")
_TRAILING = re.compile(r"[。.!！?？;；,，\s]+$")

_NON_NAME_VALUES = frozenset(
    {
        "顺其自然",
        "随便",
        "随意",
        "无所谓",
        "都行",
        "都可以",
        "看情况",
        "开心就好",
        "你看着办",
        "自己",
        "本人",
        "用户",
    }
)
_BAD_NAME_PREFIXES = ("自己", "你", "您", "我", "俺", "咱")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalise_entity_key(value: Any) -> str:
    """Stable, local entity key used by the derived fact index.

    It is a recall key, never an identity claim: two different strings may
    resolve to ``self`` without meaning the same fact. Every write is still
    decided on evidence and relation.
    """
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = _PUNCT.sub("", text)
    if not text:
        return "self"
    if text in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[text]
    for alias, canonical in _ENTITY_ALIASES.items():
        if text == alias or text.endswith(alias):
            return canonical
    return text[:MAX_FACT_ENTITY_CHARS]


def normalise_attribute(value: Any, *, field_key: str = "") -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = _PUNCT.sub("", text)
    if text in _ATTRIBUTE_ALIASES:
        return _ATTRIBUTE_ALIASES[text]
    for alias, canonical in _ATTRIBUTE_ALIASES.items():
        if alias and alias in text:
            return canonical
    if text:
        return text[:80]
    # Derive from the storage field only as a fallback: "interests.likes" ->
    # "likes", "knowledge.skills" -> "skills". A category alone is not an
    # attribute, so the suffix is the useful part.
    suffix = str(field_key or "").rsplit(".", 1)[-1]
    return suffix[:80] or "unknown"


def canonical_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = _TRAILING.sub("", text)
    return text[:MAX_FACT_VALUE_CHARS]


def looks_like_person_name(value: Any) -> bool:
    """Conservative name-likeness check for strict identity values.

    This is not a semantic name validator; it only rejects the phrases a broad
    regex otherwise captures from ordinary speech ("叫我顺其自然就好",
    "我叫自己顺其自然"). Rejection leaves the existing identity untouched.
    """
    text = _text(value).strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    if not text or len(text) > 40:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if text.casefold() in _NON_NAME_VALUES:
        return False
    if text.startswith(_BAD_NAME_PREFIXES):
        return False
    if any(word in text for word in _NON_NAME_VALUES):
        return False
    return True


class ExtractedPersonFact(BaseModel):
    """One fact exactly as the model extracted it from the person's words.

    ``fact_id`` is local-only bookkeeping: the model may omit it and the engine
    assigns ``f1``, ``f2`` ... in output order. ``classification`` is the
    model's claim; :func:`classify_fact` re-derives it locally and may only
    make it stricter.
    """

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(default="", max_length=64)
    entity: str = Field(default="self", max_length=MAX_FACT_ENTITY_CHARS)
    attribute: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=MAX_FACT_VALUE_CHARS)
    assertion_type: Literal[
        "DIRECT", "CORRECTION", "OBSERVED", "UNCERTAIN", "HYPOTHETICAL",
        "THIRD_PARTY",
    ]
    temporal_scope: Literal["STABLE", "ONGOING", "TEMPORARY", "EVENT", "FUTURE"]
    classification: Optional[
        Literal["PROFILE", "OBSERVATION", "SCENE_MEMORY"]
    ] = None
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)

    @field_validator("evidence_event_ids")
    @classmethod
    def _bounded_events(cls, value: list[str]) -> list[str]:
        return [
            str(item)[:128]
            for item in value
            if str(item or "").strip()
        ][:MAX_FACT_EVIDENCE_IDS]

    @field_validator("evidence_quotes")
    @classmethod
    def _bounded_quotes(cls, value: list[str]) -> list[str]:
        return [
            _text(item)[:MAX_FACT_QUOTE_CHARS]
            for item in value
            if _text(item)
        ][:MAX_FACT_QUOTES]


class FactRelationDecision(BaseModel):
    """How one extracted fact relates to the stored Profile.

    ``SAME`` and ``UNCERTAIN`` are terminal for the write path: ``SAME`` may
    attach evidence but never rewrites content, ``UNCERTAIN`` is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    relation: Literal[
        "SAME", "EXTENDS", "STATE_CHANGE", "CORRECTS", "DISTINCT", "UNCERTAIN"
    ]
    target_entry_ids: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=500)

    @field_validator("target_entry_ids")
    @classmethod
    def _bounded_targets(cls, value: list[str]) -> list[str]:
        return [
            str(item)[:128] for item in value if str(item or "").strip()
        ][:16]

    @property
    def operation(self) -> str:
        return RELATION_OPERATIONS.get(self.relation, "SAME")


def classify_fact(
    fact: ExtractedPersonFact, field_key: str
) -> FactClass:
    """Local classification of one extracted fact.

    The rules are intentionally conservative: when the model's claim and the
    local rule disagree, the stricter class wins. A one-off event, a temporary
    state, a hypothetical, a third-party report or a vague inference can never
    be classed as ``PROFILE`` by the model alone.
    """
    attribute = normalise_attribute(fact.attribute, field_key=field_key)
    field = str(field_key or "")

    # One-off shared context is exactly what used to pollute the Profile.
    if field == "interaction.shared_context":
        return FactClass.SCENE_MEMORY

    if fact.assertion_type in (
        AssertionType.HYPOTHETICAL,
        AssertionType.THIRD_PARTY,
        AssertionType.UNCERTAIN,
    ):
        return FactClass.OBSERVATION

    if fact.temporal_scope in (TemporalScope.TEMPORARY, TemporalScope.EVENT):
        return FactClass.SCENE_MEMORY

    if fact.assertion_type == AssertionType.OBSERVED:
        # Only a behaviour pattern seen across several independent messages is
        # profile-worthy. Everything else is an observation, even if the model
        # labelled it PROFILE.
        if (
            attribute in OBSERVABLE_PROFILE_ATTRIBUTES
            and fact.temporal_scope == TemporalScope.STABLE
            and len(set(fact.evidence_event_ids)) >= 2
        ):
            return FactClass.PROFILE
        return FactClass.OBSERVATION

    if fact.temporal_scope == TemporalScope.FUTURE:
        if attribute in _FUTURE_PROFILE_ATTRIBUTES:
            return FactClass.PROFILE
        return FactClass.OBSERVATION

    return FactClass.PROFILE


def fact_index_row(
    *,
    person_id: str,
    entry_id: str,
    field_key: str,
    content: str,
    fact: Optional[ExtractedPersonFact] = None,
) -> dict[str, Any]:
    """Derived row for ``person_profile_fact_index``.

    The index is a cache. It is rebuilt from ACTIVE entries whenever it may be
    stale, so a missing or incorrect row cannot authorize a write.
    """
    spec = profile_field_spec(field_key)
    temporal = (
        fact.temporal_scope if fact is not None else TemporalScope.STABLE
    )
    entity = fact.entity if fact is not None else "self"
    attribute = fact.attribute if fact is not None else (spec.label if spec else field_key)
    value = fact.value if fact is not None else content
    aliases = []
    if fact is not None and fact.value and fact.value != content:
        aliases.append(content)
    return {
        "entry_id": entry_id,
        "person_id": person_id,
        "entity_key": normalise_entity_key(entity),
        "attribute": normalise_attribute(attribute, field_key=field_key),
        "temporal_scope": str(temporal),
        "canonical_value": canonical_value(value),
        "aliases_json": aliases,
        "updated_at": "",
    }


__all__ = [
    "ASSERTION_TYPES",
    "AssertionType",
    "ExtractedPersonFact",
    "FACT_CLASSES",
    "FACT_RELATIONS",
    "FactClass",
    "FactRelation",
    "FactRelationDecision",
    "MAX_FACT_VALUE_CHARS",
    "OBSERVABLE_PROFILE_ATTRIBUTES",
    "RELATION_OPERATIONS",
    "TEMPORAL_SCOPES",
    "TemporalScope",
    "canonical_value",
    "classify_fact",
    "fact_index_row",
    "looks_like_person_name",
    "normalise_attribute",
    "normalise_entity_key",
]
