"""Deterministic dialogue-style compilation (V0.4).

The model should not be handed a pile of raw integer traits and left to invent
a personality. This module converts the *existing* effective persona (global
persona + relationship preset + overrides) into a small, stable set of
expressive tendencies and short guidance lines.

Nothing here creates a new identity source, calls a model, or writes a memory.
A current-turn explicit request outranks stored preferences, which outrank the
trait thresholds. High humour or high empathy never becomes an obligation to
joke or to psychoanalyse the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel, Field

from companion.agent.models import EffectiveAgentContext


@dataclass(frozen=True)
class StyleThresholds:
    low: int = 35
    high: int = 70

    def band(self, value) -> str:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 50
        if number < self.low:
            return "LOW"
        if number >= self.high:
            return "HIGH"
        return "MODERATE"


class DialogueStyleProfile(BaseModel):
    """Provider-neutral expressive tendencies for one turn."""

    formality: str = "MODERATE"
    directness: str = "BALANCED"
    typical_length: str = "MEDIUM"
    humor_tendency: str = "WHEN_NATURAL"
    playfulness: str = "MODERATE"
    warmth_expression: str = "MODERATE"
    questioning_style: str = "WHEN_USEFUL"
    name_addressing: str = "RARELY"
    disagreement_style: str = "DIRECT_BUT_RESPECTFUL"
    empathy_expression: str = "MODERATE"
    explicit_request: Optional[str] = Field(default=None, max_length=500)
    personal_preferences: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict:
        return self.model_dump()


_LENGTH_HINTS = re.compile(
    r"(简短|简洁|短一点|说短|别啰嗦|不要太长|不用太长|一两句|brief|short)", re.I
)
_LONG_HINTS = re.compile(r"(详细|展开说|多说一点|讲详细|完整一点|longer|detail)", re.I)
_DIRECT_HINTS = re.compile(r"(直接|别绕|不用铺垫|直说|get to the point)", re.I)
_WARM_HINTS = re.compile(r"(温柔|耐心一点|语气好一点|gentle|kind tone)", re.I)
_NO_HUMOR_HINTS = re.compile(r"(别开玩笑|不要开玩笑|别闹|严肃一点|no jokes?)", re.I)
_NO_NAME_HINTS = re.compile(r"(别叫我|不要叫我|别喊我|不要喊我|不用叫我)", re.I)
_USE_NAME_HINTS = re.compile(r"(叫我|喊我)", re.I)
_NO_QUESTION_HINTS = re.compile(r"(别老问|不要总问|少问|别追问|不要再问)", re.I)
_SHORT_MEMORY_HINTS = re.compile(r"(喜欢简短|回复简短|简短回复|别啰嗦|不喜欢太长)", re.I)
_LONG_MEMORY_HINTS = re.compile(r"(喜欢详细|希望详细|多讲一点|喜欢长一点)", re.I)
_DIRECT_MEMORY_HINTS = re.compile(r"(喜欢直接|直接说|不要拐弯|说话直接)", re.I)
_NO_FORMAL_HINTS = re.compile(r"(不要敬语|别用敬语|不用客气|随意一点|别这么客气)", re.I)
_NO_HUMOR_MEMORY_HINTS = re.compile(r"(不喜欢开玩笑|不要开玩笑|别开玩笑)", re.I)
_NO_EMPATHY_ANALYSIS = re.compile(r"(不要分析我|别分析我|不要心理分析)", re.I)


def _texts(values: Iterable) -> list[str]:
    out: list[str] = []
    for value in values or []:
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, Mapping):
            content = value.get("content")
            if isinstance(content, str) and content.strip():
                out.append(content)
    return out


def compile_dialogue_style(
    effective: EffectiveAgentContext,
    *,
    memories: Optional[Iterable] = None,
    current_text: str = "",
    thresholds: StyleThresholds = StyleThresholds(),
) -> DialogueStyleProfile:
    """Compile a style profile from the already-resolved effective persona."""
    traits = dict(effective.traits or {})
    formality_band = thresholds.band(traits.get("formality", 50))
    assertiveness = thresholds.band(traits.get("assertiveness", 45))
    verbosity = thresholds.band(traits.get("verbosity", 45))
    humor = thresholds.band(traits.get("humor", 50))
    playfulness = thresholds.band(traits.get("playfulness", 50))
    warmth = thresholds.band(traits.get("warmth", 60))
    curiosity = thresholds.band(traits.get("curiosity", 60))
    empathy = thresholds.band(traits.get("empathy", 65))

    profile = DialogueStyleProfile(
        formality=formality_band,
        directness=(
            "HIGH"
            if assertiveness == "HIGH"
            else "LOW"
            if assertiveness == "LOW"
            else "BALANCED"
        ),
        typical_length=(
            "SHORT" if verbosity == "LOW" else "LONG" if verbosity == "HIGH" else "MEDIUM"
        ),
        humor_tendency=(
            "OCCASIONAL"
            if humor == "LOW"
            else "FREQUENT_WHEN_FITTING"
            if humor == "HIGH"
            else "WHEN_NATURAL"
        ),
        playfulness=playfulness,
        warmth_expression=warmth,
        questioning_style=(
            "RARELY"
            if curiosity == "LOW"
            else "OFTEN_BUT_NOT_REFLEXIVE"
            if curiosity == "HIGH"
            else "WHEN_USEFUL"
        ),
        empathy_expression=empathy,
        disagreement_style=(
            "DIRECT_BUT_RESPECTFUL"
            if assertiveness != "LOW"
            else "GENTLE_HONESTY"
        ),
    )

    preferences: list[str] = []
    for content in _texts(memories):
        if _NO_NAME_HINTS.search(content):
            profile.name_addressing = "NEVER"
            preferences.append("avoid_name_address")
        if _SHORT_MEMORY_HINTS.search(content):
            profile.typical_length = "SHORT"
            preferences.append("prefers_short")
        if _LONG_MEMORY_HINTS.search(content):
            profile.typical_length = "LONG"
            preferences.append("prefers_detail")
        if _DIRECT_MEMORY_HINTS.search(content):
            profile.directness = "HIGH"
            preferences.append("prefers_direct")
        if _NO_FORMAL_HINTS.search(content):
            profile.formality = "LOW"
            preferences.append("prefers_informal")
        if _NO_HUMOR_MEMORY_HINTS.search(content):
            profile.humor_tendency = "OCCASIONAL"
            preferences.append("avoid_humor")
        if _NO_QUESTION_HINTS.search(content):
            profile.questioning_style = "RARELY"
            preferences.append("avoid_questions")

    text = current_text or ""
    overrides: list[str] = []
    if _LENGTH_HINTS.search(text):
        profile.typical_length = "SHORT"
        overrides.append("current_request_short")
    if _LONG_HINTS.search(text):
        profile.typical_length = "LONG"
        overrides.append("current_request_detail")
    if _DIRECT_HINTS.search(text):
        profile.directness = "HIGH"
        overrides.append("current_request_direct")
    if _WARM_HINTS.search(text):
        profile.warmth_expression = "HIGH"
        overrides.append("current_request_warm")
    if _NO_HUMOR_HINTS.search(text):
        profile.humor_tendency = "OCCASIONAL"
        overrides.append("current_request_no_humor")
    if _NO_QUESTION_HINTS.search(text):
        profile.questioning_style = "RARELY"
        overrides.append("current_request_no_questions")
    if _NO_NAME_HINTS.search(text):
        profile.name_addressing = "NEVER"
        overrides.append("current_request_no_name")
    elif _USE_NAME_HINTS.search(text):
        profile.name_addressing = "WHEN_NATURAL"
        overrides.append("current_request_use_name")
    if _NO_EMPATHY_ANALYSIS.search(text):
        profile.empathy_expression = "LOW"
        overrides.append("current_request_no_analysis")

    if overrides:
        profile.explicit_request = ", ".join(overrides)
    profile.personal_preferences = sorted(set(preferences))
    profile.guidance = _guidance(profile)
    return profile


def _guidance(profile: DialogueStyleProfile) -> list[str]:
    lines: list[str] = [
        "先回应眼前这句话，不要机械复述对方或背诵人设。",
        "避免客服腔、模板开场白和每轮都总结。",
    ]
    if profile.typical_length == "SHORT":
        lines.append("默认一到三句话；除非对方明确要展开，不写长篇。")
    elif profile.typical_length == "LONG":
        lines.append("需要时可以把关键点讲完整，但仍避免空话和重复。")
    else:
        lines.append("长度随内容需要，保持自然，不刻意拉长。")
    if profile.formality == "LOW":
        lines.append("用自然口语，不需要敬语或过度客套。")
    elif profile.formality == "HIGH":
        lines.append("措辞礼貌完整，但不要变成公文或客服模板。")
    if profile.directness == "HIGH":
        lines.append("先给结论或直接回应，再补充必要背景。")
    elif profile.directness == "LOW":
        lines.append("表达可以柔和，但不要含糊回避真实立场。")
    if profile.humor_tendency == "FREQUENT_WHEN_FITTING":
        lines.append("语境合适时可以幽默；不要为了幽默硬讲笑话。")
    elif profile.humor_tendency == "OCCASIONAL":
        lines.append("本轮少用玩笑，避免把认真话题轻浮化。")
    if profile.questioning_style == "RARELY":
        lines.append("不要每轮都用反问或追问推进对话。")
    elif profile.questioning_style == "OFTEN_BUT_NOT_REFLEXIVE":
        lines.append("可以好奇，但不要连续盘问或强行找话题。")
    if profile.name_addressing == "NEVER":
        lines.append("不要用称呼开头，也不要用名字制造亲密感。")
    if profile.empathy_expression == "HIGH":
        lines.append("可以共情，但不要做心理诊断、贴标签或替对方下结论。")
    lines.append("不知道或不记得时就自然承认，不要编造记忆或现实细节。")
    return lines
