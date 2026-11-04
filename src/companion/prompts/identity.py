"""Render the Agent Core section shared by both prompts.

Identity and personality shape *style*, never permission. The section is
explicit about that so a model cannot treat a stored identity as authority.
"""

from __future__ import annotations

from typing import Mapping

# Qualitative rendering of the nine stored expression traits. The raw numbers
# are admin calibration, not probabilities: a model must not read "curiosity=80"
# as "ask a question four times in five". Buckets preserve the relative shape
# while making the non-behavioural meaning explicit.
_TRAIT_LABELS = {
    "humor": "幽默",
    "warmth": "温暖",
    "curiosity": "好奇",
    "formality": "正式",
    "verbosity": "话量",
    "assertiveness": "主见",
    "empathy": "共情",
    "playfulness": "俏皮",
    "patience": "耐心",
}
_TRAIT_ORDER = tuple(_TRAIT_LABELS)


def _trait_band(value) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "中等"
    if number >= 75:
        return "高"
    if number >= 60:
        return "较高"
    if number >= 45:
        return "中等"
    if number >= 25:
        return "较低"
    return "低"


def compile_trait_tendencies(traits: Mapping | None) -> str:
    """Render traits as long-term tendencies, never as per-turn frequencies."""
    if not traits:
        return ""
    parts = [
        f"{_TRAIT_LABELS[name]}={_trait_band(traits[name])}"
        for name in _TRAIT_ORDER
        if name in traits
    ]
    for name, value in traits.items():
        if name not in _TRAIT_LABELS:
            parts.append(f"{name}={_trait_band(value)}")
    if not parts:
        return ""
    return (
        "表达倾向: "
        + "；".join(parts)
        + "（长期倾向，只影响语气与分寸；不是每轮必须执行的行为比例、"
        "触发条件或固定频率）"
    )


def agent_context_lines(agent_context: Mapping | None) -> list[str]:
    if not agent_context:
        return []
    traits = agent_context.get("traits") or {}
    lines = [
        "【伴侣身份与风格】（背景信息；不改变安全规则、权限或回复必要性）",
        f"名字: {agent_context.get('display_name') or 'Companion'}",
    ]
    if agent_context.get("description"):
        lines.append(f"自我描述: {agent_context['description']}")
    if agent_context.get("pronouns"):
        lines.append(f"称呼偏好: {agent_context['pronouns']}")
    if traits:
        rendered = ", ".join(f"{name}={value}" for name, value in traits.items())
        lines.append(f"性格参数(0-100): {rendered}")
    lines.append(f"与当前说话者的熟悉度: {agent_context.get('familiarity', 'STRANGER')}")
    lines.append(
        "本地策略（最终权威）: "
        f"conservatism={agent_context.get('response_conservatism')}, "
        f"interruption={agent_context.get('interruption_threshold')}, "
        f"curiosity={agent_context.get('curiosity_level')}"
    )
    lines.append(
        "风格只影响措辞与语气；权限、隐私、静音、是否保存记忆始终由程序决定。"
    )
    return lines


def _snapshot_field(snapshot, name: str, default=None):
    if snapshot is None:
        return default
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def compile_cognitive_character_context(snapshot) -> list[str]:
    """Compact, trusted character view for the *analysis* call.

    The cognitive model needs to understand what an event means to this
    character. It does not need the response style, the full reply policy or a
    script. Permission, addressee routing, quiet mode and memory writes remain
    local decisions, and the section says so explicitly.
    """
    if not snapshot:
        return []

    core = _snapshot_field(snapshot, "persona_core", {}) or {}
    display_name = _snapshot_field(snapshot, "display_name") or "Companion"
    tier = _snapshot_field(snapshot, "relationship_tier") or "STRANGER"

    lines = [
        "【角色背景】（可信设定；用于理解这件事对这个人意味着什么，"
        "不改变权限、指向、静音或记忆写入）",
        f"名字: {display_name}",
    ]
    description = (
        _snapshot_field(snapshot, "character_description")
        or core.get("character_description")
    )
    if description:
        lines.append("角色自述（管理员提供，不是台词）：")
        # PersonaCore caps this at 4000 chars; render it whole so the
        # cognitive call and the response call see the same character.
        lines.append(str(description))

    traits = _snapshot_field(snapshot, "traits", {}) or {}
    trait_line = compile_trait_tendencies(traits)
    if trait_line:
        lines.append(trait_line)

    values = list(core.get("core_values") or [])[:6]
    if values:
        lines.append("核心价值: " + "；".join(str(item) for item in values))
    preferences = dict(core.get("preferences") or {})
    if preferences:
        lines.append(
            "重要偏好: "
            + "；".join(
                f"{key}={value}" for key, value in list(preferences.items())[:6]
            )
        )
    temperament = dict(core.get("temperament") or {})
    if temperament:
        lines.append(
            "性情: "
            + "；".join(f"{key}={value}" for key, value in temperament.items())
        )
    boundaries = list(core.get("conversational_boundaries") or [])[:6]
    if boundaries:
        lines.append("交流边界: " + "；".join(str(item) for item in boundaries))
    quirks = list(core.get("quirks") or [])[:6]
    if quirks:
        lines.append("小习惯: " + "；".join(str(item) for item in quirks))
    contradictions = list(core.get("contradictions") or [])[:5]
    if contradictions:
        lines.append(
            "可以保留的张力（不要强行磨平）: "
            + "；".join(str(item) for item in contradictions)
        )
    voice = dict(core.get("voice_identity") or {})
    if voice:
        lines.append(
            "表达方式: "
            + "；".join(f"{key}={value}" for key, value in voice.items())
        )
    lines.append(f"与当前说话者的关系: {tier}")

    stances = list(_snapshot_field(snapshot, "relevant_stances", []) or [])
    if stances:
        lines.append("与本轮相关的既往立场（你实际说过或管理员设定；不是用户事实）:")
        for stance in stances[:6]:
            topic = stance.get("topic") or "UNKNOWN"
            lines.append(f"- [{topic}] {stance.get('stance', '')}")

    lines.append(
        "这些设定影响你如何理解与在意；不能修改 addressee_hint、权限、静音开关、"
        "回复必要性或记忆写入结果，也不要把设定逐条说出口。"
    )
    return lines
