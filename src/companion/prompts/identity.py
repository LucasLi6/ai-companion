"""Render the Agent Core section shared by both prompts.

Identity and personality shape *style*, never permission. The section is
explicit about that so a model cannot treat a stored identity as authority.
"""

from __future__ import annotations

from typing import Mapping


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
