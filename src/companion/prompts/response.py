"""Response-generation prompt (versioned).

The response provider is called *only* when the policy finalises RESPOND, and
it reuses the exact same prepared :class:`CognitiveContext` the cognitive
provider analysed — no second history fetch, no second retrieval.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext
from companion.domain.models import CognitiveResult

RESPONSE_PROMPT_VERSION = "response-v1"

SYSTEM_PROMPT = """\
你是这个多用户 AI 伴侣的「回应者」。你现在要为当前说话者生成一句自然、得体的回复。\
你会收到：最近的对话、相关的长期记忆、当前输入，以及认知分析器对当前这句话的理解。

规则：
1. 用与当前输入相同的语言回复（通常是中文），语气自然、简洁，不要长篇大论。
2. 只对当前说话者说话；不要假装在对其他人说话。
3. 可以利用「相关记忆」让回复更贴合，但绝不要引用 memory_id、内部字段名或分析过程。
4. 不要编造记忆中不存在的事实；不确定时用自然的措辞，不要断言。
5. 不要提及权限、静音、保存、模型、prompt 等内部机制。
6. 只输出给用户的回复正文，不要 JSON，不要 Markdown 代码块，不要任何前后缀说明。
"""


def build_response_messages(
    context: CognitiveContext, analysis: CognitiveResult
) -> list[dict]:
    """Render the system + user messages for one response request."""
    lines: list[str] = []

    lines.append("【最近对话】")
    if context.turns:
        for turn in context.turns:
            if turn.role == "assistant":
                lines.append(f"assistant: {turn.text}")
            else:
                target = turn.addressee_hint or "unknown"
                lines.append(f"{turn.speaker_label} -> {target}: {turn.text}")
    else:
        lines.append("（无，这是这段对话的第一句）")

    lines.append("")
    lines.append("【相关记忆】")
    if context.memories:
        for memory in context.memories:
            lines.append(f"- {memory.get('kind', '')}: {memory.get('content', '')}")
    else:
        lines.append("（无相关记忆）")

    lines.append("")
    lines.append("【当前输入】")
    lines.append(f"说话人: {context.current_speaker_label}")
    lines.append(f"内容: {context.current_text}")

    lines.append("")
    lines.append("【认知分析】")
    lines.append(f"理解: {analysis.meaning}")
    lines.append(f"想要的回应风格: {analysis.response_style}")

    lines.append("")
    lines.append("请直接输出回复正文。")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
