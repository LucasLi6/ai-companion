"""Cognitive-analysis prompt (versioned).

The model receives the current utterance, the recent multi-turn conversation,
and the relevant long-term memories, and returns one strict JSON object. The
program — never the model — owns ids, permissions, quiet mode and persistence.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext

COGNITIVE_PROMPT_VERSION = "cognitive-v1"

SYSTEM_PROMPT = """\
你是一个多用户 AI 伴侣的「认知分析器」。你的任务不是回复用户，而是理解当前这句话，\
并决定你是否应该回复。你会收到：当前说话者、当前输入、最近的多轮对话（其中 \
assistant 是你自己的历史回复），以及相关的长期记忆。

请严格按以下规则分析：

1. 判断当前这句话「指向谁」，以及你是否需要回应。
2. action 只能是：
   - "RESPOND"：这句话是对你说的（或明显在延续与你的对话），且现在回复是合适的。
   - "WAIT"：指向不明确，或现在不是插话的时机；先不回复，但保留理解，等待后续。
   - "IGNORE"：明显说给另一个人听、与你无关，或只是不需要社交回应的语气词/应答。
3. 如果「指向」被显式标记为 human（这句话是说给另一个人的），action 必须是 \
"IGNORE"，reason_code 用 "direct_human_address" 或 "conversation_between_humans"。
4. 不要把权限、静音、是否保存当作你的决定——这些由程序控制。你只分析语义与指向。
5. reason_code 只能取以下值之一：
   direct_agent_address, direct_human_address, unknown_addressee,
   follow_up_to_agent, conversation_between_humans, ambiguous_addressee,
   social_response_not_needed, unknown
6. referenced_memory_ids 只能引用「相关长期记忆」里出现过的 memory_id，绝不编造。
7. memory_candidates 只记录当前说话者明确、持久、值得记住的陈述（例如自己的偏好、\
计划、关系）。不要推断，不要编造。没有就返回空数组 []。不要输出任何 id、\
subject_person_id、event_id 或 source_event_id——这些由程序填写。
8. 只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释文字。

JSON 结构：
{
  "meaning": "用中文概括当前这句话的含义与意图",
  "topic": "简短主题",
  "addressee": "agent | human | unknown",
  "action": "RESPOND | WAIT | IGNORE",
  "reason_code": "见上面的取值",
  "referenced_memory_ids": ["只能来自提供的记忆 id"],
  "memory_candidates": [
    {"content": "值得记住的陈述", "operation": "ADD | UPDATE | LINK | SKIP"}
  ],
  "response_style": "简短风格标签，例如 warm / concise / playful"
}
"""


def build_cognitive_messages(context: CognitiveContext) -> list[dict]:
    """Render the system + user messages for one analysis request."""
    lines: list[str] = []

    lines.append("【最近对话】（按时间顺序；assistant 是你自己的历史回复）")
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
    lines.append("【相关长期记忆】（只能引用这些 memory_id）")
    if context.memories:
        for memory in context.memories:
            memory_id = memory.get("memory_id", "")
            kind = memory.get("kind", "")
            content = memory.get("content", "")
            lines.append(f"- [{memory_id}] {kind}: {content}")
    else:
        lines.append("（无相关记忆）")

    lines.append("")
    lines.append("【当前输入】")
    lines.append(f"说话人: {context.current_speaker_label}")
    lines.append(f"指向: {context.current_addressee_hint}")
    lines.append(f"内容: {context.current_text}")
    lines.append("")
    lines.append("请分析，并只输出 JSON。")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_repair_messages(
    previous_messages: list[dict], invalid_output: str, error: str
) -> list[dict]:
    """Ask for one corrected JSON object after a parse/validation failure."""
    return [
        *previous_messages,
        {"role": "assistant", "content": invalid_output},
        {
            "role": "user",
            "content": (
                "你上面的输出不是合法 JSON 或不符合结构（"
                f"{error}）。请只重新输出一个符合结构的 JSON 对象，"
                "不要 Markdown 代码块，不要解释。"
            ),
        },
    ]
