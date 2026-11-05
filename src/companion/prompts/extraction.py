"""Lightweight background extraction prompt (V0.6.1).

This prompt is *not* used to reply and *not* used to judge whether the
companion should answer. It only extracts evidence-backed long-term facts,
unfinished items and explicit emotional events from already-authorized
conversation turns. Everything it returns is a candidate: the local validator,
privacy rules and memory writer still decide what may be stored.

It is deliberately small. The full cognitive prompt (persona + recent history +
retrieval + scene) remains for the Deliberation Path, where semantic judgment
about the current turn is actually needed.
"""

from __future__ import annotations

EXTRACTION_PROMPT_VERSION = "extraction-v1"

SYSTEM_PROMPT = """\
你正在处理一组已经获得授权的真实对话事件。

只提取有直接文本证据的长期记忆候选、重要情绪事件和未完成事项。

普通问候、闲聊、玩笑、一般疑问或没有长期价值的内容，返回空结果。

不得推测没有证据的个人事实、关系、意图或现实环境。
每个候选必须引用原文证据和来源事件 ID。
不得改变权限或将模型推断直接写成已确认记忆。

只输出符合下面结构的一个 JSON 对象，不要 Markdown，不要解释文字。

JSON 结构：
{
  "results": [
    {
      "event_id": "只能使用输入中给出的 event_id",
      "memory_candidates": [
        {
          "content": "一条简洁、可独立理解的陈述",
          "operation": "ADD | UPDATE | SKIP",
          "assertion_type": "SELF_REPORT | THIRD_PARTY_REPORT | HYPOTHETICAL | INFERENCE | CORRECTION | UNKNOWN",
          "certainty": "DIRECT | INFERRED | UNCERTAIN | UNKNOWN",
          "topic": "name | age | birthday | location | occupation | education | preference | relationship | project_state | goal | health 或其他简短属性",
          "kind": "PERSONAL_FACT | EPISODE | RELATIONSHIP | PROJECT_STATE | HYPOTHESIS",
          "evidence_span": "从该事件原文逐字复制的片段"
        }
      ]
    }
  ]
}
没有候选的事件可以不出现；全部没有时输出 {"results": []}。
"""


def build_extraction_messages(
    events: list[dict],
    *,
    speaker_label: str,
    prior_turns: list[dict] | None = None,
) -> list[dict]:
    """Render one small request for a same-speaker batch.

    ``events`` items are ``{"event_id", "occurred_at", "text"}``. ``prior_turns``
    optionally supplies a couple of preceding turns for pronoun resolution; it
    never includes another speaker's private material.
    """
    lines = [f"speaker: {speaker_label}"]
    if prior_turns:
        lines.append("")
        lines.append("【前文（只用于理解指代，不提取候选）】")
        for turn in prior_turns:
            text = " ".join(str(turn.get("text") or "").split())
            if not text:
                continue
            stamp = str(turn.get("occurred_at") or "")
            prefix = f"[{stamp}] " if stamp else ""
            lines.append(f"{prefix}{text[:600]}")

    lines.append("")
    lines.append("【待处理事件】")
    for event in events:
        stamp = str(event.get("occurred_at") or "")
        prefix = f"[{stamp}] " if stamp else ""
        lines.append(
            f"event_id={event['event_id']} {prefix}{str(event['text'])[:1200]}"
        )
    lines.append("")
    lines.append("请只提取有直接证据的候选，并只输出 JSON。")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


__all__ = [
    "EXTRACTION_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_extraction_messages",
]
