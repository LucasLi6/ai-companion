"""Lightweight background extraction prompt (V0.6.1, episodes in V0.9.9.3).

This prompt is *not* used to reply and *not* used to judge whether the
companion should answer. It only extracts evidence-backed long-term facts,
unfinished items and explicit emotional events from already-authorized
conversation turns, and groups genuinely multi-turn work into episode
summaries. Everything it returns is a candidate: the local validator, privacy
rules and memory writer still decide what may be stored.

It is deliberately small. The full cognitive prompt (persona + recent history +
retrieval + scene) remains for the Deliberation Path, where semantic judgment
about the current turn is actually needed.
"""

from __future__ import annotations

EXTRACTION_PROMPT_VERSION = "extraction-v2"

SYSTEM_PROMPT = """\
你正在处理一组已经获得授权的真实对话事件。每个事件可能包含用户原话（USER）
和 Ava 真实发出的回复（AVA）。Ava 的回复只用于理解对话过程，不是用户做过的事。

只提取有直接文本证据的长期记忆候选、重要情绪事件和未完成事项。

普通问候、闲聊、玩笑、一般疑问或没有长期价值的内容，返回空结果。

不得推测没有证据的个人事实、关系、意图或现实环境。
每个候选必须引用原文证据和来源事件 ID。
不得改变权限或将模型推断直接写成已确认记忆。

## 事件梗概（episodes）

当这组事件里存在一段真实的多轮经历时（出现了一个问题、做过尝试、出现转折、
产生结果，或明确留下未完成事项），把它概括成一条事件梗概。

不要为问候、寒暄、简单问答、单一事实陈述生成梗概。

梗概必须遵守：
1. 只依据输入中 USER 的原话和 AVA 的真实回复；不得编造输入中没有的细节。
2. 不得把 Ava 的提议或建议写成用户已经执行的动作。
3. 不得把“打算 / 想 / 怀疑 / 可能 / 试试”改写成“已完成 / 已证实”。
4. 只有文本明确表明问题已经解决时，status 才可为 RESOLVED；
   仍在进行或只做了部分尝试用 ONGOING；无法判断用 UNKNOWN。
5. source_event_ids 只能使用输入中真实出现的 event_id，不能编造。
6. 不输出 person_id、权限、数据库 ID 或任何存储键。
7. 如果这组事件延续了【已有事件梗概】中的某一项，填写相同的 continues_title；
   否则省略该字段。不要为不同事件强行合并。
8. 最多输出 3 条梗概；没有就输出空数组。

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
  ],
  "episodes": [
    {
      "title": "简短、可识别的标题",
      "summary": "问题 / 尝试 / 转折 / 结果 / 未完成事项，只用有证据的内容",
      "topic": "这段经历的主题，如 机器狗调试",
      "status": "RESOLVED | ONGOING | UNKNOWN",
      "source_event_ids": ["真实 event_id"],
      "continues_title": "可选；仅在延续【已有事件梗概】时填写相同标题"
    }
  ]
}
没有候选的事件可以不出现；全部没有时输出 {"results": [], "episodes": []}。
"""


def build_extraction_messages(
    events: list[dict],
    *,
    speaker_label: str,
    prior_turns: list[dict] | None = None,
    replies: dict[str, str] | None = None,
    recent_episodes: list[dict] | None = None,
) -> list[dict]:
    """Render one small request for a same-speaker batch.

    ``events`` items are ``{"event_id", "occurred_at", "text"}``. ``prior_turns``
    optionally supplies a couple of preceding turns for pronoun resolution; it
    never includes another speaker's private material. ``replies`` maps an
    event id to the real text Ava delivered for that turn — never an invented
    or undelivered reply. ``recent_episodes`` supplies only short titles/topics
    so the model can say "this continues that one" without seeing, or being
    able to choose, any database id.
    """
    replies = replies or {}
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
            role = str(turn.get("role") or "").upper()
            label = f"{role}: " if role in ("USER", "AVA") else ""
            lines.append(f"{prefix}{label}{text[:600]}")

    if recent_episodes:
        lines.append("")
        lines.append("【已有事件梗概（只用于判断是否延续，不得编造 ID）】")
        for episode in recent_episodes:
            title = " ".join(str(episode.get("title") or "").split())[:120]
            topic = " ".join(str(episode.get("topic") or "").split())[:80]
            status = str(episode.get("status") or "").strip()
            if not title:
                continue
            bits = [f"title={title}"]
            if topic:
                bits.append(f"topic={topic}")
            if status:
                bits.append(f"status={status}")
            lines.append(" | ".join(bits))

    lines.append("")
    lines.append("【待处理事件】")
    for event in events:
        stamp = str(event.get("occurred_at") or "")
        prefix = f"[{stamp}] " if stamp else ""
        # Keep the "event_id=..." token first: it is the stable machine handle
        # used by local validation and by existing parsers.
        lines.append(
            f"event_id={event['event_id']} {prefix}USER: {str(event['text'])[:1200]}"
        )
        reply = " ".join(str(replies.get(event["event_id"]) or "").split())
        if reply:
            lines.append(f"AVA: {reply[:1200]}")
    lines.append("")
    lines.append("请只提取有直接证据的候选和事件梗概，并只输出 JSON。")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


__all__ = [
    "EXTRACTION_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_extraction_messages",
]
