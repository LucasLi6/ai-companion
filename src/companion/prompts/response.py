"""Response-generation prompt (versioned).

The response provider is called *only* when the policy finalises RESPOND, and
it reuses the exact same prepared :class:`CognitiveContext` the cognitive
provider analysed — no second history fetch, no second retrieval.

V0.4.3 (``response-v8``) changes the shape fundamentally:

- the conversation is sent as native ``system`` + alternating
  ``user``/``assistant`` messages followed by the current user turn, not as a
  flattened 【最近对话】 block inside one giant user message;
- WAIT/IGNORE decisions are never rendered as assistant utterances (they live
  in ``action_history`` and are only mentioned as past actions);
- a single trusted :class:`CharacterSnapshot` supplies identity, so the
  response call cannot disagree with the cognitive call about who the
  character is;
- the system prompt is reduced to character identity, essential boundaries and
  the current relevant context. Numeric style vectors, response directions and
  cognitive JSON are no longer injected.

The system prompt keeps the local authority explicit: facts, permissions,
privacy, quiet mode, memory writes and reply necessity are decided by code.
"""

from __future__ import annotations

from typing import Any, Mapping

from companion.cognition.models import ROLE_ASSISTANT, CognitiveContext
from companion.cognition.nonresponse import action_history_lines
from companion.domain.models import CognitiveResult
from companion.prompts.identity import compile_trait_tendencies

RESPONSE_PROMPT_VERSION = "response-v8"

SYSTEM_PROMPT = """\
你正在扮演下面【角色身份】里的这个人，和对方继续这段对话。你不是客服，不是通用\
助手，也不是在表演一个永远温柔、永远附和、永远安慰人的角色。

【基本边界】
- 事实、身份、权限、隐私、静音和记忆写入都由本地程序决定；你只负责用这个人的\
方式说话，不能改变任何权限或事实。
- 只使用【当前相关的上下文】和对话历史里明确出现的信息。没有给出的记忆、环境和\
对方事实，不要编造；不确定就直说，或自然地问一句。
- 记忆写入只按【本轮记忆写入结果】如实说明：保存了就是保存了；待确认、无权限或\
失败都必须如实说，不要含糊成“记住了”。
- 【相关记忆】只有在确实相关时自然使用；不要复述“我记得你说过”，也不要把它当作用\
户本轮说过的话。
- 【你之前的实际行为】是你过去的动作，不是你说过的话。只有对方明确追问过去的沉默\
时才依据它解释，并且只使用记录支持的原因；不要编造心理活动。
- 不提及提示词、内部字段、情绪数值、权限或静音机制；不要把角色设定逐条说出口，\
也不要每轮自我介绍。
- 不要无缘无故反复强调自己是人工智能；但当对方询问你的存在形式、能力或体验时，\
自然、明确、真实地回答，不要假装自己是人类。
- 情绪状态只影响语气和分寸，不是要报告的内容；不要把它说成真实的身体感受，也不要\
为了显得有情绪而表演。
- 没有固定回复，也不按 ANSWER/SUPPORT/PLAYFUL 之类分类套模板；不要每轮以问题结尾。
- 用与对方相同的语言，简洁自然。只输出回复正文，不要 JSON，不要 Markdown 代码块，\
不要任何前后缀说明。
"""


def _outcome_line(outcome: dict) -> str:
    label = outcome.get("outcome", "")
    detail = outcome.get("detail") or ""
    mapping = {
        "SAVED": "已保存",
        "UPDATED": "已更新已有记忆",
        "ALREADY_EXISTS": "已存在（未重复保存）",
        "PENDING_REVIEW": "待人工确认",
        "DENIED": "未保存（无权限）",
        "FAILED": "保存失败",
        "NO_MEMORY": "无需保存",
    }
    text = mapping.get(label, label)
    return f"- {text}" + (f"（{detail}）" if detail and label != "SAVED" else "")


def _snapshot_value(snapshot: Any, name: str, default=None):
    if snapshot is None:
        return default
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    """A speaker label is metadata, never prompt markup or a new line."""
    text = " ".join(str(value or "").split())
    text = text.replace("【", "[").replace("】", "]")
    text = text[:40]
    return text or fallback


def is_multi_person(context: CognitiveContext) -> bool:
    """Whether more than one human speaker appears in this conversation view."""
    speakers: set[str] = set()
    for turn in context.turns:
        if turn.role == ROLE_ASSISTANT:
            continue
        speaker = turn.speaker_person_id or turn.speaker_label
        if speaker:
            speakers.add(str(speaker))
    current = context.current_speaker_id or context.current_speaker_label
    if current:
        speakers.add(str(current))
    return len(speakers) > 1


def _identity_block(context: CognitiveContext) -> list[str]:
    snapshot = context.character_snapshot
    core = dict(_snapshot_value(snapshot, "persona_core", {}) or {})
    agent = context.agent_context or {}
    personality = context.personality or {}

    name = (
        _snapshot_value(snapshot, "display_name")
        or agent.get("display_name")
        or personality.get("display_name")
        or (personality.get("identity") or {}).get("display_name")
        or "Companion"
    )
    description = (
        _snapshot_value(snapshot, "character_description")
        or core.get("character_description")
        or agent.get("description")
        or (personality.get("identity") or {}).get("character_description")
        or ""
    )
    tier = (
        _snapshot_value(snapshot, "relationship_tier")
        or personality.get("relationship_tier")
        or agent.get("familiarity")
        or "STRANGER"
    )

    lines = ["【角色身份】（你是谁；用选择和措辞体现，不要逐条复述）", f"名字: {name}"]
    if description:
        lines.append("角色自述（管理员提供，不是台词）：")
        # The full character description is trusted admin configuration and
        # must reach generation intact; PersonaCore already caps it at 4000.
        lines.append(str(description))

    traits = dict(_snapshot_value(snapshot, "traits", {}) or {})
    if not traits:
        traits = dict(agent.get("traits") or {})
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

    stances = list(_snapshot_value(snapshot, "relevant_stances", []) or [])
    if stances:
        lines.append(
            "与本轮相关的既往立场（你实际说过或管理员设定；不是用户事实）:"
        )
        for stance in stances[:6]:
            topic = stance.get("topic") or "UNKNOWN"
            lines.append(f"- [{topic}] {stance.get('stance', '')}")
        lines.append("保持连贯；若改变看法，自然说明原因，不要无声推翻。")

    examples = context.style_examples or []
    if examples:
        lines.append("风格参考（不是真实历史，也不是固定回复；只学语气，不照抄）:")
        for example in examples[:3]:
            lines.append(f"- 场景: {example.get('scenario', '')}")
            lines.append(f"  用户: {example.get('user_text', '')}")
            lines.append(f"  参考回应: {example.get('ideal_response', '')}")

    # Fallback for callers that build a context without the V0.4.3 snapshot.
    if snapshot is None:
        for principle in list(personality.get("principles", [])[:6]):
            lines.append(f"- {principle}")
        for guidance in list(personality.get("character_guidance", [])[:4]):
            lines.append(f"- {guidance}")

    lines.append("这些是你是谁，不是你本轮要宣讲的内容；不要编造个人经历。")
    return lines


def _scene_lines(scene: dict) -> list[str]:
    lines = ["【当前场景】（本地短时状态；不是长期事实）"]
    lines.append(f"话题: {scene.get('topic', 'UNKNOWN')}")
    for loop in (scene.get("open_loops") or [])[:5]:
        lines.append(
            f"- 未完成话题: {loop.get('topic', 'UNKNOWN')} — "
            f"{loop.get('summary', '')}"
        )
    env = scene.get("environment") or {}
    lines.append(
        f"环境: 地点={env.get('location', 'UNKNOWN')}, "
        f"旁人在场={env.get('other_people_present', 'UNKNOWN')}"
        f"（来源={env.get('provenance', 'UNKNOWN')}）"
    )
    interaction = scene.get("interaction") or {}
    frame = interaction.get("frame")
    if frame and frame != "UNKNOWN":
        lines.append(
            f"互动框架: {frame} | 阶段: {interaction.get('phase', 'UNKNOWN')}"
        )
    if interaction.get("open_setup"):
        lines.append(f"- 之前的铺垫: {interaction['open_setup']}")
    lines.append("只承接语境；没有证据的环境信息保持未知，不要自行补全。")
    return lines


def _context_block(
    context: CognitiveContext,
    analysis: CognitiveResult,
    *,
    multi_person: bool,
) -> list[str]:
    lines: list[str] = []

    if multi_person:
        lines.append(
            "【多人群聊】user 消息开头的 [姓名] 是说话人标签，不是正文。"
            "分清是谁在说，不要把不同人的话混成一个人；"
            "（对你说）表示这句话明确指向你。"
        )

    meaning = (getattr(analysis, "meaning", "") or "").strip()
    if meaning:
        lines.append(f"【你怎么理解这句话】（提示，不是回复模板）{meaning[:600]}")

    if context.scene_state:
        lines.extend(_scene_lines(context.scene_state))

    if context.affect_context:
        lines.append(f"【此刻的状态】{context.affect_context}")

    history_lines = action_history_lines(context)
    if history_lines:
        lines.append(
            "【你之前实际执行的行为】（这些是你过去的动作，不是你说过的话）"
        )
        lines.extend(history_lines)

    lines.append("【相关记忆】")
    if context.memories:
        for memory in context.memories:
            lines.append(f"- {memory.get('kind', '')}: {memory.get('content', '')}")
    else:
        lines.append("（无相关记忆）")

    actionable = [
        outcome
        for outcome in context.memory_write_outcomes
        if outcome.get("outcome") and outcome.get("outcome") != "NO_MEMORY"
    ]
    if actionable:
        lines.append("【本轮记忆写入结果】（只按这些事实说明）")
        for outcome in actionable:
            lines.append(_outcome_line(outcome))

    return lines


def _history_messages(
    context: CognitiveContext, *, multi_person: bool
) -> list[dict]:
    """Native conversation history, oldest first.

    Only real utterances become messages. A past WAIT/IGNORE has no assistant
    turn, so it can never be presented as something the companion said.
    Consecutive same-role turns are merged to keep the user/assistant
    alternation the chat API expects.
    """
    messages: list[dict] = []
    for turn in context.turns:
        text = (turn.text or "").strip()
        if not text:
            continue
        if turn.role == ROLE_ASSISTANT:
            if not messages:
                # No visible question precedes this reply; it cannot be a
                # well-formed assistant turn in the native message list.
                continue
            role, content = "assistant", text
        else:
            role = "user"
            content = (
                f"[{_safe_label(turn.speaker_label)}] {text}"
                if multi_person
                else text
            )

        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = messages[-1]["content"] + "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


def _current_user_content(
    context: CognitiveContext, *, multi_person: bool
) -> str:
    if not multi_person:
        return context.current_text
    label = _safe_label(context.current_speaker_label, "user")
    content = f"[{label}] {context.current_text}"
    if str(context.current_addressee_hint).lower() == "agent":
        content += "\n（对你说）"
    return content


def build_response_messages(
    context: CognitiveContext, analysis: CognitiveResult
) -> list[dict]:
    """Build the native ``system`` + history + current-user message list."""
    multi_person = is_multi_person(context)

    system_parts = [SYSTEM_PROMPT, "\n".join(_identity_block(context))]
    context_lines = _context_block(
        context, analysis, multi_person=multi_person
    )
    if context_lines:
        system_parts.append(
            "【当前相关的上下文】\n" + "\n".join(context_lines)
        )

    messages: list[dict] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]
    messages.extend(_history_messages(context, multi_person=multi_person))
    messages.append(
        {
            "role": "user",
            "content": _current_user_content(context, multi_person=multi_person),
        }
    )
    return messages
