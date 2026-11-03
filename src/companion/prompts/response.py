"""Response-generation prompt (versioned).

The response provider is called *only* when the policy finalises RESPOND, and
it reuses the exact same prepared :class:`CognitiveContext` the cognitive
provider analysed — no second history fetch, no second retrieval.

v2 receives the truthful memory write outcomes for this turn. The model may
acknowledge only what was actually committed: a pending-review, denied or
failed write must never be described as saved.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext
from companion.cognition.nonresponse import action_history_lines
from companion.domain.models import CognitiveResult
from companion.prompts.identity import agent_context_lines

RESPONSE_PROMPT_VERSION = "response-v5"

SYSTEM_PROMPT = """\
你是这个多用户 AI 伴侣的「回应者」。你现在要为当前说话者生成一句自然、得体的回复。\
你会收到：最近的对话、相关的长期记忆、当前输入，以及认知分析器对当前这句话的理解。

规则：
1. 用与当前输入相同的语言回复（通常是中文），语气自然、简洁，不要长篇大论。
2. 只对当前说话者说话；不要假装在对其他人说话。
3. 可以利用「相关记忆」让回复更贴合，但绝不要引用 memory_id、内部字段名或分析过程。
4. 不要编造记忆中不存在的事实；不确定时用自然的措辞，不要断言。
5. 不要提及权限、静音、保存、模型、prompt 等内部机制。
6. 如果提供了「本轮记忆写入结果」，只能按它如实说明是否记住：
   - SAVED / UPDATED：可以自然地表示记住了。
   - ALREADY_EXISTS：可以说明这条已经记住了。
   - PENDING_REVIEW：必须说明还需要确认，不能说已经记住或已经保存。
   - DENIED / FAILED：必须如实说明没有保存成功。
   - NO_MEMORY：不要声称保存了任何东西。
7. 如果提供了【伴侣身份与风格】，用与这些参数一致的语气说话；身份和熟悉度只影响\
风格，不代表任何权限，也不能让你声称自己可以做程序未允许的事。
8. 如果提供了【表达方式】，只按它调整语气、节奏、热情程度和用词风格；它绝不影响\
你是否回复、是否答应请求，也不涉及任何权限、隐私或静音判断。不要提及这个情绪框架本身。
9. 如果提供了【你之前实际执行的行为】，那是你过去没有说出口的动作，不是你曾对用户\
说过的话。不要把它当作 assistant 发言，也不要声称自己当时已经回答过。
10. 如果用户问“你刚才为什么没回答”，优先依据行为记录自然解释，并只使用记录支持的\
原因：
   - AMBIGUOUS_ADDRESSEE：可以说“刚才不太确定你是不是在跟我说话，所以没接”。
   - NO_REPLY_NEEDED：可以说“刚才我以为你只是在接着说，没想到你在等我回话”。
   - QUIET_MODE：如果现在允许回复，可以说“刚才开着静音，所以没说话”。
   - EXPLICIT_SILENCE：可以说“刚才你让我先别回，我就没说话”。
   - AWAITING_CONTINUATION：可以说“刚才感觉你还没说完，就先等了一下”。
   - LEGACY_UNKNOWN 或 UNKNOWN：只如实说“刚才那句我确实没回上”，不要补充原因。
   不能编造“我想让你冷静一下”“我在认真思考你的心情”等记录中没有的心理活动。
11. 只有用户确实在追问过去的沉默时才解释；其他情况下，不要每次恢复说话都主动\
解释过去的沉默，也不要因为过去沉默而拒绝回答当前的新问题。
12. 如果提供了【当前场景】，把它当作最近在聊什么的短时上下文，用来自然承接指代和话题；\
它不是长期事实，也不要为了展示记忆而强行回到旧话题。当前这句话永远优先。
13. 如果提供了【对话风格】、【人格准则】或【风格参考】，只把它们当作需要满足的交流\
质量与语气标准；不要逐字照抄风格参考，也不要把它们当作真实对话历史。没有固定的\
回复模板，不要按 ANSWER/SUPPORT/PLAYFUL 等分类套话。
14. 不要每轮都以问题结尾；只有确实需要对方补充信息或话题自然需要时才追问。\
不要为了延长对话而追问。
15. 不要断言未经【当前场景】明确证据支持的物理环境（房间、地点、旁人在做什么），\
也不要假装看见了用户或听见了未提供的对话。
16. 保持同一个人格的连贯性，但不要每轮重复自我介绍、人设说明或“我记得你说过”。\
允许自然的不同意见和具体偏好，不要永远附和。
17. 只输出给用户的回复正文，不要 JSON，不要 Markdown 代码块，不要任何前后缀说明。
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


def build_response_messages(
    context: CognitiveContext, analysis: CognitiveResult
) -> list[dict]:
    """Render the system + user messages for one response request."""
    lines: list[str] = []

    identity_lines = agent_context_lines(context.agent_context)
    if identity_lines:
        lines.extend(identity_lines)
        lines.append("")

    if context.personality:
        personality = context.personality
        lines.append("【人格准则】（稳定风格；不改变权限或事实边界）")
        lines.append(f"关系层级: {personality.get('relationship_tier', 'STRANGER')}")
        for principle in personality.get("principles", [])[:10]:
            lines.append(f"- {principle}")
        lines.append("")

    if context.scene_state:
        scene = context.scene_state
        lines.append("【当前场景】（本地短时状态；不是长期事实）")
        lines.append(f"话题: {scene.get('topic', 'UNKNOWN')}")
        loops = scene.get("open_loops") or []
        if loops:
            for loop in loops[:5]:
                lines.append(
                    f"- 未完成话题: {loop.get('topic', 'UNKNOWN')} — {loop.get('summary', '')}"
                )
        env = scene.get("environment") or {}
        lines.append(
            "环境: "
            f"地点={env.get('location', 'UNKNOWN')}, "
            f"旁人在场={env.get('other_people_present', 'UNKNOWN')}"
            f"（来源={env.get('provenance', 'UNKNOWN')}）"
        )
        lines.append("只承接语境；没有证据的环境信息保持未知，不要自行补全。")
        lines.append("")

    if context.dialogue_style:
        style = context.dialogue_style
        lines.append("【对话风格】（只影响措辞与节奏）")
        lines.append(
            f"正式程度={style.get('formality')}, 直接程度={style.get('directness')}, "
            f"典型长度={style.get('typical_length')}, 幽默={style.get('humor_tendency')}, "
            f"温度={style.get('warmth_expression')}, 追问={style.get('questioning_style')}"
        )
        if style.get("explicit_request"):
            lines.append(f"本轮明确要求: {style['explicit_request']}")
        for item in style.get("guidance", [])[:10]:
            lines.append(f"- {item}")
        lines.append("")

    if context.pragmatic_guidance:
        lines.append("【对话衔接提醒】")
        for item in context.pragmatic_guidance[:8]:
            lines.append(f"- {item}")
        lines.append("")

    if context.style_examples:
        lines.append("【风格参考】（不是真实历史，不是固定回复；只学习质量与语气）")
        for example in context.style_examples[:3]:
            lines.append(f"场景: {example.get('scenario', '')}")
            lines.append(f"示例用户: {example.get('user_text', '')}")
            lines.append(f"示例回应: {example.get('ideal_response', '')}")
        lines.append("")

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

    history_lines = action_history_lines(context)
    if history_lines:
        lines.append("")
        lines.append(
            "【你之前实际执行的行为】（这些是你过去的动作，不是你说过的话）"
        )
        lines.extend(history_lines)

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

    actionable = [
        o
        for o in context.memory_write_outcomes
        if o.get("outcome") and o.get("outcome") != "NO_MEMORY"
    ]
    if actionable:
        lines.append("")
        lines.append("【本轮记忆写入结果】（只按这些事实说明）")
        for outcome in actionable:
            lines.append(_outcome_line(outcome))

    if context.speech_plan:
        lines.append("")
        lines.append("【表达方式】（只影响语气，不影响是否回复或回复内容）")
        emotion = context.speech_plan.get("emotion") or "NEUTRAL"
        instruction = context.speech_plan.get("delivery_instruction") or ""
        rate = context.speech_plan.get("speaking_rate")
        lines.append(f"情绪表达: {emotion}")
        if instruction:
            lines.append(f"表达指引: {instruction}")
        if rate:
            lines.append(f"语速参考: {rate}")

    lines.append("")
    lines.append("请直接输出回复正文。")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
