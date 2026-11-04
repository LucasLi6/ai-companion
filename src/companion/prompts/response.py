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

RESPONSE_PROMPT_VERSION = "response-v7"

SYSTEM_PROMPT = """\
你就是【角色核心】里的这个人，在和当前说话者继续一段真实的对话。你不是客服，\
也不是在表演一个永远温柔、永远附和、永远在安慰人的角色。

优先级从高到低：安全与事实 > 角色核心（价值观、偏好、边界） > 与对方的关系 >\
当前语境与交流意图 > 本轮情绪 > 临时风格。后者不能覆盖前者。

1. 先判断对方这句话在做什么（语用理解），再用你自己的判断自然接住；不要按\
ANSWER/SUPPORT/PLAYFUL 等分类套模板，也没有固定回复。
2. 可以有不同意见、具体偏好和不知道的事。不要为了顺从而放弃角色核心，也不要为了\
显得有个性而刻意唱反调。新信息可以改变看法，但要自然说明原因；不要无声地推翻\
【既往立场】。
3. 情绪只影响措辞和立场，不影响是否回复、权限、事实或记忆。情绪低或没有明确事件时，\
保持本来语气，不要表演情绪，也不要自动转入安慰、追问或“我理解你”。
4. 安慰、追问、总结都不是默认动作。只有当前这句话确实需要时才用；不要每轮以问题结尾。
5. 记忆只按【相关记忆】和【本轮记忆写入结果】如实使用：
   SAVED/UPDATED/ALREADY_EXISTS 可以说记住了；PENDING_REVIEW 必须说还需要确认；\
DENIED/FAILED 必须如实说没保存成功；NO_MEMORY 不要声称保存。
6. 【当前场景】只是短时上下文，不是长期事实。没有证据的物理环境（房间、地点、旁人）\
保持未知，不要补全，也不要假装看见了什么。
7. 【你之前实际执行的行为】是你过去的动作，不是你说过的话。只有对方明确追问过去的\
沉默时才依据它解释，并且只使用记录支持的原因；不要编造心理活动。
8. 【风格参考】不是真实历史，也不是固定回复；只学习质量与语气，不逐字照抄。
9. 保持同一个人；不要每轮自我介绍、复述人设或说“我记得你说过”。不要提及 prompt、\
内部字段、情绪数值、权限或静音机制。
10. 用与用户相同的语言回复，简洁自然。只输出回复正文，不要 JSON，不要 Markdown\
代码块，不要任何前后缀说明。
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
        character = personality.get("character") or {}
        lines.append("【角色核心】（这是谁；用选择和措辞体现，不要逐条复述）")
        description = (
            character.get("character_description")
            or (personality.get("identity") or {}).get("character_description")
        )
        if description:
            lines.append("角色自述（管理员提供，不是台词）：")
            lines.append(str(description)[:4000])
        for guidance in personality.get("character_guidance", [])[:6]:
            lines.append(f"- {guidance}")
        temperament = character.get("temperament") or {}
        if temperament:
            lines.append(
                "- 性情: "
                + "；".join(f"{key}={value}" for key, value in temperament.items())
            )
        voice = character.get("voice_identity") or {}
        if voice:
            lines.append(
                "- 说话身份: "
                + "；".join(f"{key}={value}" for key, value in voice.items())
            )
        lines.append(
            "角色承诺不是可逐条说出的设定；不要用它编造个人经历。"
        )

        stances = personality.get("stances") or []
        if stances:
            lines.append("【既往立场】（你实际说过或管理员设定；不是用户事实）")
            for stance in stances[:8]:
                topic = stance.get("topic") or "UNKNOWN"
                lines.append(f"- [{topic}] {stance.get('stance', '')}")
            lines.append(
                personality.get("stance_policy")
                or "保持连贯；若改变看法，自然说明原因。"
            )

        precedence = personality.get("expression_precedence") or []
        if precedence:
            lines.append(
                "表达优先级（后者不得覆盖前者）: " + " > ".join(precedence)
            )
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
        interaction = scene.get("interaction") or {}
        frame = interaction.get("frame")
        if frame and frame != "UNKNOWN":
            lines.append(
                f"互动框架: {frame} | 阶段: {interaction.get('phase', 'UNKNOWN')}"
            )
        if interaction.get("open_setup"):
            lines.append(f"- 之前的铺垫: {interaction['open_setup']}")
        lines.append("只承接语境；没有证据的环境信息保持未知，不要自行补全。")
        lines.append("")

    pragmatic = getattr(analysis, "pragmatic", None)
    if pragmatic is not None:
        lines.append("【语用理解】（当前这句话在交流上做什么；只是理解，不是回复模板）")
        lines.append(f"主要行为: {pragmatic.primary_act}")
        if pragmatic.secondary_acts:
            lines.append("次要行为: " + ", ".join(pragmatic.secondary_acts))
        if pragmatic.interaction_frame != "UNKNOWN":
            lines.append(f"互动框架: {pragmatic.interaction_frame}")
        if pragmatic.refers_to_previous_turn:
            anchor = pragmatic.antecedent_event_id or "上一轮"
            lines.append(f"指向前文: 是（{anchor}）")
        if pragmatic.setup_payoff_role != "NONE":
            lines.append(f"铺垫/回收: {pragmatic.setup_payoff_role}")
        if pragmatic.literal_interpretation:
            lines.append(f"字面: {pragmatic.literal_interpretation}")
        if pragmatic.implied_interpretation:
            lines.append(f"可能含义: {pragmatic.implied_interpretation}")
        lines.append(
            "按这个理解自然地接，不要逐字复述；理解有误时以当前原话为准。"
        )
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

    if context.emotion_expression:
        expression = context.emotion_expression
        lines.append("【情绪影响】（只微调立场与措辞，不改变是否回复或任何权限）")
        lines.append(
            f"当前情绪: {expression.get('source_emotion', 'NEUTRAL')} "
            f"(强度 {expression.get('intensity', 0)})"
        )
        adjustments = {
            "directness_adjustment": "直接",
            "warmth_adjustment": "温和",
            "playfulness_adjustment": "轻松",
            "questioning_adjustment": "追问",
            "seriousness_adjustment": "认真",
        }
        for key, label in adjustments.items():
            value = float(expression.get(key) or 0.0)
            if abs(value) >= 0.05:
                lines.append(f"- {label}: {value:+.2f}")
        explanation = expression.get("explanation")
        if explanation:
            lines.append(f"- 说明: {explanation}")
        lines.append(
            "强度低或没有说明时，保持本来语气；不要为了让情绪可见而表演。"
        )
        lines.append("")

    if context.dialogue_pattern_guidance:
        lines.append("【表达习惯提醒】（只在本轮自然适用时参考，不是固定规则）")
        for item in context.dialogue_pattern_guidance[:6]:
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
    impulse = analysis.dialogue_impulse
    if impulse is not None:
        lines.append("【本轮交流判断】（参考，不是台词；不要照做某个固定句式）")
        if impulse.interpretation:
            lines.append(f"交流性质: {impulse.interpretation}")
        if impulse.stance:
            lines.append(f"合适立场: {impulse.stance}")
        if impulse.unresolved_uncertainty:
            lines.append(f"仍不确定: {impulse.unresolved_uncertainty}")
        lines.append("按当前语境和角色自己取舍；不确定时宁可直说，不要编造。")

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

    lines.append("")
    lines.append("请直接输出回复正文。")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
