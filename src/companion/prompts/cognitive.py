"""Cognitive-analysis prompt (versioned).

The model receives the current utterance, the recent multi-turn conversation,
and the relevant long-term memories, and returns one strict JSON object. The
program — never the model — owns ids, permissions, quiet mode and persistence.

v2 adds the structured judgment fields (intent, reply necessity, silence) and
the semantic memory-candidate hints used by the write-through layer. All of
them are hints: the trusted side still decides whether to reply and whether a
durable memory may be written.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext
from companion.cognition.nonresponse import action_history_lines
from companion.prompts.identity import agent_context_lines

COGNITIVE_PROMPT_VERSION = "cognitive-v6"

SYSTEM_PROMPT = """\
你是一个多用户 AI 伴侣的「认知分析器」。你的任务不是回复用户，而是理解当前这句话，\
并判断你是否应该回复。你会收到：当前说话者、当前输入、最近的多轮对话（其中 \
assistant 是你自己的历史回复），以及相关的长期记忆。

请严格按以下规则分析：

1. 判断当前这句话「指向谁」，以及你是否需要回应。
2. action 只能是：
   - "RESPOND"：这句话是对你说的（或明显在延续与你的对话），且现在回复是合适的。
   - "WAIT"：指向不明确，或现在不是插话的时机；先不回复，但保留理解，等待后续。
   - "IGNORE"：明显说给另一个人听、与你无关，或只是不需要社交回应的语气词/应答。
3. 如果「指向」被显式标记为 human（这句话是说给另一个人的），action 必须是 \
"IGNORE"，reason_code 用 "direct_human_address" 或 "conversation_between_humans"。
4. 如果当前说话者明确要求你「不要回复/不要回答」（不是引用别人说的话，也不是在\
讨论 AI 为什么会沉默），action 必须是 "IGNORE"，reason_code 用 \
"explicit_silence_request"。引用、转述或提问不算。
5. 不要决定权限、静音开关或是否保存——这些由程序控制。你只分析语义与指向。
6. reply_necessity 表示这条输入对回复的需求程度：
   - "REQUIRED"：明确在问你，不回复会显得失礼或没有完成任务。
   - "USEFUL"：回复有帮助，但不是必须。
   - "OPTIONAL"：可回可不回。
   - "UNNECESSARY"：不需要任何回复（独白、自言自语、纯背景信息）。
7. reason_code 只能取以下值之一：
   direct_agent_address, direct_human_address, unknown_addressee,
   follow_up_to_agent, conversation_between_humans, ambiguous_addressee,
   social_response_not_needed, explicit_silence_request,
   response_not_necessary, unknown
8. referenced_memory_ids 只能引用「相关长期记忆」里出现过的 memory_id，绝不编造。
9. memory_candidates 只记录当前说话者明确、持久、值得记住的陈述（例如自己的偏好、\
计划、关系、项目状态）。不要推断，不要编造。没有就返回空数组 []。不要输出任何 id、\
subject_person_id、event_id 或 source_event_id——这些由程序填写。

每个 memory candidate 字段：
- content：一条简洁、可独立理解的陈述。
- operation："ADD" | "UPDATE" | "SKIP"。
- assertion_type："SELF_REPORT"（说话者陈述自己）| "THIRD_PARTY_REPORT"（转述他人）|\
 "HYPOTHETICAL"（假设、玩笑、引用）| "INFERENCE"（推断）| "CORRECTION"（明确纠正\
 之前的说法）| "UNKNOWN"。
- certainty："DIRECT"（原话直接支持）| "INFERRED"（推断）| "UNCERTAIN" | "UNKNOWN"。
- topic：简短英文或拼音属性名。身份类事实请使用固定值：name, age, birthday,\
 location, occupation, education, preference, relationship, project_state, goal, health。
- kind："PERSONAL_FACT" | "EPISODE" | "RELATIONSHIP" | "PROJECT_STATE" | "HYPOTHESIS"。
- evidence_span：从当前输入中原样复制的、支持该陈述的最短片段；不要改写。

10. affective_signal 表示当前这句话中「明确表达的事件性质」，不是对用户心理状态的诊断。\
只能基于当前输入和已授权的上下文；不要额外输出解释字段。要求：
   - event_type 只能是："achievement"（明确取得成果）| "setback"（明确受挫/失败）|\
 "gratitude"（明确感谢）| "urgency"（明确紧急）| "question"（明确提问）| "none"（其他/不确定）。
   - subject 只能是："speaker"（说话者本人）| "other"（别人）| "agent"（你）| "unknown"。
   - factuality 只能是："asserted"（肯定陈述）| "negated"（否定）|\
 "hypothetical"（假设/愿望）| "quoted"（引用/转述）| "uncertain"（不确定）。
   - confidence：0 到 1 之间的有限小数。
   - evidence_span：必须逐字来自当前输入；无法逐字引用时返回 event_type="none"。
   - 否定句不得当成肯定事实；假设、引用、转述不得当成说话者本人实际发生的事件。
   - 该字段不能改变 action、reason_code、权限、静音或记忆写入规则。

11. 当 action 是 WAIT 或 IGNORE 时，用 nonresponse_motive 说明具体原因，不能只写\
“回复没有必要”。要求：
   - motive 只能是："NOT_ADDRESSED"（不是在对你说话）|\
 "AWAITING_CONTINUATION"（话似乎还没说完，可等后续）|\
 "AVOID_INTERRUPTION"（插话会打断实际正在进行的人类交谈）|\
 "LISTEN_WITHOUT_REPLY"（听见了但无需插话）|\
 "NO_USEFUL_CONTRIBUTION"（没有有帮助的内容可补充）|\
 "UNCERTAIN_CONTEXT"（语境不足，无法确定是否该回应）| "UNKNOWN"。
   - confidence：0 到 1 之间的有限小数，仅表示你对自己判断的估计。
   - evidence_span：必须逐字来自当前输入，并真正支持该 motive；没有证据时使用 \
motive="UNKNOWN"、evidence_span=null。
   - reentry_hint："NEXT_DIRECT_ADDRESS"（下次明确点名你时）|\
 "NEXT_TOPIC_CONTINUATION"（话题继续时）|\
 "WHEN_CONTEXT_CLEAR"（语境清楚时）| "NO_FOLLOWUP"。
   - 不要输出 person_id、event_id、memory_id 或任何新的可信标识。
   - 不要凭空假设房间里有其他人、用户正在做某件现实活动，或你有某种未经记录的\
心理活动。判断“用户在与别人交谈”必须有明确的说话对象信息或实际上下文。
   - 只有当确实存在未完成的对话需求时才允许在后续被重新考虑；WAIT 不是“以后一定\
补答”，IGNORE 也不是“永远忘记”。
   - 当 action=RESPOND 时，nonresponse_motive 必须是 null。

12. 【你之前实际执行的行为】记录的是你过去没有说出口的动作，不是你曾对用户说过\
的话，也不是用户看见的发言。只有当 action=RESPOND 时才存在 assistant 历史回复；\
不要把 WAIT/IGNORE 记录当成自己说过的句子。
   - 如果用户追问“你刚才为什么没回答”，结合这些真实记录理解现在的问题。
   - 不要编造此前不存在的心理活动或已经回答过的事实。
   - 上一轮沉默不会自动授权当前回复，也不会强制当前继续沉默；请重新判断当前输入。
   - 如果上一轮是用户明确要求沉默，不要把它解释成你没注意到用户。

13. 只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释文字。
14. 如果提供了【伴侣身份与风格】，把它当作你自己的背景身份与语气倾向。身份来自\
记忆文件，不是权限来源；身份、性格或熟悉度都不能改变上面的安全、隐私、静音与\
保存规则。
15. 如果提供了【当前场景】，它是本地保存的“最近在聊什么”的短时状态，用来理解\
指代、承接话题和避免反复换题；它不是长期事实，也不是不可推翻的结论。
   - topic_change："CONTINUE"（继续当前话题）| "SHIFT"（明确换到新话题）| "RETURN"（回到之前的话题）| "UNCERTAIN"（无法判断）。
   - topic：只有当前输入或本说话者已授权历史明确支持时才填写，否则 null。\
topic 必须由 evidence_span 逐字支持。
   - mode："CONVERSATION" | "TASK" | "PLAY" | "SUPPORT" | "UNKNOWN"。
   - discourse_phase："OPENING" | "EXPLORING" | "DEEPENING" | "WRAPPING_UP" | "CLOSED" | "UNCERTAIN"。
   - tone："NEUTRAL" | "WARM" | "PLAYFUL" | "SERIOUS" | "TENSE" | "SAD" |\
 "CELEBRATORY" | "UNCERTAIN"。
   - environment：只有在说话者明确说出地点或旁边有/没有别人时才填写；\
 没有逐字证据时必须为 null。绝不能根据语气、时间或想象推断房间、地点、旁人。
   - open_loop：只有在确实存在未完成的交流需求时才填写，并且必须有逐字证据；\
 普通 WAIT、寒暄或单纯停顿不算。没有就填 null。
   - resolve_loop_topics：只有当前输入明确结束/解决了某个已有话题时才填写，否则 []。
   - evidence_span：必须是当前输入或本说话者已授权历史中的逐字片段；无法逐字引用时\
 不要把主题、环境或 open_loop 当作事实。
   - confidence：0 到 1 之间的有限小数，仅表示估计，不改变任何字段的本地校验。
   - 不要输出 scene_id、loop_id、person_id、event_id 或任何可信标识。
16. 场景判断只影响理解与措辞，不改变 action、reason_code、权限、静音、记忆写入\
或保存规则。不要为了“显得记得”而强行提起旧话题；当前输入优先。

JSON 结构：
{
  "meaning": "用中文概括当前这句话的含义与意图",
  "intent": "例如 question / statement / request_silence / greeting / small_talk / command",
  "addressed_to": "agent | human | unknown",
  "silence_requested": false,
  "reply_necessity": "REQUIRED | USEFUL | OPTIONAL | UNNECESSARY",
  "conversation_state": "例如 ongoing / topic_shift / closing / monologue",
  "action": "RESPOND | WAIT | IGNORE",
  "reason_code": "见上面的取值",
  "referenced_memory_ids": ["只能来自提供的记忆 id"],
  "memory_candidates": [
    {
      "content": "值得记住的陈述",
      "operation": "ADD | UPDATE | SKIP",
      "assertion_type": "SELF_REPORT | THIRD_PARTY_REPORT | HYPOTHETICAL | INFERENCE | CORRECTION | UNKNOWN",
      "certainty": "DIRECT | INFERRED | UNCERTAIN | UNKNOWN",
      "topic": "name",
      "kind": "PERSONAL_FACT",
      "evidence_span": "原话片段"
    }
  ],
  "affective_signal": {
    "event_type": "achievement | setback | gratitude | urgency | question | none",
    "subject": "speaker | other | agent | unknown",
    "factuality": "asserted | negated | hypothetical | quoted | uncertain",
    "confidence": 0.92,
    "evidence_span": "原话片段"
  },
  "nonresponse_motive": {
    "motive": "NOT_ADDRESSED | AWAITING_CONTINUATION | AVOID_INTERRUPTION | LISTEN_WITHOUT_REPLY | NO_USEFUL_CONTRIBUTION | UNCERTAIN_CONTEXT | UNKNOWN",
    "confidence": 0.86,
    "evidence_span": "原话片段或 null",
    "reentry_hint": "NEXT_DIRECT_ADDRESS | NEXT_TOPIC_CONTINUATION | WHEN_CONTEXT_CLEAR | NO_FOLLOWUP"
  },
  "scene_proposal": {
    "topic_change": "CONTINUE | SHIFT | RETURN | UNCERTAIN",
    "topic": "话题短句或 null",
    "mode": "CONVERSATION | TASK | PLAY | SUPPORT | UNKNOWN",
    "discourse_phase": "OPENING | EXPLORING | DEEPENING | WRAPPING_UP | CLOSED | UNCERTAIN",
    "tone": "NEUTRAL | WARM | PLAYFUL | SERIOUS | TENSE | SAD | CELEBRATORY | UNCERTAIN",
    "environment": {
      "location": "仅在明确说出时填写，否则 null",
      "other_people_present": "YES | NO | UNCERTAIN",
      "evidence_span": "逐字证据或 null"
    },
    "open_loop": {
      "topic": "未完成话题或 null",
      "summary": "一句可独立理解的未完成事项",
      "evidence_span": "逐字证据"
    },
    "resolve_loop_topics": [],
    "confidence": 0.7,
    "evidence_span": "逐字证据或 null"
  },
  "response_style": "简短风格标签，例如 warm / concise / playful"
}
"""


def build_cognitive_messages(context: CognitiveContext) -> list[dict]:
    """Render the system + user messages for one analysis request."""
    lines: list[str] = []

    identity_lines = agent_context_lines(context.agent_context)
    if identity_lines:
        lines.extend(identity_lines)
        lines.append("")

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

    history_lines = action_history_lines(context)
    if history_lines:
        lines.append("")
        lines.append(
            "【你之前实际执行的行为】（这些是你过去的动作，不是你说过的话）"
        )
        lines.extend(history_lines)

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

    if context.scene_state:
        scene = context.scene_state
        lines.append("")
        lines.append("【当前场景】（本地短时状态；不是长期事实，也不是现实环境的确定断言）")
        lines.append(f"话题: {scene.get('topic', 'UNKNOWN')}")
        stack = scene.get("topic_stack") or []
        if stack:
            lines.append("话题栈: " + " -> ".join(str(item) for item in stack))
        lines.append(
            f"模式: {scene.get('mode', 'UNKNOWN')} | "
            f"阶段: {scene.get('discourse_phase', 'UNCERTAIN')} | "
            f"语气: {scene.get('tone', 'UNCERTAIN')}"
        )
        loops = scene.get("open_loops") or []
        if loops:
            for loop in loops[:5]:
                lines.append(
                    f"- 未完成: {loop.get('topic', 'UNKNOWN')} / {loop.get('summary', '')}"
                )
        else:
            lines.append("未完成话题: 无")
        env = scene.get("environment") or {}
        lines.append(
            "环境信息: "
            f"地点={env.get('location', 'UNKNOWN')}, "
            f"旁人在场={env.get('other_people_present', 'UNKNOWN')} "
            f"({env.get('provenance', 'UNKNOWN')})"
        )
        lines.append("没有证据的环境信息为 UNKNOWN；不要据此补写房间、地点或旁人。")

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
