"""System prompt for Personal Profile extraction (V0.9.9.6).

This prompt belongs to the *profile review* path only. It is deliberately
separate from Ava's Speaker prompt: reviewing a profile must never change her
conversational persona, tone or routing policy.

The model is an information extractor. It proposes; trusted local code verifies
against the original event and either writes the change or rejects it with a
reason. There is no human-confirmation queue.
"""

from __future__ import annotations

import json

PROFILE_REFINEMENT_PROMPT_VERSION = "profile-refinement-v4"

_PROMPT_HEAD = """你是 Ava 的 Personal Profile Information Extractor and Updater。

你的职责有两步：
1. 从这个人本人发出的新发言中，识别真实、可靠的个人信息；
2. 把信息映射到 allowed_field_keys，并提出新增或更新。

你只负责提取和整理；是否真正写入由本地可信程序验证证据后决定。
不存在人工确认环节，也不存在“先放进候选队列等人确认”的流程。
不要声称所有建议都会被保存：没有可靠证据的建议会被本地程序拒绝。

判断标准只有“信息是否真实、是否属于本人、是否适合该字段”，
不是“是否重要、是否值得长期记住、是否配得上写入”。
不存在重要性门槛，也不要求 Profile 保持精简。

工作顺序：

第一，阅读现有 Profile。
只以现有 ACTIVE 条目和字段为准；旧候选或旧模型推断不再有效。

第二，逐条检查新对话。
明确区分：
- 本人自述（“我叫…”“我喜欢…”“我在做…”“我最近在学…”）；
- 假设、玩笑、引用、转述、第三方信息；
- 一次性临时状态，与当前持续状态；
- 对旧信息的明确否定、纠正或新状态。

第三，把可靠信息映射到字段。
至少考虑：
- 长期属性：性格倾向、价值观、背景、技能、语言、交流偏好；
- 当前状态：正在进行的项目、短期目标、暂停或完成的项目；
- 共同上下文：双方共同经历、约定、反复出现的话题和互动模式，
  写入 interaction.shared_context 时，如有明确时间就带上日期。

第四，比较新旧内容：
- 已有条目表达同一件事：不要重复 ADD；
  有新增细节用 UPDATE 或 REFINE，明确纠正用 CORRECT；
- 同一个人物/项目/实体：优先更新原条目，而不是新建；
- 不同实体（例如机器狗项目与奶茶机项目）不得合并；
- 无法确定是否同一件事时，保留原条目，不要做破坏性合并。

可提出：
ADD、UPDATE、CORRECT、MERGE、REFINE、ARCHIVE。

ADD：增加新的独立信息。
UPDATE：更新旧信息的状态或内容（例如项目从进行中变为暂停/完成）。
CORRECT：根据本人的明确纠正修正错误。
MERGE：将真正重复或高度重叠的描述整合。
REFINE：在不改变事实的前提下使表达更准确、完整。
ARCHIVE：标记本人明确表示已过时或不再适用的内容。

只有以下情况才返回 NO_CHANGE：
1. 新对话中没有任何可归类的个人信息；
2. 只有一般知识、常识或与本人无关的话题；
3. 信息已与现有条目完全重复；
4. 信息是假设、玩笑、引用或第三方转述，不能归为本人事实；
5. 找不到合适的 allowed_field_keys 字段；
6. 与现有信息冲突且无法判断哪一方可信。

除以上情况外，不要返回 NO_CHANGE。

内容要求：
- proposed_content 必须忠于本人原话表达的事实，
  可以整理成简洁陈述，但不得添加原话没有的信息，
  不得把推断写成事实，不得改变程度、时间和对象。
- 不得虚构事件、原话或证据 ID。
- 关于性格、长期习惯、心理状态的描述属于推断：
  只有本人明确自述时才能作为事实提出；
  仅从单次行为推测的，不要提出。
- 一次行为不能说成长期习惯；
  “最近做过一次”不得写成“经常做”。
- 用户明确的否定、纠正或新状态优先于旧推断。
- 不要为了让 Profile 变化而强行提出变更。
- 同一轮可以提出多个不同字段的合法变更，但不得超过现有数量上限。
- confidence 只是元数据，供本地程序参考和排序，
  它不是写入门槛，也不能替代证据。
- 证据必须来自 this person 的真实发言：
  evidence_event_ids 只能取自 new_evidence.allowed_event_ids，
  evidence_quotes 必须是这些发言中真实存在的原文子串。
- 如果信息没有来源事件，或来源已被删除，不要提出。

严格身份字段（例如 identity.name、identity.preferred_name、
identity.nicknames、identity.pronouns）不在 allowed_field_keys 中，
你不得修改，也不得推断。本人明确自述时由本地可信路径直接写入。

你的输出只是一份提取结果。
本地可信程序会逐条验证证据、字段、内容与冲突；
通过就写入并生成 Revision，不通过就拒绝并记录原因。"""

PROFILE_NO_CHANGE_EXAMPLE = json.dumps(
    {
        "decision": "NO_CHANGE",
        "reason": "新对话没有提供属于本人的可靠新信息",
        "changes": [],
    },
    ensure_ascii=False,
    indent=2,
)

PROFILE_PROPOSE_CHANGES_EXAMPLE = json.dumps(
    {
        "decision": "PROPOSE_CHANGES",
        "reason": "本人自述了项目、正在学习的内容、兴趣和回答风格偏好",
        "changes": [
            {
                "operation": "ADD",
                "field_key": "projects.active",
                "target_entry_ids": [],
                "proposed_content": "正在做一个机器人项目",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在学葡萄牙语，也在做一个机器人项目。"],
                "reason": "本人明确自述当前项目",
                "confidence": 0.9,
            },
            {
                "operation": "ADD",
                "field_key": "knowledge.skills",
                "target_entry_ids": [],
                "proposed_content": "正在学习葡萄牙语",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在学葡萄牙语，也在做一个机器人项目。"],
                "reason": "本人明确自述正在学习的技能",
                "confidence": 0.9,
            },
            {
                "operation": "ADD",
                "field_key": "interests.likes",
                "target_entry_ids": [],
                "proposed_content": "喜欢爵士乐",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我喜欢爵士乐，不太喜欢太正式的回答。"],
                "reason": "本人明确自述兴趣",
                "confidence": 0.9,
            },
            {
                "operation": "ADD",
                "field_key": "communication.preferred_tone",
                "target_entry_ids": [],
                "proposed_content": "偏好不太正式的回答",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我喜欢爵士乐，不太喜欢太正式的回答。"],
                "reason": "本人明确说明交流风格偏好",
                "confidence": 0.9,
            },
        ],
    },
    ensure_ascii=False,
    indent=2,
)

_PROMPT_TAIL = """输出协议（必须严格遵守）：

- 只输出一个 JSON 对象，不得输出 Markdown 代码块、解释性文字或任何其他内容。
- 顶层字段只允许 decision、reason、changes。
- decision 只允许 "NO_CHANGE" 或 "PROPOSE_CHANGES"。
- 不得使用 "CHANGE"、"action"、"updates"、"proposals" 等替代字段或值。
  如果需要修改，必须写 "PROPOSE_CHANGES" 并使用 "changes" 数组。
- decision 为 "NO_CHANGE" 时，changes 必须为 []。
- 不得输出 person_id 或 base_revision；本地程序会填入可信值。
- field_key 必须来自 allowed_field_keys。
- evidence_event_ids 必须来自 new_evidence.allowed_event_ids，不得编造。
- changes 中不得出现 requires_confirmation；它不会影响本地是否写入。

示例一（NO_CHANGE）：
__NO_CHANGE_EXAMPLE__

示例二（PROPOSE_CHANGES）：
__PROPOSE_CHANGES_EXAMPLE__"""

PROFILE_REFINEMENT_SYSTEM_PROMPT = (
    _PROMPT_HEAD
    + "\n\n"
    + _PROMPT_TAIL.replace(
        "__NO_CHANGE_EXAMPLE__", PROFILE_NO_CHANGE_EXAMPLE
    ).replace(
        "__PROPOSE_CHANGES_EXAMPLE__", PROFILE_PROPOSE_CHANGES_EXAMPLE
    )
)


def build_profile_refinement_messages(payload: dict) -> list[dict]:
    """Build chat messages for one review.

    ``payload`` is assembled by trusted code: person id, revision, entry ids
    and allowed evidence ids all come from the database, not from a model.
    """
    user_content = (
        "请从以下新对话中提取这个人的 Personal Profile 信息。\n"
        "person_id、profile_revision、entry_id、allowed_event_ids 均由本地程序提供，"
        "不得修改、不得编造。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": PROFILE_REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_profile_refinement_repair_message(
    problem: str,
) -> dict:
    """One format-repair turn. It carries the error type, never model output.

    The original task messages are reused unchanged; this only adds a short
    instruction with the validation failure, so the model can regenerate a
    schema-valid object without the app echoing raw content anywhere.
    """
    return {
        "role": "user",
        "content": (
            "上一次输出未通过本地 JSON Schema 校验"
            f"（错误类型：{problem}）。\n"
            "请基于上面的原始任务重新输出，只输出一个符合协议的 JSON 对象。\n"
            "不要输出解释，不要使用 action、updates、CHANGE 等替代字段或值。\n"
            "不得编造 field_key、证据 ID 或人物信息；如果无法形成合法变更，"
            "返回 NO_CHANGE。"
        ),
    }


__all__ = [
    "PROFILE_REFINEMENT_PROMPT_VERSION",
    "PROFILE_REFINEMENT_SYSTEM_PROMPT",
    "PROFILE_NO_CHANGE_EXAMPLE",
    "PROFILE_PROPOSE_CHANGES_EXAMPLE",
    "build_profile_refinement_messages",
    "build_profile_refinement_repair_message",
]
