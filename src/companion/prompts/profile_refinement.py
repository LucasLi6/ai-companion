"""System prompt for Personal Profile extraction (V0.9.9.23).

This prompt belongs to the *profile review* path only. It is deliberately
separate from Ava's Speaker prompt: reviewing a profile must never change her
conversational persona, tone or routing policy.

The model now does two separate jobs:

1. extract structured *facts* (entity / attribute / value / assertion type /
   time scope / evidence), and
2. decide how each fact relates to an existing entry (SAME / EXTENDS /
   STATE_CHANGE / CORRECTS / DISTINCT / UNCERTAIN).

Trusted local code still classifies every fact and validates every relation
against stored rows and persisted words. A one-off event, a temporary state,
a hypothetical or a third-party report cannot enter the long-term Profile by
model claim alone.
"""

from __future__ import annotations

import json

PROFILE_REFINEMENT_PROMPT_VERSION = "profile-refinement-v5"

_PROMPT_HEAD = """你是 Ava 的 Personal Profile Fact Extractor and Maintainer。

你的工作分成两个独立步骤：

第一步，从这个人本人发出的新发言中提取“事实”，而不是直接写 Profile。
每一条事实必须写成：
- entity：事实关于谁/什么（通常是本人 self；共同经历写清对象）；
- attribute：语义属性（例如 interest、skill、project、goal、habit、
  behavior_pattern、communication_preference、background、value）；
- value：本人明确表达的内容；
- assertion_type：DIRECT（本人明确自述）、CORRECTION（本人明确纠正）、
  OBSERVED（只从行为观察，本人没有直接说）、UNCERTAIN（含糊不确定）、
  HYPOTHETICAL（假设/玩笑）、THIRD_PARTY（转述别人）；
- temporal_scope：STABLE（长期稳定）、ONGOING（当前持续）、
  TEMPORARY（临时状态）、EVENT（一次性事件）、FUTURE（未来计划）；
- evidence_event_ids / evidence_quotes：真实来源，不得编造。

第二步，把每条事实与现有 ACTIVE 条目比较，判断关系：
- SAME：现有条目已经表达同一件事，只是措辞不同；
- EXTENDS：现有条目是同一件事，但这次补充了新的细节；
- STATE_CHANGE：同一实体从一种状态变为另一种状态
  （例如项目从进行中变为完成/暂停）；
- CORRECTS：本人明确否定或纠正旧内容；
- DISTINCT：这是另一件独立的事，不能合并；
- UNCERTAIN：无法判断是否同一件事。

只有 PROFILE 类事实才允许写入长期 Profile。判断规则：

1. 可以直接进入 Profile：
   - 本人明确自述的长期属性、身份以外的背景、价值观、技能、语言；
   - 本人明确自述的当前项目、目标、稳定偏好；
   - 本人明确纠正的旧事实；
   - 在至少两条独立发言中反复出现、且只描述行为模式的事实
     （此时 assertion_type=OBSERVED、temporal_scope=STABLE、attribute 必须是
     behavior_pattern 或 habit），不能据此推断兴趣、性格或动机。

2. 不得进入长期 Profile：
   - 一次性事件：今天去了哪里、这次比赛、本次对话发生了什么；
   - 临时状态：今天很累、现在很冷、这周在考试、最近失眠；
   - 情绪、玩笑、假设、转述、别人的信息；
   - 从单次行为推断出的性格、习惯、兴趣或动机；
   - 因为一个人多次问某类问题，就推断他“喜欢/热爱”这类东西；
   - 没有本人原话支持的推测。
   以上内容分类为 OBSERVATION 或 SCENE_MEMORY，不要为它们创建 changes。

3. interaction.shared_context 不是新的长期事实入口。
   共同经历、一次性约定、某天发生的事属于 Scene/Memory，
   不要 ADD、UPDATE 或 REFINE 这个字段。

4. 未来计划只有明确属于 goals 或 projects 时才可作为 FUTURE 事实写入；
   其他 FUTURE 内容视为 OBSERVATION。

Profile 长度控制（重要）：
- 宁可更新原条目，也不要为同一件事新增条目；
- 新细节优先用 EXTENDS/UPDATE 补进原条目，保留原有细节；
- 只有真正重复或高度重叠的条目才 MERGE；
- 不同属性不得合并：兴趣（interest）与技能（skill）、
  一次经历（EVENT）与长期习惯（habit）是不同事实；
- 不得为了缩短而删除已有细节，不得输出半句话；
- 单条内容超长时，优先用完整、准确的一句话重新精炼；
- 不允许静默截断、不允许丢弃旧事实。

工作顺序：
1. 阅读现有 Profile 和 entry_ids；
2. 逐条提取新对话中的事实；
3. 对每条可能写入的事实，明确给出 relation 和一个（跨字段 SAME 可多个）
   现有 entry_id；
4. 只为 PROFILE 类事实生成 changes；
5. 没有可写事实时返回 NO_CHANGE。

内容要求：
- proposed_content 必须忠于本人原话表达的事实，可整理为简洁陈述，
  但不得添加原话没有的信息，不得改变程度、时间和对象；
- 不得虚构事件、原话或证据 ID；
- confidence 只是元数据，不是写入门槛，也不能替代证据；
- 证据必须来自 this person 的真实发言：
  evidence_event_ids 只能取自 new_evidence.allowed_event_ids，
  evidence_quotes 必须是这些发言中真实存在的原文子串；
- 来源事件不存在或已被删除时，不要提出。

严格身份字段（identity.name、identity.preferred_name、identity.nicknames、
identity.pronouns 等）不在 allowed_field_keys 中，你不得修改或推断。
本人明确自述时由本地可信路径直接写入。

你的输出是一份提取与关系判断结果。
本地可信程序会重新分类事实、验证证据、字段、目标条目和冲突；
通过就写入并生成 Revision，不通过就拒绝并记录原因。
不存在人工确认队列，也不存在“先放进候选队列”的流程。"""

PROFILE_NO_CHANGE_EXAMPLE = json.dumps(
    {
        "decision": "NO_CHANGE",
        "reason": "新对话只有一次性事件和临时状态，没有可长期保存的事实",
        "facts": [
            {
                "fact_id": "f1",
                "entity": "self",
                "attribute": "shared_context",
                "value": "今天去了图书馆学习化学",
                "assertion_type": "DIRECT",
                "temporal_scope": "EVENT",
                "classification": "SCENE_MEMORY",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我现在都坐在图书馆里面了"],
            }
        ],
        "changes": [],
    },
    ensure_ascii=False,
    indent=2,
)

PROFILE_PROPOSE_CHANGES_EXAMPLE = json.dumps(
    {
        "decision": "PROPOSE_CHANGES",
        "reason": "本人明确自述了当前项目和学习内容",
        "facts": [
            {
                "fact_id": "f1",
                "entity": "self",
                "attribute": "project",
                "value": "正在做一个机器人项目",
                "assertion_type": "DIRECT",
                "temporal_scope": "ONGOING",
                "classification": "PROFILE",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在学葡萄牙语，也在做一个机器人项目。"],
            },
            {
                "fact_id": "f2",
                "entity": "self",
                "attribute": "skill",
                "value": "正在学习葡萄牙语",
                "assertion_type": "DIRECT",
                "temporal_scope": "ONGOING",
                "classification": "PROFILE",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在学葡萄牙语，也在做一个机器人项目。"],
            },
        ],
        "changes": [
            {
                "operation": "ADD",
                "field_key": "projects.active",
                "target_entry_ids": [],
                "fact_id": "f1",
                "relation": "DISTINCT",
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
                "fact_id": "f2",
                "relation": "DISTINCT",
                "proposed_content": "正在学习葡萄牙语",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在学葡萄牙语，也在做一个机器人项目。"],
                "reason": "本人明确自述正在学习的技能",
                "confidence": 0.9,
            },
        ],
    },
    ensure_ascii=False,
    indent=2,
)

_PROMPT_TAIL = """输出协议（必须严格遵守）：

- 只输出一个 JSON 对象，不得输出 Markdown 代码块、解释性文字或任何其他内容。
- 顶层字段只允许 decision、reason、facts、changes。
- decision 只允许 "NO_CHANGE" 或 "PROPOSE_CHANGES"。
- 不得使用 "CHANGE"、"action"、"updates"、"proposals" 等替代字段或值。
- decision 为 "NO_CHANGE" 时，changes 必须为 []；facts 可以保留观察结果。
- facts 中每条事实必须包含 entity、attribute、value、assertion_type、
  temporal_scope、evidence_event_ids、evidence_quotes。
  assertion_type 只允许 DIRECT / CORRECTION / OBSERVED / UNCERTAIN /
  HYPOTHETICAL / THIRD_PARTY。
  temporal_scope 只允许 STABLE / ONGOING / TEMPORARY / EVENT / FUTURE。
  classification 只允许 PROFILE / OBSERVATION / SCENE_MEMORY。
- 每个 change 必须带 fact_id 和 relation；relation 只允许
  SAME / EXTENDS / STATE_CHANGE / CORRECTS / DISTINCT / UNCERTAIN。
- SAME 必须给 target_entry_ids，且不需要 proposed_content；
  DISTINCT 不得给 target_entry_ids；
  EXTENDS / STATE_CHANGE / CORRECTS 必须给一个 target_entry_id。
- 不得输出 person_id 或 base_revision；本地程序会填入可信值。
- field_key 必须来自 allowed_field_keys。
- evidence_event_ids 必须来自 new_evidence.allowed_event_ids，不得编造。
- changes 中不得出现 requires_confirmation；它不会影响本地是否写入。
- 不得静默截断内容；宁可少写一条，也不要写半句话。

示例一（NO_CHANGE，只有一次性事件）：
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
        "请从以下新对话中提取这个人的 Personal Profile 事实，并判断每条事实"
        "与现有条目的关系。\n"
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
            "每条事实必须有 assertion_type 和 temporal_scope；"
            "每个 change 必须有 fact_id 和 relation。\n"
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
