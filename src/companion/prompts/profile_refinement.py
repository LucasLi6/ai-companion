"""System prompt for Profile-Aware Continuous Refinement (V0.9.7).

This prompt belongs to the *profile review* path only. It is deliberately
separate from Ava's Speaker prompt: reviewing a profile must never change her
conversational persona, tone or routing policy.
"""

from __future__ import annotations

import json

PROFILE_REFINEMENT_PROMPT_VERSION = "profile-refinement-v2"

_PROMPT_HEAD = """你是 Ava 的 Personal Profile Refinement 模块。

你的职责不是收集对话中的所有个人信息，
而是结合一个人的现有完整 Profile 与新对话，
判断 Ava 对这个人的理解是否有必要改变。

工作顺序必须是：

第一，阅读并理解现有 Profile。
理解其中已有的事实、兴趣、性格倾向、项目、
行为模式、交流偏好，以及各项信息之间的关系。

第二，阅读新对话及必要上下文。
判断新内容是否真正提供了关于这个人的信息。
区分自述、假设、玩笑、引用、第三方信息、
临时状态、明确纠正和稳定个人特点。

第三，比较新旧理解。
如果现有 Profile 已经足够准确，
或者新内容没有实质意义，返回 NO_CHANGE。

第四，只有当新证据使人物理解值得改变时，
才提出最小必要修改。

可提出：
ADD、UPDATE、CORRECT、MERGE、REFINE、ARCHIVE。

ADD：增加新的独立信息。
UPDATE：更新旧信息的状态或内容。
CORRECT：根据可靠纠正修正错误。
MERGE：将真正重复或高度重叠的描述整合。
REFINE：在不改变事实的前提下使表达更准确、简洁。
ARCHIVE：标记已经过时或不再适用的内容。

不得为了显示自己完成工作而强行修改。
不得把整份 Profile 每轮改写一遍。

人物理解应该随着交流逐渐深入，而不是不断膨胀。

例如：
现有信息是“喜欢电子工程，正在开发 Ava”。
新对话深入讨论 Ava 的长期记忆设计。
你应该判断是否值得深化对现有项目的描述，
而不是机械新增一个“喜欢长期记忆”的兴趣。

例如：
现有信息是“技术决策理性”。
新对话表现出一次情绪化反应。
不能因此直接改为“性格冲动”。
单次表现不一定代表稳定特征。

例如：
现有信息描述了 STM32、RTOS、机器人控制。
可以适度精简重复内容，
但不能把具体能力全部压缩成“喜欢科技”。

必须保留有意义的区别、事实、时间和来源。

已确认身份字段不得由你直接修改。
你只能提出待本人确认的变更。

你的输出只是一份变更建议。
最终是否提交由本地可信程序决定。

不得虚构人物信息。
不得自行生成证据 ID。
不得将模型旧推断当作新的独立证据。
不得因为新信息与旧信息略有不同就假定旧信息错误。

如果没有必要修改，返回 NO_CHANGE。"""

# Two real, parseable JSON examples. They are built with json.dumps so the
# prompt can never drift from valid JSON again.
PROFILE_NO_CHANGE_EXAMPLE = json.dumps(
    {
        "decision": "NO_CHANGE",
        "reason": "现有 Profile 已足够准确，新对话没有提供稳定的新信息",
        "changes": [],
    },
    ensure_ascii=False,
    indent=2,
)

PROFILE_PROPOSE_CHANGES_EXAMPLE = json.dumps(
    {
        "decision": "PROPOSE_CHANGES",
        "reason": "本人明确说明了一个正在进行中的项目",
        "changes": [
            {
                "operation": "ADD",
                "field_key": "projects.active",
                "target_entry_ids": [],
                "proposed_content": "正在开发一个以 STM32 为主控的机器狗项目",
                "evidence_event_ids": ["evt_example_allowed_event_id"],
                "evidence_quotes": ["我最近在做一个机器狗项目，用 STM32 做主控。"],
                "reason": "本人明确自述当前项目，属于稳定信息",
                "confidence": 0.9,
                "requires_confirmation": False,
            }
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
        "请审查以下人物 Profile 与新对话。\n"
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
