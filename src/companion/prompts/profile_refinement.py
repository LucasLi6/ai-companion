"""System prompt for Profile-Aware Continuous Refinement (V0.9.7).

This prompt belongs to the *profile review* path only. It is deliberately
separate from Ava's Speaker prompt: reviewing a profile must never change her
conversational persona, tone or routing policy.
"""

from __future__ import annotations

import json

PROFILE_REFINEMENT_PROMPT_VERSION = "profile-refinement-v1"

PROFILE_REFINEMENT_SYSTEM_PROMPT = """你是 Ava 的 Personal Profile Refinement 模块。

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

如果没有必要修改，返回 NO_CHANGE。

只输出符合指定 Schema 的 JSON。"""


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


__all__ = [
    "PROFILE_REFINEMENT_PROMPT_VERSION",
    "PROFILE_REFINEMENT_SYSTEM_PROMPT",
    "build_profile_refinement_messages",
]
