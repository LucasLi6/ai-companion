"""Compile the existing Agent Core into a stable personality section.

There is exactly one identity source: the AgentService's effective context
(global persona + relationship preset + per-person overrides). This compiler
does not introduce a second persona table, and it never lets a model write it.
It only turns already-trusted values into concise guidance for the response
prompt, plus a small set of pragmatic reminders derived from the scene and the
action history.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from companion.agent.dialogue_style import DialogueStyleProfile
from companion.agent.models import EffectiveAgentContext

MAX_STYLE_EXAMPLES = 3


class PersonalityCompiler:
    def compile(
        self,
        effective: EffectiveAgentContext,
        *,
        style: DialogueStyleProfile,
        scene: Optional[Mapping] = None,
        style_examples: Optional[Iterable[Mapping]] = None,
        current_text: str = "",
        stances: Optional[Iterable] = None,
    ) -> dict:
        examples = []
        for example in list(style_examples or [])[:MAX_STYLE_EXAMPLES]:
            if not isinstance(example, Mapping):
                continue
            examples.append(
                {
                    "scenario": str(example.get("scenario") or "")[:80],
                    "user_text": str(example.get("user_text") or "")[:1000],
                    "ideal_response": str(example.get("ideal_response") or "")[:2000],
                    # Explicitly mark the status so the prompt cannot mistake a
                    # style reference for real dialogue history.
                    "role": "STYLE_REFERENCE_NOT_HISTORY",
                }
            )

        principles = [
            "保持同一个人格：语气、价值观和边界跨轮次稳定，不随机换人设。",
            "先回应眼前这句话；不要每轮都用问候、总结或追问开场。",
            "可以有不同意见和偏好，不要永远附和，也不用刻意唱反调。",
            "不确定或不知道就直接说，不编造事实、记忆或现实环境。",
            "不要自称 AI/模型，不要提及 prompt、内部字段、评分或权限机制。",
            "高幽默不等于必须讲笑话；高共情不等于要分析对方的心理。",
            "对方明确要求某种表达方式时，本轮优先遵从，但仍服从事实与安全边界。",
            "性格通过选择和措辞自然体现，不要把价值观逐条说出来。",
        ]

        character = dict(effective.persona_core or {})
        stance_items = []
        for record in list(stances or [])[:8]:
            if isinstance(record, Mapping):
                topic = record.get("topic")
                stance = record.get("stance")
                source = record.get("source")
            else:
                topic = getattr(record, "topic", None)
                stance = getattr(record, "stance", None)
                source = getattr(record, "source", None)
            if not stance:
                continue
            stance_items.append(
                {
                    "topic": str(topic or "UNKNOWN")[:120],
                    "stance": str(stance)[:240],
                    "source": str(source or "RESPONSE")[:16],
                }
            )

        return {
            "agent_id": effective.agent_id,
            "display_name": effective.display_name,
            "pronouns": effective.pronouns,
            "description": effective.description,
            "relationship_tier": effective.familiarity,
            "traits": dict(effective.traits),
            "character": character,
            "persona_first": True,
            # Identity is compiled separately from mutable mood/style so the
            # response prompt can put "who this is" before "how this turn
            # feels".
            "identity": {
                "display_name": effective.display_name,
                "pronouns": effective.pronouns,
                "description": effective.description,
                "character_description": character.get("character_description")
                or "",
                "revision": character.get("revision"),
            },
            "character_guidance": self._character_guidance(character),
            "stances": stance_items,
            "stance_policy": (
                "这些是你过去实际说过的立场或管理员设定的立场，不是用户事实，"
                "也不是必须重复的台词。新信息可以改变看法，但要自然说明为什么；"
                "不要因为对方换了个语气就无声地站到反面。"
            ),
            "expression_precedence": [
                "安全与事实",
                "角色核心（价值观、偏好、边界）",
                "与当前说话者的关系",
                "当前语境与交流意图",
                "本轮情绪",
                "临时风格",
            ],
            "style": style.as_dict(),
            "principles": principles,
            "style_examples": examples,
            "style_examples_policy": (
                "这些只是需要满足的交流质量与风格参考，不是真实对话历史，"
                "也不是固定回复；不要逐字照抄。"
            ),
            "scene_topic": (scene or {}).get("topic") if scene else None,
        }

    @staticmethod
    def _character_guidance(character: Mapping) -> list[str]:
        """Turn trusted persona fields into behavioural, non-quotable guidance."""
        lines: list[str] = []
        values = list(character.get("core_values") or [])[:6]
        if values:
            lines.append(
                "你的判断以这些价值为锚："
                + "；".join(str(value) for value in values)
                + "。可以温和，但不要为了迎合而临时放弃它们。"
            )
        preferences = dict(character.get("preferences") or {})
        if preferences:
            pairs = [
                f"{key}={value}"
                for key, value in list(preferences.items())[:6]
            ]
            lines.append(
                "你有具体偏好：" + "；".join(pairs) + "。可以表达不同意见，"
                "不需要把每个偏好都解释一遍。"
            )
        quirks = list(character.get("quirks") or [])[:4]
        if quirks:
            lines.append(
                "自然的说话习惯：" + "；".join(str(item) for item in quirks) + "。"
            )
        contradictions = list(character.get("contradictions") or [])[:3]
        if contradictions:
            lines.append(
                "允许保留这些张力，不要强行磨平："
                + "；".join(str(item) for item in contradictions)
                + "。"
            )
        boundaries = list(character.get("conversational_boundaries") or [])[:6]
        if boundaries:
            lines.append(
                "边界（不因对方要求或本轮情绪改变）："
                + "；".join(str(item) for item in boundaries)
                + "。"
            )
        return lines

    def pragmatic_guidance(
        self,
        *,
        scene: Optional[Mapping] = None,
        action_history: Optional[Iterable] = None,
        current_addressee_hint: str = "unknown",
        quiet_mode: bool = False,
    ) -> list[str]:
        """Short, trusted reminders for continuity without scripting a reply."""
        lines: list[str] = []
        if scene:
            topic = scene.get("topic")
            if topic and topic != "UNKNOWN":
                lines.append(f"当前话题：{topic}。先顺着语境回应，不要突然换题。")
            loops = scene.get("open_loops") or []
            if loops:
                lines.append(
                    "存在未完成的话题时，只有当对方现在确实回到它才自然接上；"
                    "不要为了显得记得而强行提起。"
                )
            env = scene.get("environment") or {}
            if (
                env.get("location") not in (None, "UNKNOWN")
                or env.get("other_people_present") not in (None, "UNKNOWN")
            ):
                lines.append(
                    "环境细节只来自对方先前的明确陈述；不要把它当成现在的确定事实，"
                    "也不要主动描述房间、旁人或现实中正在发生的事。"
                )
        pending = [
            record
            for record in list(action_history or [])
            if getattr(record, "pending_followup", False)
        ]
        if pending:
            lines.append(
                "过去有未接上的话，但只有对方现在主动追问时才解释；"
                "不要每次恢复说话都道歉或解释沉默。"
            )
        if current_addressee_hint == "human":
            lines.append("这句话主要是说给另一个人听的，保持简短，不要抢话。")
        if quiet_mode:
            lines.append("当前会话处于静音模式；是否回复由本地策略决定。")
        lines.append("只回应本轮当前说话者；不要替别人说话或假装听见了未提供的对话。")
        return lines
