"""The single speech-expression tag whitelist.

Only the tags defined here may act as voice instructions. The Speaker prompt is
rendered from this catalog, and :mod:`companion.speech_expression.parser` reads
the same definitions; there is no second hand-maintained list.

Control tags change how the following text is spoken and persist until the next
control tag or a real audio segment boundary. Nonverbal tags insert a sound at
their position and leave the surrounding delivery unchanged. Neither kind is an
emotion-state request: the Emotion system is independent and is never updated
from a tag.
"""

from __future__ import annotations

from dataclasses import dataclass

KIND_CONTROL = "control"
KIND_NONVERBAL = "nonverbal"

# Alibaba Cloud Qwen-Audio-3.1-TTS-Flash control tags. Exact official spelling,
# including spaces and lower case; do not add aliases.
CONTROL_TAGS: tuple[str, ...] = (
    "[sad]",
    "[amazed]",
    "[deep and loud shouting]",
    "[trembling]",
    "[angry]",
    "[excited]",
    "[sarcastic]",
    "[curious]",
    "[like dracula]",
    "[bored]",
    "[tired]",
    "[scornful]",
    "[shouting]",
    "[asmr]",
    "[panicked]",
    "[mischievously]",
    "[empathetic]",
    "[whispers]",
    "[reluctantly]",
    "[crying]",
    "[serious]",
    "[very slowly]",
    "[very fast]",
)

# Alibaba Cloud Qwen-Audio-3.1-TTS-Flash nonverbal sound tags.
NONVERBAL_TAGS: tuple[str, ...] = (
    "[gasp]",
    "[sighing]",
    "[clears throat]",
    "[giggles]",
    "[laughing]",
    "[cough]",
    "[snorts]",
)

ALL_TAGS: tuple[str, ...] = CONTROL_TAGS + NONVERBAL_TAGS
TAG_KINDS: dict[str, str] = {
    **{tag: KIND_CONTROL for tag in CONTROL_TAGS},
    **{tag: KIND_NONVERBAL for tag in NONVERBAL_TAGS},
}
MAX_TAG_LENGTH = max(len(tag) for tag in ALL_TAGS)

KIND_SEMANTICS = {
    KIND_CONTROL: (
        "控制后续文字的语气或表达方式；作用持续到下一个控制标签或模型实际的"
        "语音分段边界。不是一次性的声音动作。"
    ),
    KIND_NONVERBAL: (
        "在当前位置插入声音动作；不改变周围文字的持续语气，可以出现在句子"
        "开头、中间或结尾。"
    ),
}


@dataclass(frozen=True)
class SpeechTag:
    tag: str
    kind: str

    @property
    def semantics(self) -> str:
        return KIND_SEMANTICS[self.kind]


def tag_spec(tag: str) -> SpeechTag | None:
    kind = TAG_KINDS.get(tag)
    if kind is None:
        return None
    return SpeechTag(tag=tag, kind=kind)


def is_control_tag(tag: str) -> bool:
    return TAG_KINDS.get(tag) == KIND_CONTROL


def is_nonverbal_tag(tag: str) -> bool:
    return TAG_KINDS.get(tag) == KIND_NONVERBAL


def is_known_tag(tag: str) -> bool:
    return tag in TAG_KINDS


def is_tag_prefix(text: str) -> bool:
    """Whether ``text`` (including its ``[``) could still become a tag."""
    return any(tag.startswith(text) for tag in ALL_TAGS)


def render_tag_catalog() -> str:
    """The tag list shown to the Speaker, generated from the whitelist."""
    control = " ".join(CONTROL_TAGS)
    nonverbal = " ".join(NONVERBAL_TAGS)
    return (
        f"语气控制标签（{len(CONTROL_TAGS)} 个，控制后续文字的语气）：\n"
        f"{control}\n\n"
        f"声音动作标签（{len(NONVERBAL_TAGS)} 个，在当前位置发声）：\n"
        f"{nonverbal}"
    )


__all__ = [
    "ALL_TAGS",
    "CONTROL_TAGS",
    "KIND_CONTROL",
    "KIND_NONVERBAL",
    "KIND_SEMANTICS",
    "MAX_TAG_LENGTH",
    "NONVERBAL_TAGS",
    "SpeechTag",
    "TAG_KINDS",
    "is_control_tag",
    "is_known_tag",
    "is_nonverbal_tag",
    "is_tag_prefix",
    "render_tag_catalog",
    "tag_spec",
]
