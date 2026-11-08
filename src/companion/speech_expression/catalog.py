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

import re
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

# Markup that clearly *tries* to be a voice instruction (emotion words, delivery
# adverbs and sound actions the model invented or borrowed from other TTS
# systems) but is not in the official whitelist. These are removed from both
# channels: they must never reach TTS, persistence or memory. This is a
# high-confidence vocabulary, not a general "remove English brackets" rule.
INVALID_SPEECH_TAGS: frozenset[str] = frozenset(
    {
        # emotion / attitude aliases
        "happy", "joy", "joyful", "cheerful", "glad", "delighted", "pleased",
        "excitedly", "sadness", "unhappy", "sorrowful", "depressed", "upset",
        "annoyed", "frustrated", "irritated", "mad", "furious", "calm",
        "relaxed", "peaceful", "neutral", "flat", "monotone", "deadpan",
        "robotic", "playful", "teasing", "confused", "confusion", "thinking",
        "thoughtful", "surprised", "surprise", "shocked", "astonished",
        "worried", "anxious", "nervous", "afraid", "scared", "fearful",
        "terrified", "hopeful", "optimistic", "desperate", "disappointed",
        "proud", "embarrassed", "shy", "awkward", "apologetic", "guilty",
        "ashamed", "disgusted", "contemptuous", "mocking", "ironic", "cynical",
        "dismissive", "smug", "affectionate", "loving", "caring", "sweet",
        "cute", "flirty", "seductive", "mysterious", "dramatic",
        "enthusiastic", "energetic", "lively", "gentle", "softly", "quietly",
        "loudly", "warmly", "coldly", "stern", "firm", "determined",
        "confident", "sleepy", "exhausted", "weary", "hesitant",
        "hesitantly",
        # delivery / pacing modifiers
        "whisper", "whispering", "murmur", "mumbling", "mutter", "muttering",
        "shout", "yelling", "yell", "scream", "screaming", "loud", "soft",
        "slowly", "quickly", "fast", "pause", "silence",
        # sound actions
        "sigh", "sighs", "laugh", "laughs", "chuckle", "chuckles", "giggle",
        "giggling", "snort", "sniff", "sniffle", "sob", "sobbing", "cry",
        "cries", "weep", "weeping", "coughing", "sneeze", "sneezing", "yawn",
        "yawning", "hum", "humming", "sing", "singing", "whistle", "breath",
        "breathe", "breathing", "pant", "panting", "groan", "groans", "moan",
        "moans", "wail", "cheer", "cheering", "clap", "applause", "music",
        "noise", "beep", "static", "gulp", "burp", "hiccup",
    }
)

# Lower-case English words that are ordinary bracket content rather than a
# generated voice tag. Context rules (array subscripts, Markdown links) add
# further protection; this list keeps common prose brackets untouched.
ORDINARY_BRACKET_WORDS: frozenset[str] = frozenset(
    {
        "a", "b", "c", "i", "j", "k", "n", "x", "y", "z", "id", "key", "name",
        "link", "url", "uri", "img", "image", "note", "todo", "draft", "ref",
        "citation", "source", "figure", "table", "json", "markdown", "html",
        "css", "code", "text", "example", "http", "https", "email", "info",
        "tip", "warning", "important", "optional", "default", "main", "test",
        "tests", "args", "kwargs", "value", "values", "item", "items",
        "index", "string", "object", "list", "detail", "details", "see",
        "also", "more", "less", "appendix", "chapter", "section", "part",
    }
)

_SOUND_STEMS = frozenset(
    {
        "giggl", "laugh", "sigh", "sob", "cry", "scream", "shout", "whisper",
        "sniffl", "cough", "sneez", "yawn", "hum", "sing", "mutter",
        "stutter", "breath", "pant", "chuckl", "groan", "moan", "wail",
        "cheer", "clap", "mumbl",
    }
)

MAX_INVALID_TAG_LENGTH = max(len(tag) + 2 for tag in INVALID_SPEECH_TAGS)
MAX_BRACKET_CANDIDATE_LENGTH = max(MAX_TAG_LENGTH, MAX_INVALID_TAG_LENGTH) + 8

# V1.2 VOICE= protocol: the internal name is exactly the official control tag
# without its brackets. One list, generated from CONTROL_TAGS, so prompt and
# parser cannot drift apart. ``natural`` and the empty value mean "no tag".
VOICE_NAMES: tuple[str, ...] = tuple(tag[1:-1] for tag in CONTROL_TAGS)
VOICE_NAME_SET: frozenset[str] = frozenset(VOICE_NAMES)
NATURAL_VOICE = "natural"

_INVALID_TAG_FORMS = tuple(f"[{tag}]" for tag in INVALID_SPEECH_TAGS)
_VOICE_WORD_SHAPE = re.compile(r"^[a-z][a-z '\-]{0,30}$")


def is_valid_voice_name(name: str) -> bool:
    """Whether ``name`` is an official control tag without brackets."""
    return name.strip().lower() in VOICE_NAME_SET


def is_empty_bracket_markup(inner: str) -> bool:
    """``[]`` or ``[   ]``: model markup with no usable instruction.

    This is checked only in the ordinary-conversation speech path; callers keep
    genuine code/array/Markdown brackets intact through context rules.
    """
    return not inner.strip()


def is_invalid_speech_tag(
    inner: str,
    *,
    preceded_by: str = "",
    followed_by: str = "",
) -> bool:
    """Whether a bracketed English phrase is clearly generated voice markup.

    ``inner`` is the text between ``[`` and ``]``. The classifier is
    intentionally conservative: an unknown bracket is removed only when the
    content is a known voice/emotion/sound word or a sound-action gerund, was
    not written as an array subscript or a Markdown link, and looks like a
    model-generated tag rather than prose.
    """
    if not inner or inner != inner.strip():
        return False
    if followed_by in ("(", "[", ":"):
        # Markdown link / nested bracket / label: ordinary text.
        return False
    if (
        preceded_by
        and preceded_by.isascii()
        and (preceded_by.isalnum() or preceded_by in "_'")
    ):
        # ``a[happy]`` / ``f'[x]'``: ASCII identifier or quoted text, not a
        # tag. CJK prose before a bracket is ordinary sentence text.
        return False
    lowered = inner.lower()
    if lowered in INVALID_SPEECH_TAGS:
        return True
    if lowered in ORDINARY_BRACKET_WORDS:
        return False
    if not _VOICE_WORD_SHAPE.match(lowered):
        return False
    words = lowered.replace("-", " ").split()
    if not 1 <= len(words) <= 3:
        return False
    # Unknown gerund built from a known sound stem ("giggling", "sighing").
    if len(words) == 1 and lowered.endswith("ing"):
        stem = lowered[:-3]
        if stem in _SOUND_STEMS or (stem + "e") in _SOUND_STEMS:
            return True
    return False


def is_invalid_tag_prefix(partial: str) -> bool:
    """Whether ``partial`` (starting with ``[``) could become an invalid tag."""
    return any(form.startswith(partial) for form in _INVALID_TAG_FORMS)


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


def render_voice_catalog() -> str:
    """The VOICE= value list, generated from the same control whitelist."""
    return " ".join(VOICE_NAMES)


__all__ = [
    "ALL_TAGS",
    "CONTROL_TAGS",
    "INVALID_SPEECH_TAGS",
    "KIND_CONTROL",
    "KIND_NONVERBAL",
    "KIND_SEMANTICS",
    "MAX_BRACKET_CANDIDATE_LENGTH",
    "MAX_INVALID_TAG_LENGTH",
    "MAX_TAG_LENGTH",
    "NONVERBAL_TAGS",
    "NATURAL_VOICE",
    "ORDINARY_BRACKET_WORDS",
    "SpeechTag",
    "TAG_KINDS",
    "VOICE_NAMES",
    "VOICE_NAME_SET",
    "is_control_tag",
    "is_empty_bracket_markup",
    "is_invalid_speech_tag",
    "is_invalid_tag_prefix",
    "is_known_tag",
    "is_nonverbal_tag",
    "is_tag_prefix",
    "is_valid_voice_name",
    "render_tag_catalog",
    "render_voice_catalog",
    "tag_spec",
]
