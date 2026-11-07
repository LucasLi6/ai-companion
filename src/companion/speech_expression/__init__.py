"""Ava speech-expression labels: catalog, prompt rendering and parser.

The package only separates labels from text. It performs no TTS/ASR, no model
call and no database access, and it never touches Emotion, PersonaCore, memory,
Profile, Stance or Scene state.
"""

from .catalog import (
    ALL_TAGS,
    CONTROL_TAGS,
    KIND_CONTROL,
    KIND_NONVERBAL,
    KIND_SEMANTICS,
    NONVERBAL_TAGS,
    SpeechTag,
    is_control_tag,
    is_known_tag,
    is_nonverbal_tag,
    is_tag_prefix,
    render_tag_catalog,
    tag_spec,
)
from .parser import (
    SpeechExpressionChunk,
    SpeechExpressionResult,
    SpeechExpressionSegment,
    SpeechMarkupParser,
    parse_speech_expressions,
)

__all__ = [
    "ALL_TAGS",
    "CONTROL_TAGS",
    "KIND_CONTROL",
    "KIND_NONVERBAL",
    "KIND_SEMANTICS",
    "NONVERBAL_TAGS",
    "SpeechExpressionChunk",
    "SpeechExpressionResult",
    "SpeechExpressionSegment",
    "SpeechMarkupParser",
    "SpeechTag",
    "is_control_tag",
    "is_known_tag",
    "is_nonverbal_tag",
    "is_tag_prefix",
    "parse_speech_expressions",
    "render_tag_catalog",
    "tag_spec",
]
