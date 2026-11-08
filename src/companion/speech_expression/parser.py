"""Streaming-safe separation of speech labels from reply text.

The parser is deliberately local and deterministic: no model call, no network,
no database, no emotion state. It recognises exactly the tags in
:mod:`companion.speech_expression.catalog` and splits them into a canonical text
channel (what is saved and remembered) and a display channel (what is shown and,
later, what TTS will consume).

It also removes *invalid* voice markup — bracket content that clearly tries to
be a voice instruction but is not in the official whitelist, such as
``[happy]`` or ``[playful]``. Such markup is removed from both channels so it
can never reach TTS, history or memory. Ordinary brackets (``a[i]``,
``[草稿]``, ``[link](url)``, ``[1]``) are preserved literally; when the parser
cannot be confident, it keeps the text.

Streaming contract::

    parser = SpeechMarkupParser()
    a = parser.feed("[mis")   # nothing confirmed yet
    b = parser.feed("chievously]你好")
    c = parser.finish()
    assert a.text + b.text + c.text == parser.result().text
    assert a.display_text + b.display_text + c.display_text == (
        parser.result().display_text
    )

The canonical channel is exactly the input minus recognised whitelist tags and
invalid voice markup, except that a whitespace seam left by a removed tag is
collapsed to one space (``"我 [giggles] 好"`` becomes ``"我 好"``, not
``"我  好"``). Incomplete bracket candidates are held in a bounded buffer so a
tag split across arbitrary chunk boundaries is classified exactly once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import (
    KIND_CONTROL,
    KIND_NONVERBAL,
    MAX_BRACKET_CANDIDATE_LENGTH,
    TAG_KINDS,
    is_invalid_speech_tag,
    is_invalid_tag_prefix,
    is_tag_prefix,
)

KIND_TEXT = "text"

TAG_KINDS_SEGMENT = frozenset({KIND_CONTROL, KIND_NONVERBAL})

MAX_INVALID_SAMPLES = 10

# A bracket candidate that is still open may only contain characters a
# generated English speech tag could contain. Anything else (digits, CJK,
# punctuation) is ordinary text and is emitted immediately.
_CANDIDATE_SHAPE = re.compile(r"^[a-z '\-]*$")


@dataclass(frozen=True)
class SpeechExpressionChunk:
    """Text confirmed by one ``feed()`` or ``finish()`` call."""

    text: str = ""
    display_text: str = ""


@dataclass(frozen=True)
class SpeechExpressionSegment:
    """One ordered piece of a parsed reply.

    ``kind`` is ``"text"``, ``"control"`` or ``"nonverbal"``. Text segments carry
    ``text``; tag segments carry ``tag``. A control segment applies to the text
    that follows it, while a nonverbal segment is a point action.
    """

    kind: str
    text: str = ""
    tag: str | None = None


@dataclass(frozen=True)
class SpeechExpressionResult:
    text: str
    display_text: str
    segments: tuple[SpeechExpressionSegment, ...]
    invalid_tags: tuple[str, ...] = ()

    @property
    def has_tags(self) -> bool:
        return any(segment.kind in TAG_KINDS_SEGMENT for segment in self.segments)

    @property
    def has_valid_text(self) -> bool:
        """A reply is only usable when, after tag removal, real text remains."""
        return bool(self.text and self.text.strip())


class SpeechMarkupParser:
    """Incremental parser for the speech-expression whitelist."""

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False
        self._just_removed = False
        self._just_removed_display = False
        self._text_parts: list[str] = []
        self._display_parts: list[str] = []
        self._segments: list[SpeechExpressionSegment] = []
        self._invalid_tags: list[str] = []
        self._invalid_tag_count = 0
        self._valid_control_count = 0
        self._valid_nonverbal_count = 0

    # -- diagnostics -------------------------------------------------------

    @property
    def invalid_tag_count(self) -> int:
        return self._invalid_tag_count

    @property
    def invalid_tag_samples(self) -> tuple[str, ...]:
        return tuple(self._invalid_tags)

    @property
    def valid_control_count(self) -> int:
        return self._valid_control_count

    @property
    def valid_nonverbal_count(self) -> int:
        return self._valid_nonverbal_count

    # -- confirmed output --------------------------------------------------

    @property
    def text(self) -> str:
        """Canonical text confirmed so far (never contains a voice tag)."""
        return "".join(self._text_parts)

    @property
    def display_text(self) -> str:
        """Display text confirmed so far (valid tags kept in their place)."""
        return "".join(self._display_parts)

    @property
    def finished(self) -> bool:
        return self._finished

    def result(self) -> SpeechExpressionResult:
        return SpeechExpressionResult(
            text=self.text,
            display_text=self.display_text,
            segments=tuple(self._segments),
            invalid_tags=tuple(self._invalid_tags),
        )

    # -- streaming ---------------------------------------------------------

    def feed(self, chunk: str) -> SpeechExpressionChunk:
        """Consume one streamed delta and return newly confirmed output."""
        if self._finished:
            raise RuntimeError("feed() called after finish()")

        out_text: list[str] = []
        out_display: list[str] = []
        buffer = self._pending + chunk
        self._pending = ""
        index = 0
        length = len(buffer)

        while index < length:
            if buffer[index] != "[":
                end = buffer.find("[", index)
                if end < 0:
                    end = length
                self._append_text(buffer[index:end], out_text, out_display)
                index = end
                continue

            close = buffer.find("]", index + 1)
            if close >= 0:
                self._consume_candidate(
                    buffer, index, close, out_text, out_display
                )
                index = close + 1
                continue

            # The bracket is still open. Hold it only while it can plausibly
            # become a tag (valid or invalid); otherwise it is ordinary text.
            rest = buffer[index:]
            if (
                len(rest) > MAX_BRACKET_CANDIDATE_LENGTH
                or not self._could_be_voice_markup(rest)
            ):
                self._append_text("[", out_text, out_display)
                index += 1
                continue
            self._pending = rest
            index = length

        return SpeechExpressionChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    def finish(self) -> SpeechExpressionChunk:
        """Flush the stream. Idempotent; an incomplete voice tag is dropped."""
        if self._finished:
            return SpeechExpressionChunk()

        self._finished = True
        out_text: list[str] = []
        out_display: list[str] = []
        pending = self._pending
        self._pending = ""
        if pending:
            if is_tag_prefix(pending) or is_invalid_tag_prefix(pending):
                # An unfinished whitelist/invalid tag is model markup, not the
                # user-visible text. Dropping it is safer than leaking it.
                pass
            else:
                self._append_text(pending, out_text, out_display)
        return SpeechExpressionChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _could_be_voice_markup(piece: str) -> bool:
        if not piece.startswith("["):
            return False
        if len(piece) > MAX_BRACKET_CANDIDATE_LENGTH:
            return False
        if is_tag_prefix(piece) or is_invalid_tag_prefix(piece):
            return True
        return bool(_CANDIDATE_SHAPE.match(piece[1:]))

    def _consume_candidate(
        self,
        buffer: str,
        start: int,
        close: int,
        out_text: list[str],
        out_display: list[str],
    ) -> None:
        candidate = buffer[start : close + 1]
        if candidate in TAG_KINDS:
            self._append_tag(candidate, out_display)
            return
        preceded_by = buffer[start - 1] if start > 0 else self._last_display_char()
        followed_by = buffer[close + 1] if close + 1 < len(buffer) else ""
        if is_invalid_speech_tag(
            candidate[1:-1],
            preceded_by=preceded_by,
            followed_by=followed_by,
        ):
            self._remove_invalid(candidate)
            return
        self._append_text(candidate, out_text, out_display)

    def _last_display_char(self) -> str:
        if not self._display_parts:
            return ""
        last = self._display_parts[-1]
        return last[-1] if last else ""

    def _append_text(
        self,
        piece: str,
        out_text: list[str],
        out_display: list[str],
    ) -> None:
        if not piece:
            return
        canonical = piece
        if self._just_removed and (
            not self._text_parts
            or (self._text_parts[-1] and self._text_parts[-1][-1].isspace())
        ):
            # The removed tag stood at the start, or between two whitespace
            # characters; one separator is enough in the canonical text.
            canonical = canonical.lstrip()
        self._just_removed = False

        display_piece = piece
        if self._just_removed_display and (
            not self._display_parts
            or (
                self._display_parts[-1]
                and self._display_parts[-1][-1].isspace()
            )
        ):
            # Invalid markup is removed from the display too, so collapse the
            # same seam there.
            display_piece = display_piece.lstrip()
        self._just_removed_display = False

        if canonical:
            self._text_parts.append(canonical)
            out_text.append(canonical)
            if self._segments and self._segments[-1].kind == KIND_TEXT:
                previous_segment = self._segments[-1]
                self._segments[-1] = SpeechExpressionSegment(
                    kind=KIND_TEXT,
                    text=previous_segment.text + canonical,
                )
            else:
                self._segments.append(
                    SpeechExpressionSegment(kind=KIND_TEXT, text=canonical)
                )
        if display_piece:
            self._display_parts.append(display_piece)
            out_display.append(display_piece)

    def _append_tag(self, tag: str, out_display: list[str]) -> None:
        self._display_parts.append(tag)
        out_display.append(tag)
        self._just_removed = True
        kind = TAG_KINDS[tag]
        if kind == KIND_CONTROL:
            self._valid_control_count += 1
        else:
            self._valid_nonverbal_count += 1
        self._segments.append(SpeechExpressionSegment(kind=kind, tag=tag))

    def _remove_invalid(self, candidate: str) -> None:
        self._invalid_tag_count += 1
        self._just_removed = True
        self._just_removed_display = True
        if len(self._invalid_tags) < MAX_INVALID_SAMPLES:
            self._invalid_tags.append(candidate)


def parse_speech_expressions(text: str) -> SpeechExpressionResult:
    """Parse a complete reply (non-streaming and proactive paths)."""
    parser = SpeechMarkupParser()
    parser.feed(text)
    parser.finish()
    return parser.result()


def has_valid_reply_text(text: str) -> bool:
    """A reply needs real, non-whitespace text after tags are removed."""
    return bool(text and text.strip())


__all__ = [
    "KIND_CONTROL",
    "KIND_NONVERBAL",
    "KIND_TEXT",
    "SpeechExpressionChunk",
    "SpeechExpressionResult",
    "SpeechExpressionSegment",
    "SpeechMarkupParser",
    "has_valid_reply_text",
    "parse_speech_expressions",
]
