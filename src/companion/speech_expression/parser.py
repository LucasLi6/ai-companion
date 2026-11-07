"""Streaming-safe separation of speech labels from reply text.

The parser is deliberately local and deterministic: no model call, no network,
no database, no emotion state. It recognises exactly the tags in
:mod:`companion.speech_expression.catalog` and splits them into a canonical text
channel (what is saved and remembered) and a display channel (what is shown and,
later, what TTS will consume).

Streaming contract::

    parser = SpeechMarkupParser()
    a = parser.feed("[mis")   # nothing confirmed yet
    b = parser.feed("chievously]你好")
    c = parser.finish()
    assert a.text + b.text + c.text == parser.result().text
    assert a.display_text + b.display_text + c.display_text == (
        parser.result().display_text
    )

The canonical channel is exactly the input minus recognised whitelist tags,
except that a whitespace seam left by a removed tag is collapsed to one space
(``"我 [giggles] 好"`` becomes ``"我 好"``, not ``"我  好"``). Ordinary bracket
text is preserved literally; an unrecognised tag such as ``[happy]`` is never
treated as a voice instruction, and because it cannot be separated from ordinary
prose it is kept as text. Incomplete tag prefixes left at the end of a stream
are dropped rather than leaked as raw markup.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import KIND_CONTROL, KIND_NONVERBAL, TAG_KINDS, is_tag_prefix

KIND_TEXT = "text"

TAG_KINDS_SEGMENT = frozenset({KIND_CONTROL, KIND_NONVERBAL})


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

    @property
    def has_tags(self) -> bool:
        return any(segment.kind in TAG_KINDS_SEGMENT for segment in self.segments)


class SpeechMarkupParser:
    """Incremental parser for the speech-expression whitelist."""

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False
        self._just_skipped_tag = False
        self._text_parts: list[str] = []
        self._display_parts: list[str] = []
        self._segments: list[SpeechExpressionSegment] = []

    @property
    def text(self) -> str:
        """Canonical text confirmed so far (never contains a whitelist tag)."""
        return "".join(self._text_parts)

    @property
    def display_text(self) -> str:
        """Display text confirmed so far (tags kept in their original place)."""
        return "".join(self._display_parts)

    @property
    def finished(self) -> bool:
        return self._finished

    def result(self) -> SpeechExpressionResult:
        return SpeechExpressionResult(
            text=self.text,
            display_text=self.display_text,
            segments=tuple(self._segments),
        )

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

            # A '[' starts a possible tag. Scan only as far as the whitelist
            # allows; anything that can no longer be a tag becomes literal text.
            cursor = index + 1
            while True:
                current = buffer[index:cursor]
                if current in TAG_KINDS:
                    self._append_tag(current, out_display)
                    index = cursor
                    break
                if cursor == length:
                    if is_tag_prefix(current):
                        # A tag may continue in the next chunk.
                        self._pending = current
                    else:
                        # Unreachable in practice: a non-prefix is flushed as
                        # soon as it appears. Kept as a safe fallback.
                        self._append_text(current, out_text, out_display)
                    index = length
                    break
                if not is_tag_prefix(current):
                    # This '[' cannot begin a tag; protect the bracket and let
                    # the rest be scanned as ordinary text.
                    self._append_text("[", out_text, out_display)
                    index += 1
                    break
                cursor += 1

        return SpeechExpressionChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    def finish(self) -> SpeechExpressionChunk:
        """Flush the stream. Idempotent; an incomplete tag prefix is dropped."""
        if self._finished:
            return SpeechExpressionChunk()

        self._finished = True
        out_text: list[str] = []
        out_display: list[str] = []
        pending = self._pending
        self._pending = ""
        if pending and not is_tag_prefix(pending):
            self._append_text(pending, out_text, out_display)
        return SpeechExpressionChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    # -- internals ---------------------------------------------------------

    def _append_text(
        self,
        piece: str,
        out_text: list[str],
        out_display: list[str],
    ) -> None:
        if not piece:
            return
        canonical = piece
        if self._just_skipped_tag and self._text_parts:
            previous = self._text_parts[-1]
            if (
                previous
                and previous[-1].isspace()
                and canonical[0].isspace()
            ):
                # The removed tag stood between two whitespace characters;
                # one separator is enough in the canonical text.
                canonical = canonical.lstrip()
        self._just_skipped_tag = False
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
        self._display_parts.append(piece)
        out_display.append(piece)

    def _append_tag(self, tag: str, out_display: list[str]) -> None:
        self._display_parts.append(tag)
        out_display.append(tag)
        self._just_skipped_tag = True
        self._segments.append(
            SpeechExpressionSegment(kind=TAG_KINDS[tag], tag=tag)
        )


def parse_speech_expressions(text: str) -> SpeechExpressionResult:
    """Parse a complete reply (non-streaming and proactive paths)."""
    parser = SpeechMarkupParser()
    parser.feed(text)
    parser.finish()
    return parser.result()


__all__ = [
    "KIND_CONTROL",
    "KIND_NONVERBAL",
    "KIND_TEXT",
    "SpeechExpressionChunk",
    "SpeechExpressionResult",
    "SpeechExpressionSegment",
    "SpeechMarkupParser",
    "parse_speech_expressions",
]
