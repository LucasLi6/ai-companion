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
    NATURAL_VOICE,
    TAG_KINDS,
    VOICE_NAME_SET,
    is_empty_bracket_markup,
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

    def __init__(self, *, remove_empty_brackets: bool = False) -> None:
        self._pending = ""
        self._finished = False
        self._just_removed = False
        self._just_removed_display = False
        self._text_parts: list[str] = []
        self._display_parts: list[str] = []
        self._segments: list[SpeechExpressionSegment] = []
        self._invalid_tags: list[str] = []
        self._invalid_tag_count = 0
        self._empty_markup_count = 0
        self._valid_control_count = 0
        self._valid_nonverbal_count = 0
        # V1.2: empty ``[]`` is model markup in ordinary conversation, but a
        # legitimate empty array/list inside backticks or a fenced code block
        # must survive. This state is only tracked by the speech path.
        self._remove_empty_brackets = bool(remove_empty_brackets)
        self._code_state = "text"  # text | inline | fence
        self._backtick_run = 0
        # Last non-whitespace display character across chunks. Used to tell a
        # prose ``[]`` from a code/array literal with a separating space
        # (``return []``, ``items = []``) even when the token begins a chunk.
        self._last_significant = ""

    # -- diagnostics -------------------------------------------------------

    @property
    def invalid_tag_count(self) -> int:
        return self._invalid_tag_count

    @property
    def invalid_tag_samples(self) -> tuple[str, ...]:
        return tuple(self._invalid_tags)

    @property
    def empty_markup_count(self) -> int:
        return self._empty_markup_count

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
                if (
                    close + 1 == length
                    and close - index + 1 <= MAX_BRACKET_CANDIDATE_LENGTH
                    and self._needs_follower(buffer, index, close)
                ):
                    # The candidate may be Markdown/syntax (``[happy](url)``)
                    # whose next character decides it. Hold the whole bounded
                    # candidate until one more character (or finish) arrives.
                    self._pending = buffer[index:]
                    index = length
                    break
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
            if pending.endswith("]") and pending.startswith("["):
                # A complete candidate held back only to inspect its follower.
                # At end-of-stream there is no follower, so classify it now.
                self._consume_candidate(
                    pending, 0, len(pending) - 1, out_text, out_display
                )
            elif is_tag_prefix(pending) or is_invalid_tag_prefix(pending):
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

    def _needs_follower(self, buffer: str, start: int, close: int) -> bool:
        """Whether a candidate ending the buffer needs one more character.

        ``[happy](url)`` is Markdown, not a voice tag; the ``(`` only exists in
        the next chunk. The same is true for ``[](link)``. Hold such a bounded
        candidate until the follower is known (or the stream ends).
        """
        candidate = buffer[start : close + 1]
        if candidate in TAG_KINDS:
            return False
        preceded_by = buffer[start - 1] if start > 0 else self._last_display_char()
        inner = candidate[1:-1]
        if (
            self._remove_empty_brackets
            and self._code_state == "text"
            and is_empty_bracket_markup(inner)
            and self._is_standalone_empty_markup(
                preceded_by,
                "",
                self._previous_significant_char(buffer, start),
            )
        ):
            return True
        return is_invalid_speech_tag(
            inner, preceded_by=preceded_by, followed_by=""
        )

    def _consume_candidate(
        self,
        buffer: str,
        start: int,
        close: int,
        out_text: list[str],
        out_display: list[str],
    ) -> None:
        self._flush_backticks()
        candidate = buffer[start : close + 1]
        if self._code_state != "text":
            # Inside backticks or a fenced block every bracket is literal
            # code, even one spelled exactly like an official voice tag.
            self._append_text(candidate, out_text, out_display)
            return
        if candidate in TAG_KINDS:
            self._append_tag(candidate, out_display)
            return
        preceded_by = buffer[start - 1] if start > 0 else self._last_display_char()
        followed_by = buffer[close + 1] if close + 1 < len(buffer) else ""
        if (
            self._remove_empty_brackets
            and self._code_state == "text"
            and is_empty_bracket_markup(candidate[1:-1])
            and self._is_standalone_empty_markup(
                preceded_by,
                followed_by,
                self._previous_significant_char(buffer, start),
            )
        ):
            # The model's V1.1 habit of emitting a bare ``[]`` (or ``[  ]``)
            # must never reach display, history or memory. Backticked and
            # fenced code is excluded above; identifiers/arrays/Markdown links
            # are excluded by context so legitimate ``[]`` survives.
            self._remove_invalid(candidate)
            self._empty_markup_count += 1
            return
        if is_invalid_speech_tag(
            candidate[1:-1],
            preceded_by=preceded_by,
            followed_by=followed_by,
        ):
            self._remove_invalid(candidate)
            return
        self._append_text(candidate, out_text, out_display)

    @staticmethod
    def _is_standalone_empty_markup(
        preceded_by: str,
        followed_by: str,
        previous_significant: str = "",
    ) -> bool:
        """Keep brackets that are part of code/array/Markdown syntax."""
        if followed_by in ("(", "[", ".", ":", "\\"):
            return False
        if preceded_by and (
            (preceded_by.isascii() and (preceded_by.isalnum() or preceded_by in "_'"))
            or preceded_by in ")]\\"
        ):
            return False
        if followed_by and followed_by in "\"'":
            return False
        # A space-separated code context: ``return []``, ``items = []``,
        # ``f(x, [])``. Only CJK prose or a string start makes a bare ``[]``
        # the model's empty voice markup.
        if (
            previous_significant
            and previous_significant.isascii()
            and (
                previous_significant.isalnum()
                or previous_significant in "_=([{,;"
            )
        ):
            return False
        return True

    def _previous_significant_char(self, buffer: str, start: int) -> str:
        probe = start - 1
        while probe >= 0 and buffer[probe] in " \t\r\n":
            probe -= 1
        if probe >= 0:
            return buffer[probe]
        return self._last_significant

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
        self._track_code(piece)
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
            for char in reversed(display_piece):
                if not char.isspace():
                    self._last_significant = char
                    break

    def _track_code(self, piece: str) -> None:
        """Track backticked/fenced code so empty ``[]`` inside it survives."""
        index = 0
        length = len(piece)
        while index < length:
            if piece[index] != "`":
                self._flush_backticks()
                index += 1
                continue
            end = index
            while end < length and piece[end] == "`":
                end += 1
            self._backtick_run += end - index
            index = end

    def _flush_backticks(self) -> None:
        run = self._backtick_run
        if not run:
            return
        self._backtick_run = 0
        if self._code_state == "text":
            if run >= 3:
                self._code_state = "fence"
            else:
                self._code_state = "inline"
        elif self._code_state == "inline":
            # Any backtick closes an inline code span.
            self._code_state = "text"
        elif self._code_state == "fence":
            if run >= 3:
                self._code_state = "text"

    def _append_tag(self, tag: str, out_display: list[str]) -> None:
        self._display_parts.append(tag)
        out_display.append(tag)
        self._last_significant = "]"
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


def parse_speech_expressions(
    text: str, *, remove_empty_brackets: bool = True
) -> SpeechExpressionResult:
    """Parse a complete reply (non-streaming and proactive paths)."""
    parser = SpeechMarkupParser(remove_empty_brackets=remove_empty_brackets)
    parser.feed(text)
    parser.finish()
    return parser.result()


@dataclass(frozen=True)
class VoiceResponseChunk:
    """Text confirmed by one ``VoiceResponseParser`` call."""

    text: str = ""
    display_text: str = ""


@dataclass(frozen=True)
class VoiceResponseResult:
    """A parsed VOICE/TEXT reply.

    ``voice`` is the canonical internal name of the initial control tag, or
    ``None`` for natural/empty/invalid/legacy output. It is runtime-only and is
    never persisted as a label. ``protocol_valid`` describes the header format,
    not whether the reply is usable.
    """

    text: str
    display_text: str
    voice: str | None
    segments: tuple[SpeechExpressionSegment, ...]
    invalid_tags: tuple[str, ...]
    protocol_valid: bool
    invalid_voice: str | None = None
    protocol_error: str | None = None

    @property
    def has_valid_text(self) -> bool:
        return bool(self.text and self.text.strip())


MAX_VOICE_HEADER_CHARS = 512


class VoiceResponseParser:
    """Streaming parser for the V1.2 ``VOICE=... / TEXT=...`` protocol.

    The parser buffers only the short header, then streams the body through the
    existing :class:`SpeechMarkupParser`. If the output does not start with the
    header it falls back to the V1.1/plain-text parser, so recovery, old-model
    output and normal prose remain usable. An invalid ``VOICE`` never produces
    a tag; it is recorded as a diagnostic only.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "header"  # header | await_text | body | legacy
        self._voice: str | None = None
        self._invalid_voice: str | None = None
        self._protocol_error: str | None = None
        self._protocol_valid = False
        self._voice_valid = False
        self._display_prefix = ""
        self._prefix_emitted = False
        self._body = SpeechMarkupParser(remove_empty_brackets=True)
        self._body_started = False
        self._finished = False

    # -- diagnostics -------------------------------------------------------

    @property
    def voice(self) -> str | None:
        return self._voice

    @property
    def invalid_voice(self) -> str | None:
        return self._invalid_voice

    @property
    def protocol_valid(self) -> bool:
        return self._protocol_valid

    @property
    def protocol_error(self) -> str | None:
        return self._protocol_error

    @property
    def invalid_tag_count(self) -> int:
        return self._body.invalid_tag_count

    @property
    def invalid_tag_samples(self) -> tuple[str, ...]:
        return self._body.invalid_tag_samples

    @property
    def empty_markup_count(self) -> int:
        return self._body.empty_markup_count

    # -- confirmed output --------------------------------------------------

    @property
    def text(self) -> str:
        return self._body.text

    @property
    def display_text(self) -> str:
        return self._display_prefix + self._body.display_text

    @property
    def finished(self) -> bool:
        return self._finished

    def result(self) -> VoiceResponseResult:
        body = self._body.result()
        return VoiceResponseResult(
            text=body.text,
            display_text=self._display_prefix + body.display_text,
            voice=self._voice,
            segments=body.segments,
            invalid_tags=body.invalid_tags,
            protocol_valid=self._protocol_valid,
            invalid_voice=self._invalid_voice,
            protocol_error=self._protocol_error,
        )

    # -- streaming ---------------------------------------------------------

    def feed(self, chunk: str) -> VoiceResponseChunk:
        if self._finished:
            raise RuntimeError("feed() called after finish()")
        out_text: list[str] = []
        out_display: list[str] = []
        self._buffer += chunk
        self._drain(out_text, out_display, final=False)
        return VoiceResponseChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    def finish(self) -> VoiceResponseChunk:
        if self._finished:
            return VoiceResponseChunk()
        out_text: list[str] = []
        out_display: list[str] = []
        self._drain(out_text, out_display, final=True)
        tail = self._body.finish()
        if tail.text:
            out_text.append(tail.text)
        if tail.display_text:
            out_display.append(tail.display_text)
        self._finished = True
        return VoiceResponseChunk(
            text="".join(out_text), display_text="".join(out_display)
        )

    # -- internals ---------------------------------------------------------

    def _drain(self, out_text, out_display, *, final: bool) -> None:
        while self._buffer or (final and self._mode in ("header", "await_text")):
            if self._mode in ("body", "legacy"):
                text = self._buffer
                self._buffer = ""
                self._feed_body(text, out_text, out_display)
                return

            if self._mode == "header":
                self._buffer = self._buffer.lstrip()
                if not self._buffer:
                    return
                upper = self._buffer.upper()
                if not upper.startswith("VOICE="):
                    if not final and "VOICE=".startswith(upper):
                        return
                    self._mode = "legacy"
                    self._protocol_error = "MISSING_HEADER"
                    self._protocol_valid = False
                    continue
                if len(self._buffer) < len("VOICE="):
                    return
                newline = self._buffer.find("\n")
                if newline < 0:
                    if not final and len(self._buffer) <= MAX_VOICE_HEADER_CHARS:
                        return
                    line = self._buffer.rstrip("\r")
                    self._buffer = ""
                else:
                    line = self._buffer[:newline].rstrip("\r")
                    self._buffer = self._buffer[newline + 1 :]
                self._parse_voice_line(line, out_display)
                continue

            # await_text
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                if final:
                    self._protocol_error = self._protocol_error or "MISSING_TEXT"
                    self._protocol_valid = False
                return
            upper = self._buffer.upper()
            if upper.startswith("TEXT="):
                body = self._buffer[len("TEXT=") :]
                self._buffer = ""
                self._mode = "body"
                if self._voice_valid:
                    self._protocol_valid = True
                self._emit_prefix(out_display)
                self._feed_body(body, out_text, out_display)
                continue
            if not final and "TEXT=".startswith(upper):
                return
            # A VOICE header followed by ordinary text: compatibility mode.
            self._protocol_error = self._protocol_error or "MISSING_TEXT"
            self._protocol_valid = False
            self._mode = "body"
            self._emit_prefix(out_display)
            continue

    def _parse_voice_line(self, line: str, out_display: list[str]) -> None:
        content = line.strip()
        rest = content[len("VOICE=") :] if content.upper().startswith("VOICE=") else content
        upper = rest.upper()
        marker = upper.find("TEXT=")
        if marker >= 0:
            voice_raw = rest[:marker].strip()
            body = rest[marker + len("TEXT=") :]
            self._set_voice(voice_raw)
            self._mode = "body"
            if self._voice_valid:
                self._protocol_valid = True
            self._emit_prefix(out_display)
            self._buffer = body
        else:
            self._set_voice(rest.strip())
            self._mode = "await_text"
            self._protocol_valid = False

    def _set_voice(self, raw: str) -> None:
        value = raw.strip()
        lowered = value.lower()
        if not value or lowered == NATURAL_VOICE:
            self._voice = None
            self._voice_valid = True
            self._display_prefix = ""
            return
        if lowered in VOICE_NAME_SET:
            self._voice = lowered
            self._voice_valid = True
            self._display_prefix = f"[{lowered}]"
            return
        # Never guess a replacement tag for an invalid VOICE.
        self._voice = None
        self._voice_valid = False
        self._invalid_voice = value[:64]
        self._display_prefix = ""
        self._protocol_error = "INVALID_VOICE"

    def _emit_prefix(self, out_display: list[str]) -> None:
        if self._display_prefix and not self._prefix_emitted:
            out_display.append(self._display_prefix)
            self._prefix_emitted = True

    def _feed_body(self, text: str, out_text, out_display) -> None:
        if not text:
            return
        if not self._body_started:
            text = text.lstrip()
            if not text:
                return
            self._body_started = True
        chunk = self._body.feed(text)
        if chunk.text:
            out_text.append(chunk.text)
        if chunk.display_text:
            out_display.append(chunk.display_text)


def parse_voice_response(text: str) -> VoiceResponseResult:
    """Parse one complete VOICE/TEXT (or legacy) reply."""
    parser = VoiceResponseParser()
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
    "VoiceResponseChunk",
    "VoiceResponseParser",
    "VoiceResponseResult",
    "has_valid_reply_text",
    "parse_speech_expressions",
    "parse_voice_response",
]
