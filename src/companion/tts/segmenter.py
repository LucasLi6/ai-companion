"""Deterministic streaming text segmentation for TTS Phase B.

The segmenter turns display-channel deltas (already validated by
``VoiceStreamParser``) into speakable chunks. It never tokenises character by
character, never splits a bracketed tag, and never re-emits a tag: tags keep
their original position in the stream, so a control tag's scope is preserved
by the single TTS session rather than being repeated at every segment.

Boundaries are deliberately conservative: sentence enders are preferred,
commas only after enough content, and URLs, decimal numbers, common English
abbreviations, Markdown links and backticked code are never split.
"""

from __future__ import annotations

DEFAULT_MIN_CHARS = 10
DEFAULT_MAX_CHARS = 60

_SENTENCE_END = frozenset("。！？!?…\n")
_COMMA = frozenset("，,、；;")
# A URL ends at whitespace or at CJK punctuation even without a space.
_URL_STOP = frozenset("。！？，、；：）】」』》")
_CLOSERS = frozenset("”’\"'）)】」』》")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
        "e.g", "i.e", "etc", "vs", "no", "fig", "al",
        "u.s", "a.m", "p.m",
    }
)
_URL_PREFIXES = ("http://", "https://", "www.")


class TextSegmenter:
    """Incremental, deterministic text splitter.

    ``feed`` may be called with arbitrarily small deltas (including a delta
    that ends in the middle of a tag); incomplete constructs stay buffered.
    ``finish`` emits whatever remains, including a short answer that has no
    terminating punctuation.
    """

    def __init__(
        self,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if min_chars < 1:
            raise ValueError("min_chars must be >= 1")
        if max_chars < min_chars:
            raise ValueError("max_chars must be >= min_chars")
        self._min_chars = int(min_chars)
        self._max_chars = int(max_chars)
        self._buffer = ""
        self._closed = False

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, delta: str) -> list[str]:
        if self._closed:
            raise RuntimeError("TextSegmenter is already finished")
        if delta:
            self._buffer += delta
        return self._drain()

    def finish(self) -> list[str]:
        if self._closed:
            return []
        self._closed = True
        piece = self._buffer.strip()
        self._buffer = ""
        return [piece] if piece else []

    def _drain(self) -> list[str]:
        segments: list[str] = []
        while True:
            cut = self._find_cut(self._buffer)
            if cut is None:
                break
            piece = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if piece:
                segments.append(piece)
        return segments

    # ------------------------------------------------------------- scanning
    def _find_cut(self, text: str) -> int | None:
        if not text:
            return None
        index = 0
        length = len(text)
        visible = 0
        last_space: int | None = None

        while index < length:
            char = text[index]

            if char == "`":
                index = self._skip_backticks(text, index)
                if index is None:
                    return None
                continue

            if char == "[":
                close = text.find("]", index + 1)
                if close == -1:
                    # Keep a partial tag (or any bracket text) for the next
                    # delta so it is never split or sent alone.
                    return None
                index = close + 1
                if index < length and text[index] == "(":
                    index = self._skip_link_target(text, index)
                    if index is None:
                        return None
                continue

            url_end = self._url_end(text, index)
            if url_end is not None:
                if url_end == length:
                    # The URL may continue in the next delta.
                    return None
                index = url_end
                continue

            if char.isspace():
                last_space = index + 1
                index += 1
                continue

            visible += 1

            if char in _SENTENCE_END or (
                char == "." and self._dot_is_boundary(text, index)
            ):
                if visible >= 2:
                    return self._after_closers(text, index + 1)
            elif char in _COMMA:
                if (
                    visible >= self._min_chars
                    and self._comma_is_boundary(text, index)
                ):
                    return self._after_closers(text, index + 1)

            if visible >= self._max_chars:
                if last_space is not None:
                    return last_space
                return index + 1

            index += 1

        return None

    @staticmethod
    def _after_closers(text: str, index: int) -> int:
        while index < len(text) and (
            text[index] in _CLOSERS or text[index] == "…"
        ):
            index += 1
        return index

    @staticmethod
    def _skip_backticks(text: str, index: int) -> int | None:
        run = 3 if text.startswith("```", index) else 1
        fence = "`" * run
        close = text.find(fence, index + run)
        if close == -1:
            return None
        return close + run

    @staticmethod
    def _skip_link_target(text: str, index: int) -> int | None:
        depth = 0
        cursor = index
        while cursor < len(text):
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        return None

    @staticmethod
    def _url_end(text: str, index: int) -> int | None:
        for prefix in _URL_PREFIXES:
            if not text.startswith(prefix, index):
                continue
            cursor = index
            while (
                cursor < len(text)
                and not text[cursor].isspace()
                and text[cursor] not in _URL_STOP
            ):
                cursor += 1
            return cursor
        return None

    @staticmethod
    def _dot_is_boundary(text: str, index: int) -> bool:
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return False
        if following and (following.isalnum() or following == "_"):
            return False
        token_start = index
        while token_start > 0 and (
            text[token_start - 1].isalpha() or text[token_start - 1] == "."
        ):
            token_start -= 1
        token = text[token_start:index].strip(".").lower()
        if token in _ABBREVIATIONS:
            return False
        if not following:
            return True
        return (
            following.isspace()
            or following in _CLOSERS
            or ord(following) > 127
        )

    @staticmethod
    def _comma_is_boundary(text: str, index: int) -> bool:
        char = text[index]
        if char == "，":
            return True
        following = text[index + 1] if index + 1 < len(text) else ""
        if not following:
            return True
        if following.isalnum() or following == "_":
            return False
        # ``1,000``: a digit before the comma is not a clause boundary.
        previous = text[index - 1] if index > 0 else ""
        if char == "," and previous.isdigit() and following.isdigit():
            return False
        return True


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_CHARS",
    "TextSegmenter",
]
