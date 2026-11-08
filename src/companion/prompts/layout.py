"""Response layout: chat (single paragraph) vs structured (verbatim).

The prompt is the first layer of the single-paragraph guarantee; this module is
the optional second layer for the *display/persistence adapter*. It is a local,
deterministic decision — it never calls a model and never classifies style, so
it does not add a second LLM invocation and does not add a reply-style
classifier.

- ``chat`` collapses the finished reply into one natural paragraph (blank lines,
  stray newlines and repeated spaces become ordinary single spaces). This only
  removes *visual* paragraph breaks; it cannot remove semantic filler, which is
  why the prompt remains the real fix.
- ``structured`` keeps Markdown, code fences, blank lines and multi-line layout
  exactly as the model produced them, for explicit tutorials / code / formulas.

Normalisation operates on the *whole* accumulated text, never per-delta, so a
newline that arrives split across two chunks is still handled correctly.
"""

from __future__ import annotations

import re

RESPONSE_LAYOUT_CHAT = "chat"
RESPONSE_LAYOUT_STRUCTURED = "structured"

# Explicit structured-output signals. These are requests for a deliverable that
# needs layout, not ordinary conversation, so they justify keeping newlines.
# V1.1: words such as 示例 / 步骤 / 写一个 / 解释一下原理 were removed because
# they also appear in ordinary conversation ("你举个示例说明一下？"). They no
# longer suppress the voice-expression protocol by themselves.
_STRUCTURED_MARKERS = (
    "代码",
    "代码块",
    "code",
    "json",
    "教程",
    "推导",
    "公式",
    "列表",
    "markdown",
    "文档",
    "表格",
)


def response_layout_for(text: str) -> str:
    """Decide chat vs structured from the *user's current request*, locally.

    Defaults to ``chat``. Returns ``structured`` only when the user explicitly
    asks for a deliverable that normally needs paragraphs, code or lists.
    """
    lowered = (text or "").lower()
    if any(marker.lower() in lowered for marker in _STRUCTURED_MARKERS):
        return RESPONSE_LAYOUT_STRUCTURED
    return RESPONSE_LAYOUT_CHAT


_WHITESPACE_RUN = re.compile(r"\s+")
_WHITESPACE_PART = re.compile(r"(\s+)")


def normalize_response_text(text: str, layout: str) -> str:
    """Return the reply text for persistence and display in the given layout.

    ``structured`` returns the text verbatim (only surrounding whitespace is
    trimmed). ``chat`` collapses every whitespace run — including ``\\n`` and
    ``\\n\\n`` — into a single space, yielding one natural paragraph.
    """
    if layout == RESPONSE_LAYOUT_STRUCTURED:
        return (text or "").strip()
    return _WHITESPACE_RUN.sub(" ", (text or "")).strip()


class StreamingTextNormalizer:
    """Emit chunks whose concatenation equals :func:`normalize_response_text`.

    Normalisation runs on the whole accumulated reply, but a stream surface
    displays deltas as they arrive. If the surface shows raw chunks while the
    database stores the normalised text, the two can diverge (for example a
    ``\\n\\n`` that is later collapsed). This adapter holds back a trailing
    whitespace run until the next real character, so every emitted chunk is
    already in its final form:

    - ``chat``: whitespace runs become one space between words; leading and
      trailing whitespace is dropped (matching ``normalize_response_text``).
    - ``structured``: internal whitespace and line breaks are preserved
      verbatim; only leading/trailing whitespace is dropped.

    A chunk that normalises to nothing returns ``""`` and is not emitted.
    """

    def __init__(self, layout: str) -> None:
        self.layout = layout
        self._started = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self.layout == RESPONSE_LAYOUT_STRUCTURED:
            return self._feed_structured(chunk)
        return self._feed_chat(chunk)

    def _feed_chat(self, chunk: str) -> str:
        out: list[str] = []
        for part in _WHITESPACE_PART.split(chunk):
            if not part:
                continue
            if part.isspace():
                if self._started:
                    self._pending = " "
                continue
            if self._pending:
                out.append(self._pending)
                self._pending = ""
            out.append(part)
            self._started = True
        return "".join(out)

    def _feed_structured(self, chunk: str) -> str:
        out: list[str] = []
        for part in _WHITESPACE_PART.split(chunk):
            if not part:
                continue
            if part.isspace():
                self._pending += part
                continue
            if self._started and self._pending:
                out.append(self._pending)
            self._pending = ""
            out.append(part)
            self._started = True
        return "".join(out)

    def finish(self) -> str:
        """Trailing whitespace is trimmed by both layouts, so nothing remains."""
        self._pending = ""
        return ""
