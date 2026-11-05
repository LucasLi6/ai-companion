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
_STRUCTURED_MARKERS = (
    "代码",
    "code",
    "教程",
    "推导",
    "公式",
    "步骤",
    "列表",
    "markdown",
    "文档",
    "表格",
    "示例",
    "example",
    "写一个",
    "解释一下原理",
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


def normalize_response_text(text: str, layout: str) -> str:
    """Return the reply text for persistence and display in the given layout.

    ``structured`` returns the text verbatim (only surrounding whitespace is
    trimmed). ``chat`` collapses every whitespace run — including ``\\n`` and
    ``\\n\\n`` — into a single space, yielding one natural paragraph.
    """
    if layout == RESPONSE_LAYOUT_STRUCTURED:
        return (text or "").strip()
    return _WHITESPACE_RUN.sub(" ", (text or "")).strip()
