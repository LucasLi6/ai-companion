"""Read-time cleanup for replies written before the V1.2 ``[]`` fix.

Older builds could persist a bare ``[]`` (the model's empty voice markup) at
the start of an assistant reply. Those rows stay in the database until an
explicit repair is approved, but new turns must not feed the polluted prefix
back into the router, speaker, profile or memory paths.

The helper is deliberately narrow: it only removes *leading* empty bracket
markup from assistant replies and never touches tags, code, arrays or Markdown
recognized by the parser. It is shared by the repository read path and the
offline repair tool, so both apply exactly the same rule.
"""

from __future__ import annotations

import re

__all__ = ["strip_leading_empty_bracket_markup"]

_EMPTY_BRACKET = re.compile(r"\[[ \t]*\]")
# After an empty bracket, these characters mean the bracket is syntax, not
# model markup: array/index access, a Markdown link/reference, or an operator.
_CODE_OR_LINK_FOLLOWERS = frozenset("([{}])<>,;:+=*/\\|&%^~`")


def strip_leading_empty_bracket_markup(text: str) -> str:
    """Remove leading ``[]``/``[  ]`` markers, preserving the body.

    Only a prefix made of empty bracket groups separated by spaces/tabs is
    removed. If the bracket is followed by syntax (``[](...)``, ``[],``,
    ``a[]``), it is left untouched. The returned string is right-stripped of
    the whitespace that followed removed markup, but intentional leading
    whitespace on an otherwise clean reply is preserved.
    """
    if not text:
        return text
    position = len(text) - len(text.lstrip(" \t\r\n"))
    removed = False
    while True:
        match = _EMPTY_BRACKET.match(text, position)
        if match is None:
            break
        after = match.end()
        follower = text[after : after + 1]
        if follower and follower in _CODE_OR_LINK_FOLLOWERS:
            break
        # ``[] (link)`` / ``[] = value`` are syntax even with a separating
        # space; another empty bracket is the repeated pollution pattern.
        probe = after
        while probe < len(text) and text[probe] in " \t":
            probe += 1
        if probe < len(text) and text[probe] in (_CODE_OR_LINK_FOLLOWERS - {"["}):
            break
        # ``[] foo`` at the start is the historical pollution shape. A bracket
        # directly followed by an identifier without a space (``[]foo``) is
        # more likely syntax, so require whitespace or end-of-string.
        if follower and not follower.isspace():
            break
        position = after
        while position < len(text) and text[position] in " \t":
            position += 1
        removed = True
    if not removed:
        return text
    if position < len(text) and text[position] == "\n":
        position += 1
        while position < len(text) and text[position] in " \t":
            position += 1
    return text[position:]
