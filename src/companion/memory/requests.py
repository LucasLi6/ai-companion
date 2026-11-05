"""Narrow, local detection of explicit memory/history requests.

These are *commands phrased as sentences*, not intent classification. Only a
small set of unambiguous phrasings matches; a joke, a general question or a
topical keyword discussion does not. No model call is involved, and a match
never bypasses access control: the record lookup still runs through
:class:`~companion.memory.service.MemoryService` with the speaker's
non-admin access context.
"""

from __future__ import annotations

import re

_SAMPLE_RE = re.compile(
    r"(?:"
    r"(?:举|说|讲|拿|随便来|给我)[^，。！？,.!?]{0,8}(?:一个|一条|几个|一些)[^，。！？,.!?]{0,6}记忆"
    r"|你(?:都|还)?记得(?:什么|些啥|哪些|住什么)"
    r"|(?:说|讲)(?:一个|一条)?你(?:记得|记住)的"
    r"|(?:show|tell)\s+me\s+(?:a|one|some)\s+memor"
    r"|what\s+do\s+you\s+remember"
    r")",
    re.IGNORECASE,
)

_EARLIEST_RE = re.compile(
    r"(?:"
    r"第一次(?:跟|和|对|与)你(?:说|聊|讲|问)"
    r"|(?:我们|咱俩|你和我)(?:第一次|最早就|最早)"
    r"|最早(?:的)?(?:那次|一句|对话|聊天|交流|记忆)"
    r"|我(?:最开始|第一次)(?:说|问|发)"
    r"|(?:first|earliest)\s+(?:thing|time|message|conversation)\s+(?:i|we)"
    r")",
    re.IGNORECASE,
)

# Used only to decide whether the *past silence record* should accompany the
# ordinary prompt. It does not change the reply decision.
_PAST_SILENCE_RE = re.compile(
    r"(?:"
    r"(?:刚才|刚刚|上次|之前|前天|昨天)[^，。！？,.!?]{0,8}(?:为什么|怎么|为啥)[^，。！？,.!?]{0,8}(?:没|不)(?:回|说|理|答)"
    r"|(?:为什么|怎么|为啥)(?:刚才|刚刚|上次|之前)?[^，。！？,.!?]{0,6}(?:没|不)(?:回(?:我|消息)?|说话|理我|回答)"
    r"|(?:why\s+did(?:n'?t)?\s+you|why\s+not)[^.!?]{0,30}(?:reply|answer|respond|say\s+anything)"
    r")",
    re.IGNORECASE,
)


def is_memory_sample_request(text: str | None) -> bool:
    """An explicit request to surface a real memory record."""
    return bool(text and _SAMPLE_RE.search(text))


def is_earliest_history_request(text: str | None) -> bool:
    """An explicit request for the first thing the speaker said."""
    return bool(text and _EARLIEST_RE.search(text))


def is_past_silence_question(text: str | None) -> bool:
    """An explicit question about why the companion did not reply before."""
    return bool(text and _PAST_SILENCE_RE.search(text))


__all__ = [
    "is_memory_sample_request",
    "is_earliest_history_request",
    "is_past_silence_question",
]
