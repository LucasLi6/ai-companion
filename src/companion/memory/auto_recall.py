"""Local, conditional auto-recall for semantic turns (V0.9.9.3).

The semantic router decides whether long-term memory is *needed* and may call
``memory.search`` itself. This module adds a narrow safety net for the case the
router under-calls it: the user clearly continues an earlier experience, the
router still says RESPOND, and a concrete topic can be recovered from trusted
context.

It is deliberately local and cheap — no LLM call, no agent, no new store:

- it runs only after a validated route, only for RESPOND turns;
- it never runs when the router already asked a memory tool (successful or
  not: a failed/empty tool result is never replaced by a hidden success);
- it never runs for non-communicative input, WAIT/IGNORE, or quiet mode;
- the query is built from the current input, the router's turn state, the
  trusted scene topic and this speaker's own last one or two turns. Another
  person's turns and the profile's whole interest list are never dumped in;
- retrieval still goes through :class:`SqliteMemoryStore` (same access checks);
- results are bounded, de-duplicated and filtered for actual term overlap.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from companion.memory.store import _CJK_FUNCTION_CHARS, extract_keywords
from companion.routing.models import (
    InputSemanticStatus,
    RouteAction,
    TurnState,
)

MEMORY_TOOL_NAMES = frozenset({"memory.search", "memory.browse"})
MAX_AUTO_RECALL = 3
MAX_QUERY_CHARS = 120
MAX_QUERY_TERMS = 8
MAX_EPISODES_IN_RECALL = 2

# Words that describe the *act* of remembering or continuing, not a topic.
# Searching for these alone would return arbitrary memories, so they never
# qualify as a recall topic.
_DISCOURSE_TERMS = frozenset(
    {
        "这次", "上次", "那次", "这次", "那次", "之前", "以前", "后来",
        "现在", "刚才", "刚刚", "今天", "昨天", "前天", "明天", "最近",
        "还是", "继续", "接着", "然后", "但是", "因为", "所以", "如果",
        "这个", "那个", "这些", "那些", "事情", "问题", "东西", "方面",
        "我们", "你们", "他们", "自己", "什么", "怎么", "怎样", "如何",
        "为什么", "可以", "不行", "好的", "知道", "觉得", "感觉", "记得",
        "已经", "没有", "一个", "一些", "真的", "其实", "可能", "应该",
        "需要", "时候", "解决", "处理", "出现", "发生", "开始", "结束",
        "一样", "这样", "那样", "有点", "比较", "非常", "特别", "一直",
        "上次那个", "之前那个", "以前那个",
    }
)

# (A) The user explicitly points back at a past experience.
_PAST_REFERENCE = (
    re.compile(r"(?:还|仍然|仍)?记得.{0,12}(?:上次|之前|以前|那次|那天|昨天|上次那个)"),
    re.compile(r"(?:上次|之前|以前|那次|那天|昨天).{0,12}(?:那个|那件|问题|事情|后来|聊过)"),
    re.compile(r"后来(?:怎么|怎么样|怎样|如何|呢|解决|结果)"),
    re.compile(r"(?:我们)?(?:以前|之前|上次).{0,10}聊过"),
    re.compile(r"(?:那个|那件|上次的).{0,6}(?:后来|现在|最后)"),
)

# (C) The current input itself carries continuation semantics. Plain "还是"
# ("我还是喜欢…") or a bare "又" is too weak; require an unfinished-work or
# repeat marker.
_CONTINUATION = re.compile(
    r"(?:还是(?:不行|没有|没用|没|不|老样子|一样)|"
    r"(?:又|再次)(?:出|坏|试|卡|失败|不)|"
    r"继续|接着|后来)"
)


def router_requested_memory(route, tool_results: Iterable[object]) -> bool:
    """True when the router involved a memory tool this turn.

    A request is enough, even if execution was denied or returned empty: the
    router's decision stands and auto-recall must not paper over a failure.
    """
    if route is None:
        return False
    for request in getattr(route, "tool_requests", ()) or ():
        if getattr(request, "tool_name", None) in MEMORY_TOOL_NAMES:
            return True
    for result in tool_results or ():
        if getattr(result, "tool_name", None) in MEMORY_TOOL_NAMES:
            return True
    return False


def should_auto_recall(route, text: str, routing_view) -> bool:
    """Whether this turn's wording is a legitimate recall trigger.

    Only the three v1 trigger families qualify. "我 / 你 / 这个" or a generic
    greeting is not a memory need.
    """
    if route is None:
        return False
    if getattr(route, "action", None) != RouteAction.RESPOND:
        return False
    if (
        getattr(route, "input_semantics", None)
        == InputSemanticStatus.NON_COMMUNICATIVE
    ):
        return False
    if routing_view is not None and getattr(routing_view, "quiet_mode", False):
        return False
    text = (text or "").strip()
    if not text:
        return False

    # (A) explicit reference to a past experience.
    if any(pattern.search(text) for pattern in _PAST_REFERENCE):
        return True

    context_terms = _context_terms(routing_view)
    state = getattr(route, "turn_state", None)

    # (B) the router says this continues a recent exchange, and trusted
    # context supplies the actual project/entity — not just "这次还是不行".
    if state in (TurnState.FOLLOW_UP, TurnState.CONTINUING) and context_terms:
        return True

    # (C) the input itself both marks continuation and names a concrete topic.
    if _CONTINUATION.search(text) and _current_terms(text):
        return True
    return False


def build_recall_query(text: str, routing_view) -> Optional[str]:
    """A short topic query, or ``None`` when no reliable topic exists.

    Priority: current input terms, trusted scene topic/open loops, then this
    speaker's own last one or two turns. The router's turn state never injects
    text of its own; it only decides *whether* the context is relevant.
    """
    terms: list[str] = []
    for term in _current_terms(text):
        _add(terms, term)
    for term in _context_terms(routing_view):
        _add(terms, term)
    if not terms:
        return None
    query = " ".join(terms)
    return query[:MAX_QUERY_CHARS] or None


def select_auto_recalled(
    memories: list[dict],
    query: str,
    *,
    limit: int = MAX_AUTO_RECALL,
    exclude_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Pick at most ``limit`` genuinely relevant memories.

    The store already enforced visibility and ACTIVE status. This layer adds:

    - de-duplication, including against memories the router already provided;
    - a relevance floor: at least one query term must occur in the content
      (or a vector hit must exist). An *assigned* memory that shares no topic
      is not recalled just because it is assigned;
    - episode preference with a reserved slot for a relevant non-episode fact,
      so a long story never hides a concrete personal fact.
    """
    excluded = {str(item) for item in (exclude_ids or ()) if item}
    terms = _query_terms(query)
    if not terms:
        return []

    scored: list[tuple[float, int, dict]] = []
    seen: set[str] = set()
    for order, memory in enumerate(memories or []):
        memory_id = str(memory.get("memory_id") or "")
        if not memory_id or memory_id in seen or memory_id in excluded:
            continue
        content = str(memory.get("content") or "")
        overlap = _overlap(content, terms)
        if overlap == 0 and memory.get("vector_rank") is None:
            continue
        relevance = float(overlap)
        if memory.get("identity"):
            relevance += 2.0
        if memory.get("vector_rank") is not None:
            relevance += 0.5
        seen.add(memory_id)
        scored.append((relevance, order, memory))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [memory for _, _, memory in scored]

    episodes = [
        memory
        for memory in ordered
        if str(memory.get("kind") or "").upper() == "EPISODE"
    ]
    others = [
        memory
        for memory in ordered
        if str(memory.get("kind") or "").upper() != "EPISODE"
    ]

    selected: list[dict] = []
    if others:
        selected.extend(episodes[:MAX_EPISODES_IN_RECALL])
        selected.extend(others)
        if len(selected) < limit:
            selected.extend(
                episodes[MAX_EPISODES_IN_RECALL : limit]
            )
    else:
        selected.extend(episodes[:limit])

    # De-duplicate again (slice boundaries can overlap) and preserve order.
    out: list[dict] = []
    chosen: set[str] = set()
    for memory in selected:
        memory_id = str(memory.get("memory_id") or "")
        if not memory_id or memory_id in chosen:
            continue
        chosen.add(memory_id)
        out.append(memory)
        if len(out) >= limit:
            break
    return out


# -- term helpers ----------------------------------------------------------


def _add(terms: list[str], term: str) -> None:
    term = (term or "").strip()
    if not term or term in terms:
        return
    if len(terms) >= MAX_QUERY_TERMS:
        return
    terms.append(term[:64])


def _usable(term: str) -> bool:
    term = (term or "").strip()
    if len(term) < 2 or term in _DISCOURSE_TERMS:
        return False
    if term.upper() in ("UNKNOWN", "NONE"):
        return False
    # A window containing a pronoun/particle ("我准", "一下") is the store's
    # last-resort search key, not a recall topic.
    if any(ch in _CJK_FUNCTION_CHARS for ch in term):
        return False
    return True


def _current_terms(text: str) -> list[str]:
    return [term for term in extract_keywords(text, max_keywords=8) if _usable(term)]


def _context_terms(routing_view) -> list[str]:
    if routing_view is None:
        return []
    terms: list[str] = []

    scene = getattr(routing_view, "compact_scene", None) or {}
    if isinstance(scene, dict):
        for value in (scene.get("topic"), *(scene.get("topic_stack") or [])):
            if isinstance(value, str):
                for term in extract_keywords(value, max_keywords=4):
                    if _usable(term):
                        _add(terms, term)
        for loop in scene.get("open_loops") or []:
            if isinstance(loop, dict) and isinstance(loop.get("topic"), str):
                for term in extract_keywords(loop["topic"], max_keywords=3):
                    if _usable(term):
                        _add(terms, term)

    # Only this speaker's own real user turns; never another person's words.
    speaker = getattr(routing_view, "current_speaker_id", None)
    human_turns = []
    for turn in getattr(routing_view, "recent_turns", ()) or ():
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").lower()
        if role not in ("human", "user"):
            continue
        if turn.get("speaker_person_id") != speaker:
            continue
        human_turns.append(turn)
    for turn in reversed(human_turns[-2:]):
        for term in extract_keywords(str(turn.get("text") or ""), max_keywords=8):
            if _usable(term):
                _add(terms, term)

    return terms


def _query_terms(query: str) -> list[str]:
    terms = [term for term in extract_keywords(query, max_keywords=8) if _usable(term)]
    # Keep the longest/most specific forms first; bigrams follow anyway.
    return terms


def _overlap(content: str, terms: list[str]) -> int:
    haystack = (content or "").lower()
    if not haystack:
        return 0
    return sum(1 for term in terms if term.lower() in haystack)


__all__ = [
    "MAX_AUTO_RECALL",
    "MEMORY_TOOL_NAMES",
    "build_recall_query",
    "router_requested_memory",
    "select_auto_recalled",
    "should_auto_recall",
]
