"""Turn executed read-only tool results into the existing speaker context.

Semantic mode never retrieves long-term memory on its own. Memory reaches the
speaker only through ``memory.search`` / ``memory.browse`` results that the
router requested and the executor authorised. This adapter preserves the real
record fields instead of copying a clipped prose string, so the speaker prompt
still has ``memory_id``, ``kind``, verified ``visibility`` and provenance.

The adapter never re-reads a memory by an id supplied by the model: every
record here came out of the access-checked store.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from companion.cognition.models import CognitiveContext
from companion.routing.models import ToolStatus

_MEMORY_TOOLS = frozenset({"memory.search", "memory.browse"})
_HISTORY_RECENT = "history.recent"
_HISTORY_EARLIEST = "history.earliest"
_HISTORY_STATS = "history.stats"


def tool_results_to_memory_context(
    results: Iterable[Any],
) -> list[dict]:
    """Real, access-checked memory records from successful memory tools."""
    memories: list[dict] = []
    seen: set[str] = set()
    for result in results:
        if getattr(result, "tool_name", None) not in _MEMORY_TOOLS:
            continue
        if getattr(result, "status", None) != ToolStatus.SUCCESS:
            continue
        data = getattr(result, "data", None)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, Mapping):
                continue
            memory_id = item.get("memory_id")
            content = str(item.get("content") or "").strip()
            if not memory_id or not content or str(memory_id) in seen:
                continue
            seen.add(str(memory_id))
            memories.append(
                {
                    "memory_id": str(memory_id),
                    "content": content,
                    "kind": item.get("kind"),
                    "visibility": item.get("visibility"),
                    "status": item.get("status"),
                    "subjects": item.get("subjects") or [],
                    "sources": item.get("sources") or [],
                    "tool_call_id": getattr(result, "tool_call_id", None),
                    "tool_name": getattr(result, "tool_name", None),
                    "source_refs": list(getattr(result, "source_refs", ()) or []),
                }
            )
    return memories


def attach_semantic_tool_signals(
    context: CognitiveContext,
    routing_view,
    results: Iterable[Any],
) -> CognitiveContext:
    """Fill retrieval/history fields strictly from executed tool results.

    If the router did not request a fact, the field stays ``none``/``None``.
    A denied or failed tool is never reported as an empty successful lookup.
    """
    results = list(results or [])
    memory_results = [
        item for item in results
        if getattr(item, "tool_name", None) in _MEMORY_TOOLS
    ]
    status = "none"
    if memory_results:
        status = _aggregate_status(
            [getattr(item, "status", None) for item in memory_results]
        )

    memories = tool_results_to_memory_context(memory_results)
    samples = [
        {
            "memory_id": item.get("memory_id"),
            "kind": item.get("kind"),
            "visibility": item.get("visibility"),
            "content": item.get("content"),
        }
        for item in memories[:3]
    ]

    history_excerpt: list[dict] = []
    conversation_stats = None
    for result in results:
        if getattr(result, "status", None) != ToolStatus.SUCCESS:
            continue
        if getattr(result, "tool_name", None) == _HISTORY_EARLIEST:
            data = getattr(result, "data", None)
            if isinstance(data, list):
                history_excerpt = [
                    dict(item) for item in data if isinstance(item, Mapping)
                ][:3]
        elif getattr(result, "tool_name", None) == _HISTORY_STATS:
            data = getattr(result, "data", None)
            if isinstance(data, Mapping):
                conversation_stats = dict(data)

    history_loaded = len(getattr(routing_view, "recent_turns", ()) or ())
    return context.model_copy(
        update={
            "memory_retrieval_status": status,
            "memory_samples": samples,
            "history_excerpt": history_excerpt,
            "history_loaded": history_loaded,
            "conversation_stats": conversation_stats,
        }
    )


def _aggregate_status(statuses: Iterable[Any]) -> str:
    values = {getattr(item, "value", str(item or "")).upper() for item in statuses}
    if "SUCCESS" in values:
        return "hits"
    if "EMPTY" in values:
        return "empty"
    if "DENIED" in values:
        return "denied"
    if "TIMEOUT" in values:
        return "timeout"
    if "ERROR" in values:
        return "error"
    if "DISABLED" in values:
        return "disabled"
    return "none"
