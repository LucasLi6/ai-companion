"""DeepSeek-shaped reasoning provider (mock-only for this milestone).

Per the V0.1 constraints, no real DeepSeek client or API key is introduced here:
the idle-learning engines keep driving against a deterministic mock. The class
still exposes the *structured* interface the engines need — ``reflect``,
``ask_questions`` and ``synthesize`` — so a real HTTP-backed implementation can
later drop in behind the same surface without touching the engines.

The mock is deliberately honest: every output is clearly labelled ``[mock]``
and is derived from the input metadata (counts, kinds, short content previews),
never from a pretence of semantic understanding.
"""

from __future__ import annotations

from typing import Any


def _preview(memory: dict[str, Any], n: int = 24) -> str:
    content = str(memory.get("content", ""))
    return content[:n] + ("…" if len(content) > n else "")


class DeepSeekProvider:
    """Deterministic mock for reflection, curiosity and research synthesis."""

    name = "mock-deepseek"

    async def reflect(self, memories: list[dict]) -> list[dict]:
        """Propose at most one reflection when there is something to reflect on."""
        if not memories:
            return []
        # Pair two distinct recent memories as a clearly-labelled reflection.
        sample = memories[:2]
        previews = "；".join(_preview(m) for m in sample)
        return [
            {
                "content": f"[mock] 最近记忆（{previews}）之间可能存在联系，值得进一步核对。",
                "kind": "REFLECTION",
                "conflicts_with": None,
            }
        ]

    async def ask_questions(self, memories: list[dict]) -> list[str]:
        """Derive one question per factual/state memory, capped at three."""
        proposals = await self.ask_question_proposals(memories)
        return [item["question"] for item in proposals]

    async def ask_question_proposals(self, memories: list[dict]) -> list[dict]:
        """Questions with the exact memory ids each one was derived from."""
        proposals: list[dict] = []
        for m in memories:
            kind = m.get("kind")
            if kind in ("PROJECT_STATE", "WORLD_KNOWLEDGE", "PERSONAL_FACT"):
                memory_id = m.get("memory_id")
                proposals.append(
                    {
                        "question": f"[mock] 关于「{_preview(m)}」是否仍然准确、有无更新？",
                        "source_memory_ids": [str(memory_id)] if memory_id else [],
                    }
                )
            if len(proposals) >= 3:
                break
        return proposals

    async def mode(self) -> str:
        return "mock"

    async def describe(self) -> tuple[str, str]:
        return "mock", self.name

    async def synthesize(self, question: str, sources: list[dict]) -> str:
        """Combine search results into a short, clearly-labelled summary."""
        titles = "、".join(s.get("title", "") for s in sources[:3])
        if not titles:
            titles = "（无来源）"
        return f"[mock] 综合 {len(sources)} 个来源（{titles}），对「{question}」的初步回答尚待人工确认。"
