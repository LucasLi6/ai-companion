"""Replacement boundaries for the cognitive engine and memory.

The Orchestrator depends only on these protocols, never on a concrete
provider implementation. V0.1 swaps ``MockCognitiveProvider`` for a real
``DeepSeekCognitiveProvider`` without touching the Orchestrator.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .models import CognitiveResult, ConversationContext, UtteranceEvent


class CognitiveProvider(Protocol):
    async def analyze(
        self,
        event: UtteranceEvent,
        context: ConversationContext,
        memories: list,
    ) -> CognitiveResult:
        """Produce a structured analysis (a *suggestion*) for one event."""
        ...


class ResponseProvider(Protocol):
    async def generate(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context: ConversationContext,
    ) -> str:
        """Generate the response text for a finalised RESPOND decision."""
        ...


class MemoryStore(Protocol):
    async def retrieve(self, query: str, person_id: Optional[str]) -> list:
        """Return memory items relevant to a query, optionally per person."""
        ...
