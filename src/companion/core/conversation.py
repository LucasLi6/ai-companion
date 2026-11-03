"""Conversation state management.

Quiet mode and ``last_action`` are persisted on the ``conversations`` table and
survive restarts. Recent utterance ids are *derived* from the database on load
and held only briefly in memory — the full chat history is never kept in RAM.
"""

from __future__ import annotations

from companion.domain.models import ConversationContext
from companion.storage.repositories import ConversationRepository, UtteranceRepository


class ConversationManager:
    def __init__(
        self,
        repo: ConversationRepository,
        utterance_repo: UtteranceRepository,
        recent_limit: int = 20,
    ) -> None:
        self.repo = repo
        self.utterance_repo = utterance_repo
        self.recent_limit = recent_limit

    async def get_or_create(self, conversation_id: str) -> ConversationContext:
        context = await self.repo.get_or_create(conversation_id)
        context.recent_utterance_ids = await self.utterance_repo.recent(
            conversation_id, self.recent_limit
        )
        return context

    async def set_quiet_mode(self, conversation_id: str, enabled: bool) -> bool:
        await self.repo.set_quiet_mode(conversation_id, enabled)
        return enabled

    async def record_last_action(self, conversation_id: str, action: str | None) -> None:
        await self.repo.update_last_action(conversation_id, action)
