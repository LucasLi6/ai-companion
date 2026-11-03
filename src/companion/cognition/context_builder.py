"""Conversation context builder (Phase 3).

Assembles the ordered multi-turn history a real DeepSeek provider needs to
understand the *current* utterance in light of the *recent* conversation:

- Fetches recent human utterances (the current event is always excluded).
- Fetches the agent's responses for those events in **one batch query** (never
  one query per event), and interleaves them with the human turns in
  chronological order.
- Labels every human turn with the speaker's display name (multi-person
  labelling), resolving names through the registry so the model sees *who*
  said what, not bare person ids.
- Applies a bounded truncation (max turns, then max characters) that keeps the
  most recent turns and drops the oldest first, so the prompt never grows
  without limit.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.core.policy import load_policy_config
from companion.domain.enums import CognitiveAction
from companion.domain.models import UtteranceEvent
from companion.people.registry import PersonRegistry
from companion.storage.repositories import Repositories

from .models import (
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    CognitiveContext,
    CognitiveTurn,
)
from .nonresponse import build_action_history_record

DEFAULT_HISTORY_LIMIT = 20
DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_CHARS = 8000


class ContextBuilder:
    def __init__(
        self,
        *,
        repos: Repositories,
        registry: PersonRegistry,
        privacy_gateway=None,
        agent_service=None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        max_turns: Optional[int] = DEFAULT_MAX_TURNS,
        max_chars: Optional[int] = DEFAULT_MAX_CHARS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.privacy_gateway = privacy_gateway
        self.agent_service = agent_service
        self.history_limit = history_limit
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.logger = logger or logging.getLogger("companion.context")
        self._people: dict[str, object] = {}

    async def build(
        self, event: UtteranceEvent, memories: list[dict]
    ) -> CognitiveContext:
        self._people = {}

        history = await self.repos.utterance.recent_context(
            event.conversation_id,
            exclude_event_id=event.event_id,
            limit=self.history_limit,
        )
        if self.privacy_gateway is not None:
            # External-AI gate: only history from authorized speakers and only
            # memories whose PERSON subjects are all authorized may reach the
            # model. The *current* speaker is checked separately by the provider
            # before any network call.
            history = await self.privacy_gateway.filter_utterances(history)
            memories, _dropped = await self.privacy_gateway.filter_memories(
                memories, self._subject_ids_of
            )
        actions = await self.repos.analysis.get_actions_for_events(
            [u.event_id for u in history]
        )

        labels = await self._resolve_labels(history, event)
        turns = self._interleave(history, actions, labels)
        turns = self._truncate(turns)
        action_history = self._action_history(history, actions, turns)

        current_label = labels.get(
            event.speaker_person_id, event.speaker_label or "unknown"
        )
        agent_context = None
        if self.agent_service is not None:
            policy_config = await load_policy_config(self.repos.settings)
            effective = await self.agent_service.effective(
                event.speaker_person_id, policy=policy_config
            )
            agent_context = effective.as_dict()
        return CognitiveContext(
            conversation_id=event.conversation_id,
            current_event_id=event.event_id,
            current_speaker_id=event.speaker_person_id,
            current_speaker_label=current_label,
            current_text=event.text,
            current_addressee_hint=event.addressee_hint.value,
            quiet_mode=False,
            turns=turns,
            action_history=action_history,
            memories=memories,
            agent_context=agent_context,
        )

    # -- internals ----------------------------------------------------------

    async def _resolve_labels(
        self, history: list[UtteranceEvent], event: UtteranceEvent
    ) -> dict:
        """Resolve every speaker's display name in one pass (cached lookups)."""
        labels: dict = {}
        for u in [*history, event]:
            pid = u.speaker_person_id
            if pid is None:
                labels[None] = u.speaker_label or "unknown"
                continue
            if pid in labels:
                continue
            person = await self._person(pid)
            labels[pid] = person.display_name if person is not None else pid
        return labels

    def _interleave(
        self, utterances, actions_by_event: dict, labels: dict
    ) -> list[CognitiveTurn]:
        """Merge human utterances and agent responses into one ordered stream.

        ``utterances`` are expected oldest-first (as ``recent_context``
        returns). An assistant turn is emitted only when the agent actually
        produced a non-empty response for that event.
        """
        turns: list[CognitiveTurn] = []
        for u in utterances:
            label = labels.get(u.speaker_person_id, u.speaker_label or "unknown")
            turns.append(
                CognitiveTurn(
                    event_id=u.event_id,
                    role=ROLE_HUMAN,
                    speaker_person_id=u.speaker_person_id,
                    speaker_label=label,
                    addressee_hint=u.addressee_hint.value,
                    text=u.text,
                )
            )
            action = actions_by_event.get(u.event_id)
            # Only a real, visible RESPOND is assistant dialogue. A WAIT/IGNORE
            # must never be rendered as if the companion said "I stayed quiet".
            if (
                action is not None
                and action.action_type == CognitiveAction.RESPOND
                and action.response_text
            ):
                turns.append(
                    CognitiveTurn(
                        event_id=u.event_id,
                        role=ROLE_ASSISTANT,
                        speaker_person_id=None,
                        speaker_label="assistant",
                        addressee_hint=None,
                        text=action.response_text,
                    )
                )
        return turns

    def _action_history(
        self, utterances, actions_by_event: dict, turns: list[CognitiveTurn]
    ):
        """Build recent WAIT/IGNORE records from the same filtered history.

        The source utterance must still be present after privacy filtering and
        turn truncation, so an action can never arrive in the model context
        without the human turn it belongs to. At most eight records are kept,
        bounding the prompt even across a long run of silence.
        """
        kept_event_ids = {
            turn.event_id for turn in turns if turn.role == ROLE_HUMAN
        }
        records = []
        for utterance in utterances:
            if utterance.event_id not in kept_event_ids:
                continue
            action = actions_by_event.get(utterance.event_id)
            if action is None:
                continue
            record = build_action_history_record(action)
            if record is not None:
                records.append(record)
        return records[-8:]

    def _truncate(self, turns: list[CognitiveTurn]) -> list[CognitiveTurn]:
        turns = list(turns)
        if self.max_turns is not None and len(turns) > self.max_turns:
            turns = turns[-self.max_turns:]
        if self.max_chars is not None:
            total = sum(len(t.text) for t in turns)
            while total > self.max_chars and len(turns) > 1:
                total -= len(turns[0].text)
                turns.pop(0)
        return turns

    async def _person(self, person_id: str):
        if person_id not in self._people:
            self._people[person_id] = await self.registry.get(person_id)
        return self._people[person_id]

    async def _subject_ids_of(self, memory: dict) -> list[str]:
        """PERSON subject ids of a retrieved memory, for the privacy gateway."""
        return [
            s["subject_id"]
            for s in memory.get("subjects", [])
            if s.get("subject_type") == "PERSON"
        ]
