"""Local scene assembly for V0.4.

The model proposes a :class:`~companion.domain.models.SceneProposal`; this
module decides what the trusted scene becomes. Nothing here calls a model or
writes to the database (persistence is :class:`SceneRepository`'s job).

Rules enforced locally:

- Topic changes, environment details and open loops need a verbatim evidence
  span from the current input or this person's authorized recent turns.
- Physical environment is never inferred. Without an explicit statement it is
  UNKNOWN, and an old observation expires from the prompt after a few turns.
- An open loop is only created when the derived nonresponse trace says there
  is a pending follow-up. A mere WAIT is not an engagement promise.
- Topic stack and open-loop list are bounded, and resolved loops are removed
  instead of accumulating.
- On load, every source utterance is re-checked against the person's current
  consent. A revoked source invalidates the derived scene rather than leaking
  an old summary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from companion.cognition.scene_models import (
    ENVIRONMENT_TTL_TURNS,
    MAX_EVIDENCE_ITEMS,
    MAX_OPEN_LOOPS,
    MAX_SOURCE_EVENTS,
    MAX_TOPIC_STACK,
    OPEN_LOOP_TTL_TURNS,
    UNKNOWN,
    OpenLoop,
    SceneEvidence,
    SceneState,
    default_environment,
)
from companion.cognition.scene_repository import SceneRepository, SceneSnapshotRecord
from companion.domain.models import (
    SceneProposal,
    new_id,
    utcnow,
    validate_scene_proposal,
)


class SceneManager:
    def __init__(
        self,
        *,
        repos,
        registry=None,
        require_external_ai: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.require_external_ai = require_external_ai
        self.scene_repo = SceneRepository(repos.db)
        self.logger = logger or logging.getLogger("companion.scene")

    # -- load --------------------------------------------------------------

    async def load(
        self, conversation_id: str, person_id: str
    ) -> Optional[SceneSnapshotRecord]:
        """Load a snapshot only if every source is still authorized.

        The scene is derived from persistent utterances, so a withdrawn
        processing/persistence grant (or, for an external-AI turn, a withdrawn
        external-AI grant) invalidates it. The derived state is deleted; the
        caller receives ``None`` and rebuilds from an unknown scene.
        """
        record = await self.scene_repo.load(conversation_id, person_id)
        if record is None:
            return None

        reason = await self._invalidation_reason(record, person_id)
        if reason is not None:
            self.logger.info(
                "scene_invalidated conversation_id=%s person_id=%s reason=%s",
                conversation_id,
                person_id,
                reason,
            )
            await self.scene_repo.delete_scope(conversation_id, person_id)
            return None
        return record

    async def _invalidation_reason(
        self, record: SceneSnapshotRecord, person_id: str
    ) -> Optional[str]:
        if self.registry is None:
            return None
        consent = await self.registry.get_consent(person_id)
        if consent is None or not consent.allow_processing:
            return "processing_withdrawn"
        if not consent.allow_persistence:
            return "persistence_withdrawn"
        if self.require_external_ai and not consent.allow_external_ai:
            return "external_ai_withdrawn"

        ids = record.source_event_ids or []
        events = await self.repos.utterance.get_many(ids)
        if len(events) != len(set(ids)):
            return "source_missing"
        for event_id in ids:
            event = events.get(event_id)
            if event is None:
                return "source_missing"
            if (
                event.conversation_id != record.conversation_id
                or event.speaker_person_id != person_id
            ):
                return "source_scope_mismatch"
        return None

    # -- transition --------------------------------------------------------

    def preview(
        self,
        previous: Optional[SceneSnapshotRecord],
        proposal,
        event,
        *,
        context=None,
        action_history: Optional[Iterable] = None,
        pending_followup: bool = False,
    ) -> SceneState:
        """Build the candidate next scene without persisting it.

        The proposal is re-validated against this person's authorized text even
        though the provider already validated it once. That second check is
        deliberate: a compromised or stale provider payload must not be able to
        set a topic or claim an environment detail without evidence.
        """
        prior = previous.state if previous is not None else None
        authorized = self._authorized_texts(event, context, previous, action_history)
        clean = validate_scene_proposal(
            proposal,
            current_text=event.text,
            authorized_texts=authorized,
        )

        state = (
            prior.model_copy(deep=True)
            if prior is not None
            else SceneState(
                conversation_id=event.conversation_id,
                person_id=event.speaker_person_id,
                revision=0,
                environment=default_environment(),
            )
        )
        state.revision = (previous.revision + 1) if previous is not None else 0
        state.last_committed_event_id = event.event_id
        state.updated_at = utcnow()
        state.source_event_ids = self._append_source(
            state.source_event_ids, event.event_id
        )

        # Old, explicitly observed environment details expire. This system has
        # no sensors, so "one person was nearby six turns ago" is not a fact
        # about the current room.
        env = dict(state.environment or default_environment())
        env["age_turns"] = int(env.get("age_turns", 0) or 0) + 1
        if env["age_turns"] > ENVIRONMENT_TTL_TURNS:
            env = default_environment()

        if clean is None:
            state.environment = env
            self._age_open_loops(state)
            return state

        self._resolve_loops(state, clean.resolve_loop_topics)
        self._apply_topic(state, clean)
        if clean.mode != "UNKNOWN":
            state.mode = clean.mode
        if clean.discourse_phase != "UNCERTAIN":
            state.discourse_phase = clean.discourse_phase
        if clean.tone != "UNCERTAIN":
            state.tone = clean.tone

        if clean.environment is not None:
            env = {
                "location": clean.environment.location or UNKNOWN,
                "other_people_present": self._presence(
                    clean.environment.other_people_present
                ),
                "status": "EXPLICIT",
                "evidence_event_id": event.event_id,
                "age_turns": 0,
            }
        state.environment = env

        if clean.evidence_span:
            state.evidence = [
                *state.evidence,
                SceneEvidence(
                    event_id=event.event_id,
                    quote=clean.evidence_span,
                    kind="SCENE",
                ),
            ][-MAX_EVIDENCE_ITEMS:]

        if clean.open_loop is not None and pending_followup:
            self._add_loop(state, clean, event)

        self._age_open_loops(state)
        return state

    def finalize(
        self,
        candidate: SceneState,
        *,
        event,
        action,
        response_text: Optional[str] = None,
        trace=None,
    ) -> SceneState:
        """Apply the final-action consequences to a candidate scene.

        There is intentionally no "WAIT means remember forever" rule. A loop
        survives only while the local trace reports a pending follow-up or it
        was proposed with evidence this turn. Resolved loops are removed.
        """
        state = candidate.model_copy(deep=True)
        state.last_committed_event_id = event.event_id
        if state.topic != UNKNOWN:
            for loop in state.open_loops:
                if loop.topic.strip().casefold() == state.topic.strip().casefold():
                    loop.last_event_id = event.event_id
                    loop.turns_since_touch = 0
        state.open_loops = state.open_loops[:MAX_OPEN_LOOPS]
        if (
            action is not None
            and getattr(action, "value", str(action)) in ("WAIT", "IGNORE")
            and (trace is None or not getattr(trace, "pending_followup", False))
        ):
            # A non-pending silence may mean a thread is no longer active; it
            # never creates one. Existing, evidenced loops are left to expire
            # naturally rather than being silently reinterpreted.
            pass
        return state

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _append_source(ids: list[str], event_id: str) -> list[str]:
        out = [str(v) for v in ids if v]
        if event_id in out:
            out = [v for v in out if v != event_id]
        out.append(event_id)
        return out[-MAX_SOURCE_EVENTS:]

    @staticmethod
    def _presence(value: Optional[str]) -> str:
        if value == "YES":
            return "YES"
        if value == "NO":
            return "NO"
        return UNKNOWN

    def _authorized_texts(
        self,
        event,
        context,
        previous: Optional[SceneSnapshotRecord],
        action_history,
    ) -> list[str]:
        person_id = event.speaker_person_id
        texts: list[str] = []
        for turn in getattr(context, "turns", []) or []:
            if getattr(turn, "role", None) != "human":
                continue
            speaker = getattr(turn, "speaker_person_id", None)
            # An unregistered speaker's words are not evidence for this
            # person's scene, even if they appear in the same conversation.
            if speaker is None or speaker != person_id:
                continue
            if getattr(turn, "text", None):
                texts.append(turn.text)
        return texts

    def _apply_topic(self, state: SceneState, proposal: SceneProposal) -> None:
        topic = (proposal.topic or "").strip()
        change = proposal.topic_change
        old = (state.topic or UNKNOWN).strip()

        if change == "SHIFT" and topic:
            if topic.casefold() != old.casefold():
                self._push_topic(state, old)
            state.topic = topic
            return
        if change == "RETURN" and topic:
            lowered = topic.casefold()
            stack = [t for t in state.topic_stack if t != UNKNOWN]
            match_at = None
            for index in range(len(stack) - 1, -1, -1):
                if stack[index].casefold() == lowered:
                    match_at = index
                    break
            if match_at is not None:
                state.topic_stack = stack[:match_at]
                state.topic = stack[match_at]
            else:
                # Returning to a topic that was never on the stack is a shift,
                # not a licence to invent history.
                self._push_topic(state, old)
                state.topic = topic
            return
        if topic and (old == UNKNOWN or change == "CONTINUE"):
            state.topic = topic

    @staticmethod
    def _push_topic(state: SceneState, topic: str) -> None:
        if not topic or topic == UNKNOWN:
            return
        stack = [t for t in state.topic_stack if t and t != UNKNOWN]
        if stack and stack[-1] == topic:
            state.topic_stack = stack[-MAX_TOPIC_STACK:]
            return
        if topic in stack:
            stack = [t for t in stack if t != topic]
        stack.append(topic)
        state.topic_stack = stack[-MAX_TOPIC_STACK:]

    def _resolve_loops(self, state: SceneState, topics: list[str]) -> None:
        if not topics:
            return
        wanted = {t.strip().casefold() for t in topics if t.strip()}
        state.open_loops = [
            loop for loop in state.open_loops
            if loop.topic.strip().casefold() not in wanted
        ]

    def _add_loop(self, state: SceneState, proposal: SceneProposal, event) -> None:
        loop_spec = proposal.open_loop
        if loop_spec is None:
            return
        summary = loop_spec.summary.strip()
        topic = (loop_spec.topic or state.topic or UNKNOWN).strip()
        if not summary:
            return
        for loop in state.open_loops:
            if (
                loop.summary.strip().casefold() == summary.casefold()
                and loop.topic.strip().casefold() == topic.casefold()
            ):
                loop.last_event_id = event.event_id
                loop.turns_since_touch = 0
                return
        state.open_loops.append(
            OpenLoop(
                loop_id=new_id("loop"),
                topic=topic[:120],
                summary=summary[:300],
                opened_event_id=event.event_id,
                last_event_id=event.event_id,
                evidence_span=loop_spec.evidence_span,
                opened_at=_iso_now(),
                turns_since_touch=0,
            )
        )
        state.open_loops = state.open_loops[-MAX_OPEN_LOOPS:]

    @staticmethod
    def _age_open_loops(state: SceneState) -> None:
        kept: list[OpenLoop] = []
        for loop in state.open_loops:
            if loop.last_event_id != state.last_committed_event_id:
                loop.turns_since_touch += 1
            if loop.turns_since_touch <= OPEN_LOOP_TTL_TURNS:
                kept.append(loop)
        state.open_loops = kept[-MAX_OPEN_LOOPS:]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
