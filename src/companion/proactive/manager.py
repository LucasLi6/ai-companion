"""Dual-stage ProactiveManager (V0.9.9).

The manager owns *timing, staleness and delivery* only. It never decides
whether a topic is interesting: :class:`ProactiveJudge` returns one boolean,
and the existing Speaker decides what Ava actually says.

Lifecycle per triggering turn:

    normal reply committed + delivered
        -> on_response_delivered()            (non-blocking)
        -> STAGE 1 IMMEDIATE: judge once, right away
             (no fixed delay, no observation of Profile Review / Turn Analysis)
        -> (YES) existing Speaker, one message, persisted with mode=IMMEDIATE
        -> wait a random 120-240 s from Ava's last delivered message
        -> STAGE 2 LONG_IDLE: judge once whether to start a *new* exchange
        -> (YES) existing Speaker, one message, persisted with mode=LONG_IDLE
        -> stop (a proactive message never schedules another cycle)

New user input, Quiet Mode, revoked authorization or shutdown invalidate the
whole cycle. Timers live only in this process and are never persisted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from dataclasses import dataclass
from typing import Optional

from companion.core.response_events import ResponseEvent, ResponseEventSink
from companion.domain.models import new_id, utcnow
from companion.proactive.departure import detect_explicit_stop
from companion.proactive.judge import (
    ProactiveJudge,
    ProactiveJudgeContext,
)
from companion.storage.repositories import ProactiveMessageRecord

ORIGIN_AVA_INITIATED = "AVA_INITIATED"
MODE_IMMEDIATE = "IMMEDIATE"
MODE_LONG_IDLE = "LONG_IDLE"

MIN_IDLE_SECONDS = 120
MAX_IDLE_SECONDS = 240


@dataclass
class _ConversationState:
    generation: int = 0
    person_id: Optional[str] = None
    source_event_id: Optional[str] = None
    latest_user_event_id: Optional[str] = None
    last_delivered_at: Optional[float] = None
    task: Optional[asyncio.Task] = None


@dataclass(frozen=True)
class _Cycle:
    conversation_id: str
    person_id: str
    source_event_id: str
    generation: int


class ProactiveManager:
    def __init__(
        self,
        *,
        repos,
        registry,
        conversations,
        orchestrator,
        judge: ProactiveJudge,
        idle_delay_seconds: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.conversations = conversations
        self.orchestrator = orchestrator
        self.judge = judge
        # ``None`` means the real 120-240 s random wait. An explicit value is
        # only for tests/embedding; production never selects a fixed delay.
        self.idle_delay_seconds = idle_delay_seconds
        self.logger = logger or logging.getLogger("companion.proactive")

        self._states: dict[str, _ConversationState] = {}
        self._events: asyncio.Queue[Optional[ResponseEvent]] = asyncio.Queue()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def on_response_delivered(
        self,
        *,
        conversation_id: str,
        person_id: str,
        source_event_id: str,
    ) -> None:
        """Start one dual-stage cycle after a delivered normal reply.

        This method is synchronous and never blocks the chat surface. Callers
        must invoke it only after the reply is fully displayed; WAIT/IGNORE,
        failures, cancellations and truncated replies must not call it.
        """
        if self._closed or not conversation_id or not source_event_id:
            return
        state = self._state_for(conversation_id)
        if state.source_event_id == source_event_id:
            # A duplicate completion event must never schedule a second cycle.
            return

        state.generation += 1
        state.person_id = person_id
        state.source_event_id = source_event_id
        state.latest_user_event_id = source_event_id
        state.last_delivered_at = asyncio.get_running_loop().time()
        self._cancel_task(state.task)
        cycle = _Cycle(
            conversation_id=conversation_id,
            person_id=person_id,
            source_event_id=source_event_id,
            generation=state.generation,
        )
        state.task = asyncio.create_task(self._run_cycle(cycle, state))

    def note_user_input(self, conversation_id: str, source_event_id: str) -> None:
        """Invalidate any pending/running cycle as soon as a user turn starts.

        The generation bump makes every later ``_still_valid`` check fail, and
        cancelling the task stops an in-flight Judge, idle wait or Speaker
        call. The new turn's delivered reply starts a fresh cycle.
        """
        state = self._state_for(conversation_id)
        state.generation += 1
        state.latest_user_event_id = source_event_id
        self._cancel_task(state.task)
        state.task = None
        # Any event still queued belongs to the previous turn and must not be
        # shown after the user has started a new one.
        self._discard_pending_events()

    async def shutdown(self) -> None:
        """Cancel every background cycle; no timer is persisted across restart."""
        if self._closed:
            return
        self._closed = True
        tasks = []
        for state in self._states.values():
            state.generation += 1
            if state.task is not None and not state.task.done():
                state.task.cancel()
                tasks.append(state.task)
            state.task = None
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._events.put_nowait(None)

    # -- delivery surface --------------------------------------------------

    async def next_event(self) -> Optional[ResponseEvent]:
        """Wait for the next proactive response event (None on shutdown)."""
        return await self._events.get()

    def poll_event(self) -> Optional[ResponseEvent]:
        """Non-blocking drain used when the user submits a line concurrently."""
        try:
            item = self._events.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if item is None:
            self._events.put_nowait(None)
        return item

    def _emit(self, event: ResponseEvent) -> None:
        if not self._closed:
            self._events.put_nowait(event)

    def _discard_pending_events(self) -> None:
        while True:
            try:
                item = self._events.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                # Preserve the shutdown sentinel for a waiting consumer.
                self._events.put_nowait(None)
                return

    # -- the cycle ---------------------------------------------------------

    async def _run_cycle(
        self, cycle: _Cycle, state: _ConversationState
    ) -> None:
        try:
            if not await self._still_valid(cycle):
                return
            # Local hard floor: an explicit departure or refusal to continue
            # blocks both stages, independently of the judge's answer.
            if await self._explicit_stop(cycle):
                self.logger.info(
                    "proactive_blocked_explicit_stop conversation_id=%s"
                    " source_event_id=%s",
                    cycle.conversation_id,
                    cycle.source_event_id,
                )
                return
            # Stage 1: right after the delivered normal reply. If Ava has just
            # asked an identity-confirmation question, the immediate judge must
            # not interject or urge; the long-idle stage may still start a
            # different topic later.
            if not await self._confirmation_pending(cycle):
                await self._attempt(cycle, state, MODE_IMMEDIATE)

            if not self._is_current(state, cycle.generation):
                return
            await self._wait_for_idle(state)

            if not self._is_current(state, cycle.generation):
                return
            if not await self._still_valid(cycle):
                return
            # Stage 2: exactly one long-idle judgment per idle cycle.
            await self._attempt(cycle, state, MODE_LONG_IDLE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "proactive_cycle_failed conversation_id=%s source_event_id=%s"
                " error=%s",
                cycle.conversation_id,
                cycle.source_event_id,
                type(exc).__name__,
            )
        finally:
            if state.task is asyncio.current_task():
                state.task = None

    async def _wait_for_idle(self, state: _ConversationState) -> None:
        """Wait 120-240 s from Ava's last actually delivered message."""
        loop = asyncio.get_running_loop()
        base = state.last_delivered_at or loop.time()
        deadline = base + self._next_idle_delay()
        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    def _next_idle_delay(self) -> float:
        if self.idle_delay_seconds is not None:
            return max(0.0, float(self.idle_delay_seconds))
        return float(
            secrets.randbelow(MAX_IDLE_SECONDS - MIN_IDLE_SECONDS + 1)
            + MIN_IDLE_SECONDS
        )

    async def _attempt(
        self,
        cycle: _Cycle,
        state: _ConversationState,
        mode: str,
    ) -> None:
        """One judge + (possibly) one delivered proactive message."""
        if not await self._still_valid(cycle):
            return

        # One utterance per mode per triggering turn, including across a
        # restart that lost the in-memory state.
        if await self.repos.proactive.get_for_source_event(
            cycle.source_event_id, mode
        ):
            return

        source_event = await self.repos.utterance.get(cycle.source_event_id)
        if source_event is None:
            return
        if not await self._still_valid(cycle):
            return

        data = await self.orchestrator.build_proactive_judge_context(
            source_event=source_event, person_id=cycle.person_id, mode=mode
        )
        if data is None:
            return
        decision = await self.judge.decide(ProactiveJudgeContext(**data))
        if decision is None:
            return
        self.logger.info(
            "proactive_judge mode=%s decision=%s reason=%s",
            mode,
            decision.should_initiate,
            decision.reason,
        )
        if not decision.should_initiate:
            return
        if not self._is_current(state, cycle.generation):
            return
        if not await self._still_valid(cycle):
            return

        generated = await self.orchestrator.generate_proactive_reply(
            source_event=source_event, person_id=cycle.person_id, mode=mode
        )
        text = str(generated.get("text") or "").strip()
        if not text:
            return
        finish_reason = generated.get("finish_reason")
        if finish_reason == "length":
            # An incomplete proactive utterance is not delivered.
            self.logger.info(
                "proactive_discarded_truncated conversation_id=%s"
                " source_event_id=%s mode=%s",
                cycle.conversation_id,
                cycle.source_event_id,
                mode,
            )
            return
        if not self._is_current(state, cycle.generation):
            return
        if not await self._still_valid(cycle):
            return

        message = ProactiveMessageRecord(
            message_id=new_id("pmsg"),
            conversation_id=cycle.conversation_id,
            person_id=cycle.person_id,
            source_event_id=cycle.source_event_id,
            mode=mode,
            origin=ORIGIN_AVA_INITIATED,
            text=text,
            status="COMPLETED",
            finish_reason=finish_reason,
            model=str(generated.get("model") or ""),
            created_at=utcnow(),
        )
        await self.repos.proactive.save(message)

        # A new user message may have arrived while the row was being written.
        # Stale output must not be shown; remove the un-delivered record so it
        # also cannot become forged later history.
        if not self._is_current(state, cycle.generation) or not await self._still_valid(
            cycle
        ):
            await self.repos.proactive.delete(message.message_id)
            return

        sink = ResponseEventSink(
            cycle.source_event_id, cycle.conversation_id, self._emit
        )
        sink.completed(
            message.message_id,
            text=text,
            finish_reason=finish_reason,
            report_status="COMPLETED",
            action="RESPOND",
        )
        state.last_delivered_at = asyncio.get_running_loop().time()
        self.logger.info(
            "proactive_delivered conversation_id=%s source_event_id=%s"
            " mode=%s message_id=%s chars=%d",
            cycle.conversation_id,
            cycle.source_event_id,
            mode,
            message.message_id,
            len(text),
        )

    async def _explicit_stop(self, cycle: _Cycle) -> bool:
        """Local departure/refusal floor read from the triggering turn."""
        event = await self.repos.utterance.get(cycle.source_event_id)
        return bool(
            event is not None
            and detect_explicit_stop(getattr(event, "text", ""))
        )

    async def _confirmation_pending(self, cycle: _Cycle) -> bool:
        """True while Ava's identity-confirmation question is still open."""
        repo = getattr(self.repos, "profile_confirmation", None)
        if repo is None:
            return False
        try:
            record = await repo.get_active(
                cycle.person_id, conversation_id=cycle.conversation_id
            )
        except Exception as exc:
            self.logger.warning(
                "proactive_confirmation_state_failed conversation_id=%s"
                " error=%s",
                cycle.conversation_id,
                type(exc).__name__,
            )
            return False
        return record is not None

    # -- validity ----------------------------------------------------------

    def _is_current(self, state: _ConversationState, generation: int) -> bool:
        return not self._closed and state.generation == generation

    async def _still_valid(self, cycle: _Cycle) -> bool:
        state = self._states.get(cycle.conversation_id)
        if state is None or state.generation != cycle.generation:
            return False
        if cycle.source_event_id != state.latest_user_event_id:
            return False
        conversation = await self.repos.conversation.get(cycle.conversation_id)
        if conversation is None or conversation.quiet_mode:
            return False
        if self._orchestrator_is_processing(cycle.conversation_id):
            return False
        if not await self.registry.is_authorized(cycle.person_id):
            return False
        if not await self.registry.is_external_ai_authorized(cycle.person_id):
            return False
        consent = await self.repos.person.get_consent(cycle.person_id)
        if consent is None or not consent.allow_persistence:
            return False
        latest = await self.repos.utterance.recent_context(
            cycle.conversation_id, exclude_event_id=None, limit=1
        )
        if not latest or latest[-1].event_id != cycle.source_event_id:
            return False
        return True

    def _orchestrator_is_processing(self, conversation_id: str) -> bool:
        checker = getattr(self.orchestrator, "is_processing", None)
        return bool(checker(conversation_id)) if callable(checker) else False

    # -- helpers -----------------------------------------------------------

    def _state_for(self, conversation_id: str) -> _ConversationState:
        state = self._states.get(conversation_id)
        if state is None:
            state = _ConversationState()
            self._states[conversation_id] = state
        return state

    @staticmethod
    def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is not None and not task.done():
            task.cancel()


__all__ = [
    "ProactiveManager",
    "ORIGIN_AVA_INITIATED",
    "MODE_IMMEDIATE",
    "MODE_LONG_IDLE",
]
