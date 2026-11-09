"""Continuous ProactiveManager (V0.9.9.21).

The manager owns *timing, staleness and delivery* only. It never decides
whether a topic is interesting: :class:`ProactiveJudge` returns one boolean,
and the existing Speaker decides what Ava actually says.

Lifecycle per triggering turn:

    normal reply committed + delivered
        -> on_response_delivered()            (non-blocking)
        -> IMMEDIATE: judge once, right away
             (no fixed delay, no observation of Profile Review / Turn Analysis)
        -> then a continuous, adaptive loop while the user stays silent:
             LONG_IDLE round 1 after 20-40 s        (high activity)
             LONG_IDLE round 2 after 45-90 s
             LONG_IDLE round 3 after 2-5 min
             later rounds after 5-10 min
        -> the loop never stops by itself; only a new user turn, Quiet Mode,
           revoked authorization, an explicit stop or shutdown ends it.

Each round is a *judgment attempt*, not a forced send. The wait is measured
from Ava's last **actually delivered** message (delivery is acknowledged by
the surface that showed/spoke it), never from model generation completion.

Persistence rounds use ``LONG_IDLE_1``, ``LONG_IDLE_2``, ... while the
semantic generation mode stays ``LONG_IDLE``. ``proactive_messages`` has
``UNIQUE(source_event_id, mode)``; the round suffix keeps every round distinct
without weakening that protection. Old rows persisted as plain ``LONG_IDLE``
remain readable; round 1 checks them for compatibility and is never re-sent.

New user input, Quiet Mode, revoked authorization or shutdown invalidate the
whole cycle. Timers live only in this process and are never persisted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from companion.core.orchestrator import EmptySpeechOutputError
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

ACTIVITY_HIGH = "high"
ACTIVITY_NORMAL = "normal"
ACTIVITY_LOW = "low"
DEFAULT_ACTIVITY = ACTIVITY_HIGH
ACTIVITY_ENV = "AVA_PROACTIVE_ACTIVITY"

# (min, max) seconds per unanswered proactive round. The last entry repeats
# forever: activity slows down but never stops.
ACTIVITY_INTERVALS: dict[str, tuple[tuple[int, int], ...]] = {
    ACTIVITY_HIGH: (
        (20, 40),
        (45, 90),
        (120, 300),
        (180, 300),
        (300, 600),
    ),
    ACTIVITY_NORMAL: (
        (120, 240),
        (180, 300),
        (300, 600),
        (300, 600),
    ),
    ACTIVITY_LOW: (
        (300, 600),
        (600, 900),
        (600, 900),
    ),
}

# Kept for embedding compatibility; the activity profiles above are the real
# source of truth in production.
MIN_IDLE_SECONDS = 120
MAX_IDLE_SECONDS = 240


@dataclass
class _ConversationState:
    conversation_id: str = ""
    generation: int = 0
    person_id: Optional[str] = None
    source_event_id: Optional[str] = None
    latest_user_event_id: Optional[str] = None
    last_delivered_at: Optional[float] = None
    last_user_input_at: Optional[float] = None
    last_attempt_at: Optional[float] = None
    next_attempt_at: Optional[float] = None
    unanswered: int = 0
    last_ack_message_id: Optional[str] = None
    # message_id -> persistence mode, so an acknowledgement can tell an
    # immediate remark apart from an unanswered long-idle round.
    pending_modes: dict[str, str] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None
    # Set when the surface confirms a proactive message really reached the
    # user; waiting cycles recompute their next attempt from that moment.
    delivery_event: asyncio.Event = field(default_factory=asyncio.Event)


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
        delivery_available: Optional[Callable[[str], bool]] = None,
        activity: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.conversations = conversations
        self.orchestrator = orchestrator
        self.judge = judge
        # ``None`` means the adaptive activity profile. An explicit value is
        # only for tests/embedding; production never selects a fixed delay.
        self.idle_delay_seconds = idle_delay_seconds
        # Optional surface hook (used by the web chat coordinator): when it
        # returns False, no message may be generated or delivered because no
        # client could receive it. The CLI leaves it unset, so its behaviour
        # is unchanged.
        self.delivery_available = delivery_available
        self.logger = logger or logging.getLogger("companion.proactive")
        self.activity = _normalise_activity(
            activity
            if activity is not None
            else os.environ.get(ACTIVITY_ENV, DEFAULT_ACTIVITY),
            self.logger,
        )

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
        """Start a continuous cycle after a delivered normal reply.

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

        now = _loop_time()
        state.generation += 1
        state.person_id = person_id
        state.source_event_id = source_event_id
        state.latest_user_event_id = source_event_id
        state.last_delivered_at = now
        if state.last_user_input_at is None:
            state.last_user_input_at = now
        state.last_attempt_at = None
        state.next_attempt_at = None
        state.unanswered = 0
        state.last_ack_message_id = None
        state.pending_modes.clear()
        state.delivery_event.clear()
        self._cancel_task(state.task)
        cycle = _Cycle(
            conversation_id=conversation_id,
            person_id=person_id,
            source_event_id=source_event_id,
            generation=state.generation,
        )
        state.task = asyncio.create_task(self._run_cycle(cycle, state))
        self.logger.info(
            "proactive_cycle_started conversation_id=%s source_event_id=%s"
            " activity=%s generation=%d",
            conversation_id,
            source_event_id,
            self.activity,
            state.generation,
        )

    def note_user_input(self, conversation_id: str, source_event_id: str) -> None:
        """Invalidate any pending/running cycle as soon as a user turn starts.

        The generation bump makes every later ``_still_valid`` check fail, and
        cancelling the task stops an in-flight Judge, idle wait or Speaker
        call. The new turn's delivered reply starts a fresh cycle.
        """
        state = self._state_for(conversation_id)
        state.generation += 1
        state.latest_user_event_id = source_event_id
        state.last_user_input_at = time.monotonic()
        state.next_attempt_at = None
        state.unanswered = 0
        self._cancel_task(state.task)
        state.task = None
        state.delivery_event.set()
        # Any event still queued for *this* conversation belongs to the
        # previous turn and must not be shown after the user has started a new
        # one. Events for other conversations (other web tabs, other people)
        # stay queued and remain deliverable.
        self._discard_pending_events(conversation_id)
        self.logger.info(
            "proactive_cycle_cancelled_by_user conversation_id=%s"
            " source_event_id=%s",
            conversation_id,
            source_event_id,
        )

    def cancel_cycle(self, conversation_id: str) -> None:
        """Invalidate a conversation's pending/running cycle without a new turn.

        Used when the last web subscriber for that conversation goes away: the
        cycle has nowhere to deliver, so its timers and judge work are
        pointless. Other conversations are unaffected.
        """
        if self._closed:
            return
        state = self._states.get(conversation_id)
        if state is None:
            return
        state.generation += 1
        state.next_attempt_at = None
        self._cancel_task(state.task)
        state.task = None
        state.delivery_event.set()
        self._discard_pending_events(conversation_id)

    async def shutdown(self) -> None:
        """Cancel every background cycle; no timer is persisted across restart."""
        if self._closed:
            return
        self._closed = True
        tasks = []
        for state in self._states.values():
            state.generation += 1
            state.next_attempt_at = None
            state.delivery_event.set()
            if state.task is not None and not state.task.done():
                state.task.cancel()
                tasks.append(state.task)
            state.task = None
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._events.put_nowait(None)

    # -- delivery acknowledgements ----------------------------------------

    def note_proactive_delivered(self, event: Optional[ResponseEvent]) -> bool:
        """The surface really showed/spoke this proactive message.

        Only this acknowledgement advances ``last_delivered_at`` and the
        unanswered counter, so the next interval is measured from real
        delivery instead of model completion. Returns False for duplicate,
        stale or foreign events.
        """
        if self._closed or event is None:
            return False
        conversation_id = getattr(event, "conversation_id", None)
        source_event_id = getattr(event, "event_id", None)
        message_id = getattr(event, "response_id", None)
        state = self._states.get(conversation_id or "")
        if state is None:
            self.logger.info(
                "proactive_ack_ignored conversation_id=%s reason=no_state",
                conversation_id,
            )
            return False
        if source_event_id != state.latest_user_event_id:
            self.logger.info(
                "proactive_ack_ignored conversation_id=%s reason=stale_event",
                conversation_id,
            )
            return False
        if message_id is not None and message_id == state.last_ack_message_id:
            return False
        state.last_ack_message_id = message_id
        state.last_delivered_at = _loop_time()
        state.next_attempt_at = None
        delivered_mode = (
            state.pending_modes.pop(message_id, "") if message_id else ""
        )
        if delivered_mode.startswith(MODE_LONG_IDLE):
            # Only unanswered long-idle rounds make the schedule slower. An
            # immediate remark, a duplicate ack or an unknown row never does.
            state.unanswered += 1
        state.delivery_event.set()
        self.logger.info(
            "proactive_delivery_confirmed conversation_id=%s source_event_id=%s"
            " message_id=%s unanswered=%d next_interval=%s",
            conversation_id,
            source_event_id,
            message_id,
            state.unanswered,
            self._interval_bounds(state.unanswered),
        )
        return True

    def note_proactive_cancelled(
        self, event: Optional[ResponseEvent], reason: str = "cancelled"
    ) -> None:
        """The surface could not deliver this proactive message."""
        if event is None:
            return
        message_id = getattr(event, "response_id", None)
        conversation_id = getattr(event, "conversation_id", None) or ""
        state = self._states.get(conversation_id)
        if state is not None and message_id:
            state.pending_modes.pop(message_id, None)
        self.logger.info(
            "proactive_delivery_cancelled conversation_id=%s source_event_id=%s"
            " message_id=%s reason=%s",
            conversation_id,
            getattr(event, "event_id", None),
            message_id,
            reason,
        )

    async def can_deliver(self, event: Optional[ResponseEvent]) -> bool:
        """Re-check a queued event just before its deferred delivery.

        Used by a surface that had to wait (Ava was speaking): the user may
        have started a new turn, entered Quiet Mode, withdrawn authorization
        or shut the session down while the message waited.
        """
        if self._closed or event is None:
            return False
        conversation_id = getattr(event, "conversation_id", None) or ""
        state = self._states.get(conversation_id)
        if state is None or not state.person_id:
            return False
        if state.task is None or state.task.done():
            # The cycle that produced this message was cancelled (user speech,
            # a new turn, quiet mode, revocation) or already ended.
            return False
        cycle = _Cycle(
            conversation_id=conversation_id,
            person_id=state.person_id,
            source_event_id=getattr(event, "event_id", "") or "",
            generation=state.generation,
        )
        return await self._still_valid(cycle)

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

    def _discard_pending_events(self, conversation_id: Optional[str] = None) -> None:
        """Drop queued proactive events, optionally only for one conversation.

        The manager is shared by every conversation in a process when the web
        UI owns it, so invalidating conversation A must never remove an
        already-delivered message waiting to be shown in conversation B.
        """
        kept: list[ResponseEvent] = []
        sentinel = False
        while True:
            try:
                item = self._events.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                sentinel = True
                continue
            if conversation_id is None or item.conversation_id == conversation_id:
                continue
            kept.append(item)
        for item in kept:
            self._events.put_nowait(item)
        if sentinel:
            # Preserve the shutdown sentinel for a waiting consumer.
            self._events.put_nowait(None)

    # -- the cycle ---------------------------------------------------------

    async def _run_cycle(
        self, cycle: _Cycle, state: _ConversationState
    ) -> None:
        try:
            if not await self._still_valid(cycle):
                return
            # Local hard floor: an explicit departure or refusal to continue
            # blocks every round, independently of the judge's answer.
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
            # not interject or urge; the long-idle rounds may still start a
            # different topic later.
            if not await self._confirmation_pending(cycle):
                await self._attempt(cycle, state, MODE_IMMEDIATE)

            # Continuous adaptive loop: every round is a judgment attempt, so
            # an answered "no" or an unanswered message merely lengthens the
            # next wait. Only fatal conditions end the cycle.
            round_number = 0
            while self._is_current(state, cycle.generation):
                delay = self._next_idle_delay(state.unanswered)
                self.logger.info(
                    "proactive_wait conversation_id=%s source_event_id=%s"
                    " activity=%s round=%d unanswered=%d delay=%.1fs"
                    " last_user_input_at=%s last_delivered_at=%s",
                    cycle.conversation_id,
                    cycle.source_event_id,
                    self.activity,
                    round_number + 1,
                    state.unanswered,
                    delay,
                    _clock_label(state.last_user_input_at),
                    _clock_label(state.last_delivered_at),
                )
                await self._wait_for_idle(state, delay)
                if not self._is_current(state, cycle.generation):
                    return
                if await self._fatal_stop(cycle):
                    self.logger.info(
                        "proactive_cycle_stopped conversation_id=%s"
                        " source_event_id=%s reason=fatal_state",
                        cycle.conversation_id,
                        cycle.source_event_id,
                    )
                    return
                if await self._explicit_stop(cycle):
                    self.logger.info(
                        "proactive_blocked_explicit_stop conversation_id=%s"
                        " source_event_id=%s",
                        cycle.conversation_id,
                        cycle.source_event_id,
                    )
                    return
                if not await self._still_valid(cycle):
                    # Transient (orchestrator busy, delivery surface briefly
                    # unavailable): retry on the next interval.
                    continue
                round_number += 1
                await self._attempt(
                    cycle,
                    state,
                    MODE_LONG_IDLE,
                    f"{MODE_LONG_IDLE}_{round_number}",
                )
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

    async def _wait_for_idle(
        self, state: _ConversationState, delay: Optional[float] = None
    ) -> None:
        """Wait ``delay`` seconds from Ava's last actually delivered message.

        The wait is a cancellable ``asyncio`` wait rather than a blocking
        sleep, and it is woken early when a delivery acknowledgement moves the
        baseline so the next attempt is measured from the real delivery.
        """
        interval = self._next_idle_delay(state.unanswered) if delay is None else delay
        loop = asyncio.get_running_loop()
        logged_at: Optional[float] = None
        while True:
            now = loop.time()
            if state.last_delivered_at is None:
                state.last_delivered_at = now
            remaining = (state.last_delivered_at + interval) - now
            if remaining <= 0:
                state.next_attempt_at = None
                return
            state.next_attempt_at = now + remaining
            if state.next_attempt_at != logged_at:
                logged_at = state.next_attempt_at
                self.logger.info(
                    "proactive_next_attempt conversation_id=%s delay=%.1fs"
                    " at=%.1f last_delivered_at=%s",
                    state.conversation_id,
                    remaining,
                    state.next_attempt_at,
                    _clock_label(state.last_delivered_at),
                )
            try:
                await asyncio.wait_for(
                    state.delivery_event.wait(), timeout=remaining
                )
            except asyncio.TimeoutError:
                state.next_attempt_at = None
                return
            # A real delivery moved the baseline and may have increased the
            # unanswered count; recompute both the baseline and the interval.
            state.delivery_event.clear()
            interval = self._next_idle_delay(state.unanswered)

    def interval_plan(self) -> tuple[tuple[int, int], ...]:
        """The (min, max) seconds plan for the selected activity profile."""
        return ACTIVITY_INTERVALS[self.activity]

    def _interval_bounds(self, unanswered: int) -> tuple[int, int]:
        plan = self.interval_plan()
        index = min(max(0, int(unanswered)), len(plan) - 1)
        return plan[index]

    def _next_idle_delay(self, unanswered: int = 0) -> float:
        if self.idle_delay_seconds is not None:
            return max(0.0, float(self.idle_delay_seconds))
        low, high = self._interval_bounds(unanswered)
        return float(secrets.randbelow(high - low + 1) + low)

    async def _attempt(
        self,
        cycle: _Cycle,
        state: _ConversationState,
        mode: str,
        storage_mode: Optional[str] = None,
    ) -> None:
        """One judge + (possibly) one delivered proactive message."""
        storage_mode = storage_mode or mode
        if not await self._still_valid(cycle):
            return

        # One utterance per persistence round per triggering turn, including
        # across a restart that lost the in-memory state.
        if await self.repos.proactive.get_for_source_event(
            cycle.source_event_id, storage_mode
        ):
            return
        if storage_mode == f"{MODE_LONG_IDLE}_1":
            # Compatibility with rows written before round suffixes existed.
            if await self.repos.proactive.get_for_source_event(
                cycle.source_event_id, MODE_LONG_IDLE
            ):
                return

        source_event = await self.repos.utterance.get(cycle.source_event_id)
        if source_event is None:
            return
        if not await self._still_valid(cycle):
            return

        state.last_attempt_at = asyncio.get_running_loop().time()
        self.logger.info(
            "proactive_judge_attempt conversation_id=%s source_event_id=%s"
            " semantic_mode=%s storage_mode=%s",
            cycle.conversation_id,
            cycle.source_event_id,
            mode,
            storage_mode,
        )

        data = await self.orchestrator.build_proactive_judge_context(
            source_event=source_event, person_id=cycle.person_id, mode=mode
        )
        if data is None:
            return
        decision = await self.judge.decide(ProactiveJudgeContext(**data))
        if decision is None:
            return
        self.logger.info(
            "proactive_judge mode=%s storage_mode=%s decision=%s reason=%s",
            mode,
            storage_mode,
            decision.should_initiate,
            decision.reason,
        )
        if not decision.should_initiate:
            return
        if not self._is_current(state, cycle.generation):
            return
        if not await self._still_valid(cycle):
            return

        try:
            generated = await self.orchestrator.generate_proactive_reply(
                source_event=source_event, person_id=cycle.person_id, mode=mode
            )
        except EmptySpeechOutputError:
            # One plain-text recovery already ran inside the orchestrator and
            # still produced no prose. Discard this initiation: no record, no
            # delivery event, and ``last_delivered_at`` is untouched.
            self.logger.warning(
                "proactive_discarded_empty conversation_id=%s"
                " source_event_id=%s mode=%s",
                cycle.conversation_id,
                cycle.source_event_id,
                storage_mode,
            )
            return
        text = str(generated.get("text") or "").strip()
        if not text:
            self.logger.warning(
                "proactive_discarded_empty conversation_id=%s"
                " source_event_id=%s mode=%s",
                cycle.conversation_id,
                cycle.source_event_id,
                storage_mode,
            )
            return
        display_text = generated.get("display_text")
        if display_text is not None:
            display_text = str(display_text)
        finish_reason = generated.get("finish_reason")
        if finish_reason == "length":
            # An incomplete proactive utterance is not delivered.
            self.logger.info(
                "proactive_discarded_truncated conversation_id=%s"
                " source_event_id=%s mode=%s",
                cycle.conversation_id,
                cycle.source_event_id,
                storage_mode,
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
            mode=storage_mode,
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
            state.pending_modes.pop(message.message_id, None)
            return

        sink = ResponseEventSink(
            cycle.source_event_id, cycle.conversation_id, self._emit
        )
        sink.completed(
            message.message_id,
            text=text,
            display_text=display_text,
            finish_reason=finish_reason,
            report_status="COMPLETED",
            action="RESPOND",
        )
        state.pending_modes[message.message_id] = storage_mode
        state.next_attempt_at = None
        self.logger.info(
            "proactive_generated conversation_id=%s source_event_id=%s"
            " mode=%s message_id=%s chars=%d awaiting_delivery_ack=true",
            cycle.conversation_id,
            cycle.source_event_id,
            storage_mode,
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

    async def _fatal_stop(self, cycle: _Cycle) -> bool:
        """Conditions that end the whole cycle (not just one attempt)."""
        state = self._states.get(cycle.conversation_id)
        if state is None or state.generation != cycle.generation:
            return True
        if self._closed:
            return True
        if cycle.source_event_id != state.latest_user_event_id:
            return True
        if self.delivery_available is not None:
            # No attached surface means the message would be generated and
            # saved with nowhere to go.
            try:
                if not self.delivery_available(cycle.conversation_id):
                    return True
            except Exception:
                return True
        conversation = await self.repos.conversation.get(cycle.conversation_id)
        if conversation is None or conversation.quiet_mode:
            return True
        if not await self.registry.is_authorized(cycle.person_id):
            return True
        if not await self.registry.is_external_ai_authorized(cycle.person_id):
            return True
        consent = await self.repos.person.get_consent(cycle.person_id)
        if consent is None or not consent.allow_persistence:
            return True
        latest = await self.repos.utterance.recent_context(
            cycle.conversation_id, exclude_event_id=None, limit=1
        )
        if not latest or latest[-1].event_id != cycle.source_event_id:
            return True
        return False

    async def _still_valid(self, cycle: _Cycle) -> bool:
        if self._closed:
            return False
        if await self._fatal_stop(cycle):
            return False
        if self._orchestrator_is_processing(cycle.conversation_id):
            return False
        return True

    def _orchestrator_is_processing(self, conversation_id: str) -> bool:
        checker = getattr(self.orchestrator, "is_processing", None)
        return bool(checker(conversation_id)) if callable(checker) else False

    # -- helpers -----------------------------------------------------------

    def _state_for(self, conversation_id: str) -> _ConversationState:
        state = self._states.get(conversation_id)
        if state is None:
            state = _ConversationState(conversation_id=conversation_id)
            self._states[conversation_id] = state
        return state

    @staticmethod
    def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is not None and not task.done():
            task.cancel()


def _normalise_activity(
    value: Optional[str], logger: logging.Logger
) -> str:
    text = str(value or "").strip().lower()
    if text in ACTIVITY_INTERVALS:
        return text
    if text:
        logger.warning(
            "proactive_activity_invalid value=%s using=%s",
            text,
            DEFAULT_ACTIVITY,
        )
    return DEFAULT_ACTIVITY


def _loop_time() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()


def _clock_label(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "-"


__all__ = [
    "ProactiveManager",
    "ORIGIN_AVA_INITIATED",
    "MODE_IMMEDIATE",
    "MODE_LONG_IDLE",
    "ACTIVITY_ENV",
    "ACTIVITY_HIGH",
    "ACTIVITY_NORMAL",
    "ACTIVITY_LOW",
    "ACTIVITY_INTERVALS",
]
