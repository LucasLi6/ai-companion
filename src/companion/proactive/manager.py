"""One-shot ProactiveManager (V0.9.8).

The manager owns *timing, staleness and delivery* only. It never decides
whether a topic is interesting: :class:`ProactiveJudge` returns one boolean,
and the existing Speaker decides what Ava actually says.

Lifecycle per triggering turn:

    normal reply committed + delivered
        -> on_response_delivered()        (non-blocking)
        -> sleep ~2 s
        -> observe short jobs for at most 1 s
        -> judge once
        -> (YES) existing Speaker, one message
        -> persist independently and emit one response_completed event
        -> stop (a proactive message never schedules another check)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from companion.core.response_events import ResponseEvent, ResponseEventSink
from companion.domain.models import new_id, utcnow
from companion.proactive.judge import (
    ProactiveJudge,
    ProactiveJudgeContext,
    ProactiveJudgeDecision,
)
from companion.storage.repositories import ProactiveMessageRecord

ORIGIN_AVA_INITIATED = "AVA_INITIATED"
SHORT_JOB_TYPES = ("TURN_ANALYSIS", "PROFILE_REVIEW")


@dataclass
class _ConversationState:
    generation: int = 0
    person_id: Optional[str] = None
    source_event_id: Optional[str] = None
    latest_user_event_id: Optional[str] = None
    check_started: bool = False
    check_finished: bool = False
    task: Optional[asyncio.Task] = None


@dataclass(frozen=True)
class _Check:
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
        delay_seconds: float = 2.0,
        short_task_timeout: float = 1.0,
        short_task_poll_seconds: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.conversations = conversations
        self.orchestrator = orchestrator
        self.judge = judge
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.short_task_timeout = max(0.0, float(short_task_timeout))
        self.short_task_poll_seconds = max(0.001, float(short_task_poll_seconds))
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
        """Schedule one check after a user-triggered normal reply was delivered.

        This method is synchronous and never blocks the chat surface. Callers
        must invoke it only after the reply is fully displayed; WAIT/IGNORE,
        failures, cancellations and truncated replies must not call it.
        """
        if self._closed or not conversation_id or not source_event_id:
            return
        state = self._state_for(conversation_id)
        if state.source_event_id == source_event_id:
            # A duplicate completion event must never schedule a second check.
            return

        state.generation += 1
        state.person_id = person_id
        state.source_event_id = source_event_id
        state.latest_user_event_id = source_event_id
        state.check_started = True
        state.check_finished = False
        self._cancel_task(state.task)
        check = _Check(
            conversation_id=conversation_id,
            person_id=person_id,
            source_event_id=source_event_id,
            generation=state.generation,
        )
        state.task = asyncio.create_task(self._run_check(check, state))

    def note_user_input(self, conversation_id: str, source_event_id: str) -> None:
        """Invalidate any pending/running check as soon as a user turn starts.

        The generation bump makes every later ``_still_valid`` check fail, and
        cancelling the task stops an in-flight Judge or Speaker call. No long
        lived database state machine is involved.
        """
        state = self._state_for(conversation_id)
        state.generation += 1
        state.latest_user_event_id = source_event_id
        state.check_finished = True
        self._cancel_task(state.task)
        state.task = None
        # Any event still queued belongs to the previous turn and must not be
        # shown after the user has started a new one.
        self._discard_pending_events()

    async def shutdown(self) -> None:
        """Cancel every background check; no timer is persisted across restart."""
        if self._closed:
            return
        self._closed = True
        tasks = []
        for state in self._states.values():
            state.generation += 1
            state.check_finished = True
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

    # -- the one check -----------------------------------------------------

    async def _run_check(
        self, check: _Check, state: _ConversationState
    ) -> None:
        try:
            await asyncio.sleep(self.delay_seconds)
            if not self._is_current(state, check.generation):
                return
            await self._wait_short_tasks(check)
            if not self._is_current(state, check.generation):
                return

            source_event = await self.repos.utterance.get(check.source_event_id)
            if source_event is None:
                return
            if not await self._still_valid(check):
                return

            data = await self.orchestrator.build_proactive_judge_context(
                source_event=source_event, person_id=check.person_id
            )
            if data is None:
                return
            decision = await self.judge.decide(ProactiveJudgeContext(**data))
            if decision is None:
                return
            self.logger.info(
                "proactive_judge conversation_id=%s source_event_id=%s"
                " should_initiate=%s",
                check.conversation_id,
                check.source_event_id,
                decision.should_initiate,
            )
            if not decision.should_initiate:
                return
            if not self._is_current(state, check.generation):
                return
            if not await self._still_valid(check):
                return

            # At most one proactive message per triggering turn. A duplicate
            # completion event is already filtered in on_response_delivered;
            # this also covers a restart/retry that lost the in-memory state.
            if await self.repos.proactive.get_for_source_event(
                check.source_event_id
            ):
                return

            generated = await self.orchestrator.generate_proactive_reply(
                source_event=source_event, person_id=check.person_id
            )
            text = str(generated.get("text") or "").strip()
            if not text:
                return
            finish_reason = generated.get("finish_reason")
            if finish_reason == "length":
                # An incomplete proactive utterance is not delivered.
                self.logger.info(
                    "proactive_discarded_truncated conversation_id=%s"
                    " source_event_id=%s",
                    check.conversation_id,
                    check.source_event_id,
                )
                return
            if not self._is_current(state, check.generation):
                return
            if not await self._still_valid(check):
                return

            message = ProactiveMessageRecord(
                message_id=new_id("pmsg"),
                conversation_id=check.conversation_id,
                person_id=check.person_id,
                source_event_id=check.source_event_id,
                origin=ORIGIN_AVA_INITIATED,
                text=text,
                status="COMPLETED",
                finish_reason=finish_reason,
                model=str(generated.get("model") or ""),
                created_at=utcnow(),
            )
            await self.repos.proactive.save(message)

            # A new user message may have arrived while the row was being
            # written. Stale output must not be shown; remove the un-delivered
            # record so it also cannot become forged later history.
            if not self._is_current(state, check.generation) or not await self._still_valid(
                check
            ):
                await self.repos.proactive.delete(message.message_id)
                return

            sink = ResponseEventSink(
                check.source_event_id, check.conversation_id, self._emit
            )
            sink.completed(
                message.message_id,
                text=text,
                finish_reason=finish_reason,
                report_status="COMPLETED",
                action="RESPOND",
            )
            self.logger.info(
                "proactive_delivered conversation_id=%s source_event_id=%s"
                " message_id=%s chars=%d",
                check.conversation_id,
                check.source_event_id,
                message.message_id,
                len(text),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "proactive_check_failed conversation_id=%s source_event_id=%s"
                " error=%s",
                check.conversation_id,
                check.source_event_id,
                type(exc).__name__,
            )
        finally:
            if state.task is asyncio.current_task():
                state.task = None
                state.check_finished = True

    async def _wait_short_tasks(self, check: _Check) -> None:
        """Observe TURN_ANALYSIS/PROFILE_REVIEW for at most the configured time.

        Long jobs (memory consolidation, embedding, research, idle learning)
        are never observed, so a 60 s extraction delay cannot hold the judge.
        A PENDING job scheduled far in the future is still just observed; the
        deadline is absolute.
        """
        deadline = asyncio.get_running_loop().time() + self.short_task_timeout
        keys = (
            f"turn_analysis:{check.source_event_id}",
            f"profile-review:{check.person_id}:{check.source_event_id}",
        )
        while True:
            active = await self._short_jobs_active(keys)
            if not active:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(self.short_task_poll_seconds, remaining))

    async def _short_jobs_active(self, keys) -> bool:
        rows = await self.repos.job.list_active_by_keys(keys)
        return bool(rows)

    # -- validity ----------------------------------------------------------

    def _is_current(self, state: _ConversationState, generation: int) -> bool:
        return not self._closed and state.generation == generation

    async def _still_valid(self, check: _Check) -> bool:
        state = self._states.get(check.conversation_id)
        if state is None or state.generation != check.generation:
            return False
        if check.source_event_id != state.latest_user_event_id:
            return False
        conversation = await self.repos.conversation.get(check.conversation_id)
        if conversation is None or conversation.quiet_mode:
            return False
        if self._orchestrator_is_processing(check.conversation_id):
            return False
        if not await self.registry.is_authorized(check.person_id):
            return False
        if not await self.registry.is_external_ai_authorized(check.person_id):
            return False
        consent = await self.repos.person.get_consent(check.person_id)
        if consent is None or not consent.allow_persistence:
            return False
        latest = await self.repos.utterance.recent_context(
            check.conversation_id, exclude_event_id=None, limit=1
        )
        if not latest or latest[-1].event_id != check.source_event_id:
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


__all__ = ["ProactiveManager", "ORIGIN_AVA_INITIATED"]
