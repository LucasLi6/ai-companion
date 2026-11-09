"""Server-lifetime chat runtime and proactive event fan-out (V0.9.9.2).

The NiceGUI management pages are request-scoped: every page opens a short
lived SQLite connection, does its work and closes it. A proactive cycle cannot
live in that scope because it has to keep running *after* the reply has been
delivered:

    normal reply finished
        -> IMMEDIATE judge (right away)
        -> [maybe] one proactive message
        -> wait 120-240 s
        -> LONG_IDLE judge
        -> [maybe] one more proactive message

This module owns exactly one long-lived :class:`~companion.runtime.Runtime`
for the chat surface, one :class:`~companion.proactive.manager.ProactiveManager`
(which is already keyed by conversation), and one background consumer that
routes each delivered ``ResponseEvent`` to every subscribed browser tab.

It deliberately does **not** create a second speaker, persona, memory store or
database. The shared runtime is the same object graph the CLI uses, built by
:func:`~companion.runtime.build_runtime`.

Isolation rules:

- one subscription table per conversation id, so two tabs on the same
  conversation both see the same proactive message instead of competing for
  one queue item;
- a conversation with no subscribers is not "deliverable": its pending cycle
  is cancelled and the manager refuses to generate new messages for it;
- events are routed by the ``conversation_id`` carried on the event itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

from companion.core.response_events import RESPONSE_COMPLETED, ResponseEvent
from companion.runtime import Runtime, build_runtime
from companion.storage.database import Database

ProactiveCallback = Callable[[ResponseEvent], Optional[Awaitable[None]]]


class ChatCoordinator:
    """One shared chat runtime plus the proactive delivery fan-out."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        provider: str = "mock",
        deepseek_client_factory: Optional[Callable[[], object]] = None,
        runtime_factory: Optional[Callable[[Database], Runtime]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.provider = (provider or "mock").lower()
        self.deepseek_client_factory = deepseek_client_factory
        self.runtime_factory = runtime_factory
        self.logger = logger or logging.getLogger("companion.ui.chat")
        self._db: Optional[Database] = None
        self._runtime: Optional[Runtime] = None
        self._dispatcher: Optional[asyncio.Task] = None
        self._subscribers: dict[str, dict[str, ProactiveCallback]] = {}
        self._latest_input: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def runtime(self) -> Optional[Runtime]:
        return self._runtime

    @property
    def started(self) -> bool:
        return self._runtime is not None

    async def start(self) -> Runtime:
        """Connect the long-lived database and build the shared runtime.

        Idempotent; safe to call from server startup and from a page that
        arrives before startup finished.
        """
        async with self._lock:
            if self._runtime is not None:
                return self._runtime
            self._closed = False
            db = await Database(self.db_path).connect()
            try:
                await db.initialize()
                runtime = self._build_runtime(db)
            except BaseException:
                await db.close()
                raise
            manager = getattr(runtime, "proactive_manager", None)
            if manager is not None:
                # The manager must never generate a message whose only
                # purpose is to be displayed when no tab is listening.
                manager.delivery_available = self.has_subscribers
            confirmation = getattr(runtime, "profile_confirmation", None)
            if confirmation is not None:
                # Resume no-reply timers once for the process, on this loop.
                # A request-scoped runtime would create duplicate timers and
                # lose them when the request ends.
                with contextlib.suppress(Exception):
                    await confirmation.resume_pending()
            self._db = db
            self._runtime = runtime
            if manager is not None:
                self._dispatcher = asyncio.create_task(
                    self._dispatch_loop(manager), name="chat-proactive-dispatch"
                )
            return runtime

    def _build_runtime(self, db: Database) -> Runtime:
        if self.runtime_factory is not None:
            return self.runtime_factory(db)
        client = None
        if self.provider == "deepseek" and self.deepseek_client_factory is not None:
            client = self.deepseek_client_factory()
        return build_runtime(
            db,
            with_worker=False,
            provider=self.provider,
            deepseek_client=client,
        )

    async def close(self) -> None:
        """Stop timers, the dispatcher and the shared connection."""
        async with self._lock:
            self._closed = True
            dispatcher, self._dispatcher = self._dispatcher, None
            runtime, self._runtime = self._runtime, None
            db, self._db = self._db, None
        if dispatcher is not None:
            dispatcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await dispatcher
        if runtime is not None:
            manager = getattr(runtime, "proactive_manager", None)
            if manager is not None:
                with contextlib.suppress(Exception):
                    await manager.shutdown()
            confirmation = getattr(runtime, "profile_confirmation", None)
            if confirmation is not None:
                with contextlib.suppress(Exception):
                    await confirmation.shutdown()
        if db is not None:
            await db.close()
        self._subscribers.clear()

    # -- subscriptions -----------------------------------------------------

    async def subscribe(
        self,
        conversation_id: str,
        client_id: str,
        callback: ProactiveCallback,
    ) -> None:
        await self.start()
        self._subscribers.setdefault(conversation_id, {})[client_id] = callback

    async def unsubscribe(self, conversation_id: str, client_id: str) -> None:
        subscribers = self._subscribers.get(conversation_id)
        if not subscribers:
            return
        subscribers.pop(client_id, None)
        if subscribers:
            return
        self._subscribers.pop(conversation_id, None)
        manager = self._manager()
        if manager is not None:
            # Nothing can be delivered for this conversation any more;
            # cancel only its cycle and leave other tabs untouched.
            manager.cancel_cycle(conversation_id)

    def has_subscribers(self, conversation_id: str) -> bool:
        return bool(self._subscribers.get(conversation_id))

    def subscriber_count(self, conversation_id: str) -> int:
        return len(self._subscribers.get(conversation_id, {}))

    # -- turn hooks --------------------------------------------------------

    def note_user_input(self, conversation_id: str, source_event_id: str) -> None:
        """Invalidate pending proactive work before a new turn starts."""
        self._latest_input[conversation_id] = source_event_id
        manager = self._manager()
        if manager is not None:
            manager.note_user_input(conversation_id, source_event_id)

    def on_normal_response_delivered(
        self,
        *,
        conversation_id: str,
        person_id: str,
        source_event_id: str,
    ) -> None:
        """Schedule the dual-stage cycle after a real, delivered reply."""
        if not self.has_subscribers(conversation_id):
            return
        # A slower earlier turn may finish after a newer turn has already
        # started. Its completion must not schedule work for a turn the user
        # has moved on from; only the latest input may start a cycle.
        latest = self._latest_input.get(conversation_id)
        if latest is not None and latest != source_event_id:
            return
        manager = self._manager()
        if manager is not None:
            manager.on_response_delivered(
                conversation_id=conversation_id,
                person_id=person_id,
                source_event_id=source_event_id,
            )

    # -- delivery ----------------------------------------------------------

    async def _dispatch_loop(self, manager) -> None:
        try:
            while True:
                event = await manager.next_event()
                if event is None:
                    return
                await self._deliver(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a broken loop must not take the server down
            self.logger.warning(
                "chat_proactive_dispatch_failed error=%s", type(exc).__name__
            )

    async def _deliver(self, event: ResponseEvent) -> None:
        manager = self._manager()
        if event.type != RESPONSE_COMPLETED:
            if manager is not None:
                manager.note_proactive_cancelled(event, "not_completed")
            return
        if not (event.text or "").strip():
            if manager is not None:
                manager.note_proactive_cancelled(event, "empty_text")
            return
        subscribers = list(
            self._subscribers.get(event.conversation_id, {}).values()
        )
        if not subscribers:
            # The manager checks before saving; this is only a display-time
            # race. The message remains in history for the next page load.
            if manager is not None:
                manager.note_proactive_cancelled(event, "no_subscriber")
            return
        delivered = False
        for callback in subscribers:
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
                delivered = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "chat_proactive_subscriber_failed conversation_id=%s"
                    " error=%s",
                    event.conversation_id,
                    type(exc).__name__,
                )
        # Delivery timing must be based on a message that really reached a
        # surface, never on model generation alone.
        if manager is not None:
            if delivered:
                manager.note_proactive_delivered(event)
            else:
                manager.note_proactive_cancelled(
                    event, "subscriber_failed"
                )

    # -- helpers -----------------------------------------------------------

    def _manager(self):
        runtime = self._runtime
        return getattr(runtime, "proactive_manager", None) if runtime else None


__all__ = ["ChatCoordinator", "ProactiveCallback"]
