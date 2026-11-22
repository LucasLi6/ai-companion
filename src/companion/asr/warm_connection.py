"""Optional background WebSocket preconnect for streaming ASR.

DashScope 2.0.4 can open one shared duplex websocket and run each
``audio.asr.recognize`` task on it (``connection=...``). Most of a cold
turn's delay is that handshake, so this module prepares the transport while
Ava is only listening. It owns the transport only: recognition state, VAD,
audio and transcripts stay in :class:`ASRSession` / :class:`ASRManager`.

One connection carries at most one binary audio upload at a time, so a
connection is leased to exactly one sender. It becomes available again only
after that task finished cleanly and its ``Recognition`` was detached; a
failed, cancelled or unhealthy task closes the transport instead of risking
a second task on a half-used socket.

All long operations (the handshake, ``close()``, joining worker threads) run
outside the internal lock. A generation counter prevents an older in-flight
preconnect from publishing itself over a newer attempt or after shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional

from companion.asr.client import _default_client_factory
from companion.asr.config import ASRConfig

# One background attempt at a time; failures back off instead of hammering
# the service while Ava is idle.
PRECONNECT_RETRY_DELAYS_SECONDS = (10.0, 30.0, 60.0)


class WarmConnectionState(str, Enum):
    """Transport lifecycle; only ``READY`` may be leased."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    IN_USE = "IN_USE"
    INVALID = "INVALID"


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


class WarmConnectionLease:
    """One recognition task's exclusive claim on a warm transport.

    ``release`` is idempotent. ``healthy=True`` means the recognition task
    reached ``task-finished`` and its ``Recognition`` was detached; only then
    may the socket be reused. Any other outcome invalidates it.
    """

    def __init__(
        self,
        manager: "ASRWarmConnectionManager",
        *,
        client: Any,
        connection: Any,
        generation: int,
        preconnect_ms: Optional[float],
    ) -> None:
        self.client = client
        self.connection = connection
        self.generation = int(generation)
        self.preconnect_ms = preconnect_ms
        self._manager = manager
        self._release_lock = threading.Lock()
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self, *, healthy: bool = True) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._manager._release_lease(self, healthy=bool(healthy))


class ASRWarmConnectionManager:
    """Owns at most one preconnected DashScope websocket.

    ``client_factory`` and ``connection_factory`` are injection points for
    offline tests; production uses the same client as the cold ASR path and
    ``client.ws_connection(module="audio")``.
    """

    def __init__(
        self,
        config: ASRConfig,
        *,
        client_factory: Optional[Callable[[ASRConfig], Any]] = None,
        connection_factory: Optional[Callable[[Any], Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._connection_factory = connection_factory
        self._logger = logger or logging.getLogger("companion.asr.warm")
        self._lock = threading.Lock()
        self._state = WarmConnectionState.DISCONNECTED
        self._client: Any = None
        self._connection: Any = None
        self._generation = 0
        self._preconnect_ms: Optional[float] = None
        self._idle_since: Optional[float] = None
        self._last_failure_at: Optional[float] = None
        self._failures = 0
        # Set when the installed SDK lacks the shared-connection API. The
        # cold per-turn path keeps working; no further handshakes are tried.
        self._disabled = False
        self._thread: Optional[threading.Thread] = None
        self._retry_timer: Optional[threading.Timer] = None
        self._expire_timer: Optional[threading.Timer] = None
        self._shutdown = threading.Event()

    # ------------------------------------------------------------ observable
    @property
    def state(self) -> WarmConnectionState:
        with self._lock:
            return self._state

    def stats(self) -> dict:
        """Non-secret diagnostics for tests and debug logs."""
        with self._lock:
            connection = self._connection
            return {
                "state": self._state.value,
                "generation": self._generation,
                "preconnect_ms": self._preconnect_ms,
                "failures": self._failures,
                "in_use": self._state is WarmConnectionState.IN_USE,
                "closed": bool(getattr(connection, "closed", False)),
                "disabled": self._disabled,
            }

    # ------------------------------------------------------------- preconnect
    def start_preconnect(self) -> bool:
        """Start one background handshake; never blocks on the network."""
        if self._shutdown.is_set() or self._disabled:
            return False
        now = time.monotonic()
        with self._lock:
            if self._state in (
                WarmConnectionState.CONNECTING,
                WarmConnectionState.READY,
                WarmConnectionState.IN_USE,
                # INVALID while a lease is out: the connection is still in
                # active use, and the release path owns the rebuild.
                WarmConnectionState.INVALID,
            ):
                return False
            if self._failures and self._last_failure_at is not None:
                delay = PRECONNECT_RETRY_DELAYS_SECONDS[
                    min(
                        self._failures - 1,
                        len(PRECONNECT_RETRY_DELAYS_SECONDS) - 1,
                    )
                ]
                remaining = delay - (now - self._last_failure_at)
                if remaining > 0.0:
                    self._schedule_retry_locked(remaining)
                    return False
            self._generation += 1
            generation = self._generation
            self._state = WarmConnectionState.CONNECTING
            self._preconnect_ms = None
            self._idle_since = None
            stale = (self._client, self._connection)
            self._client = None
            self._connection = None
            thread = threading.Thread(
                target=self._connect_worker,
                args=(generation,),
                name=f"asr-warm-connect-{generation}",
                daemon=True,
            )
            self._thread = thread
        _close_quietly(stale[0])
        _close_quietly(stale[1])
        thread.start()
        return True

    def try_acquire(self) -> Optional[WarmConnectionLease]:
        """Atomically lease the ready transport, or return ``None``.

        Never waits for a handshake that is still in progress: the caller
        must fall back to a cold connection instead.
        """
        if self._shutdown.is_set():
            return None
        now = time.monotonic()
        expired: tuple[Any, Any] = (None, None)
        rebuild = False
        with self._lock:
            if self._state is not WarmConnectionState.READY:
                return None
            connection = self._connection
            ttl = float(getattr(self._config, "warm_idle_ttl_seconds", 0.0))
            stale = getattr(connection, "closed", False)
            idle = (
                self._idle_since is not None
                and ttl > 0.0
                and now - self._idle_since > ttl
            )
            if connection is None or stale or idle:
                expired = (self._client, self._connection)
                self._client = None
                self._connection = None
                self._preconnect_ms = None
                self._idle_since = None
                self._state = WarmConnectionState.DISCONNECTED
                rebuild = True
            else:
                self._state = WarmConnectionState.IN_USE
                self._cancel_timer_locked("_expire_timer")
                return WarmConnectionLease(
                    self,
                    client=self._client,
                    connection=self._connection,
                    generation=self._generation,
                    preconnect_ms=self._preconnect_ms,
                )
        _close_quietly(expired[0])
        _close_quietly(expired[1])
        if rebuild:
            self.start_preconnect()
        return None

    def invalidate(self, reason: str = "") -> None:
        """Stop handing out this transport; close it once it is free."""
        with self._lock:
            if self._shutdown.is_set():
                return
            if self._state is WarmConnectionState.IN_USE:
                # The active lease closes it on release.
                self._state = WarmConnectionState.INVALID
                return
            close = (self._client, self._connection)
            self._client = None
            self._connection = None
            self._preconnect_ms = None
            self._idle_since = None
            self._state = WarmConnectionState.DISCONNECTED
            self._cancel_timer_locked("_expire_timer")
        _close_quietly(close[0])
        _close_quietly(close[1])
        if reason:
            self._logger.debug("asr_warm_invalidated reason=%s", reason)
        self.start_preconnect()

    def disable(self, reason: str = "") -> None:
        """Permanently fall back to per-turn cold connections.

        Used when this installed SDK cannot share a socket across tasks. The
        cold path is untouched; this only stops the optional optimization.
        """
        with self._lock:
            if self._disabled:
                return
            self._disabled = True
        if reason:
            self._logger.warning(
                "asr_preconnect_disabled reason=%s", reason
            )
        self.invalidate("sdk_without_shared_connection")

    def shutdown(self) -> None:
        """Stop rebuilding and close every owned resource."""
        self._shutdown.set()
        with self._lock:
            self._generation += 1
            close = (self._client, self._connection)
            self._client = None
            self._connection = None
            self._preconnect_ms = None
            self._idle_since = None
            self._state = WarmConnectionState.DISCONNECTED
            self._cancel_timer_locked("_retry_timer")
            self._cancel_timer_locked("_expire_timer")
            thread = self._thread
            self._thread = None
        _close_quietly(close[1])
        _close_quietly(close[0])
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

    # ------------------------------------------------------------- internals
    def _connect_worker(self, generation: int) -> None:
        started = time.monotonic()
        client: Any = None
        connection: Any = None
        try:
            client = self._client_factory(self._config)
            if self._connection_factory is not None:
                connection = self._connection_factory(client)
            else:
                connection = client.ws_connection(module="audio")
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        except Exception as exc:  # noqa: BLE001 - report and retry with backoff
            if isinstance(exc, AttributeError):
                # e.g. an older dashscope without ``Client.ws_connection``.
                self.disable("sdk_missing_ws_connection")
            self._preconnect_failed(generation, exc, client, connection)
            return
        with self._lock:
            stale = (
                self._shutdown.is_set()
                or self._disabled
                or generation != self._generation
            )
            if not stale:
                self._client = client
                self._connection = connection
                self._preconnect_ms = elapsed_ms
                self._idle_since = time.monotonic()
                self._state = WarmConnectionState.READY
                self._failures = 0
                self._last_failure_at = None
                self._thread = None
                self._schedule_expire_locked()
        if stale:
            _close_quietly(connection)
            _close_quietly(client)

    def _preconnect_failed(
        self,
        generation: int,
        exc: BaseException,
        client: Any,
        connection: Any,
    ) -> None:
        _close_quietly(connection)
        _close_quietly(client)
        with self._lock:
            if (
                self._shutdown.is_set()
                or generation != self._generation
            ):
                return
            self._state = WarmConnectionState.DISCONNECTED
            self._thread = None
            self._failures += 1
            self._last_failure_at = time.monotonic()
            delay = PRECONNECT_RETRY_DELAYS_SECONDS[
                min(self._failures - 1, len(PRECONNECT_RETRY_DELAYS_SECONDS) - 1)
            ]
            if not self._disabled:
                self._schedule_retry_locked(delay)
        self._logger.debug(
            "asr_warm_preconnect_failed type=%s retry_in=%.0fs",
            type(exc).__name__,
            delay,
        )

    def _release_lease(
        self, lease: WarmConnectionLease, *, healthy: bool
    ) -> None:
        close: tuple[Any, Any] = (None, None)
        restart = False
        with self._lock:
            current = (
                self._state is WarmConnectionState.IN_USE
                and lease.generation == self._generation
                and lease.connection is self._connection
            )
            if not current:
                # Shutdown, a newer generation or an explicit invalidation
                # already detached this transport. Never publish it again.
                # After shutdown the manager owns teardown (it closed the
                # connection this lease held), so do not close it twice.
                if not self._shutdown.is_set():
                    close = (lease.client, lease.connection)
                if self._state is WarmConnectionState.INVALID:
                    self._state = WarmConnectionState.DISCONNECTED
                    restart = not self._shutdown.is_set()
            else:
                connection = self._connection
                healthy = (
                    healthy
                    and not self._shutdown.is_set()
                    and not getattr(connection, "closed", False)
                )
                if healthy:
                    self._state = WarmConnectionState.READY
                    self._idle_since = time.monotonic()
                    self._schedule_expire_locked()
                else:
                    close = (self._client, self._connection)
                    self._client = None
                    self._connection = None
                    self._preconnect_ms = None
                    self._idle_since = None
                    self._state = WarmConnectionState.DISCONNECTED
                    self._cancel_timer_locked("_expire_timer")
                    restart = not self._shutdown.is_set()
        _close_quietly(close[1])
        _close_quietly(close[0])
        if restart:
            self.start_preconnect()

    def _schedule_retry_locked(self, delay: float) -> None:
        self._cancel_timer_locked("_retry_timer")
        timer = threading.Timer(
            max(0.01, float(delay)), self._retry_preconnect
        )
        timer.daemon = True
        self._retry_timer = timer
        timer.start()

    def _retry_preconnect(self) -> None:
        with self._lock:
            self._retry_timer = None
        if not self._shutdown.is_set():
            self.start_preconnect()

    def _schedule_expire_locked(self) -> None:
        """Replace an idle connection once its TTL is reached."""
        self._cancel_timer_locked("_expire_timer")
        ttl = float(getattr(self._config, "warm_idle_ttl_seconds", 0.0))
        if ttl <= 0.0:
            return
        timer = threading.Timer(ttl, self._expire_idle)
        timer.daemon = True
        self._expire_timer = timer
        timer.start()

    def _expire_idle(self) -> None:
        close: tuple[Any, Any] = (None, None)
        rebuild = False
        now = time.monotonic()
        with self._lock:
            self._expire_timer = None
            ttl = float(getattr(self._config, "warm_idle_ttl_seconds", 0.0))
            if (
                self._shutdown.is_set()
                or self._state is not WarmConnectionState.READY
                or self._idle_since is None
                or ttl <= 0.0
                or now - self._idle_since < ttl
            ):
                return
            close = (self._client, self._connection)
            self._client = None
            self._connection = None
            self._preconnect_ms = None
            self._idle_since = None
            self._state = WarmConnectionState.DISCONNECTED
            rebuild = True
        _close_quietly(close[1])
        _close_quietly(close[0])
        if rebuild:
            self.start_preconnect()

    def _cancel_timer_locked(self, attribute: str) -> None:
        timer = getattr(self, attribute)
        if timer is not None:
            timer.cancel()
            setattr(self, attribute, None)


__all__ = [
    "ASRWarmConnectionManager",
    "PRECONNECT_RETRY_DELAYS_SECONDS",
    "WarmConnectionLease",
    "WarmConnectionState",
]
