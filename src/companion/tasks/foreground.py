"""Foreground activity signal.

A counter, not a boolean: overlapping foreground events (a streamed reply and
an explicit foreground memory write in the same turn) keep the signal active
until the *last* one ends. Background handlers consult it only at safe
checkpoints; a database transaction or an in-flight provider request is never
killed.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

from companion.domain.models import utcnow


class ForegroundActivity:
    def __init__(self) -> None:
        self._count = 0
        self._last_active_at: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._count > 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def last_active_at(self) -> Optional[datetime]:
        with self._lock:
            return self._last_active_at

    def begin(self) -> None:
        with self._lock:
            self._count += 1
            self._last_active_at = utcnow()

    def end(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1
            self._last_active_at = utcnow()

    def idle_seconds(self, *, now: Optional[datetime] = None) -> float:
        with self._lock:
            if self._count > 0 or self._last_active_at is None:
                return 0.0
            current = now or utcnow()
            return max(0.0, (current - self._last_active_at).total_seconds())

    @contextmanager
    def scope(self) -> Iterator[None]:
        self.begin()
        try:
            yield
        finally:
            self.end()
