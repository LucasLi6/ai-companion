"""Unified response lifecycle events.

Every response surface (CLI, web, future voice transport) consumes the same
small event vocabulary instead of raw provider chunks:

``response_started`` / ``response_delta`` / ``response_completed`` /
``response_failed`` / ``response_cancelled``

Each event carries stable identity (``event_id``, ``conversation_id``,
``response_id``) and a per-response ``sequence`` so a receiver can reassemble
the stream, detect gaps and render a truthful interruption state. A partial
reply is never reported as ``COMPLETED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

RESPONSE_STARTED = "response_started"
RESPONSE_DELTA = "response_delta"
RESPONSE_COMPLETED = "response_completed"
RESPONSE_FAILED = "response_failed"
RESPONSE_CANCELLED = "response_cancelled"

EVENT_TYPES = (
    RESPONSE_STARTED,
    RESPONSE_DELTA,
    RESPONSE_COMPLETED,
    RESPONSE_FAILED,
    RESPONSE_CANCELLED,
)


@dataclass(frozen=True)
class ResponseEvent:
    type: str
    event_id: str
    conversation_id: str
    response_id: str
    sequence: int
    status: str
    delta: Optional[str] = None
    text: Optional[str] = None
    display_delta: Optional[str] = None
    display_text: Optional[str] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None
    report_status: Optional[str] = None
    action: Optional[str] = None

    def as_dict(self) -> dict:
        data = {
            "type": self.type,
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "response_id": self.response_id,
            "sequence": self.sequence,
            "status": self.status,
        }
        for name in (
            "delta",
            "text",
            "display_delta",
            "display_text",
            "finish_reason",
            "error",
            "usage",
            "report_status",
            "action",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data


class ResponseEventSink:
    """Synchronous event emitter used inside the orchestrator.

    It assigns sequence numbers and forwards immutable events to a callback
    (usually an ``asyncio.Queue.put_nowait``). It never buffers the response
    itself; the caller receives each delta as it is produced.
    """

    def __init__(self, event_id: str, conversation_id: str, emit) -> None:
        self.event_id = event_id
        self.conversation_id = conversation_id
        self._emit = emit
        self._sequence = 0

    def _next(self, type_: str, response_id: str, status: str, **fields) -> ResponseEvent:
        event = ResponseEvent(
            type=type_,
            event_id=self.event_id,
            conversation_id=self.conversation_id,
            response_id=response_id,
            sequence=self._sequence,
            status=status,
            **fields,
        )
        self._sequence += 1
        self._emit(event)
        return event

    def started(self, response_id: str) -> ResponseEvent:
        return self._next(RESPONSE_STARTED, response_id, "STREAMING")

    def delta(
        self,
        response_id: str,
        text: str,
        *,
        display_delta: Optional[str] = None,
    ) -> ResponseEvent:
        return self._next(
            RESPONSE_DELTA,
            response_id,
            "STREAMING",
            delta=text,
            display_delta=display_delta,
        )

    def completed(
        self,
        response_id: str,
        *,
        text: str,
        display_text: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[dict] = None,
        report_status: str = "COMPLETED",
        action: Optional[str] = None,
    ) -> ResponseEvent:
        status = "TRUNCATED" if finish_reason == "length" else "COMPLETED"
        return self._next(
            RESPONSE_COMPLETED,
            response_id,
            status,
            text=text,
            display_text=display_text,
            finish_reason=finish_reason,
            usage=usage,
            report_status=report_status,
            action=action,
        )

    def failed(
        self,
        response_id: str,
        *,
        error: str,
        partial_text: Optional[str] = None,
    ) -> ResponseEvent:
        return self._next(
            RESPONSE_FAILED,
            response_id,
            "FAILED",
            error=error,
            text=partial_text,
        )

    def cancelled(
        self, response_id: str, *, partial_text: Optional[str] = None
    ) -> ResponseEvent:
        return self._next(
            RESPONSE_CANCELLED,
            response_id,
            "CANCELLED",
            text=partial_text,
        )
