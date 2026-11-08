"""Terminal rendering for live ASR.

Only confirmed text becomes a transcript line. Provisional ASR text is shown
on the last line and replaced in place with ``\\r\\x1b[2K``; it is never
treated as final. A non-TTY stream falls back to plain line output, and status
or error messages go to a separate status stream so they never pollute the
transcript.
"""

from __future__ import annotations

import shutil
import sys
import unicodedata
from dataclasses import dataclass
from typing import Optional, TextIO

CLEAR_LINE = "\r\x1b[2K"


@dataclass(frozen=True)
class RenderEvent:
    """One renderer instruction produced by the manager."""

    kind: str  # status | error | partial | final | clear_partial
    utterance_id: int = 0
    text: str = ""
    latency_ms: Optional[float] = None
    replace_partial: bool = True


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    category = unicodedata.category(char)
    if category in ("Cc", "Cf", "Cs"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Terminal cell width of ``text`` (CJK/emoji wide, combining marks zero)."""
    return sum(_char_width(char) for char in text)


def fit_line(text: str, width: int) -> str:
    """Fit one physical line, keeping the newest text when it is too long."""
    text = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    target = max(0, width - 1)  # reserve one cell for the ellipsis
    kept: list[str] = []
    used = 0
    for char in reversed(text):
        char_width = _char_width(char)
        if used + char_width > target:
            break
        kept.append(char)
        used += char_width
    return "…" + "".join(reversed(kept))


class TerminalRenderer:
    """Single-writer renderer; the manager owns it from one thread."""

    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        status_stream: Optional[TextIO] = None,
        is_tty: Optional[bool] = None,
        width: Optional[int] = None,
        show_partials: bool = False,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._status_stream = (
            status_stream if status_stream is not None else sys.stderr
        )
        self._is_tty = (
            bool(getattr(self._stream, "isatty", lambda: False)())
            if is_tty is None
            else is_tty
        )
        self._width = width
        self._show_partials = show_partials
        self._current_uid: Optional[int] = None
        self._partial_text = ""
        self._partial_active = False
        self.confirmed: list[str] = []

    @property
    def is_tty(self) -> bool:
        return self._is_tty

    @property
    def partial_text(self) -> str:
        return self._partial_text if self._partial_active else ""

    # ------------------------------------------------------------------ public
    def render(self, event: RenderEvent) -> None:
        if event.kind == "partial":
            self.partial(event.utterance_id, event.text)
        elif event.kind == "final":
            self.final(
                event.utterance_id,
                event.text,
                replace_partial=event.replace_partial,
            )
        elif event.kind == "clear_partial":
            self.clear_partial(event.utterance_id)
        elif event.kind == "error":
            self.status(event.text, is_error=True)
        else:
            self.status(event.text, is_error=False)

    def status(self, text: str, *, is_error: bool = False) -> None:
        text = str(text or "").strip()
        if not text:
            return
        prefix = "[error]" if is_error else "[status]"
        if self._is_tty and self._partial_active:
            self._write_stream(CLEAR_LINE)
            self._write_status(f"{prefix} {text}\n")
            self._write_stream(self._fit(self._partial_text))
        else:
            self._write_status(f"{prefix} {text}\n")

    def partial(self, utterance_id: int, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        if not self._is_tty:
            if self._show_partials:
                self._write_status(f"[partial] {text}\n")
            return
        if self._partial_active and utterance_id < (self._current_uid or 0):
            # A late partial from an older utterance must never overwrite the
            # newer utterance's provisional text.
            return
        self._current_uid = utterance_id
        self._partial_text = text
        self._write_stream(CLEAR_LINE + self._fit(text))
        self._partial_active = True

    def final(
        self,
        utterance_id: int,
        text: str,
        *,
        replace_partial: bool = True,
    ) -> None:
        text = str(text or "").strip()
        if not text:
            return
        if not self._is_tty:
            self._stream.write(text + "\n")
            self._stream.flush()
            self.confirmed.append(text)
            return

        pending: Optional[tuple[int, str]] = None
        if self._partial_active:
            pending = (self._current_uid or 0, self._partial_text)
            self._write_stream(CLEAR_LINE)
            self._partial_active = False
        # A final transcript is never truncated: the terminal may soft-wrap
        # it, but every character is written.
        self._write_stream(text + "\n")
        self.confirmed.append(text)
        if pending is not None and (pending[0] > utterance_id or not replace_partial):
            # A newer sentence/utterance's provisional line was on screen;
            # put it back below the confirmed line.
            self._current_uid, self._partial_text = pending
            self._write_stream(self._fit(self._partial_text))
            self._partial_active = True
        elif pending is not None:
            self._current_uid = None
            self._partial_text = ""

    def clear_partial(self, utterance_id: int) -> None:
        if not self._partial_active:
            return
        if self._current_uid not in (None, utterance_id):
            return
        if self._is_tty:
            self._write_stream(CLEAR_LINE)
        self._partial_active = False
        self._current_uid = None
        self._partial_text = ""

    def close(self) -> None:
        if self._is_tty and self._partial_active:
            self._write_stream(CLEAR_LINE)
            self._partial_active = False
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001 - closed pipe
            pass
        try:
            self._status_stream.flush()
        except Exception:  # noqa: BLE001 - closed pipe
            pass

    # ----------------------------------------------------------------- helpers
    def _terminal_width(self) -> int:
        if self._width is not None:
            return max(1, self._width)
        try:
            return max(
                1, shutil.get_terminal_size(fallback=(80, 24)).columns
            )
        except Exception:  # noqa: BLE001 - unusual stream
            return 80

    def _fit(self, text: str) -> str:
        return fit_line(text, self._terminal_width())

    def _write_stream(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def _write_status(self, text: str) -> None:
        self._status_stream.write(text)
        self._status_stream.flush()


__all__ = [
    "CLEAR_LINE",
    "RenderEvent",
    "TerminalRenderer",
    "display_width",
    "fit_line",
]
