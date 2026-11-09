"""Terminal rendering for live ASR.

Only confirmed text becomes a transcript line. Provisional ASR text is shown
on the last line and replaced in place with ``\\r\\x1b[2K``; it is never
treated as final. A non-TTY stream falls back to plain line output, and status
or error messages go to a separate status stream so they never pollute the
transcript.

When speaker identification is enabled, an identity line can sit directly
below the live partial. Both lines are owned by this single renderer thread;
the voice-print worker only posts events. Clearing therefore moves the cursor
up over the whole block, never over a confirmed transcript line belonging to
another utterance.
"""

from __future__ import annotations

import shutil
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, TextIO

CLEAR_LINE = "\r\x1b[2K"
CURSOR_UP_CLEAR = "\x1b[1A\r\x1b[2K"


@dataclass(frozen=True)
class RenderEvent:
    """One renderer instruction produced by the manager."""

    kind: str  # status | error | partial | final | clear_partial | speaker_status
    utterance_id: int = 0
    text: str = ""
    latency_ms: Optional[float] = None
    replace_partial: bool = True
    # An immutable identity snapshot (companion.speaker_id.models.SpeakerResult)
    # or None when speaker identification is disabled. Kept as ``Any`` so the
    # ASR renderer never imports the optional speaker package eagerly.
    speaker_result: Any = None


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


def _speaker_status_tag(result: Any) -> str:
    from companion.speaker_id.models import status_tag

    return status_tag(result)


def _speaker_final_tag(result: Any) -> str:
    from companion.speaker_id.models import final_tag

    return final_tag(result)


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
        # Identity line shown under the live partial. ``_drawn_lines`` is the
        # number of physical lines the renderer last wrote without a trailing
        # newline, so a redraw can erase exactly its own block.
        self._speaker_uid: Optional[int] = None
        self._speaker_text = ""
        self._speaker_active = False
        self._drawn_lines = 0
        self._last_final_uid: Optional[int] = None
        self.confirmed: list[str] = []

    @property
    def is_tty(self) -> bool:
        return self._is_tty

    @property
    def partial_text(self) -> str:
        return self._partial_text if self._partial_active else ""

    @property
    def speaker_text(self) -> str:
        return self._speaker_text if self._speaker_active else ""

    # ------------------------------------------------------------------ public
    def render(self, event: RenderEvent) -> None:
        if event.kind == "partial":
            self.partial(event.utterance_id, event.text)
        elif event.kind == "final":
            self.final(
                event.utterance_id,
                event.text,
                replace_partial=event.replace_partial,
                speaker_result=event.speaker_result,
            )
        elif event.kind == "clear_partial":
            self.clear_partial(event.utterance_id)
        elif event.kind == "speaker_status":
            self.speaker_status(event.utterance_id, event.speaker_result)
        elif event.kind == "error":
            self.status(event.text, is_error=True)
        else:
            self.status(event.text, is_error=False)

    def status(self, text: str, *, is_error: bool = False) -> None:
        text = str(text or "").strip()
        if not text:
            return
        prefix = "[error]" if is_error else "[status]"
        if self._is_tty and (self._partial_active or self._speaker_active):
            self._erase_block()
            self._write_status(f"{prefix} {text}\n")
            self._redraw()
        else:
            self._write_status(f"{prefix} {text}\n")

    def speaker_status(self, utterance_id: int, result: Any) -> None:
        """Show/update the identity line for the live utterance."""
        if result is None:
            return
        text = _speaker_status_tag(result)
        uid = int(utterance_id or 0)
        if not self._is_tty:
            self._write_status(f"[speaker] #{uid} {text}\n")
            return
        if self._partial_active and uid < (self._current_uid or 0):
            # A late identity update from an older utterance must not replace
            # the newer one's live status; print it as its own line instead.
            self._write_status(f"[speaker] #{uid} {text}\n")
            return
        if self._last_final_uid is not None and uid <= self._last_final_uid:
            # That transcript is already confirmed; never reprint it.
            self._write_status(f"[speaker] #{uid} {text}\n")
            return
        self._speaker_uid = uid
        self._speaker_text = text
        self._speaker_active = True
        self._redraw()

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
        if self._speaker_active and self._speaker_uid != utterance_id:
            # The identity line belongs to a previous utterance.
            self._speaker_active = False
            self._speaker_text = ""
            self._speaker_uid = None
        self._current_uid = utterance_id
        self._partial_text = text
        self._partial_active = True
        if self._speaker_active:
            self._redraw()
            return
        if self._drawn_lines > 1:
            self._erase_block()
        self._write_stream(CLEAR_LINE + self._fit(text))
        self._drawn_lines = 1

    def final(
        self,
        utterance_id: int,
        text: str,
        *,
        replace_partial: bool = True,
        speaker_result: Any = None,
    ) -> None:
        text = str(text or "").strip()
        if not text:
            return
        if not self._is_tty:
            self._stream.write(text + "\n")
            if speaker_result is not None:
                self._stream.write(_speaker_final_tag(speaker_result) + "\n")
            self._stream.flush()
            self.confirmed.append(text)
            return

        pending: Optional[tuple[int, str]] = None
        if self._partial_active:
            pending = (self._current_uid or 0, self._partial_text)
        pending_speaker: Optional[tuple[Optional[int], str]] = None
        if self._speaker_active:
            pending_speaker = (self._speaker_uid, self._speaker_text)
        if self._drawn_lines > 0:
            self._erase_block()
        elif self._partial_active:
            self._write_stream(CLEAR_LINE)
        self._partial_active = False
        self._speaker_active = False
        self._speaker_text = ""
        self._speaker_uid = None
        # A final transcript is never truncated: the terminal may soft-wrap
        # it, but every character is written.
        self._write_stream(text + "\n")
        if speaker_result is not None:
            self._write_stream(_speaker_final_tag(speaker_result) + "\n")
        self._last_final_uid = utterance_id
        self.confirmed.append(text)
        if pending is not None and (pending[0] > utterance_id or not replace_partial):
            # A newer sentence/utterance's provisional line was on screen;
            # put it back below the confirmed line.
            self._current_uid, self._partial_text = pending
            self._partial_active = True
            if (
                pending_speaker is not None
                and pending_speaker[0] == pending[0]
            ):
                self._speaker_uid, self._speaker_text = pending_speaker
                self._speaker_active = True
            self._redraw()
        elif pending is not None:
            self._current_uid = None
            self._partial_text = ""

    def clear_partial(self, utterance_id: int) -> None:
        if not self._partial_active and not self._speaker_active:
            return
        if self._partial_active and self._current_uid not in (None, utterance_id):
            return
        if self._speaker_active and self._speaker_uid not in (None, utterance_id):
            return
        if self._drawn_lines > 0:
            self._erase_block()
        elif self._is_tty:
            self._write_stream(CLEAR_LINE)
        self._partial_active = False
        self._current_uid = None
        self._partial_text = ""
        self._speaker_active = False
        self._speaker_text = ""
        self._speaker_uid = None

    def close(self) -> None:
        if self._is_tty and (self._partial_active or self._speaker_active):
            self._erase_block()
            self._partial_active = False
            self._speaker_active = False
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001 - closed pipe
            pass
        try:
            self._status_stream.flush()
        except Exception:  # noqa: BLE001 - closed pipe
            pass

    # ----------------------------------------------------------------- layout
    def _block_lines(self) -> list[str]:
        lines: list[str] = []
        if self._partial_active:
            lines.append(self._fit(self._partial_text))
        if self._speaker_active:
            lines.append(self._fit(self._speaker_text))
        return lines

    def _erase_block(self) -> None:
        if self._drawn_lines <= 0:
            return
        self._write_stream(
            CLEAR_LINE + CURSOR_UP_CLEAR * (self._drawn_lines - 1)
        )
        self._drawn_lines = 0

    def _redraw(self) -> None:
        self._erase_block()
        lines = self._block_lines()
        if not lines:
            return
        self._write_stream("\n".join(lines))
        self._drawn_lines = len(lines)

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
