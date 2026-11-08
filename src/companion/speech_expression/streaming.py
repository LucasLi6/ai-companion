"""One streaming adapter: raw token -> VOICE/TEXT parser -> two channels.

The Speaker protocol is line-oriented: the ``VOICE=`` header ends at the first
newline. Normalising a raw delta before parsing (the pre-V1.2.1 order) turns
that newline into a space, so the parser cannot tell where the header ends and
the first visible text can be delayed or misclassified.

This adapter fixes the order and keeps the two output channels final:

    raw token -> VoiceResponseParser
              -> canonical normaliser -> canonical delta (stored)
              -> display normaliser   -> display delta   (shown / future TTS)

Both normalisers are :class:`StreamingTextNormalizer`, so the concatenation of
the emitted deltas is exactly the normalised channel text. The display channel
keeps valid voice tags but normalises the prose between them.

The surface must not flash a bubble that only contains labels: while no real
canonical text has been confirmed, display pieces (leading tags) are held and
released together with the first prose delta. A tag-only reply therefore emits
nothing, which lets the existing empty-reply recovery decide what to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from companion.prompts.layout import (
    RESPONSE_LAYOUT_CHAT,
    StreamingTextNormalizer,
)

from .parser import VoiceResponseParser

__all__ = ["VoiceStreamAdapter", "VoiceStreamResult"]


@dataclass(frozen=True)
class VoiceStreamResult:
    """Already-normalised deltas produced by one ``feed()``/``finish()``."""

    text: str = ""
    display_text: str = ""


class VoiceStreamAdapter:
    """Drive :class:`VoiceResponseParser` and normalise both channels."""

    def __init__(self, layout: str = RESPONSE_LAYOUT_CHAT) -> None:
        self.layout = layout
        self.parser = VoiceResponseParser()
        self._canonical = StreamingTextNormalizer(layout)
        self._display = StreamingTextNormalizer(layout)
        self._canonical_parts: list[str] = []
        self._display_parts: list[str] = []
        self._held_display = ""
        self._first_text_emitted = False

    # -- parser diagnostics ------------------------------------------------

    @property
    def voice(self):
        return self.parser.voice

    @property
    def invalid_voice(self):
        return self.parser.invalid_voice

    @property
    def protocol_valid(self) -> bool:
        return self.parser.protocol_valid

    @property
    def protocol_error(self):
        return self.parser.protocol_error

    @property
    def invalid_tag_count(self) -> int:
        return self.parser.invalid_tag_count

    @property
    def empty_markup_count(self) -> int:
        return self.parser.empty_markup_count

    # -- confirmed channels ------------------------------------------------

    @property
    def text(self) -> str:
        """Canonical text emitted so far (safe to persist on cancel)."""
        return "".join(self._canonical_parts)

    @property
    def display_text(self) -> str:
        """Display text emitted so far (valid tags kept, header removed)."""
        return "".join(self._display_parts)

    @property
    def has_valid_text(self) -> bool:
        return bool(self.text and self.text.strip())

    @property
    def finished(self) -> bool:
        return self.parser.finished

    # -- streaming ---------------------------------------------------------

    def feed(self, raw: str) -> VoiceStreamResult:
        """Consume one raw provider delta (newlines intact)."""
        if not raw:
            return VoiceStreamResult()
        parsed = self.parser.feed(raw)
        return self._absorb(parsed.text, parsed.display_text)

    def finish(self) -> VoiceStreamResult:
        if self.parser.finished:
            return VoiceStreamResult()
        tail = self.parser.finish()
        result = self._absorb(tail.text, tail.display_text)
        # ``StreamingTextNormalizer.finish`` always returns "" (a trailing
        # whitespace run is trimmed by both layouts), but call it so the
        # adapter's contract matches the normaliser it delegates to.
        self._canonical.finish()
        self._display.finish()
        return result

    # -- internals ---------------------------------------------------------

    def _absorb(self, text: str, display_text: str) -> VoiceStreamResult:
        canonical = self._canonical.feed(text)
        display_piece = self._display.feed(display_text)

        if not self._first_text_emitted:
            self._held_display += display_piece
            if not canonical:
                # Only the header and/or leading labels have arrived. Publish
                # nothing yet: an empty bubble must not flash on the surface.
                return VoiceStreamResult()
            display_piece = self._held_display
            self._held_display = ""
            self._first_text_emitted = True
        elif not canonical and not display_piece:
            return VoiceStreamResult()

        self._canonical_parts.append(canonical)
        self._display_parts.append(display_piece)
        return VoiceStreamResult(text=canonical, display_text=display_piece)
