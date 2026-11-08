"""Shared exception hierarchy for the optional TTS output.

Kept in its own module so the client, player, config and manager all agree on
one ``TTSError`` base. Messages are display-safe: they never include the API
key, raw audio or the reply text.
"""

from __future__ import annotations


class TTSError(Exception):
    """Base TTS failure."""


class TTSClientError(TTSError):
    """DashScope request, response or transport failure."""


class TTSEmptyAudioError(TTSError):
    """The provider completed without producing any audio."""


class TTSPlayerError(TTSError):
    """Audio device / PyAudio failure."""


__all__ = [
    "TTSClientError",
    "TTSEmptyAudioError",
    "TTSError",
    "TTSPlayerError",
]
