"""Per-event runtime context (V0.6).

This package turns one input event into a small, trusted snapshot of *when*
and *where* it arrived, plus the temporal gaps since the last contact. The
snapshot is data for the generation prompt — never an instruction, never a
reply template, and never a replacement for the local policy decision.

Modules:

- :mod:`timezone` resolves an IANA time zone from trusted configuration;
- :mod:`connection` extracts a public client IP from real connection metadata
  (never from the model or the message text);
- :mod:`geolocation` optionally resolves a city-level location from a local
  MaxMind database;
- :mod:`temporal` computes Last Heard / Last Interaction / Last Meaningful
  Context gaps and the re-entry state;
- :mod:`snapshot` assembles and persists one immutable snapshot per event;
- :mod:`assembler` compresses the snapshot into a low-token prompt block.
"""
