"""Trusted-side handling of model-proposed memory candidates.

The model may repeat itself: a single analysis can contain two candidates whose
content differs only in whitespace or letter case, and a retried request can
propose the same statement again. Candidate rows are *not* long-term memory —
they are inert proposals that only a later, human-confirmed step could ever
promote — but duplicated rows still make the review queue noisy.

Deduplication is therefore done here, on the trusted side, using a normalised
form of the content. This module never talks to the model and never promotes a
candidate to ``ACTIVE``.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def normalize_candidate_content(text: str) -> str:
    """Collapse whitespace and case so trivial variants compare equal."""
    return " ".join(str(text).split()).strip().casefold()


def dedupe_candidates(candidates: Iterable[T]) -> list[T]:
    """Drop blank and duplicate candidates, preserving first-seen order.

    Duplicates are compared by normalised ``content`` only; two proposals with
    the same text are the same proposal regardless of which generation produced
    them.
    """
    seen: set[str] = set()
    unique: list[T] = []
    for candidate in candidates:
        key = normalize_candidate_content(getattr(candidate, "content", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
