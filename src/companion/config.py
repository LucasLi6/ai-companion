"""Runtime configuration and logging setup.

All persisted timestamps are stored as timezone-aware UTC ISO-8601 strings;
durations and performance timings use ``time.monotonic()`` (see the
orchestrator and job worker, where durations are measured but never persisted
as wall-clock deltas).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_NAME = "companion.db"


@dataclass
class Config:
    """Runtime configuration for a single CLI / test invocation."""

    db_path: Path
    debug: bool = False


def default_db_path() -> Path:
    """Resolve the default database path.

    Honour ``COMPANION_DB`` when set, otherwise use ``data/companion.db``
    relative to the current working directory.
    """
    env = os.environ.get("COMPANION_DB")
    if env:
        return Path(env)
    return Path("data") / DEFAULT_DB_NAME


def configure_logging(debug: bool = False) -> None:
    """Configure process-wide structured logging.

    Messages carry ``key=value`` fields (``event_id``, ``conversation_id``,
    ``speaker_person_id``, ``provider``, ``processing_stage``, ``duration_ms``,
    ``status``, ``reason_code``) so a single event can be traced end-to-end.
    Raw utterance text is intentionally *not* copied into logs.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
