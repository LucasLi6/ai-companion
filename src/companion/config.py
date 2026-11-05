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
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_DB_NAME = "companion.db"
DEFAULT_LOG_NAME = "companion.log"


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


def default_log_path() -> Path:
    """Resolve the default rotating log file.

    Honour ``COMPANION_LOG`` when set, otherwise use ``logs/companion.log``
    relative to the current working directory.
    """
    env = os.environ.get("COMPANION_LOG")
    if env:
        return Path(env)
    return Path("logs") / DEFAULT_LOG_NAME


def configure_logging(debug: bool = False, *, log_file: Path | None = None) -> None:
    """Configure process-wide structured logging without stealing the console.

    Run-time logs go to a rotating file (``logs/companion.log`` by default) so
    streaming replies and user input are never interleaved with ``INFO``
    diagnostics. The terminal stays a pure conversation surface: nothing is
    written to ``stdout``/``stderr`` unless ``debug`` is set, in which case a
    short ``stderr`` handler is added for local troubleshooting.

    Messages still carry ``key=value`` fields (``event_id``, ``conversation_id``,
    ``speaker_person_id``, ``provider``, ``processing_stage``, ``duration_ms``,
    ``status``, ``reason_code``) so a single event can be traced end-to-end.
    Raw utterance text, API keys, raw IP addresses and private content are
    intentionally never written to logs.

    ``httpx``/``httpcore`` request logs are suppressed so provider transport
    chatter never reaches the conversation surface, even in debug mode.
    """
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any handler configured by a previous call (idempotent in tests).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    path = Path(log_file) if log_file is not None else default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(file_handler)

    if debug:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root.addHandler(stream_handler)

    # Third-party transports are implementation detail, never conversation UI.
    for noisy in ("httpx", "httpcore", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
