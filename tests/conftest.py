"""Shared async fixtures for the test suite."""

from __future__ import annotations

import socket

import pytest
import pytest_asyncio

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.runtime import build_runtime
from companion.storage.database import Database


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Every test is offline: an INET connect is a test bug, not a fixture.

    Local files (SQLite) do not use sockets, so this also catches a fake
    provider accidentally falling back to a real HTTP client.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("real network access is disabled in tests")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = await Database(tmp_path / "test.db").connect()
    await database.initialize()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def rt(db):
    yield build_runtime(db)


@pytest_asyncio.fixture
def register(rt):
    async def _register(name, role="friend", *, processing=True, persistence=True, learning=True, external_ai=False):
        return await rt.registry.register(
            name,
            role,
            allow_processing=processing,
            allow_persistence=persistence,
            allow_memory_learning=learning,
            allow_external_ai=external_ai,
        )

    return _register


@pytest_asyncio.fixture
def say(rt):
    async def _say(person_id, to, text, conversation="default"):
        event = make_utterance_event(
            conversation_id=conversation,
            text=text,
            addressee_hint=to,
            speaker_person_id=person_id,
            source=EventSource.TEST,
        )
        return await rt.orchestrator.process_utterance(event)

    return _say
