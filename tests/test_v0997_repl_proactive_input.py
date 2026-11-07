"""V0.9.9.7: REPL input survives proactive messages.

The old REPL read ``readline.get_line_buffer()`` from the event loop while
``input()`` was blocked, then reprinted ``you> `` plus that buffer after a
proactive line. On a real terminal the buffer could still hold the *previous*
submitted line, so the old line reappeared under the new prompt.

The fix gives the input line one owner: a prompt_toolkit session on a TTY (or
one ``input()`` thread off-TTY). Proactive output is printed through that owner
and never reads or restores a buffer. These tests cover the queue/render logic
with pipe input and drive the real terminal path under a pseudo-terminal.
"""

from __future__ import annotations

import asyncio
import os
import pty
import select
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from companion.cli import (
    _build_prompt_session,
    _read_line_with_proactive,
    _render_proactive_event,
)
from companion.core.response_events import ResponseEvent

SRC = Path(__file__).resolve().parents[1] / "src"


class QueueManager:
    """Minimal stand-in for the proactive manager's event queue."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    async def next_event(self):
        return await self._queue.get()

    def push(self, text: str, *, type_: str = "response_completed") -> None:
        self._queue.put_nowait(
            ResponseEvent(
                type=type_,
                event_id="evt_proactive",
                conversation_id="c1",
                response_id="rsp_1",
                sequence=0,
                status="COMPLETED",
                text=text,
            )
        )


def _view(manager):
    return SimpleNamespace(proactive_manager=manager)


async def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


def _session(pipe):
    return PromptSession(input=pipe, output=DummyOutput())


# -- prompt_toolkit logic (deterministic, no terminal) -----------------------


@pytest.mark.asyncio
async def test_proactive_message_does_not_lose_text_typed_so_far(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr("companion.cli._print_proactive_line", rendered.append)

    with create_pipe_input() as pipe:
        session = _session(pipe)
        manager = QueueManager()
        manager.push("先喝口水吧")
        pipe.send_text("par")

        task = asyncio.create_task(
            _read_line_with_proactive(_view(manager), session)
        )
        await _wait_for(lambda: rendered)
        assert rendered == ["先喝口水吧"]

        pipe.send_text("tial\r")
        assert await asyncio.wait_for(task, 5) == "partial"
        assert rendered == ["先喝口水吧"]


@pytest.mark.asyncio
async def test_each_proactive_message_is_drawn_once_and_in_order(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr("companion.cli._print_proactive_line", rendered.append)

    with create_pipe_input() as pipe:
        session = _session(pipe)
        manager = QueueManager()
        manager.push("Immediate：先喝口水")
        manager.push("LONG_IDLE：早点休息")

        task = asyncio.create_task(
            _read_line_with_proactive(_view(manager), session)
        )
        await _wait_for(lambda: len(rendered) == 2)
        pipe.send_text("ok\r")

        assert await asyncio.wait_for(task, 5) == "ok"
        assert rendered == ["Immediate：先喝口水", "LONG_IDLE：早点休息"]


@pytest.mark.asyncio
async def test_submitted_input_wins_a_simultaneous_completion(monkeypatch):
    """If input and a proactive event complete together, input is returned.

    The event belongs to the turn being submitted, so drawing it after the
    input line would show a message the caller is about to invalidate.
    """
    rendered: list[str] = []
    monkeypatch.setattr("companion.cli._print_proactive_line", rendered.append)

    class ImmediateSession:
        app = SimpleNamespace(is_running=False)

        async def prompt_async(self, message):
            return "hello"

    class ImmediateManager:
        async def next_event(self):
            return ResponseEvent(
                type="response_completed",
                event_id="evt_proactive",
                conversation_id="c1",
                response_id="rsp_1",
                sequence=0,
                status="COMPLETED",
                text="过期消息",
            )

    line = await _read_line_with_proactive(
        _view(ImmediateManager()), ImmediateSession()
    )
    assert line == "hello"
    assert rendered == []


@pytest.mark.asyncio
async def test_non_response_events_are_not_printed(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr("companion.cli._print_proactive_line", rendered.append)

    manager = QueueManager()
    event = ResponseEvent(
        type="response_delta",
        event_id="evt_proactive",
        conversation_id="c1",
        response_id="rsp_1",
        sequence=0,
        status="STREAMING",
        text="partial",
    )
    await _render_proactive_event(event, None)
    await _render_proactive_event(
        ResponseEvent(
            type="response_completed",
            event_id="evt_proactive",
            conversation_id="c1",
            response_id="rsp_2",
            sequence=1,
            status="COMPLETED",
            text="   ",
        ),
        None,
    )
    assert rendered == []


@pytest.mark.asyncio
async def test_off_tty_falls_back_to_a_single_plain_read(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    assert await _read_line_with_proactive(SimpleNamespace()) == "hello"
    # A non-TTY never builds an interactive session.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _build_prompt_session() is None


# -- real terminal (pseudo-terminal) ----------------------------------------


HARNESS = textwrap.dedent(
    """
    import asyncio, sys
    from types import SimpleNamespace

    from companion.cli import _build_prompt_session, _read_line_with_proactive
    from companion.core.response_events import ResponseEvent
    from companion.domain.models import make_utterance_event
    from companion.runtime import build_runtime
    from companion.storage.database import Database


    class Manager:
        def __init__(self):
            self._queue = asyncio.Queue()

        async def next_event(self):
            return await self._queue.get()

        def push(self, text):
            self._queue.put_nowait(
                ResponseEvent(
                    type="response_completed",
                    event_id="evt_proactive",
                    conversation_id="c1",
                    response_id="rsp_1",
                    sequence=0,
                    status="COMPLETED",
                    text=text,
                )
            )


    async def save(rt, person, text):
        event = make_utterance_event(
            conversation_id="c1",
            text=text,
            addressee_hint="agent",
            speaker_person_id=person.person_id,
        )
        await rt.repos.utterance.save(event)


    async def main(db_path):
        db = await Database(db_path).connect()
        await db.initialize()
        rt = build_runtime(db, with_worker=False)
        person = await rt.registry.register(
            "Alice",
            "owner",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=True,
        )
        await rt.conversations.get_or_create("c1")

        manager = Manager()
        view = SimpleNamespace(proactive_manager=manager)
        session = _build_prompt_session()
        if session is None:
            print("NO_SESSION", flush=True)
            return 2

        first = (await _read_line_with_proactive(view, session)).strip()
        await save(rt, person, first)
        print("FIRST:" + first, flush=True)

        # Immediate: already delivered before the next prompt is read.
        manager.push("Immediate-消息")
        # LONG_IDLE: delivered while the person is halfway through typing.
        asyncio.get_running_loop().call_later(
            1.5, manager.push, "LongIdle-消息"
        )

        second = (await _read_line_with_proactive(view, session)).strip()
        await save(rt, person, second)
        print("SECOND:" + second, flush=True)

        await db.close()
        print("DONE", flush=True)
        return 0


    sys.exit(asyncio.run(main(sys.argv[1])))
    """
)


class Terminal:
    def __init__(self, master: int, process: subprocess.Popen):
        self.master = master
        self.process = process
        self.buffer = b""
        self.cursor = 0

    def expect(self, marker: str, timeout: float = 15.0) -> str:
        data = marker.encode()
        deadline = time.monotonic() + timeout
        while data not in self.buffer[self.cursor :]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"timed out waiting for {marker!r}; output so far:\n"
                    + self.text(self.cursor)
                )
            ready, _, _ = select.select(
                [self.master], [], [], min(0.05, remaining)
            )
            if not ready:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                code = self.process.poll()
                raise AssertionError(
                    f"terminal closed while waiting for {marker!r} "
                    f"(exit={code}); output so far:\n" + self.text(self.cursor)
                )
            self.buffer += chunk
        found = self.buffer.index(data, self.cursor) + len(data)
        self.cursor = found
        return self.text(found)

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())

    def text(self, start: int = 0) -> str:
        return self.buffer[start:].decode("utf-8", "replace")

    def close(self) -> None:
        try:
            os.close(self.master)
        except OSError:
            pass


@pytest.fixture
def terminal(tmp_path):
    harness = tmp_path / "repl_harness.py"
    harness.write_text(HARNESS)
    db_path = tmp_path / "companion.db"

    master, slave = pty.openpty()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = "120"
    env["LINES"] = "40"
    process = subprocess.Popen(
        [sys.executable, str(harness), str(db_path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    term = Terminal(master, process)
    try:
        yield term, db_path
    finally:
        term.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_repl_keeps_partial_input_and_never_revives_old_input(terminal):
    term, db_path = terminal

    term.expect("you>")
    term.send("old-input\n")
    term.expect("FIRST:old-input")

    # Immediate message is shown before the person types anything.
    term.expect("assistant> Immediate-消息")

    # Type a new line in two parts around the LONG_IDLE delivery.
    term.send("new")
    term.expect("assistant> LongIdle-消息")
    marker = len(term.buffer)
    term.send("-text\n")

    term.expect("SECOND:new-text")
    term.expect("DONE")

    output = term.text()
    # The previous submission is never reprinted: nothing after the second
    # proactive line brings ``old-input`` back into the current prompt.
    after_proactive = term.text(marker)
    assert "old-input" not in after_proactive
    # The first submission is echoed once, when it is typed; no later prompt
    # restores it into the current input line.
    assert output.count("you> old-input") == 1
    # Both proactive messages were drawn exactly once.
    assert output.count("assistant> Immediate-消息") == 1
    assert output.count("assistant> LongIdle-消息") == 1

    assert term.process.wait(timeout=15) == 0

    # One Enter produced exactly one user event, in order, with no duplicate.
    with sqlite3.connect(db_path) as con:
        rows = [row[0] for row in con.execute(
            "SELECT text FROM utterances ORDER BY rowid"
        )]
    assert rows == ["old-input", "new-text"]
