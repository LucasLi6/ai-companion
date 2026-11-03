"""Command-line interface.

Usage (see README for the canonical, tested commands):

    python -m companion init
    python -m companion person add --name Owner --role owner \
        --allow-processing --allow-persistence --allow-memory-learning
    python -m companion say --person <PERSON_ID> --to agent --text "你好"
    python -m companion inspect <EVENT_ID>
    python -m companion quiet on
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from companion.config import Config, configure_logging, default_db_path
from companion.domain.enums import AddresseeHint, EventSource
from companion.domain.models import make_utterance_event
from companion.runtime import build_runtime
from companion.storage.database import Database

DEFAULT_CONVERSATION = "default"


def _common_parser() -> argparse.ArgumentParser:
    """Global options accepted both before and after a subcommand."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--db", type=Path, default=argparse.SUPPRESS, help="SQLite database path")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Enable debug logging")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion",
        parents=[_common_parser()],
        description="AI Companion V0.0 — multi-person identity, event persistence, pluggable cognition.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", parents=[_common_parser()], help="Initialise the database schema").set_defaults(
        func=cmd_init
    )

    p_person = sub.add_parser("person", parents=[_common_parser()], help="Manage registered people")
    p_person_sub = p_person.add_subparsers(dest="person_command")
    p_add = p_person_sub.add_parser("add", parents=[_common_parser()], help="Register a person")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--role", choices=["owner", "friend", "guest"], default="guest")
    p_add.add_argument("--allow-processing", action="store_true", default=False)
    p_add.add_argument("--allow-persistence", action="store_true", default=False)
    p_add.add_argument("--allow-memory-learning", action="store_true", default=False)
    p_add.set_defaults(func=cmd_person_add)
    p_list = p_person_sub.add_parser("list", parents=[_common_parser()], help="List people")
    p_list.set_defaults(func=cmd_person_list)
    p_show = p_person_sub.add_parser("show", parents=[_common_parser()], help="Show one person")
    p_show.add_argument("person_id")
    p_show.set_defaults(func=cmd_person_show)

    p_say = sub.add_parser("say", parents=[_common_parser()], help="Submit one utterance")
    p_say.add_argument("--person", required=True, help="Registered person id of the speaker")
    p_say.add_argument(
        "--to", choices=["agent", "human", "unknown"], default="agent",
        help="Addressee hint (who the utterance is aimed at)",
    )
    p_say.add_argument("--text", required=True)
    p_say.add_argument("--conversation", default=DEFAULT_CONVERSATION)
    p_say.set_defaults(func=cmd_say)

    p_inspect = sub.add_parser("inspect", parents=[_common_parser()], help="Show a full processing trace for an event")
    p_inspect.add_argument("event_id")
    p_inspect.set_defaults(func=cmd_inspect)

    p_history = sub.add_parser("history", parents=[_common_parser()], help="List events")
    p_history.add_argument("--person", default=None)
    p_history.add_argument("--limit", type=int, default=50)
    p_history.set_defaults(func=cmd_history)

    p_quiet = sub.add_parser("quiet", parents=[_common_parser()], help="Toggle quiet mode for a conversation")
    p_quiet.add_argument("state", choices=["on", "off"])
    p_quiet.add_argument("--conversation", default=DEFAULT_CONVERSATION)
    p_quiet.set_defaults(func=cmd_quiet)

    p_jobs = sub.add_parser("jobs", parents=[_common_parser()], help="Inspect background jobs")
    p_jobs_sub = p_jobs.add_subparsers(dest="jobs_command")
    p_jl = p_jobs_sub.add_parser("list", parents=[_common_parser()], help="List jobs")
    p_jl.add_argument("--status", choices=["PENDING", "RUNNING", "COMPLETED", "FAILED"], default=None)
    p_jl.set_defaults(func=cmd_jobs_list)
    p_jr = p_jobs_sub.add_parser("retry", parents=[_common_parser()], help="Requeue a failed job")
    p_jr.add_argument("job_id")
    p_jr.set_defaults(func=cmd_jobs_retry)

    sub.add_parser("demo", parents=[_common_parser()], help="Run a scripted demo on an isolated database").set_defaults(
        func=cmd_demo
    )
    p_repl = sub.add_parser("repl", parents=[_common_parser()], help="Start an interactive text session")
    p_repl.add_argument("--conversation", default=DEFAULT_CONVERSATION)
    p_repl.set_defaults(func=cmd_repl)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "db"):
        args.db = default_db_path()
    if not hasattr(args, "debug"):
        args.debug = False
    configure_logging(args.debug)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return asyncio.run(args.func(args))


async def _connect(db_path: Path, initialize: bool = True) -> Database:
    if initialize:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await Database(db_path).connect()
    if initialize:
        await db.initialize()
    return db


# -- commands -------------------------------------------------------------


async def cmd_init(args) -> int:
    db = await _connect(args.db)
    try:
        print(f"Database initialised at {args.db}")
    finally:
        await db.close()
    return 0


async def cmd_person_add(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        person = await rt.registry.register(
            args.name,
            args.role,
            allow_processing=args.allow_processing,
            allow_persistence=args.allow_persistence,
            allow_memory_learning=args.allow_memory_learning,
        )
        print(person.person_id)
    finally:
        await db.close()
    return 0


async def cmd_person_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        for p in await rt.registry.list():
            c = await rt.registry.get_consent(p.person_id)
            flags = (
                f"processing={int(c.allow_processing)}"
                f" persistence={int(c.allow_persistence)}"
                f" learning={int(c.allow_memory_learning)}"
                if c
                else "no consent"
            )
            print(f"{p.person_id}  {p.display_name}  role={p.role.value}  {flags}")
    finally:
        await db.close()
    return 0


async def cmd_person_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        p = await rt.registry.get(args.person_id)
        if p is None:
            print(f"person not found: {args.person_id}")
            return 1
        c = await rt.registry.get_consent(p.person_id)
        print(f"person_id: {p.person_id}")
        print(f"name: {p.display_name}")
        print(f"role: {p.role.value}")
        print(f"registration_status: {p.registration_status.value}")
        print(f"created_at: {p.created_at.isoformat()}")
        if c:
            print(f"allow_processing: {c.allow_processing}")
            print(f"allow_persistence: {c.allow_persistence}")
            print(f"allow_memory_learning: {c.allow_memory_learning}")
    finally:
        await db.close()
    return 0


async def cmd_say(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        event = make_utterance_event(
            conversation_id=args.conversation,
            text=args.text,
            addressee_hint=args.to,
            speaker_person_id=args.person,
            source=EventSource.CLI,
        )
        report = await rt.orchestrator.process_utterance(event)
        for step in report.steps:
            print(step)
        # Drain the just-queued memory job so inspect shows it completed.
        if rt.worker is not None:
            await rt.worker.run_once()
    finally:
        await db.close()
    return 0


async def cmd_inspect(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        event = await rt.repos.utterance.get(args.event_id)
        if event is None:
            print(f"event not found: {args.event_id}")
            return 1
        print(f"== event {event.event_id} ==")
        print(f"text: {event.text}")
        print(f"speaker_person_id: {event.speaker_person_id}")
        if event.speaker_person_id:
            p = await rt.registry.get(event.speaker_person_id)
            if p:
                print(f"speaker: {p.display_name} (role={p.role.value})")
        print(f"addressee_hint: {event.addressee_hint.value}")
        print(f"source: {event.source.value}")
        print(f"conversation_id: {event.conversation_id}")
        print(f"processing_status: {event.processing_status.value}")
        print(f"occurred_at: {event.occurred_at.isoformat()}")

        context = await rt.repos.conversation.get(event.conversation_id)
        print(f"quiet_mode: {context.quiet_mode if context else False}")

        analysis = await rt.repos.analysis.get_analysis(args.event_id)
        if analysis:
            print(f"-- analysis --")
            print(f"provider: {analysis.provider}")
            print(f"suggestion: {analysis.action.value}")
            print(f"reason_code: {analysis.reason_code.value}")
            print(f"meaning: {analysis.meaning}")

        action = await rt.repos.analysis.get_action(args.event_id)
        if action:
            print(f"-- action --")
            print(f"decision: {action.action_type.value}")
            print(f"reason_code: {action.reason_code.value}")
            print(f"response_text: {action.response_text or '(none)'}")

        candidates = await rt.repos.analysis.get_candidates(args.event_id)
        print(f"-- memory candidates: {len(candidates)} --")
        for c in candidates:
            print(f"  {c.candidate_id} op={c.operation.value} status={c.status.value}")

        job = await rt.repos.job.get_by_key(f"memory:{args.event_id}")
        if job:
            print(f"-- memory job: {job.job_id} status={job.status.value} attempts={job.attempts}")
        else:
            print("-- memory job: none")
    finally:
        await db.close()
    return 0


async def cmd_history(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        events = await rt.repos.utterance.list_events(args.person, args.limit)
        for e in events:
            action = await rt.repos.analysis.get_action(e.event_id)
            decision = action.action_type.value if action else e.processing_status.value
            print(
                f"{e.event_id}  {e.speaker_person_id or '-'}  to={e.addressee_hint.value}"
                f"  {decision}  {e.text}"
            )
    finally:
        await db.close()
    return 0


async def cmd_quiet(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        enabled = args.state == "on"
        await rt.conversations.set_quiet_mode(args.conversation, enabled)
        print(f"quiet_mode={'on' if enabled else 'off'} conversation={args.conversation}")
    finally:
        await db.close()
    return 0


async def cmd_jobs_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        status = None
        if args.status:
            from companion.domain.enums import JobStatus

            status = JobStatus(args.status)
        for j in await rt.repos.job.list_jobs(status):
            print(
                f"{j.job_id}  {j.job_type.value}  {j.status.value}"
                f"  attempts={j.attempts}/{j.max_attempts}"
                f"  key={j.job_key}"
                + (f"  error={j.last_error}" if j.last_error else "")
            )
    finally:
        await db.close()
    return 0


async def cmd_jobs_retry(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        ok = await rt.repos.job.retry(args.job_id)
        if not ok:
            print(f"job not found: {args.job_id}")
            return 1
        await rt.worker.run_once()
        j = await rt.repos.job.get(args.job_id)
        print(f"job {j.job_id} status={j.status.value}")
    finally:
        await db.close()
    return 0


async def cmd_demo(args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        db = await _connect(db_path)
        try:
            rt = build_runtime(db)
            owner = await rt.registry.register(
                "宿主", "owner",
                allow_processing=True, allow_persistence=True, allow_memory_learning=True,
            )
            friend = await rt.registry.register(
                "朋友", "friend",
                allow_processing=True, allow_persistence=True, allow_memory_learning=True,
            )
            print(f"registered owner={owner.person_id} friend={friend.person_id}")
            print()

            async def say(person, to, text):
                event = make_utterance_event(
                    conversation_id=DEFAULT_CONVERSATION,
                    text=text,
                    addressee_hint=to,
                    speaker_person_id=person,
                    source=EventSource.CLI,
                )
                report = await rt.orchestrator.process_utterance(event)
                for step in report.steps:
                    print(step)
                if rt.worker is not None:
                    await rt.worker.run_once()
                print()

            print("--- owner -> agent (expect RESPOND) ---")
            await say(owner.person_id, "agent", "你好，今天怎么样？")
            print("--- owner -> human (expect IGNORE) ---")
            await say(owner.person_id, "human", "我昨天完成了一个项目。")
            print("--- owner -> unknown (expect WAIT) ---")
            await say(owner.person_id, "unknown", "自言自语。")
            print("--- quiet on, then owner -> agent (expect WAIT, no response) ---")
            await rt.conversations.set_quiet_mode(DEFAULT_CONVERSATION, True)
            await say(owner.person_id, "agent", "安静模式下说话。")
            print("--- quiet off, then friend -> agent (expect RESPOND) ---")
            await rt.conversations.set_quiet_mode(DEFAULT_CONVERSATION, False)
            await say(friend.person_id, "agent", "机器人你好。")
        finally:
            await db.close()
    return 0


async def cmd_repl(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        person = await rt.registry.register(
            "repl-user", "owner",
            allow_processing=True, allow_persistence=True, allow_memory_learning=True,
        )
        addressee = "agent"
        print("AI Companion REPL. Type /help for commands.")
        print(f"speaking as {person.person_id}")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                parts = line.split()
                cmd = parts[0].lower()
                if cmd == "/quit" or cmd == "/exit":
                    break
                elif cmd == "/quiet" and len(parts) == 2:
                    await rt.conversations.set_quiet_mode(
                        args.conversation, parts[1] == "on"
                    )
                    print(f"quiet_mode={'on' if parts[1] == 'on' else 'off'}")
                elif cmd == "/to" and len(parts) == 2 and parts[1] in ("agent", "human", "unknown"):
                    addressee = parts[1]
                    print(f"addressee -> {addressee}")
                elif cmd == "/history":
                    for e in await rt.repos.utterance.list_events(None, 20):
                        print(f"  {e.event_id} to={e.addressee_hint.value} {e.text}")
                elif cmd == "/help":
                    print("commands: /to agent|human|unknown  /quiet on|off  /history  /quit")
                else:
                    print("unknown command (see /help)")
                continue
            event = make_utterance_event(
                conversation_id=args.conversation,
                text=line,
                addressee_hint=addressee,
                speaker_person_id=person.person_id,
                source=EventSource.CLI,
            )
            report = await rt.orchestrator.process_utterance(event)
            for step in report.steps:
                print(step)
            if rt.worker is not None:
                await rt.worker.run_once()
    finally:
        await db.close()
    return 0
