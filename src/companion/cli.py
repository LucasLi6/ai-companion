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

from companion.config import configure_logging, default_db_path
from companion.core.policy import (
    INTERRUPTION_THRESHOLD_VALUES,
    RESPONSE_CONSERVATISM_VALUES,
    CURIOSITY_LEVEL_VALUES,
    SETTING_CURIOSITY_LEVEL,
    SETTING_INTERRUPTION_THRESHOLD,
    SETTING_RESPONSE_CONSERVATISM,
    load_policy_config,
)
from companion.domain.enums import (
    EventSource,
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
)
from companion.domain.models import make_utterance_event
from companion.memory.models import (
    AccessContext,
    MemoryNotFoundError,
    MemoryPermissionDeniedError,
    MemoryVersionConflictError,
)
from companion.providers.deepseek_client import (
    DeepSeekConfigError,
    DeepSeekError,
    build_deepseek_client,
    load_deepseek_config,
)
from companion.recovery import RecoveryError, RecoveryManager
from companion.runtime import build_runtime
from companion.storage.database import Database
from companion.storage.repositories import make_repositories

DEFAULT_CONVERSATION = "default"

# Memory-management commands act as the trusted local admin, not as a person.
ADMIN = AccessContext("admin", is_admin=True, role="cli")


def _add_provider_arg(parser: argparse.ArgumentParser) -> None:
    """Attach ``--provider`` to commands that process utterances.

    Default stays ``mock`` on purpose: no command can make a paid API call
    unless the user explicitly asks for ``deepseek`` (and has a key).
    """
    parser.add_argument(
        "--provider",
        choices=["mock", "deepseek"],
        default="mock",
        help="Cognition backend (default: mock; deepseek requires DEEPSEEK_API_KEY)",
    )


def _build_foreground_runtime(db, provider: str):
    """Build a runtime for utterance-processing commands.

    A missing key for ``--provider deepseek`` prints a clear error and returns
    ``None`` — it never falls back to mock and never makes a network call.
    """
    try:
        return build_runtime(db, provider=provider)
    except DeepSeekConfigError as exc:
        print(f"deepseek error: {exc}")
        return None


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
    p_add.add_argument(
        "--allow-external-ai",
        action="store_true",
        default=False,
        help="Permit this person's context to be sent to an external AI (needed for --provider deepseek)",
    )
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
    _add_provider_arg(p_say)
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

    p_memory = sub.add_parser("memory", parents=[_common_parser()], help="Manage the unified memory store")
    p_mem_sub = p_memory.add_subparsers(dest="memory_command")
    p_ma = p_mem_sub.add_parser("add", parents=[_common_parser()], help="Create a memory")
    p_ma.add_argument("--kind", choices=[k.value for k in MemoryKind], default="PERSONAL_FACT")
    p_ma.add_argument("--content", required=True)
    p_ma.add_argument("--status", choices=[s.value for s in MemoryStatus], default="ACTIVE")
    p_ma.add_argument("--visibility", choices=[v.value for v in MemoryVisibility], default="PRIVATE")
    p_ma.add_argument("--human-confirmed", action="store_true", default=False)
    p_ma.add_argument("--subject", action="append", default=None, metavar="TYPE:id",
                      help="Associate a subject (repeatable), e.g. PERSON:person_123")
    p_ma.set_defaults(func=cmd_memory_add)
    p_ms = p_mem_sub.add_parser("search", parents=[_common_parser()], help="Search memories")
    p_ms.add_argument("--query", default=None)
    p_ms.add_argument("--kind", choices=[k.value for k in MemoryKind], default=None)
    p_ms.add_argument("--status", choices=[s.value for s in MemoryStatus], default=None)
    p_ms.add_argument("--include-retracted", action="store_true", default=False)
    p_ms.add_argument("--limit", type=int, default=50)
    p_ms.set_defaults(func=cmd_memory_search)
    p_mv = p_mem_sub.add_parser("show", parents=[_common_parser()], help="Show one memory in full")
    p_mv.add_argument("memory_id")
    p_mv.set_defaults(func=cmd_memory_show)
    p_mc = p_mem_sub.add_parser("confirm", parents=[_common_parser()], help="Human-confirm a candidate memory")
    p_mc.add_argument("memory_id")
    p_mc.set_defaults(func=cmd_memory_confirm)
    p_mr = p_mem_sub.add_parser("retract", parents=[_common_parser()], help="Retract a memory (kept, excluded from search)")
    p_mr.add_argument("memory_id")
    p_mr.set_defaults(func=cmd_memory_retract)
    p_md = p_mem_sub.add_parser("delete", parents=[_common_parser()], help="Permanently delete a memory")
    p_md.add_argument("memory_id")
    p_md.set_defaults(func=cmd_memory_delete)
    p_ml = p_mem_sub.add_parser("link", parents=[_common_parser()], help="Link two memories")
    p_ml.add_argument("memory_id")
    p_ml.add_argument("other_id")
    p_ml.add_argument("--type", default="related_to")
    p_ml.set_defaults(func=cmd_memory_link)
    p_mas = p_mem_sub.add_parser(
        "assign", parents=[_common_parser()],
        help="Associate a memory with a person for retrieval",
    )
    p_mas.add_argument("memory_id")
    p_mas.add_argument("--person", required=True)
    p_mas.set_defaults(func=cmd_memory_assign)
    p_muas = p_mem_sub.add_parser(
        "unassign", parents=[_common_parser()],
        help="Remove a memory/person association",
    )
    p_muas.add_argument("memory_id")
    p_muas.add_argument("--person", required=True)
    p_muas.set_defaults(func=cmd_memory_unassign)
    p_mal = p_mem_sub.add_parser(
        "assignments", parents=[_common_parser()],
        help="List the people a memory is associated with",
    )
    p_mal.add_argument("memory_id")
    p_mal.set_defaults(func=cmd_memory_assignments)

    p_policy = sub.add_parser("policy", parents=[_common_parser()], help="Show or set behaviour policy")
    p_policy_sub = p_policy.add_subparsers(dest="policy_command")
    p_policy_show = p_policy_sub.add_parser("show", parents=[_common_parser()], help="Show policy settings")
    p_policy_show.set_defaults(func=cmd_policy_show)
    p_policy_set = p_policy_sub.add_parser("set", parents=[_common_parser()], help="Set one policy setting")
    p_policy_set.add_argument(
        "key",
        choices=[
            SETTING_RESPONSE_CONSERVATISM,
            SETTING_INTERRUPTION_THRESHOLD,
            SETTING_CURIOSITY_LEVEL,
        ],
    )
    p_policy_set.add_argument("value")
    p_policy_set.set_defaults(func=cmd_policy_set)

    p_ui = sub.add_parser("ui", parents=[_common_parser()], help="Start the management web UI (NiceGUI)")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8080)
    p_ui.set_defaults(func=cmd_ui)

    p_research = sub.add_parser("research", parents=[_common_parser()], help="Manage idle research tasks")
    p_res_sub = p_research.add_subparsers(dest="research_command")
    p_rl = p_res_sub.add_parser("list", parents=[_common_parser()], help="List research tasks")
    p_rl.add_argument("--status", choices=["PENDING", "RUNNING", "COMPLETED", "FAILED"], default=None)
    p_rl.set_defaults(func=cmd_research_list)
    p_rq = p_res_sub.add_parser("queue", parents=[_common_parser()], help="Queue a research question")
    p_rq.add_argument("--question", required=True)
    p_rq.add_argument("--priority", type=int, default=0)
    p_rq.add_argument("--source-kind", choices=["curiosity", "manual", "reflection"], default="manual")
    p_rq.set_defaults(func=cmd_research_queue)
    p_rr = p_res_sub.add_parser("run", parents=[_common_parser()], help="Execute pending research tasks")
    p_rr.add_argument("--limit", type=int, default=3)
    p_rr.set_defaults(func=cmd_research_run)

    p_idle = sub.add_parser("idle", parents=[_common_parser()], help="Run idle cognitive learning")
    p_idle_sub = p_idle.add_subparsers(dest="idle_command")
    p_ir = p_idle_sub.add_parser("run", parents=[_common_parser()], help="Run one idle cycle")
    p_ir.set_defaults(func=cmd_idle_run)

    p_demo = sub.add_parser("demo", parents=[_common_parser()], help="Run a scripted demo on an isolated database")
    _add_provider_arg(p_demo)
    p_demo.set_defaults(func=cmd_demo)
    p_repl = sub.add_parser("repl", parents=[_common_parser()], help="Start an interactive text session")
    p_repl.add_argument("--conversation", default=DEFAULT_CONVERSATION)
    p_repl.add_argument("--person", default=None, help="Registered person id of the speaker")
    _add_provider_arg(p_repl)
    p_repl.set_defaults(func=cmd_repl)

    p_recovery = sub.add_parser("recovery", parents=[_common_parser()], help="Recover unfinished events after a crash")
    p_recovery_sub = p_recovery.add_subparsers(dest="recovery_command")
    p_rec_list = p_recovery_sub.add_parser("list", parents=[_common_parser()], help="List unfinished events")
    p_rec_list.set_defaults(func=cmd_recovery_list)
    p_rec_retry = p_recovery_sub.add_parser("retry", parents=[_common_parser()], help="Safely resume one event")
    p_rec_retry.add_argument("event_id")
    _add_provider_arg(p_rec_retry)
    p_rec_retry.set_defaults(func=cmd_recovery_retry)

    p_ds = sub.add_parser("deepseek", parents=[_common_parser()], help="Inspect the DeepSeek configuration")
    p_ds_sub = p_ds.add_subparsers(dest="deepseek_command")
    p_ds_check = p_ds_sub.add_parser(
        "check",
        parents=[_common_parser()],
        help="Validate DeepSeek configuration offline (add --live for one real call)",
    )
    p_ds_check.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Make exactly one minimal API call to verify the key (costs tokens)",
    )
    p_ds_check.set_defaults(func=cmd_deepseek_check)

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
    # The UI server owns its own event loop; run it synchronously.
    if args.command == "ui":
        return args.func(args)
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
            allow_external_ai=args.allow_external_ai,
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
                f" external_ai={int(c.allow_external_ai)}"
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
            print(f"allow_external_ai: {c.allow_external_ai}")
    finally:
        await db.close()
    return 0


async def cmd_say(args) -> int:
    db = await _connect(args.db)
    try:
        rt = _build_foreground_runtime(db, getattr(args, "provider", "mock"))
        if rt is None:
            return 1
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
            print(f"intent: {analysis.intent}")
            print(f"addressed_to: {analysis.addressed_to or '-'}")
            print(f"silence_requested: {analysis.silence_requested}")
            print(f"reply_necessity: {analysis.reply_necessity.value}")
            print(f"conversation_state: {analysis.conversation_state}")

        action = await rt.repos.analysis.get_action(args.event_id)
        if action:
            print(f"-- action --")
            print(f"decision: {action.action_type.value}")
            print(f"reason_code: {action.reason_code.value}")
            print(
                "model_suggested_action: "
                f"{action.model_suggested_action.value if action.model_suggested_action else '-'}"
            )
            print(
                "policy_final_action: "
                f"{action.policy_final_action.value if action.policy_final_action else '-'}"
            )
            print(f"override_reason: {action.override_reason or '-'}")
            print(f"response_text: {action.response_text or '(none)'}")

        candidates = await rt.repos.analysis.get_candidates(args.event_id)
        writes = {
            w["candidate_id"]: w
            for w in await rt.repos.analysis.list_candidate_writes(args.event_id)
        }
        print(f"-- memory candidates: {len(candidates)} --")
        for c in candidates:
            write = writes.get(c.candidate_id, {})
            print(
                f"  {c.candidate_id} op={c.operation.value} status={c.status.value}"
                f" assertion={c.assertion_type.value} certainty={c.certainty_category.value}"
                f" topic={c.topic or '-'}"
                f" memory_write_outcome={write.get('outcome', '(none)')}"
                f" memory_id={write.get('memory_id') or '-'}"
            )

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
        existing = await rt.repos.job.get(args.job_id)
        if existing is None:
            print(f"job not found: {args.job_id}")
            return 1
        ok = await rt.repos.job.retry(args.job_id)
        if not ok:
            print(
                f"job {args.job_id} status={existing.status.value} is not retryable"
                " (only FAILED jobs can be retried)"
            )
            return 1
        await rt.worker.run_once()
        j = await rt.repos.job.get(args.job_id)
        print(f"job {j.job_id} status={j.status.value}")
    finally:
        await db.close()
    return 0


async def cmd_memory_add(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        subjects = []
        for spec in args.subject or []:
            if ":" in spec:
                subject_type, subject_id = spec.split(":", 1)
            else:
                subject_type, subject_id = "PERSON", spec
            subjects.append((subject_type, subject_id))
        item = await rt.memory_service.create(
            kind=args.kind,
            content=args.content,
            access=ADMIN,
            status=args.status,
            visibility=args.visibility,
            human_confirmed=args.human_confirmed,
            subjects=subjects,
        )
        print(item.memory_id)
    finally:
        await db.close()
    return 0


async def cmd_memory_search(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        results = await rt.memory_service.search(
            access=ADMIN,
            query=args.query,
            kinds=[args.kind] if args.kind else None,
            statuses=[args.status] if args.status else None,
            include_retracted=args.include_retracted,
            limit=args.limit,
        )
        for r in results:
            print(
                f"{r['memory_id']}  {r['kind']}  {r['status']}  {r['visibility']}"
                f"  {r['content'][:60]}"
            )
        if not results:
            print("no memories match")
    finally:
        await db.close()
    return 0


async def cmd_memory_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        view = await rt.memory_service.get_view(args.memory_id, ADMIN)
        if view is None:
            print(f"memory not found: {args.memory_id}")
            return 1
        item = view["item"]
        print(f"memory_id: {item['memory_id']}")
        print(f"kind: {item['kind']}")
        print(f"status: {item['status']}")
        print(f"visibility: {item['visibility']}")
        print(f"version: {item['version']}")
        print(f"created_by: {item['created_by']}")
        print(f"human_confirmed: {item['human_confirmed']}")
        print(f"source_supported: {item['source_supported']}")
        print(f"self_reported: {item['self_reported']}")
        print(f"third_party_reported: {item['third_party_reported']}")
        print(f"model_inferred: {item['model_inferred']}")
        print(f"content: {item['content']}")
        for s in view["subjects"]:
            print(f"subject: {s['subject_type']}:{s['subject_id']}")
        for s in view["sources"]:
            print(f"source: {s['source_id']} type={s['source_type']} title={s['title']}")
        for l in view["links"]:
            print(f"link: {l['memory_id']} -> {l['linked_memory_id']} ({l['link_type']})")
        for a in view["assignments"]:
            print(f"assignment: {a['person_id']} by={a['created_by']} at={a['created_at']}")
        prov = view.get("provenance")
        if prov:
            print(
                "provenance:"
                f" source_event={prov.get('source_event_id') or '-'}"
                f" candidate={prov.get('candidate_id') or '-'}"
                f" speaker={prov.get('speaker_person_id') or '-'}"
                f" subject={prov.get('subject_person_id') or '-'}"
                f" assertion={prov.get('assertion_type')}"
                f" topic={prov.get('topic') or '-'}"
            )
        for r in view["revisions"]:
            print(f"revision: v{r['version']} {r['status']} by={r['changed_by']} {r['reason'] or ''}")
    finally:
        await db.close()
    return 0


async def _memory_action(args, action: str) -> int:
    """Shared driver for confirm / retract / delete, mapping errors to messages."""
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            if action == "confirm":
                await rt.memory_service.confirm(args.memory_id, access=ADMIN)
            elif action == "retract":
                await rt.memory_service.retract(args.memory_id, access=ADMIN)
            else:  # delete
                await rt.memory_service.delete(args.memory_id, access=ADMIN)
        except MemoryNotFoundError:
            print(f"memory not found: {args.memory_id}")
            return 1
        except MemoryPermissionDeniedError as exc:
            print(f"permission denied: {exc}")
            return 1
        except MemoryVersionConflictError as exc:
            print(f"version conflict: {exc}")
            return 1
        print(f"{action} {args.memory_id}")
    finally:
        await db.close()
    return 0


async def cmd_memory_confirm(args) -> int:
    return await _memory_action(args, "confirm")


async def cmd_memory_retract(args) -> int:
    return await _memory_action(args, "retract")


async def cmd_memory_delete(args) -> int:
    return await _memory_action(args, "delete")


async def cmd_memory_link(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            await rt.memory_service.link(args.memory_id, args.other_id, args.type, ADMIN)
        except MemoryNotFoundError:
            print(f"memory not found: {args.memory_id} or {args.other_id}")
            return 1
        except MemoryPermissionDeniedError as exc:
            print(f"permission denied: {exc}")
            return 1
        print(f"linked {args.memory_id} -> {args.other_id} ({args.type})")
    finally:
        await db.close()
    return 0


def _memory_error(exc: Exception) -> int:
    if isinstance(exc, MemoryNotFoundError):
        print(f"not found: {exc}")
    elif isinstance(exc, MemoryPermissionDeniedError):
        print(f"permission denied: {exc}")
    elif isinstance(exc, MemoryVersionConflictError):
        print(f"version conflict: {exc}")
    else:
        print(f"error: {type(exc).__name__}: {exc}")
    return 1


async def cmd_memory_assign(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            created = await rt.memory_service.assign_memory(
                args.memory_id, args.person, ADMIN
            )
        except Exception as exc:
            return _memory_error(exc)
        status = "assigned" if created else "already assigned"
        print(f"{status} {args.memory_id} -> {args.person}")
    finally:
        await db.close()
    return 0


async def cmd_memory_unassign(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            removed = await rt.memory_service.unassign_memory(
                args.memory_id, args.person, ADMIN
            )
        except Exception as exc:
            return _memory_error(exc)
        status = "unassigned" if removed else "was not assigned"
        print(f"{status} {args.memory_id} -> {args.person}")
    finally:
        await db.close()
    return 0


async def cmd_memory_assignments(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            assignments = await rt.memory_service.list_memory_assignments(
                args.memory_id, ADMIN
            )
        except Exception as exc:
            return _memory_error(exc)
        for a in assignments:
            print(f"{a['memory_id']}  {a['person_id']}  by={a['created_by']}  at={a['created_at']}")
        if not assignments:
            print("no assignments")
    finally:
        await db.close()
    return 0


async def cmd_policy_show(args) -> int:
    db = await _connect(args.db)
    try:
        config = await load_policy_config(make_repositories(db).settings)
        print(f"{SETTING_RESPONSE_CONSERVATISM}: {config.response_conservatism}")
        print(f"{SETTING_INTERRUPTION_THRESHOLD}: {config.interruption_threshold}")
        print(f"{SETTING_CURIOSITY_LEVEL}: {config.curiosity_level}")
        print("defaults: conservative=balanced threshold=medium curiosity=low")
    finally:
        await db.close()
    return 0


async def cmd_policy_set(args) -> int:
    allowed = {
        SETTING_RESPONSE_CONSERVATISM: RESPONSE_CONSERVATISM_VALUES,
        SETTING_INTERRUPTION_THRESHOLD: INTERRUPTION_THRESHOLD_VALUES,
        SETTING_CURIOSITY_LEVEL: CURIOSITY_LEVEL_VALUES,
    }
    value = args.value.strip().lower()
    if value not in allowed[args.key]:
        print(f"invalid value {args.value!r}; expected one of {allowed[args.key]}")
        return 1
    db = await _connect(args.db)
    try:
        repos = make_repositories(db)
        await repos.settings.set(args.key, value)
        print(f"{args.key} = {value}")
    finally:
        await db.close()
    return 0


def cmd_ui(args) -> int:
    """Start the management UI.

    NiceGUI owns its own event loop, so this runs synchronously (the ``ui``
    command is dispatched without ``asyncio.run`` in :func:`main`).
    """
    from companion.ui.app import serve

    args.db.parent.mkdir(parents=True, exist_ok=True)
    try:
        serve(args.db, host=args.host, port=args.port)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    return 0


async def _build_idle_scheduler(rt):
    """Build the idle scheduler with a settings-aware (mock-first) web search."""
    from companion.idle import build_idle_scheduler
    from companion.providers.websearch import make_web_search

    web_search = await make_web_search(rt.repos.settings)
    return build_idle_scheduler(rt, web_search=web_search)


async def cmd_idle_run(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        scheduler = await _build_idle_scheduler(rt)
        result = await scheduler.run_cycle()
        if result["skipped"]:
            print("idle learning is disabled (set idle.enabled=true to enable)")
        else:
            print(
                f"reflections={result['reflections']} questions={result['questions']}"
                f" research={result['research']}"
            )
    finally:
        await db.close()
    return 0


async def cmd_research_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        tasks = await rt.repos.research.list(status=args.status)
        for t in tasks:
            print(
                f"{t['task_id']}  {t['status']}  p={t['priority']}"
                f"  {t['source_kind']}  {t['question']}"
            )
        if not tasks:
            print("no research tasks")
    finally:
        await db.close()
    return 0


async def cmd_research_queue(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        task_id = await rt.repos.research.create(
            args.question, source_kind=args.source_kind, priority=args.priority
        )
        print(task_id)
    finally:
        await db.close()
    return 0


async def cmd_research_run(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        if not await rt.repos.settings.get_bool("research.enabled", False):
            print("research is disabled (set research.enabled=true to enable)")
            return 0
        scheduler = await _build_idle_scheduler(rt)
        tasks = await scheduler.research.run_all(limit=args.limit)
        for t in tasks:
            print(f"{t['task_id']}  {t['status']}  {t['question']}")
        if not tasks:
            print("no pending research tasks")
    finally:
        await db.close()
    return 0


async def cmd_recovery_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        recovery = RecoveryManager(rt.repos, rt.orchestrator)
        events = await recovery.list_unfinished()
        if not events:
            print("no unfinished events")
        for e in events:
            print(f"{e.event_id}  {e.processing_status.value}  {e.conversation_id}  {e.text}")
    finally:
        await db.close()
    return 0


async def cmd_recovery_retry(args) -> int:
    db = await _connect(args.db)
    try:
        rt = _build_foreground_runtime(db, getattr(args, "provider", "mock"))
        if rt is None:
            return 1
        recovery = RecoveryManager(rt.repos, rt.orchestrator)
        try:
            report = await recovery.retry(args.event_id)
        except RecoveryError as exc:
            print(f"recovery failed: {exc}")
            return 1
        for step in report.steps:
            print(step)
        if rt.worker is not None:
            await rt.worker.run_once()
        print(f"event {args.event_id} status={report.status}")
    finally:
        await db.close()
    return 0


async def cmd_demo(args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        db = await _connect(db_path)
        try:
            rt = _build_foreground_runtime(db, getattr(args, "provider", "mock"))
            if rt is None:
                return 1
            external_ai = getattr(args, "provider", "mock") == "deepseek"
            owner = await rt.registry.register(
                "宿主", "owner",
                allow_processing=True, allow_persistence=True, allow_memory_learning=True,
                allow_external_ai=external_ai,
            )
            friend = await rt.registry.register(
                "朋友", "friend",
                allow_processing=True, allow_persistence=True, allow_memory_learning=True,
                allow_external_ai=external_ai,
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


async def _select_repl_person(rt, person_id):
    """Choose the REPL speaker without ever implicitly creating a new person.

    Returns ``(person, error_message)``. ``error_message`` is ``None`` on
    success. Display names are never used to merge people (duplicates are
    allowed).
    """
    if person_id is not None:
        person = await rt.registry.get(person_id)
        if person is None:
            return None, f"person not found: {person_id}"
        return person, None
    people = await rt.registry.list()
    if not people:
        return None, "no registered person — run `person add` first"
    if len(people) > 1:
        return None, "multiple people registered — pass --person <PERSON_ID>"
    return people[0], None


async def cmd_repl(args) -> int:
    db = await _connect(args.db)
    try:
        provider = getattr(args, "provider", "mock")
        rt = _build_foreground_runtime(db, provider)
        if rt is None:
            return 1
        person, err = await _select_repl_person(rt, getattr(args, "person", None))
        if person is None:
            print(err)
            return 1
        # Real DeepSeek starts conservative: an utterance without an explicit
        # /to agent is treated as unclear, so the model cannot talk over a
        # conversation it may not be part of. Mock keeps its old default.
        addressee = "unknown" if provider == "deepseek" else "agent"
        print("AI Companion REPL. Type /help for commands.")
        print(f"speaking as {person.person_id} ({person.display_name})")
        print(f"provider: {rt.provider}")
        if provider == "deepseek":
            print(f"addressee: {addressee} (use /to agent to address me explicitly)")
        if provider == "deepseek":
            authorized = await rt.registry.is_external_ai_authorized(person.person_id)
            if not authorized:
                print(
                    "warning: this person has not granted --allow-external-ai;"
                    " every DeepSeek call will be denied"
                )
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
                elif cmd == "/status":
                    ctx = await rt.repos.conversation.get(args.conversation)
                    print(f"provider: {rt.provider}")
                    print(f"person: {person.person_id} ({person.display_name})")
                    print(f"conversation: {args.conversation}")
                    print(f"addressee: {addressee}")
                    print(f"quiet_mode: {'on' if ctx and ctx.quiet_mode else 'off'}")
                elif cmd == "/help":
                    print(
                        "commands: /to agent|human|unknown  /quiet on|off"
                        "  /history  /status  /quit"
                    )
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
            if report.response_text:
                print(f"assistant> {report.response_text}")
            elif report.status != "COMPLETED":
                print(f"[{report.status}] {report.error or ''}".rstrip())
            else:
                reason = f" ({report.reason_code.value})" if report.reason_code else ""
                action = report.action.value if report.action else "no action"
                print(f"[{action}{reason}]")
            if rt.worker is not None:
                await rt.worker.run_once()
    finally:
        await db.close()
    return 0


async def cmd_deepseek_check(args) -> int:
    """Validate DeepSeek configuration without revealing the key.

    Offline by default. ``--live`` makes exactly one minimal API call; it is
    only attempted when a key is configured, and any failure is reported
    without ever printing the key or the response body.
    """
    config = load_deepseek_config()
    print("== deepseek configuration ==")
    print(f"api_key: {'configured' if config.api_key else 'MISSING'}")
    print(f"base_url: {config.base_url}")
    print(f"model: {config.model}")
    print(f"timeout_seconds: {config.timeout_seconds}")
    print(f"max_retries: {config.max_retries}")
    print(f"analysis_max_tokens: {config.analysis_max_tokens}")
    print(f"response_max_tokens: {config.response_max_tokens}")
    print(f"analysis_reasoning_effort: {config.analysis_reasoning_effort}")
    print(f"response_reasoning_effort: {config.response_reasoning_effort}")

    if not config.api_key:
        print("error: DEEPSEEK_API_KEY is not set; --provider deepseek cannot start")
        return 1

    if not args.live:
        print("config check passed (offline; pass --live to make one real call)")
        return 0

    try:
        client = build_deepseek_client(config)
        response = await client.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            request_type="response",
            max_tokens=8,
        )
    except DeepSeekError as exc:
        print(f"live check failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"live check passed: model={response.model}"
        f" tokens={response.total_tokens}"
        f" reply={response.content[:20]!r}"
    )
    return 0
