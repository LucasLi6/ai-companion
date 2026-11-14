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
import contextlib
import ipaddress
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from companion.config import configure_logging, default_db_path
from companion.agent.constants import (
    FAMILIARITY_TIERS,
    PERSONA_TRAITS,
    PersonalityValidationError,
    RelationshipValidationError,
)
from companion.agent.style_examples import StyleExampleError
from companion.cmf.service import (
    CmfError,
    CmfFileExistsError,
    CmfFileService,
    CmfValidationError,
)
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
from companion.people.profile_models import (
    ProfileFieldStorage,
    profile_field_spec,
)
from companion.people.profile_candidate_migration import (
    migrate_profile_candidates,
)
from companion.people.profile_service import ProfileUnauthorized
from companion.jobs.profile_review import (
    ProfileReviewScheduler,
    repair_skipped_reviews,
)
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
from companion.providers.websearch import (
    BRAVE_KEY_ENV,
    BraveWebSearchProvider,
    HttpWebSearchProvider,
    WebSearchConfigError,
    WebSearchError,
)
from companion.recovery import RecoveryError, RecoveryManager
from companion.runtime import build_runtime
from companion.runtime_context.assembler import compile_runtime_context
from companion.runtime_context.timezone import TimeZoneService, valid_zone
from companion.storage.database import Database
from companion.storage.repositories import make_repositories
from companion.tts import SpeechJob, TTSManager
from companion.tts.config import load_local_env, tts_enabled_by_default

DEFAULT_CONVERSATION = "default"

# Runtime-context settings. GeoIP stays off unless explicitly enabled here (or
# at the process level) *and* a locally licensed database is present.
RUNTIME_SETTING_TIMEZONE = "runtime.timezone"
RUNTIME_SETTING_REENTRY_GAP = "runtime.reentry_gap_seconds"
RUNTIME_SETTING_GEOIP_ENABLED = "privacy.geoip.enabled"
RUNTIME_SETTING_GEOIP_TRUSTED_PROXIES = "privacy.geoip.trusted_proxies"
RUNTIME_SETTING_KEYS = (
    RUNTIME_SETTING_TIMEZONE,
    RUNTIME_SETTING_REENTRY_GAP,
    RUNTIME_SETTING_GEOIP_ENABLED,
    RUNTIME_SETTING_GEOIP_TRUSTED_PROXIES,
)

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


async def _build_foreground_runtime(db, provider: str):
    """Build a runtime for utterance-processing commands.

    Embedding follows the persistent ``memory.embedding.*`` settings: disabled
    by default, and only ``enabled=true`` with an explicit backend wires the
    hybrid retriever. A requested-but-unavailable backend is a clear error, not
    a silent fallback. A missing key for ``--provider deepseek`` likewise prints
    an error and returns ``None`` — never a mock fallback, never a network call.
    """
    from companion.memory.embeddings import (
        EmbeddingBackendUnavailable,
        build_embedding_provider_from_settings,
    )

    settings = make_repositories(db).settings
    try:
        embedding_provider = await build_embedding_provider_from_settings(settings)
    except EmbeddingBackendUnavailable as exc:
        print(f"embedding error: {exc}")
        return None
    try:
        return build_runtime(
            db, provider=provider, embedding_provider=embedding_provider
        )
    except DeepSeekConfigError as exc:
        print(f"deepseek error: {exc}")
        return None


async def _get_foreground_runtime(db, provider: str):
    """Await the builder, tolerating a synchronous test double."""
    import inspect

    result = _build_foreground_runtime(db, provider)
    if inspect.isawaitable(result):
        return await result
    return result


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

    p_profile = sub.add_parser(
        "profile", parents=[_common_parser()],
        help="Inspect and review one person's understanding profile",
    )
    p_profile_sub = p_profile.add_subparsers(dest="profile_command")
    pp_show = p_profile_sub.add_parser(
        "show", parents=[_common_parser()], help="Show the confirmed profile"
    )
    pp_show.add_argument("--person", required=True)
    pp_show.set_defaults(func=cmd_profile_show)
    pp_history = p_profile_sub.add_parser(
        "history", parents=[_common_parser()], help="Show profile revisions"
    )
    pp_history.add_argument("--person", required=True)
    pp_history.set_defaults(func=cmd_profile_history)
    pp_migrate = p_profile_sub.add_parser(
        "migrate-candidates", parents=[_common_parser()],
        help=(
            "Revalidate retired profile candidates and write only legal"
            " values (dry-run by default)"
        ),
    )
    pp_migrate.add_argument("--person", required=True)
    pp_migrate_mode = pp_migrate.add_mutually_exclusive_group()
    pp_migrate_mode.add_argument(
        "--dry-run", dest="apply", action="store_false",
        help="Report what would be written without writing (default)",
    )
    pp_migrate_mode.add_argument(
        "--apply", dest="apply", action="store_true",
        help="Back up the database, then write validated changes",
    )
    pp_migrate.set_defaults(apply=False)
    pp_migrate.add_argument(
        "--backup-path", default=None,
        help="Backup file for --apply (default: <db>.backup-<timestamp>)",
    )
    pp_migrate.set_defaults(func=cmd_profile_migrate_candidates)
    pp_review = p_profile_sub.add_parser(
        "review", parents=[_common_parser()],
        help="Run or check a formal profile review for one turn",
    )
    pp_review.add_argument("--person", required=True)
    pp_review.add_argument(
        "--event", default=None,
        help="Source event id (default: this person's latest turn)",
    )
    _add_provider_arg(pp_review)
    pp_review.set_defaults(func=cmd_profile_review)
    pp_repair = p_profile_sub.add_parser(
        "repair-reviews", parents=[_common_parser()],
        help=(
            "Requeue reviews skipped because the profile row was missing"
            " (dry-run by default)"
        ),
    )
    pp_repair.add_argument("--person", required=True)
    pp_repair_mode = pp_repair.add_mutually_exclusive_group()
    pp_repair_mode.add_argument(
        "--dry-run", dest="apply", action="store_false",
        help="Report what would be repaired without writing (default)",
    )
    pp_repair_mode.add_argument(
        "--apply", dest="apply", action="store_true",
        help="Initialise the profile and requeue matching reviews",
    )
    pp_repair.set_defaults(apply=False)
    pp_repair.add_argument(
        "--limit", type=int, default=30,
        help="Maximum reviews to repair in this run (default: 30)",
    )
    pp_repair.set_defaults(func=cmd_profile_repair_reviews)

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
    p_ms.add_argument(
        "--mode",
        choices=["keyword", "hybrid"],
        default="keyword",
        help="hybrid fuses keyword + vector ranking (requires an embedding provider)",
    )
    p_ms.add_argument(
        "--fake-provider",
        action="store_true",
        default=False,
        help="Use the deterministic offline embedding provider (no model download)",
    )
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

    p_mix = p_mem_sub.add_parser(
        "index", parents=[_common_parser()],
        help="Manage the semantic (vector) index for hybrid retrieval",
    )
    p_mix_sub = p_mix.add_subparsers(dest="index_command")
    p_mix_status = p_mix_sub.add_parser(
        "status", parents=[_common_parser()], help="Show index model + vector counts"
    )
    p_mix_status.set_defaults(func=cmd_memory_index_status)
    p_mix_backfill = p_mix_sub.add_parser(
        "backfill", parents=[_common_parser()],
        help="Embed every ACTIVE memory missing an up-to-date vector",
    )
    p_mix_backfill.add_argument(
        "--fake-provider", action="store_true", default=False,
        help="Use the deterministic offline embedding provider (no model download)",
    )
    p_mix_backfill.set_defaults(func=cmd_memory_index_backfill)
    p_mix_rebuild = p_mix_sub.add_parser(
        "rebuild", parents=[_common_parser()],
        help="Mark all vectors stale and re-embed every ACTIVE memory",
    )
    p_mix_rebuild.add_argument(
        "--fake-provider", action="store_true", default=False,
        help="Use the deterministic offline embedding provider (no model download)",
    )
    p_mix_rebuild.set_defaults(func=cmd_memory_index_rebuild)

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

    p_runtime = sub.add_parser(
        "runtime",
        parents=[_common_parser()],
        help="Show or set runtime context settings (time zone, GeoIP)",
    )
    p_runtime_sub = p_runtime.add_subparsers(dest="runtime_command")
    p_runtime_show = p_runtime_sub.add_parser(
        "show", parents=[_common_parser()], help="Show runtime context settings"
    )
    p_runtime_show.set_defaults(func=cmd_runtime_show)
    p_runtime_set = p_runtime_sub.add_parser(
        "set", parents=[_common_parser()], help="Set one runtime context setting"
    )
    p_runtime_set.add_argument("key")
    p_runtime_set.add_argument("value")
    p_runtime_set.set_defaults(func=cmd_runtime_set)
    p_runtime_inspect = p_runtime_sub.add_parser(
        "inspect",
        parents=[_common_parser()],
        help="Diagnose the runtime block used for one event",
    )
    p_runtime_inspect.add_argument("--event", required=True, dest="event_id")
    p_runtime_inspect.set_defaults(func=cmd_runtime_inspect)

    p_ui = sub.add_parser("ui", parents=[_common_parser()], help="Start the management web UI (NiceGUI)")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8080)
    _add_provider_arg(p_ui)
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
    p_ir.add_argument(
        "--real-provider",
        action="store_true",
        help="Use the real DeepSeek idle provider (also requires"
        " idle.real_deepseek.enabled=true and DEEPSEEK_API_KEY)",
    )
    p_ir.set_defaults(func=cmd_idle_run)

    p_goal = sub.add_parser("goal", parents=[_common_parser()], help="Manage durable goals")
    p_goal_sub = p_goal.add_subparsers(dest="goal_command")
    p_ga = p_goal_sub.add_parser("add", parents=[_common_parser()], help="Create a goal")
    p_ga.add_argument("title")
    p_ga.add_argument("--description", default=None)
    p_ga.add_argument("--priority", type=int, default=0)
    p_ga.add_argument("--next", dest="next_action", default=None)
    p_ga.set_defaults(func=cmd_goal_add)
    p_gl = p_goal_sub.add_parser("list", parents=[_common_parser()], help="List goals")
    p_gl.add_argument("--status", choices=["OPEN", "IN_PROGRESS", "BLOCKED", "DONE", "CANCELLED"], default=None)
    p_gl.set_defaults(func=cmd_goal_list)
    p_gs = p_goal_sub.add_parser("show", parents=[_common_parser()], help="Show a goal")
    p_gs.add_argument("goal_id")
    p_gs.set_defaults(func=cmd_goal_show)
    p_gst = p_goal_sub.add_parser("set-status", parents=[_common_parser()], help="Set a goal's status")
    p_gst.add_argument("goal_id")
    p_gst.add_argument("status", choices=["OPEN", "IN_PROGRESS", "BLOCKED", "DONE", "CANCELLED"])
    p_gst.add_argument("--next", dest="next_action", default=None)
    p_gst.set_defaults(func=cmd_goal_set_status)
    p_gc = p_goal_sub.add_parser("checkpoint", parents=[_common_parser()], help="Record a goal checkpoint")
    p_gc.add_argument("goal_id")
    p_gc.add_argument("--status", choices=["OPEN", "IN_PROGRESS", "BLOCKED", "DONE", "CANCELLED"], default="IN_PROGRESS")
    p_gc.add_argument("--summary", default=None)
    p_gc.set_defaults(func=cmd_goal_checkpoint)
    p_gd = p_goal_sub.add_parser("depend", parents=[_common_parser()], help="Declare a goal dependency")
    p_gd.add_argument("goal_id")
    p_gd.add_argument("--on", dest="depends_on", required=True)
    p_gd.set_defaults(func=cmd_goal_depend)
    p_gr = p_goal_sub.add_parser("ready", parents=[_common_parser()], help="List goals whose dependencies are done")
    p_gr.set_defaults(func=cmd_goal_ready)

    p_demo = sub.add_parser("demo", parents=[_common_parser()], help="Run a scripted demo on an isolated database")
    _add_provider_arg(p_demo)
    p_demo.set_defaults(func=cmd_demo)
    p_repl = sub.add_parser("repl", parents=[_common_parser()], help="Start an interactive text session")
    p_repl.add_argument("--conversation", default=DEFAULT_CONVERSATION)
    p_repl.add_argument("--person", default=None, help="Registered person id of the speaker")
    p_repl.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help="Enable microphone listening at startup (VAD + Qwen ASR)",
    )
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

    p_mf = sub.add_parser(
        "memory-file",
        parents=[_common_parser()],
        help="Create, inspect, validate, export or clone a CMF memory file",
    )
    p_mf_sub = p_mf.add_subparsers(dest="memory_file_command")
    p_mfc = p_mf_sub.add_parser("create", parents=[_common_parser()], help="Create a new .memory file")
    p_mfc.add_argument("path")
    p_mfc.add_argument("--name", default="Companion")
    p_mfc.add_argument("--overwrite", action="store_true", default=False)
    p_mfc.set_defaults(func=cmd_memory_file_create)
    p_mfi = p_mf_sub.add_parser("info", parents=[_common_parser()], help="Show a memory file summary")
    p_mfi.add_argument("path")
    p_mfi.set_defaults(func=cmd_memory_file_info)
    p_mfv = p_mf_sub.add_parser("validate", parents=[_common_parser()], help="Validate integrity, schema and manifest")
    p_mfv.add_argument("path")
    p_mfv.set_defaults(func=cmd_memory_file_validate)
    p_mfe = p_mf_sub.add_parser("export", parents=[_common_parser()], help="Write a consistent copy")
    p_mfe.add_argument("source")
    p_mfe.add_argument("destination")
    p_mfe.add_argument("--overwrite", action="store_true", default=False)
    p_mfe.set_defaults(func=cmd_memory_file_export)
    p_mfcl = p_mf_sub.add_parser("clone", parents=[_common_parser()], help="Clone with a new agent_id and lineage")
    p_mfcl.add_argument("source")
    p_mfcl.add_argument("destination")
    p_mfcl.add_argument("--name", default=None)
    p_mfcl.add_argument("--overwrite", action="store_true", default=False)
    p_mfcl.set_defaults(func=cmd_memory_file_clone)
    p_mfm = p_mf_sub.add_parser("migrate", parents=[_common_parser()], help="Upgrade a legacy .db in place")
    p_mfm.add_argument("path")
    p_mfm.set_defaults(func=cmd_memory_file_migrate)

    p_agent = sub.add_parser("agent", parents=[_common_parser()], help="Manage agent identity and personality")
    p_agent_sub = p_agent.add_subparsers(dest="agent_command")
    p_as = p_agent_sub.add_parser("show", parents=[_common_parser()], help="Show identity and persona")
    p_as.set_defaults(func=cmd_agent_show)
    p_asn = p_agent_sub.add_parser("set-name", parents=[_common_parser()], help="Set the agent display name")
    p_asn.add_argument("name")
    p_asn.add_argument("--pronouns", default=None)
    p_asn.add_argument("--description", default=None)
    p_asn.add_argument("--reason", default=None)
    p_asn.set_defaults(func=cmd_agent_set_name)
    p_asp = p_agent_sub.add_parser("set-persona", parents=[_common_parser()], help="Set one personality trait")
    p_asp.add_argument("field", choices=list(PERSONA_TRAITS))
    p_asp.add_argument("value", type=int)
    p_asp.add_argument("--reason", default=None)
    p_asp.set_defaults(func=cmd_agent_set_persona)
    p_aset = p_agent_sub.add_parser("set", parents=[_common_parser()], help="Alias for set-persona")
    p_aset.add_argument("field", choices=list(PERSONA_TRAITS))
    p_aset.add_argument("value", type=int)
    p_aset.add_argument("--reason", default=None)
    p_aset.set_defaults(func=cmd_agent_set_persona)
    p_ar = p_agent_sub.add_parser("revisions", parents=[_common_parser()], help="List persona revisions")
    p_ar.add_argument("--limit", type=int, default=20)
    p_ar.set_defaults(func=cmd_agent_revisions)
    p_ae = p_agent_sub.add_parser("effective", parents=[_common_parser()], help="Show the effective context for one person")
    p_ae.add_argument("--person", required=True)
    p_ae.set_defaults(func=cmd_agent_effective)
    p_pc = p_agent_sub.add_parser(
        "persona-core", parents=[_common_parser()],
        help="Show or set the admin-maintained character core",
    )
    p_pc_sub = p_pc.add_subparsers(dest="persona_core_command")
    p_pcs = p_pc_sub.add_parser("show", parents=[_common_parser()], help="Show the character core")
    p_pcs.set_defaults(func=cmd_agent_persona_core_show)
    p_pcset = p_pc_sub.add_parser("set", parents=[_common_parser()], help="Set character core fields from JSON")
    p_pcset.add_argument("--json", required=True, dest="core_json",
                         help="JSON object with core_values/temperament/preferences/quirks/contradictions/boundaries")
    p_pcset.add_argument("--reason", default=None)
    p_pcset.set_defaults(func=cmd_agent_persona_core_set)
    p_style = p_agent_sub.add_parser(
        "style-example", parents=[_common_parser()],
        help="Manage trusted style references (never fixed replies)",
    )
    p_style_sub = p_style.add_subparsers(dest="style_command")
    p_sa = p_style_sub.add_parser("add", parents=[_common_parser()], help="Add a style reference")
    p_sa.add_argument("--scenario", required=True)
    p_sa.add_argument("--user", required=True, dest="user_text")
    p_sa.add_argument("--response", required=True, dest="ideal_response")
    p_sa.add_argument("--person", default=None, help="Optional person scope; omit for global")
    p_sa.set_defaults(func=cmd_agent_style_add)
    p_sl = p_style_sub.add_parser("list", parents=[_common_parser()], help="List style references")
    p_sl.add_argument("--person", default=None)
    p_sl.add_argument("--include-disabled", action="store_true", default=False)
    p_sl.add_argument("--limit", type=int, default=100)
    p_sl.set_defaults(func=cmd_agent_style_list)
    p_sd = p_style_sub.add_parser("disable", parents=[_common_parser()], help="Disable a style reference")
    p_sd.add_argument("example_id")
    p_sd.set_defaults(func=cmd_agent_style_disable)

    p_rel = sub.add_parser("relationship", parents=[_common_parser()], help="Manage familiarity and per-person style")
    p_rel_sub = p_rel.add_subparsers(dest="relationship_command")
    p_rs = p_rel_sub.add_parser("show", parents=[_common_parser()], help="Show one relationship")
    p_rs.add_argument("person_id")
    p_rs.set_defaults(func=cmd_relationship_show)
    p_rset = p_rel_sub.add_parser("set", parents=[_common_parser()], help="Set familiarity tier")
    p_rset.add_argument("person_id")
    p_rset.add_argument("tier", type=lambda value: value.upper(), choices=list(FAMILIARITY_TIERS))
    p_rset.add_argument("--reason", default=None)
    p_rset.set_defaults(func=cmd_relationship_set)
    p_rov = p_rel_sub.add_parser("override", parents=[_common_parser()], help="Absolute per-person trait override")
    p_rov.add_argument("person_id")
    p_rov.add_argument("field", choices=list(PERSONA_TRAITS))
    p_rov.add_argument("value", type=int)
    p_rov.set_defaults(func=cmd_relationship_override)
    p_rcl = p_rel_sub.add_parser("clear-override", parents=[_common_parser()], help="Remove a per-person override")
    p_rcl.add_argument("person_id")
    p_rcl.add_argument("field", choices=list(PERSONA_TRAITS))
    p_rcl.set_defaults(func=cmd_relationship_clear_override)
    p_rh = p_rel_sub.add_parser("history", parents=[_common_parser()], help="Show relationship history")
    p_rh.add_argument("person_id")
    p_rh.set_defaults(func=cmd_relationship_history)

    p_emotion = sub.add_parser("emotion", parents=[_common_parser()], help="Inspect or configure the companion's emotion framework")
    p_emotion_sub = p_emotion.add_subparsers(dest="emotion_command")
    p_es = p_emotion_sub.add_parser("show", parents=[_common_parser()], help="Show the emotion profile")
    p_es.set_defaults(func=cmd_emotion_show)
    p_ei = p_emotion_sub.add_parser("inspect", parents=[_common_parser()], help="Inspect events, snapshots and feedback")
    p_ei.add_argument("--person", default=None, help="Filter snapshots/events to one person id")
    p_ei.add_argument("--limit", type=int, default=100)
    p_ei.set_defaults(func=cmd_emotion_inspect)
    p_er = p_emotion_sub.add_parser("rules", parents=[_common_parser()], help="List appraisal rules")
    p_er_sub = p_er.add_subparsers(dest="rules_command")
    p_erl = p_er_sub.add_parser("list", parents=[_common_parser()], help="List appraisal rules")
    p_erl.set_defaults(func=cmd_emotion_rules_list)
    p_ec = p_emotion_sub.add_parser("config", parents=[_common_parser()], help="Show or set the emotion profile")
    p_ec_sub = p_ec.add_subparsers(dest="config_command")
    p_ecs = p_ec_sub.add_parser("show", parents=[_common_parser()], help="Show the emotion profile config")
    p_ecs.set_defaults(func=cmd_emotion_config_show)
    p_ecset = p_ec_sub.add_parser("set", parents=[_common_parser()], help="Update one profile field")
    p_ecset.add_argument(
        "field",
        choices=["baseline_valence", "baseline_arousal", "default_half_life_seconds", "responsiveness"],
    )
    p_ecset.add_argument("value")
    p_ecset.set_defaults(func=cmd_emotion_config_set)
    p_esim = p_emotion_sub.add_parser(
        "simulate", parents=[_common_parser()],
        help="Show the decayed state for one person at an injected clock time",
    )
    p_esim.add_argument("--person", required=True, help="Person id to scope the state to")
    p_esim.add_argument("--elapsed-seconds", type=int, required=True,
                        help="Seconds to advance the clock before decaying")
    p_esim.set_defaults(func=cmd_emotion_simulate)

    p_web = sub.add_parser("web", parents=[_common_parser()], help="Inspect and control real web search")
    p_web_sub = p_web.add_subparsers(dest="web_command")
    p_ws = p_web_sub.add_parser("status", parents=[_common_parser()], help="Show web-search state and config")
    p_ws.set_defaults(func=cmd_web_status)
    p_we = p_web_sub.add_parser("enable", parents=[_common_parser()], help="Turn real web search on")
    p_we.add_argument(
        "--provider",
        choices=["brave", "duckduckgo"],
        default=None,
        help="Provider to use (default: brave if BRAVE_SEARCH_API_KEY is set, else duckduckgo)",
    )
    p_we.set_defaults(func=cmd_web_enable)
    p_wd = p_web_sub.add_parser("disable", parents=[_common_parser()], help="Turn real web search off")
    p_wd.set_defaults(func=cmd_web_disable)
    p_wt = p_web_sub.add_parser("test", parents=[_common_parser()], help="Run one real search, bypassing the Router")
    p_wt.add_argument("query")
    p_wt.add_argument(
        "--provider",
        choices=["brave", "duckduckgo"],
        default=None,
        help="Provider to test (default: the configured provider, or duckduckgo when no Brave key)",
    )
    p_wt.add_argument("--limit", type=int, default=5)
    p_wt.set_defaults(func=cmd_web_test)
    p_wd = p_web_sub.add_parser(
        "diagnose",
        parents=[_common_parser()],
        help="Show recent web/weather tool calls and likely causes",
    )
    p_wd.add_argument("--limit", type=int, default=20)
    p_wd.set_defaults(func=cmd_web_diagnose)

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


async def cmd_profile_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        snapshot = await rt.profile_service.get_full_snapshot(args.person)
        if snapshot is None:
            print(
                "profile not available: unknown person or persistence consent"
                " is not granted"
            )
            return 1
        print(f"person_id: {snapshot.person_id}")
        print(f"registration_label: {snapshot.display_name}")
        print(f"revision: {snapshot.revision}")
        print(
            "identity_confirmed: name={name} preferred_name={preferred_name}"
            " pronouns={pronouns}".format(
                name=snapshot.identity.get("name", "UNKNOWN"),
                preferred_name=snapshot.identity.get("preferred_name", "UNKNOWN"),
                pronouns=snapshot.identity.get("pronouns", "UNKNOWN"),
            )
        )
        print(
            f"relationship: familiarity={snapshot.relationship.get('familiarity')}"
            f" label={snapshot.relationship.get('label') or []}"
        )
        entry_count = sum(len(items) for items in snapshot.entries.values())
        print(f"understanding_entries: {entry_count}")
        for field_key, items in snapshot.entries.items():
            for entry in items:
                print(
                    f"  [{field_key}] {entry.content}"
                    f" ({entry.status}, source={entry.source},"
                    f" confidence={entry.confidence:.2f})"
                )
        receipt = await rt.repos.profile.latest_review_receipt(args.person)
        if receipt is not None:
            print(
                "last_review: status={status} decision={decision}"
                " revision={base}->{result} applied={applied}"
                " rejected={rejected} source_event={event}".format(
                    status=receipt["status"],
                    decision=receipt["decision"] or "-",
                    base=receipt["base_revision"] or "-",
                    result=receipt["result_revision"] or "-",
                    applied=receipt.get("applied_count", 0),
                    rejected=receipt.get("rejected_count", 0),
                    event=receipt["source_event_id"],
                )
            )
    finally:
        await db.close()
    return 0


def _print_revision(revision: dict) -> None:
    print(
        f"revision {revision['revision']}  by={revision['changed_by']}"
        f"  event={revision['source_event_id'] or '-'}"
        f"  reason={revision['reason'] or '-'}"
    )
    changes = revision["changes"] or {}
    rows = changes.get("changes") if isinstance(changes, dict) else None
    if not rows and isinstance(changes, dict):
        rows = [changes] if changes.get("field_key") else []
    for change in rows or []:
        if not isinstance(change, dict):
            continue
        operation = change.get("operation") or "-"
        field_key = change.get("field_key") or "-"
        before = change.get("before")
        after = change.get("after")
        targets = change.get("target_entry_ids") or []
        print(
            f"  {operation} {field_key} targets={targets}"
            f" before={before!r} after={after!r}"
            f" status={change.get('status', '-')}"
        )


async def cmd_profile_history(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        if not await rt.profile_service.can_persist(args.person):
            print("profile history unavailable: no persistence consent")
            return 1
        revisions = await rt.repos.profile.list_revisions(args.person)
        if not revisions:
            print("no profile revisions")
            return 0
        for revision in revisions:
            _print_revision(revision)
    finally:
        await db.close()
    return 0


async def cmd_profile_review(args) -> int:
    db = await _connect(args.db)
    try:
        rt = await _get_foreground_runtime(db, getattr(args, "provider", "mock"))
        if rt is None:
            return 1
        if not await rt.profile_service.can_use_external_ai(args.person):
            print(
                "profile review unavailable: persistence and external-AI"
                " consent are required"
            )
            return 1
        event_id = args.event
        if not event_id:
            events = await rt.repos.utterance.list_events(args.person, limit=1)
            if not events:
                print("profile review unavailable: this person has no turns")
                return 1
            event_id = events[0].event_id
        scheduler = ProfileReviewScheduler(
            rt.repos.profile, rt.repos.job, persons=rt.repos.person
        )
        async with db.transaction():
            review_id = await scheduler.request(args.person, event_id)
        if review_id is None:
            print("profile review skipped: consent is not granted")
            return 0
        if rt.worker is not None:
            await rt.worker.run_once()
        receipt = await rt.repos.profile.get_review_receipt(args.person, event_id)
        if receipt is None:
            print("profile review receipt missing")
            return 1
        print(
            "review: status={status} decision={decision}"
            " revision={base}->{result} event={event}".format(
                status=receipt["status"],
                decision=receipt["decision"] or "-",
                base=receipt["base_revision"] or "-",
                result=receipt["result_revision"] or "-",
                event=event_id,
            )
        )
        if receipt["reason"]:
            print(f"reason: {receipt['reason']}")
        for change in receipt["changes"] or []:
            print(
                f"  {change.get('operation', '-')} {change.get('field_key', '-')}"
                f" status={change.get('status', '-')}"
                f" before={change.get('before')!r}"
                f" after={change.get('after')!r}"
            )
    finally:
        await db.close()
    return 0


async def cmd_profile_migrate_candidates(args) -> int:
    """One-time migration of the retired Person Profile candidate table.

    Dry-run is the default and performs no writes. ``--apply`` makes a backup
    first, then revalidates every old row locally; model confidence is not
    trusted and no model is called.
    """
    if args.apply:
        source = Path(args.db)
        if not source.exists():
            print(f"database not found: {args.db}")
            return 1
        backup = (
            Path(args.backup_path)
            if args.backup_path
            else source.with_name(
                f"{source.name}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
        )
        if backup.exists():
            print(f"backup already exists, refusing to overwrite: {backup}")
            return 1
        shutil.copy2(source, backup)
        print(f"backup: {backup}")
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        summary = await migrate_profile_candidates(
            rt.profile_service,
            args.person,
            dry_run=not args.apply,
        )
        print(
            "mode={mode} person={person} scanned={scanned} applied={applied}"
            " already_present={present} already_migrated={migrated}"
            " skipped={skipped}".format(
                mode="apply" if args.apply else "dry-run",
                person=args.person,
                scanned=summary["scanned"],
                applied=summary["applied"],
                present=summary["already_present"],
                migrated=summary["already_migrated"],
                skipped=summary["skipped"],
            )
        )
        for item in summary["results"]:
            print(
                f"  {item['status']} {item.get('field_key') or '-'}"
                f" reason={item.get('reason') or '-'}"
                f" revision={item.get('revision') or '-'}"
            )
    finally:
        await db.close()
    return 0


async def cmd_profile_repair_reviews(args) -> int:
    """Repair legacy SKIPPED reviews for one person.

    Default mode is a read-only dry run. ``--apply`` initialises the profile
    when it is legally missing and requeues the matching reviews through the
    normal scheduler; the background worker then runs them.
    """
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        summary = await repair_skipped_reviews(
            rt.repos,
            person_id=args.person,
            dry_run=not args.apply,
            limit=args.limit,
        )
        print(
            "mode={mode} person={person} scanned={scanned} matched={matched}"
            " would_retry={would} retried={retried} skipped={skipped}".format(
                mode="apply" if args.apply else "dry-run",
                person=args.person,
                scanned=summary["scanned"],
                matched=summary["matched"],
                would=len(summary["would_retry"]),
                retried=len(summary["retried"]),
                skipped=len(summary["skipped"]),
            )
        )
        for item in summary["would_retry"]:
            print(f"  would retry {item['review_id']} event={item['event_id']}")
        for item in summary["retried"]:
            print(f"  retried {item['review_id']} event={item['event_id']}")
        for item in summary["skipped"]:
            print(f"  skip {item['review_id'] or '-'} reason={item['reason']}")
        if args.apply and summary["retried"]:
            print("requeued; the background worker will process these reviews")
    finally:
        await db.close()
    return 0


async def cmd_say(args) -> int:
    db = await _connect(args.db)
    try:
        rt = await _get_foreground_runtime(db, getattr(args, "provider", "mock"))
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

        snapshot = await rt.repos.runtime_snapshot.get(args.event_id)
        if snapshot:
            print("-- runtime context --")
            print(
                f"local_time: {snapshot.local_time}"
                f" ({snapshot.timezone}, source={snapshot.timezone_source})"
            )
            print(
                f"last_heard_gap_seconds: {snapshot.last_heard_gap_seconds}"
                f" last_interaction_gap_seconds: {snapshot.last_interaction_gap_seconds}"
                f" last_meaningful_context_gap_seconds:"
                f" {snapshot.last_meaningful_context_gap_seconds}"
            )
            print(f"reentry_state: {snapshot.reentry_state}")
            print(f"ava_emotion: {snapshot.ava_emotion or '-'}")
            if snapshot.location is not None:
                place = ", ".join(
                    part
                    for part in (
                        snapshot.location.city,
                        snapshot.location.subdivision,
                        snapshot.location.country,
                    )
                    if part
                )
                print(f"network_egress_location: {place or '(unknown)'}")
            else:
                print("network_egress_location: (none)")

        decisions = await rt.repos.routing.list_decisions(args.event_id)
        print(f"-- routing decisions: {len(decisions)} --")
        for decision_row in decisions:
            decision = decision_row.get("decision") or {}
            print(
                f"  step={decision_row['step_index']}"
                f" phase={decision_row['phase']}"
                f" validation={decision_row['validation_status']}"
                f" provider={decision_row['provider']}"
                f" model={decision_row['model']}"
            )
            print(
                f"    action={decision.get('action')}"
                f" needs_followup_planning="
                f"{decision.get('needs_followup_planning')}"
                f" reason={str(decision.get('reason') or '-')[:160]}"
            )
            for request in (decision.get("tool_requests") or [])[:8]:
                print(
                    "    tool_request: "
                    f"{request.get('tool_name')}"
                    f" args={json.dumps(request.get('arguments'), ensure_ascii=False)}"
                    f" required={request.get('required')}"
                )

        calls = await rt.repos.routing.list_tool_calls(args.event_id)
        print(f"-- tool calls: {len(calls)} --")
        for call in calls:
            print(
                f"  {call['tool_call_id']} step={call['route_step_index']}"
                f" tool={call['tool_name']} status={call['status']}"
                f" execution={call['execution_status']}"
                f" error={call['error_code'] or '-'}"
            )
            refs = (call.get("result_reference") or {}).get("source_refs") or []
            for ref in refs[:5]:
                print(f"    source: {ref}")

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
    mode = getattr(args, "mode", "keyword")
    if mode == "hybrid" and not args.query:
        print("hybrid search requires --query")
        return 1
    db = await _connect(args.db)
    try:
        if mode == "hybrid":
            rt = _build_embedding_runtime(db, args)
            if rt is None:
                return 1
            if rt.hybrid_retriever is None:
                print("internal error: hybrid retriever was not wired")
                return 1
        else:
            rt = build_runtime(db)
        keyword_results = await rt.memory_service.search(
            access=ADMIN,
            query=args.query,
            kinds=[args.kind] if args.kind else None,
            statuses=[args.status] if args.status else None,
            include_retracted=args.include_retracted,
            limit=args.limit,
        )
        if mode == "hybrid":
            results = await rt.hybrid_retriever.retrieve(
                args.query,
                keyword_results=keyword_results,
                access=ADMIN,
                identity_ids=set(),
                limit=args.limit,
            )
            for r in results:
                diag = (
                    f" kw={r.get('keyword_rank')} vec={r.get('vector_rank')}"
                    f" fusion={r.get('fusion_score')} idx={r.get('index_status')}"
                )
                print(
                    f"{r['memory_id']}  {r['kind']}  {r['status']}  {r['visibility']}"
                    f"  {r['content'][:40]}{diag}"
                )
            if not results:
                print("no memories match")
        else:
            for r in keyword_results:
                print(
                    f"{r['memory_id']}  {r['kind']}  {r['status']}  {r['visibility']}"
                    f"  {r['content'][:60]}"
                )
            if not keyword_results:
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


async def cmd_runtime_show(args) -> int:
    db = await _connect(args.db)
    try:
        repos = make_repositories(db)
        settings = repos.settings
        user_zone = await settings.get(RUNTIME_SETTING_TIMEZONE)
        resolved = TimeZoneService().resolve(user_timezone=user_zone)
        geo_enabled = await settings.get_bool(RUNTIME_SETTING_GEOIP_ENABLED, False)
        proxies = await settings.get(RUNTIME_SETTING_GEOIP_TRUSTED_PROXIES)
        raw_gap = await settings.get(RUNTIME_SETTING_REENTRY_GAP)

        print(f"{RUNTIME_SETTING_TIMEZONE}: {user_zone or '(not set)'}")
        print(f"resolved time zone: {resolved.name} (source={resolved.source})")
        print(f"{RUNTIME_SETTING_REENTRY_GAP}: {raw_gap or '(default 21600)'}")
        print(f"{RUNTIME_SETTING_GEOIP_ENABLED}: {str(geo_enabled).lower()}")
        print(f"GEOIP_DB_PATH: {'configured' if os.environ.get('GEOIP_DB_PATH') else '(not set)'}")
        print(
            f"{RUNTIME_SETTING_GEOIP_TRUSTED_PROXIES}: {proxies or '(none)'}"
        )
        print(
            "note: GeoIP is opt-in and local-only; it is only used when the"
            " speaker permits external AI use. A location is never stored as"
            " a raw IP."
        )
    finally:
        await db.close()
    return 0


async def cmd_runtime_set(args) -> int:
    key, value = args.key, args.value.strip()
    person_zone_prefix = f"{RUNTIME_SETTING_TIMEZONE}.person."
    if key == RUNTIME_SETTING_TIMEZONE or key.startswith(person_zone_prefix):
        if key.startswith(person_zone_prefix) and not key[len(person_zone_prefix):]:
            print("person time zone key needs a person id")
            return 1
        zone = valid_zone(value)
        if zone is None:
            print(
                f"invalid IANA time zone {value!r}; example: Asia/Shanghai"
            )
            return 1
        normalised = zone
    elif key == RUNTIME_SETTING_REENTRY_GAP:
        try:
            seconds = int(value)
        except ValueError:
            print("reentry gap must be a whole number of seconds")
            return 1
        if seconds <= 0:
            print("reentry gap must be positive")
            return 1
        normalised = str(seconds)
    elif key == RUNTIME_SETTING_GEOIP_ENABLED:
        lowered = value.lower()
        if lowered not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            print("expected true or false")
            return 1
        normalised = "true" if lowered in ("true", "1", "yes", "on") else "false"
    elif key == RUNTIME_SETTING_GEOIP_TRUSTED_PROXIES:
        items = [item.strip() for item in value.split(",") if item.strip()]
        normalised_items: list[str] = []
        for item in items:
            try:
                normalised_items.append(
                    str(ipaddress.ip_network(item, strict=False))
                )
            except ValueError:
                print(f"invalid proxy network {item!r}")
                return 1
        normalised = ",".join(normalised_items)
    else:  # argparse restricts the choice; defensive only
        print(f"unknown runtime setting {key!r}")
        return 1

    db = await _connect(args.db)
    try:
        await make_repositories(db).settings.set(key, normalised)
        print(f"{key} = {normalised}")
    finally:
        await db.close()
    return 0


async def cmd_runtime_inspect(args) -> int:
    """Explain what runtime data one event could have given the reply.

    Shows trusted metadata only: timestamps, the resolved zone and its source,
    and the rendered block. It never prints prompt text, conversation content or
    a raw IP address.
    """
    db = await _connect(args.db)
    try:
        repos = make_repositories(db)
        event = await repos.utterance.get(args.event_id)
        if event is None:
            print(f"event not found: {args.event_id}")
            return 1
        snapshot = await repos.runtime_snapshot.get(args.event_id)
        if snapshot is None:
            print(f"event: {event.event_id}")
            print("runtime snapshot: (none)")
            print(
                "error: no snapshot was persisted for this event; the reply"
                " could not include a runtime block."
            )
            return 0

        block = compile_runtime_context(snapshot)
        action = await repos.analysis.get_action(args.event_id)
        print(f"event: {snapshot.event_id}")
        print(f"occurred_at_utc: {snapshot.occurred_at_utc.isoformat()}")
        print(f"received_at_utc: {snapshot.received_at_utc.isoformat()}")
        print(f"timezone: {snapshot.timezone} (source={snapshot.timezone_source})")
        print(f"utc_offset_seconds: {snapshot.utc_offset_seconds}")
        print(f"snapshot_local_time: {snapshot.local_time or '-'}")
        print(
            "last_interaction_at_utc: "
            f"{snapshot.last_interaction_at_utc.isoformat() if snapshot.last_interaction_at_utc else '-'}"
        )
        print(
            "last_interaction_gap_seconds: "
            f"{snapshot.last_interaction_gap_seconds}"
        )
        print(f"runtime_block_chars: {len(block)}")
        if block:
            # The block is trusted local data, not conversation content.
            print("-- rendered block --")
            print(block)
        else:
            print("runtime_block: (empty)")
        if action is not None and action.action_type.value == "RESPOND":
            print(
                "final_request: the response provider builds the system message"
                " from this snapshot; a non-empty block above is included."
            )
        elif action is not None:
            print(
                f"final_request: no response was generated"
                f" (action={action.action_type.value}); the block was not sent."
            )
        else:
            print("final_request: no action recorded yet.")
        errors = []
        if not block:
            errors.append("rendered block is empty")
        if snapshot.timezone_source == "utc" and not snapshot.utc_offset_seconds:
            errors.append("time zone fell back to UTC")
        print(f"errors: {'; '.join(errors) if errors else 'none'}")
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
        serve(
            args.db,
            host=args.host,
            port=args.port,
            provider=getattr(args, "provider", "mock"),
        )
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    return 0


async def _build_idle_scheduler(rt, *, real_provider: bool = False):
    """Reuse the runtime's single research engine.

    Providers are resolved by the runtime (settings-aware, offline by
    default). The caller must build the runtime with ``real_idle_provider``
    when a real idle provider is wanted; this function must never construct a
    second engine or replace the worker's global handler.
    """
    from companion.idle import build_idle_scheduler

    return build_idle_scheduler(rt)


async def _provider_report_line(rt) -> str:
    engine = getattr(rt, "research_engine", None)
    if engine is None or not hasattr(engine, "provider_report"):
        return "search=mock synthesis=mock"
    try:
        report = await engine.provider_report()
    except Exception as exc:  # a missing credential is reported, not hidden
        return f"provider_error={type(exc).__name__}"
    return (
        f"search={report.get('search_mode')}:{report.get('search_provider')}"
        f" synthesis={report.get('synthesis_mode')}:{report.get('synthesis_provider')}"
        f" real_research={report.get('real_research')}"
    )


async def _idle_provider_status(rt) -> list[str]:
    """Truthful per-stage provider status, read from the live providers.

    Reflection, curiosity and research synthesis share one idle provider, so
    all three report the same synthesis mode. Nothing here is hardcoded: an
    unauthorised real adapter reports ``unauthorised`` rather than claiming to
    be real.
    """
    engine = getattr(rt, "research_engine", None)
    if engine is None or not hasattr(engine, "provider_report"):
        return ["Idle Provider: mock (mock)"]
    try:
        report = await engine.provider_report()
    except Exception as exc:  # a config/credential error is surfaced, not hidden
        return [f"provider_error={type(exc).__name__}: {exc}"]
    synthesis_mode = report.get("synthesis_mode") or "mock"
    synthesis_provider = report.get("synthesis_provider") or "mock-deepseek"
    search_mode = report.get("search_mode") or "mock"
    search_provider = report.get("search_provider") or "mock-web-search"
    return [
        f"Idle Provider: {synthesis_provider} ({synthesis_mode})",
        f"Reflection: {synthesis_mode}",
        f"Curiosity: {synthesis_mode}",
        f"Research Search: {search_provider} ({search_mode})",
        f"Research Synthesis: {synthesis_provider} ({synthesis_mode})",
    ]


async def cmd_idle_run(args) -> int:
    db = await _connect(args.db)
    real_provider = bool(getattr(args, "real_provider", False))
    try:
        try:
            rt = build_runtime(db, real_idle_provider=real_provider)
        except Exception as exc:  # missing DEEPSEEK_API_KEY / config
            print(f"error: {exc}")
            return 1
        if real_provider and not await rt.repos.settings.get_bool(
            "idle.real_deepseek.enabled", False
        ):
            # No silent real->mock downgrade: refuse instead of pretending.
            print(
                "error: real idle provider is not authorised"
                " (set idle.real_deepseek.enabled=true)"
            )
            return 1
        try:
            scheduler = await _build_idle_scheduler(rt)
            for line in await _idle_provider_status(rt):
                print(line)
        except Exception as exc:  # missing key / unauthorised real provider
            print(f"error: {exc}")
            return 1
        try:
            result = await scheduler.run_cycle()
            if result["skipped"]:
                print("idle learning is disabled (set idle.enabled=true to enable)")
            else:
                print(
                    f"reflections={result['reflections']}"
                    f" questions={result['questions']}"
                    f" research={result['research']}"
                )
                # Execution stays on the worker path. Draining here makes the
                # CLI useful end-to-end without introducing a second executor.
                if rt.worker is not None:
                    processed = await rt.worker.run_until_idle()
                    print(f"worker_processed={processed}")
        except Exception as exc:
            print(f"error: {exc}")
            return 1
    finally:
        await db.close()
    return 0


async def cmd_research_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        tasks = await rt.repos.research.list(status=args.status)
        for t in tasks:
            detail = f"  error={t['error']}" if t.get("error") else ""
            print(
                f"{t['task_id']}  {t['status']}  p={t['priority']}"
                f"  {t['source_kind']}  {t['question']}{detail}"
            )
        if not tasks:
            print("no research tasks")
    finally:
        await db.close()
    return 0


async def cmd_research_queue(args) -> int:
    from companion.idle.scheduling import schedule_research

    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        report = await schedule_research(
            rt.db,
            rt.repos.research,
            rt.task_service,
            args.question,
            origin="user",
            source_kind=args.source_kind,
            priority=args.priority,
        )
        if not report.get("task_id"):
            print(f"not queued: {report.get('reason')}")
            return 0
        job = (
            await rt.repos.job.get(report["job_id"])
            if report.get("job_id")
            else None
        )
        print(report["task_id"])
        print(
            f"created={report.get('created')} reason={report.get('reason')}"
            f" job={report.get('job_id') or '-'}"
            f" job_status={job.status.value if job is not None else '-'}"
        )
    finally:
        await db.close()
    return 0


async def cmd_research_run(args) -> int:
    from companion.idle.scheduling import enqueue_existing_research

    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        print(f"provider: {await _provider_report_line(rt)}")
        if not await rt.repos.settings.get_bool("research.enabled", False):
            print("research is disabled (set research.enabled=true to enable)")
            return 0
        pending = await rt.repos.research.list(status="PENDING", limit=args.limit)
        task_ids: list[str] = []
        for task in pending:
            task_id = str(task["task_id"])
            try:
                await enqueue_existing_research(rt.task_service, task_id)
            except Exception as exc:  # TaskRejected or config error
                print(f"{task_id}  NOT_SCHEDULED  {type(exc).__name__}: {exc}")
                continue
            task_ids.append(task_id)
        if rt.worker is not None:
            await rt.worker.run_until_idle(max_iterations=50)
        for task_id in task_ids:
            task = await rt.repos.research.get(task_id)
            if task is None:
                continue
            job = await rt.repos.job.get_by_key(f"research:{task_id}")
            job_status = job.status.value if job is not None else "-"
            job_error = (job.last_error or "") if job is not None else ""
            print(
                f"{task_id}  task={task['status']}  job={job_status}"
                f"  error={task.get('error') or job_error or '-'}"
                f"  {task['question']}"
            )
        if not task_ids:
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
        rt = await _get_foreground_runtime(db, getattr(args, "provider", "mock"))
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
            rt = await _get_foreground_runtime(db, getattr(args, "provider", "mock"))
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


VOICE_PERSON_BINDING_ENV = "AVA_VOICE_PERSON_BINDING"
_DEFAULT_PERSON = object()


def voice_person_binding_enabled(env=None) -> bool:
    """Whether voice turns resolve their speaker through explicit bindings.

    Off by default: the REPL keeps the single startup person for both keyboard
    and microphone input. When enabled, a microphone turn must resolve through
    ``person_speaker_bindings`` or stay anonymous; it never falls back to the
    startup person.
    """
    source = os.environ if env is None else env
    return str(source.get(VOICE_PERSON_BINDING_ENV, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def resolve_voice_person_id(
    rt,
    speaker_result,
    *,
    default_person_id: Optional[str],
    enabled: Optional[bool] = None,
) -> Optional[str]:
    """Resolve one voice final's speaker snapshot to a person id.

    With the bridge disabled this returns ``default_person_id`` exactly as
    before. With it enabled the answer comes only from
    :class:`SpeakerBindingService`: an unbound, ambiguous, provisional or
    unauthorized speaker resolves to ``None`` (anonymous), never to the
    startup person. Resolution errors also mean anonymous, so a broken voice
    print can never attribute speech to someone else.
    """
    if enabled is None:
        enabled = voice_person_binding_enabled()
    if not enabled:
        return default_person_id
    service = getattr(rt, "speaker_bindings", None)
    if service is None:
        return None
    try:
        return await service.resolve_speaker(speaker_result)
    except Exception:  # noqa: BLE001 - identity must never break the turn
        return None


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
    # ``AVA_PROACTIVE_ACTIVITY`` (and other local settings) may live in the
    # git-ignored .env; load it before the runtime builds the proactive
    # manager. Exported values always win.
    load_local_env()
    db = await _connect(args.db)
    try:
        provider = getattr(args, "provider", "mock")
        rt = await _get_foreground_runtime(db, provider)
        if rt is None:
            return 1
        person, err = await _select_repl_person(rt, getattr(args, "person", None))
        if person is None:
            print(err)
            return 1
        # A single-person REPL is an explicit conversation with the companion:
        # the trusted default addressee is therefore "agent". Multi-person or
        # background ingestion must not use this default; callers set
        # "unknown"/"human" explicitly. ``/to human`` and ``/to unknown`` still
        # override this for the rest of the session, and local Policy remains
        # the final authority.
        addressee = "agent"
        print("AI Companion REPL. Type /help for commands.")
        print(f"speaking as {person.person_id} ({person.display_name})")
        print(f"provider: {rt.provider}")
        print(f"router mode: {rt.router_mode}")
        if provider == "deepseek":
            print("addressee: agent (single-person session; use /to human|unknown to override)")
        if provider == "deepseek":
            authorized = await rt.registry.is_external_ai_authorized(person.person_id)
            if not authorized:
                print(
                    "warning: this person has not granted --allow-external-ai;"
                    " every DeepSeek call will be denied"
                )
        # One managed worker for the session. Starting/stopping it once avoids
        # the old per-turn ``run_once`` which blocked the next input.
        worker_started = False
        if rt.worker is not None:
            await rt.worker.start()
            worker_started = True
        # The no-reply timer for a delivered identity-confirmation question is
        # in-memory only. Resume it for any record left AWAITING_REPLY so a
        # restart cannot leave the confirmation hanging forever.
        confirmation = getattr(rt, "profile_confirmation", None)
        if confirmation is not None:
            with contextlib.suppress(Exception):
                await confirmation.resume_pending()
        # TTS is an independent output side channel. Phase B defaults it on
        # (AVA_TTS_ENABLED=false disables it), but enabling only opens the
        # local audio device: no Alibaba connection or cost happens until the
        # first reply that actually produces speech.
        tts_manager = TTSManager()
        if tts_enabled_by_default():
            ok, message = tts_manager.enable()
            if ok:
                print(message)
            else:
                # One clear line, then normal text-only chat continues.
                print(f"tts: off ({message}) — 继续文字聊天")
        else:
            print("tts: off (AVA_TTS_ENABLED=false)")
        try:
            # Created once for the session: prompt_toolkit owns the live input
            # line, so proactive / voice output can be printed above it
            # without ever reading or restoring the typed buffer.
            voice_mode = bool(getattr(args, "voice", False))
            # Voice mode shows recognized speech under the prompt as it
            # arrives; text-only sessions keep the plain prompt.
            voice_subtitle = _PartialSubtitle() if voice_mode else None
            repl_session = _build_prompt_session(subtitle=voice_subtitle)
            voice_binding_enabled = voice_mode and voice_person_binding_enabled()
            coordinator = None
            voice_output = (
                _PromptTurnOutput(repl_session)
                if repl_session is not None
                else _DirectTurnOutput()
            )
            proactive = getattr(rt, "proactive_manager", None)

            if voice_mode:
                # Imported lazily: text-only sessions must keep working even
                # when the optional ASR dependencies are not installed.
                from companion.asr.config import load_asr_config
                from companion.asr.manager import ASRManager
                from companion.voice import VoiceCoordinator
                from companion.voice.barge_in_config import (
                    BargeInConfigError,
                    load_barge_in_config,
                )
                from companion.voice.echo_canceller import (
                    EchoCancellationPipeline,
                )

                # Optional real-time barge-in. Off by default and completely
                # additive: if anything below fails, the existing half-duplex
                # voice system runs exactly as before.
                barge_in_config = None
                echo_pipeline = None
                try:
                    barge_in_config = load_barge_in_config()
                except BargeInConfigError as exc:
                    print(f"barge-in: off ({exc}) — 使用半双工模式")
                    barge_in_config = None
                if barge_in_config is not None and barge_in_config.enabled:
                    try:
                        playback_rate = int(tts_manager.status().sample_rate)
                    except Exception:  # noqa: BLE001 - TTS status is advisory
                        playback_rate = 24000
                    try:
                        echo_pipeline = EchoCancellationPipeline(
                            playback_sample_rate=playback_rate,
                            # AEC3 device-buffer hint only.
                            stream_delay_ms=barge_in_config.aec_delay_ms,
                            # Independent, measured timeline calibration.
                            timeline_offset_ms=(
                                barge_in_config.timeline_offset_ms
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - optional extra
                        print(
                            "barge-in: off "
                            f"({type(exc).__name__}) — 使用半双工模式"
                        )
                        echo_pipeline = None
                        barge_in_config = None
                    if echo_pipeline is not None:
                        if not echo_pipeline.available:
                            print(
                                "barge-in: off "
                                f"({echo_pipeline.unavailable_reason})"
                                " — 使用半双工模式"
                            )
                            echo_pipeline = None
                            barge_in_config = None
                        else:
                            setter = getattr(
                                tts_manager, "set_playback_pcm_listener", None
                            )
                            if callable(setter):
                                setter(
                                    lambda generation, pcm, timestamp: (
                                        echo_pipeline.push_playback(
                                            pcm, timestamp, generation
                                        )
                                    )
                                )
                                # Timing is independent of the PCM
                                # reference: even a dropped reference frame
                                # still tells the AEC that playback was
                                # submitted, so real TTS silence and a
                                # reference sync fault are distinguishable.
                                timing_setter = getattr(
                                    tts_manager,
                                    "set_playback_timing_listener",
                                    None,
                                )
                                if callable(timing_setter):
                                    timing_setter(
                                        echo_pipeline.note_playback_timing
                                    )
                                event_adder = getattr(
                                    tts_manager,
                                    "add_playback_listener",
                                    None,
                                )
                                if callable(event_adder):
                                    event_adder(
                                        lambda event: (
                                            echo_pipeline.note_playback_event(
                                                getattr(event, "kind", ""),
                                                getattr(
                                                    event, "generation", None
                                                ),
                                            )
                                        )
                                    )
                            else:
                                echo_pipeline = None
                                barge_in_config = None

                def _make_asr_manager():
                    manager = ASRManager(
                        load_asr_config(
                            debug=bool(getattr(args, "debug", False))
                        )
                    )
                    if echo_pipeline is not None and barge_in_config:
                        manager.attach_echo_canceller(
                            echo_pipeline,
                            vad_threshold=barge_in_config.vad_threshold,
                            min_speech_ms=barge_in_config.min_speech_ms,
                            pre_roll_ms=barge_in_config.pre_roll_ms,
                        )
                    return manager

                def _pause_proactive() -> None:
                    if proactive is not None:
                        with contextlib.suppress(Exception):
                            # Cancel, do not merely delay: an already generated
                            # proactive line must not preempt the user.
                            proactive.cancel_cycle(args.conversation)

                def _cooldown_seconds() -> float:
                    raw = os.environ.get(
                        "AVA_VOICE_COOLDOWN_SECONDS", "0.3"
                    )
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        return 0.3
                    return max(0.0, min(2.0, value))

                def _voice_notice(text: str) -> None:
                    try:
                        asyncio.create_task(voice_output.write(text + "\n"))
                    except RuntimeError:  # loop shutting down
                        pass

                # The reply currently being generated, as an explicit task so
                # a barge-in can cancel exactly it (never the new turn). The
                # same slot also tracks a proactive delivery's task.
                active_reply_task: Optional[asyncio.Task] = None
                active_reply_generation: Optional[int] = None

                def _cancel_interrupted_reply(generation: int) -> None:
                    task = active_reply_task
                    if task is None or task.done():
                        return
                    if (
                        active_reply_generation is not None
                        and active_reply_generation != generation
                    ):
                        # A stale signal for a reply that is no longer active.
                        return
                    task.cancel()

                coordinator = VoiceCoordinator(
                    manager_factory=_make_asr_manager,
                    tts_manager=tts_manager,
                    cooldown_seconds=_cooldown_seconds(),
                    on_speech_started=_pause_proactive,
                    on_notice=_voice_notice,
                    barge_in=barge_in_config,
                    on_barge_in=_cancel_interrupted_reply,
                )
                # ``--voice`` means persistent listening from startup. A
                # missing key or device is reported once and text chat
                # continues; it never aborts the REPL.
                _mic_ok, _mic_message = await coordinator.enable()
                print(_mic_message)
                if repl_session is not None and voice_subtitle is not None:
                    # Part 3: show what ASR has recognized so far under the
                    # prompt. Runs on the loop (the coordinator consumer), and
                    # an empty text clears the line again.
                    _partial_sequence = 0

                    def _show_partial(_uid: int, text: str) -> None:
                        nonlocal _partial_sequence
                        if not text:
                            voice_subtitle.clear_if_current(_uid)
                            return
                        _partial_sequence += 1
                        voice_subtitle.set_partial(
                            _uid, _partial_sequence, text
                        )

                    coordinator.set_partial_listener(_show_partial)

            async def _prompt_once():
                if repl_session is not None:
                    return await repl_session.prompt_async("you> ")
                return await asyncio.to_thread(input, "you> ")

            async def _handle_command(line: str) -> bool:
                """Handle one slash command; return True to quit the REPL."""
                nonlocal addressee
                parts = line.split()
                cmd = parts[0].lower()
                if cmd == "/quit" or cmd == "/exit":
                    return True
                if cmd == "/quiet" and len(parts) == 2:
                    await rt.conversations.set_quiet_mode(
                        args.conversation, parts[1] == "on"
                    )
                    if parts[1] == "on" and proactive is not None:
                        # Quiet Mode stops every later proactive round now,
                        # not only after its next timer expires.
                        with contextlib.suppress(Exception):
                            proactive.cancel_cycle(args.conversation)
                    print(f"quiet_mode={'on' if parts[1] == 'on' else 'off'}")
                elif cmd == "/to" and len(parts) == 2 and parts[1] in ("agent", "human", "unknown"):
                    addressee = parts[1]
                    print(f"addressee -> {addressee}")
                elif cmd == "/speech" and len(parts) == 2 and parts[1] in ("on", "off"):
                    enabled = parts[1] == "on"
                    rt.orchestrator.speech_expression_enabled = enabled
                    provider_obj = getattr(rt.orchestrator, "response", None)
                    if hasattr(provider_obj, "speech_expression_enabled"):
                        provider_obj.speech_expression_enabled = enabled
                    print(
                        f"speech expression: {'on' if enabled else 'off'}"
                    )
                elif cmd == "/tts":
                    for output in _handle_tts_command(tts_manager, parts):
                        print(output)
                elif cmd == "/mic":
                    await _handle_mic_command(parts, voice_mode, coordinator)
                elif cmd == "/history":
                    for e in await rt.repos.utterance.list_events(None, 20):
                        print(f"  {e.event_id} to={e.addressee_hint.value} {e.text}")
                elif cmd == "/status":
                    ctx = await rt.repos.conversation.get(args.conversation)
                    print(f"provider: {rt.provider}")
                    print(f"router mode: {rt.router_mode}")
                    print(f"person: {person.person_id} ({person.display_name})")
                    print(f"conversation: {args.conversation}")
                    print(f"addressee: {addressee}")
                    print(f"quiet_mode: {'on' if ctx and ctx.quiet_mode else 'off'}")
                    print(
                        "speech expression: "
                        f"{'on' if getattr(rt.orchestrator, 'speech_expression_enabled', False) else 'off'}"
                    )
                    print(
                        f"tts: {'on' if tts_manager.enabled else 'off'}"
                        f" ({tts_manager.status().state})"
                    )
                    if coordinator is not None:
                        print(coordinator.description())
                    try:
                        web_enabled = await rt.repos.settings.get_bool(
                            "web_search.enabled", False
                        )
                        web_provider = (
                            await rt.repos.settings.get("web_search.provider") or ""
                        ).strip().lower() or "brave"
                    except Exception:
                        web_enabled, web_provider = False, "unknown"
                    print(
                        f"web search: {'on' if web_enabled else 'off'}"
                        f" (provider={web_provider},"
                        f" brave_key={'configured' if _brave_key_configured() else 'missing'})"
                    )
                elif cmd == "/help":
                    voice_help = (
                        "  /mic on|off|status|devices"
                        if voice_mode
                        else ""
                    )
                    print(
                        "commands: /to agent|human|unknown  /quiet on|off"
                        "  /speech on|off  /tts on|off|stop|status"
                        f"{voice_help}  /history  /status  /quit"
                    )
                else:
                    print("unknown command (see /help)")
                return False

            async def _voice_tts_started(generation: int) -> None:
                # Voice state is armed by _run_turn with its turn token; this
                # callback only renders the one user-visible line.
                await voice_output.write("[tts] Speaking...\n")

            async def _run_turn(
                text: str,
                source: EventSource,
                submitted_at: float,
                *,
                output=None,
                on_tts_started=None,
                person_id: Any = _DEFAULT_PERSON,
            ) -> dict:
                # Keyboard and voice both go through this one call path.
                # Keyboard callers omit ``person_id`` and keep the startup
                # person; a voice caller passes the resolved id (or ``None``
                # for an anonymous/unbound speaker).
                speaker_person_id = (
                    person.person_id
                    if person_id is _DEFAULT_PERSON
                    else person_id
                )
                if coordinator is not None:
                    turn_token = coordinator.begin_turn()
                else:
                    turn_token = None

                async def _tts_started_for_turn(generation: int) -> None:
                    nonlocal active_reply_generation
                    active_reply_generation = int(generation)
                    if coordinator is not None:
                        coordinator.on_reply_started(
                            generation, token=turn_token
                        )
                    if on_tts_started is not None:
                        await on_tts_started(generation)

                delivery = {"delivered": False}
                try:
                    delivery = await handle_user_message(
                        rt,
                        text=text,
                        source=source,
                        submitted_at=submitted_at,
                        conversation_id=args.conversation,
                        person_id=speaker_person_id,
                        addressee=addressee,
                        debug=bool(getattr(args, "debug", False)),
                        tts_manager=tts_manager,
                        output=output,
                        on_tts_started=_tts_started_for_turn,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                    message = f"[failed] {type(exc).__name__}: {exc}\n"
                    if output is not None:
                        await output.write(message)
                    else:
                        sys.stdout.write(message)
                        sys.stdout.flush()
                finally:
                    if coordinator is not None:
                        # Resume only after real playback completion (or
                        # immediately when no speech was produced). The token
                        # makes a reply superseded by a barge-in inert.
                        coordinator.on_turn_finished(
                            delivery, token=turn_token
                        )
                    if output is not None:
                        with contextlib.suppress(Exception):
                            await output.flush()
                return delivery

            if not voice_mode:
                # Legacy keyboard-only loop. With no microphone configured the
                # prompt is the only input owner and behaviour is unchanged.
                while True:
                    try:
                        line = (
                            await _read_line_with_proactive(
                                rt, repl_session, tts_manager=tts_manager
                            )
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not line:
                        continue
                    if line.startswith("/"):
                        if await _handle_command(line):
                            break
                        continue
                    await _run_turn(
                        line, EventSource.CLI, time.monotonic()
                    )
            else:
                # Unified input scheduler: one long-lived prompt task plus the
                # ASR-final, proactive-event and proactive-delivery waiters. A
                # submitted keyboard line is never cancelled or re-submitted; a
                # voice turn may run while the prompt stays active (output goes
                # above the input line), and an Enter pressed during that turn
                # is processed next.

                async def _deliver_proactive_voice_local(event) -> None:
                    await _deliver_proactive_voice(
                        event,
                        manager=proactive,
                        coordinator=coordinator,
                        tts_manager=tts_manager,
                        session=repl_session,
                        output=voice_output,
                    )

                async def _run_reply_turn(*turn_args, **turn_kwargs) -> None:
                    """Run one reply as an explicit task.

                    A confirmed barge-in cancels exactly this task (through the
                    coordinator callback). The ASR turn of the interrupting
                    utterance is untouched and arrives through the normal
                    final queue.
                    """
                    nonlocal active_reply_task, active_reply_generation
                    reply_task = asyncio.create_task(
                        _run_turn(*turn_args, **turn_kwargs),
                        name="ava-reply-turn",
                    )
                    active_reply_task = reply_task
                    active_reply_generation = None
                    try:
                        await reply_task
                    except asyncio.CancelledError:
                        if not reply_task.cancelled():
                            # This REPL task itself is being cancelled.
                            raise
                        with contextlib.suppress(Exception):
                            await voice_output.write("[interrupted]\n")
                    finally:
                        if active_reply_task is reply_task:
                            active_reply_task = None
                            active_reply_generation = None
                        if not reply_task.done():
                            reply_task.cancel()
                            with contextlib.suppress(
                                asyncio.CancelledError, Exception
                            ):
                                await reply_task

                async def _cancel_proactive() -> None:
                    nonlocal proactive_task, active_reply_task
                    nonlocal active_reply_generation
                    task = proactive_task
                    proactive_task = None
                    if task is not None:
                        await _cancel_async_task(task)
                    if active_reply_task is task:
                        active_reply_task = None
                        active_reply_generation = None

                prompt_task = asyncio.create_task(_prompt_once())
                proactive_task: Optional[asyncio.Task] = None
                quit_requested = False
                try:
                    while not quit_requested:
                        final_future = asyncio.ensure_future(
                            coordinator.next_final()
                        )
                        event_future = (
                            asyncio.ensure_future(proactive.next_event())
                            if proactive is not None and proactive_task is None
                            else None
                        )
                        waiters = {prompt_task, final_future}
                        if event_future is not None:
                            waiters.add(event_future)
                        if proactive_task is not None:
                            waiters.add(proactive_task)
                        done, _pending = await asyncio.wait(
                            waiters, return_when=asyncio.FIRST_COMPLETED
                        )

                        if prompt_task.done() and final_future not in done:
                            # A pressed Enter is already an accepted input and
                            # cancels any pending/deferred proactive delivery.
                            # A completed ASR final wins a tie (below): it is
                            # already-accepted user speech that must never be
                            # dropped, while the typed line stays queued and is
                            # consumed on the next iteration.
                            _discard_future(final_future)
                            _discard_future(event_future)
                            if proactive_task is not None:
                                await _cancel_proactive()
                            try:
                                line = prompt_task.result()
                            except (EOFError, KeyboardInterrupt):
                                print()
                                break
                            prompt_task = None
                            line = (line or "").strip()
                            if not line:
                                prompt_task = asyncio.create_task(
                                    _prompt_once()
                                )
                                continue
                            if line.startswith("/"):
                                quit_requested = await _handle_command(line)
                                if not quit_requested:
                                    prompt_task = asyncio.create_task(
                                        _prompt_once()
                                    )
                                continue
                            await _run_reply_turn(
                                line,
                                EventSource.CLI,
                                time.monotonic(),
                                output=voice_output,
                                on_tts_started=_voice_tts_started,
                            )
                            prompt_task = asyncio.create_task(
                                _prompt_once()
                            )
                            continue

                        if final_future in done:
                            _discard_future(event_future)
                            if proactive_task is not None:
                                await _cancel_proactive()
                            final = final_future.result()
                            # The ASR renderer does not print finals in the
                            # formal REPL; echo the accepted text exactly once,
                            # above the live input line. Partials are never
                            # echoed or persisted.
                            voice_person_id: Optional[str] = person.person_id
                            if voice_binding_enabled:
                                voice_person_id = await resolve_voice_person_id(
                                    rt,
                                    final.speaker,
                                    default_person_id=person.person_id,
                                    enabled=True,
                                )
                                if voice_person_id is None:
                                    # Never substitute the startup person for an
                                    # unbound speaker: anonymous handling means
                                    # no private Profile/Memory access.
                                    await voice_output.write(
                                        "[id] 当前声音未绑定到已登记的人，本条按匿名处理。\n"
                                    )
                            await voice_output.write(
                                f"you 🎤> {final.text}\n"
                            )
                            # The prompt (and its typed buffer) stays alive;
                            # output is drawn above the input line.
                            await _run_reply_turn(
                                final.text,
                                EventSource.VOICE,
                                time.monotonic(),
                                output=voice_output,
                                on_tts_started=_voice_tts_started,
                                person_id=voice_person_id,
                            )
                            continue

                        if proactive_task is not None and proactive_task in done:
                            _discard_future(final_future)
                            await _cancel_proactive()
                            continue

                        if event_future is not None and event_future in done:
                            _discard_future(final_future)
                            event = event_future.result()
                            if event is not None:
                                # Deferral is asynchronous and bounded to one
                                # proactive delivery at a time; new input
                                # cancels it without touching the manager's
                                # next scheduling time (no ack = not delivered).
                                proactive_task = asyncio.create_task(
                                    _deliver_proactive_voice_local(event),
                                    name="ava-proactive-delivery",
                                )
                                active_reply_task = proactive_task
                                active_reply_generation = None
                            continue

                        # Defensive: a waiter vanished without an item.
                        _discard_future(final_future)
                        _discard_future(event_future)
                finally:
                    if prompt_task is not None and not prompt_task.done():
                        prompt_task.cancel()
                        with contextlib.suppress(
                            asyncio.CancelledError, Exception
                        ):
                            await prompt_task
                    if proactive_task is not None:
                        await _cancel_proactive()
        finally:
            # Stop voice input, then speech/audio threads, before the event
            # loop closes; the REPL must exit without a recording, a playing
            # stream or a leftover worker.
            if coordinator is not None:
                with contextlib.suppress(Exception):
                    await coordinator.shutdown()
            await asyncio.to_thread(tts_manager.shutdown)
            confirmation = getattr(rt, "profile_confirmation", None)
            if confirmation is not None:
                await confirmation.shutdown()
            manager = getattr(rt, "proactive_manager", None)
            if manager is not None:
                await manager.shutdown()
            if worker_started:
                await rt.worker.stop()
    finally:
        await db.close()
    return 0


def _discard_future(future) -> None:
    """Cancel a pending waiter without consuming or re-submitting its data."""
    if future is not None and not future.done():
        future.cancel()


async def _cancel_async_task(task) -> None:
    """Cancel a background task and wait for its cleanup (no orphan tasks)."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _handle_mic_command(parts: list[str], voice_mode: bool, coordinator) -> None:
    """Handle ``/mic ...``; device switching stays env/config driven."""
    if not voice_mode or coordinator is None:
        print("/mic 需要在启动时加 --voice：python -m companion repl --voice")
        return
    action = parts[1].lower() if len(parts) == 2 else ""
    if action == "on":
        _ok, message = await coordinator.enable()
        print(message)
    elif action == "off":
        print(await coordinator.disable())
    elif action == "status":
        print(coordinator.description())
        stats = coordinator.stats
        print(
            "mic events: "
            f"accepted={stats.finals_accepted}"
            f"  stale={stats.stale_finals}"
            f"  errors={stats.turn_errors}"
        )
        if coordinator.last_error:
            print(f"last_error: {coordinator.last_error}")
        barge = coordinator.barge_in_status()
        if barge.get("enabled"):
            print(
                "barge-in: "
                f"armed={str(barge.get('armed')).lower()}"
                f"  candidates={barge.get('candidates', 0)}"
                f"  confirmed={barge.get('confirmed', 0)}"
                f"  fallback={barge.get('fallback_reason') or '-'}"
            )
            latency = barge.get("stop_latency_ms")
            confirm_latency = (barge.get("asr") or {}).get(
                "last_user_speech_to_confirm_ms"
            )
            print(
                "  latency: "
                f"speech_to_confirm={confirm_latency if confirm_latency is not None else 'n/a'}ms"
                # This is the stop() call time, not measured device silence.
                f"  confirm_to_stop_call={latency if latency is not None else 'n/a'}ms"
                f"  mic_dropped={barge.get('mic_dropped_chunks', 0)}"
            )
            asr_stats = barge.get("asr") or {}
            aec = asr_stats.get("aec") or {}
            if asr_stats:
                print(
                    "  mode: "
                    f"{asr_stats.get('mode')}"
                    f"  detecting={str(asr_stats.get('detecting')).lower()}"
                    f"  tail={str(asr_stats.get('tail')).lower()}"
                    f"  gaps={asr_stats.get('gaps', 0)}"
                )
            if aec:
                print(
                    "  aec: "
                    f"valid_ratio={aec.get('valid_ratio')}"
                    f"  p50={aec.get('aec_p50_ms')}ms"
                    f"  p95={aec.get('aec_p95_ms')}ms"
                    f"  rms_near={aec.get('near_rms_p50')}"
                    f"  rms_clean={aec.get('clean_rms_p50')}"
                    f"  ref_frames={aec.get('reference_frames')}"
                    f"  ref_consumed={aec.get('reference_consumed')}"
                    f"  ref_expired={aec.get('reference_expired')}"
                    f"  ref_dropped={aec.get('reference_dropped')}"
                    f"  discontinuities={aec.get('sync_discontinuities')}"
                    f"  mic_jumps={aec.get('mic_discontinuities')}"
                    f"  tail_frames={aec.get('tail_frames')}"
                    f"  drops={aec.get('playback_dropped')}"
                )
                print(
                    "  align: "
                    f"far_seq={aec.get('far_sequence')}"
                    f"  near_seq={aec.get('near_sequence')}"
                    f"  ref_age_ms={aec.get('reference_age_ms')}"
                    f"  offset_ms={aec.get('alignment_offset_ms')}"
                    f"  generation={aec.get('generation')}"
                    f"  aec_valid={str(aec.get('aec_valid')).lower()}"
                )
                reasons = aec.get("invalid_reasons") or {}
                if reasons:
                    summary = " ".join(
                        f"{key}={value}"
                        for key, value in sorted(reasons.items())
                    )
                    print(f"  aec invalid: {summary}")
    elif action == "devices":
        try:
            from companion.asr.microphone import list_input_devices

            devices = await asyncio.to_thread(list_input_devices)
        except Exception as exc:  # noqa: BLE001 - dependency/device failure
            print(f"error: {exc}")
            return
        if not devices:
            print("没有找到音频输入设备。")
            return
        for device in devices:
            flags = ["系统默认输入"] if device.is_default_input else []
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"{device.index:>2}  {device.name}  "
                f"默认采样率 {device.default_sample_rate} Hz  "
                f"输入 {device.max_input_channels}{suffix}"
            )
        print("选择设备：设置 AVA_ASR_DEVICE（索引或名称片段）后重新开启监听。")
    else:
        print("usage: /mic on|off|status|devices")


def _handle_tts_command(manager, parts: list[str]) -> list[str]:
    """Handle ``/tts ...``; returns lines for the REPL to print.

    Speech expression (``/speech``) and TTS are independent switches: this
    function never touches the orchestrator or the response provider.
    """
    action = parts[1].lower() if len(parts) == 2 else ""
    if action == "on":
        lines = [
            "注意：开启 TTS 后，Ava 的回复文本会按句流式发送到阿里云百炼"
            "（DashScope）进行语音合成，音频只在本机播放。"
        ]
        ok, message = manager.enable()
        lines.append(message)
        if not ok:
            lines.append("tts: off (未启用)")
        return lines
    if action == "off":
        manager.disable()
        return ["tts: off"]
    if action == "stop":
        manager.stop()
        suffix = "" if manager.enabled else " (tts off)"
        return [f"tts: stopped current speech{suffix}"]
    if action == "status":
        lines = manager.status().lines()
        metrics = manager.metrics()
        if metrics is not None:
            lines.append(_format_tts_metrics(metrics))
        return lines
    return ["usage: /tts on|off|stop|status"]


def _format_tts_metrics(metrics: dict) -> str:
    def value(key: str) -> str:
        number = metrics.get(key)
        return "-" if number is None else f"{number:.0f}ms"

    if metrics.get("user_to_first_text_ms") is not None:
        # Detailed Phase B report, all relative to the user's submission.
        # Concurrent phases overlap, so these rows must not be summed.
        rows: list[tuple[str, str]] = []

        def row(label: str, key: str) -> None:
            number = metrics.get(key)
            if number is not None:
                rows.append((label, f"{number:.0f} ms"))

        row("Router", "user_to_router_ms")
        row("Preparation", "router_to_response_ms")
        row("Speaker first text", "response_to_first_text_ms")
        row("First segment wait", "first_text_to_first_segment_ms")
        row("WebSocket handshake", "tts_connect_ms")
        row("TTS first PCM", "first_segment_to_first_pcm_ms")
        row("Audio queue/playback", "first_pcm_to_write_ms")
        if not rows:
            # Router/preparation are absent (legacy path): show the combined
            # user→response_started phase instead of a blank section.
            row("Router/preparation", "user_to_response_started_ms")
        totals = [
            ("User → First text", "user_to_first_text_ms"),
            ("User → First audio", "user_to_first_audio_ms"),
            ("User → Text complete", "user_to_text_done_ms"),
        ]
        lines = ["Last response timing:"]
        for label, text in rows:
            lines.append(f"  {label + ':':<24}{text:>8}")
        lines.append("")
        for label, key in totals:
            number = metrics.get(key)
            lines.append(
                f"  {label + ':':<24}"
                f"{'-' if number is None else f'{number:.0f} ms':>8}"
            )
        return "\n".join(lines)

    if metrics.get("response_started_to_first_audio_played_ms") is not None:
        # Phase B: how soon speech starts relative to the reply and its first
        # visible text, not relative to a finished reply.
        return (
            "last latency: response→first_audio "
            f"{value('response_started_to_first_audio_played_ms')}"
            "  first_text→first_audio "
            f"{value('first_display_delta_to_first_audio_played_ms')}"
            "  segment→first_audio "
            f"{value('first_segment_sent_to_first_audio_played_ms')}"
            "  first_audio→played "
            f"{value('first_audio_received_to_played_ms')}"
            "  playback "
            f"{value('response_to_playback_done_ms')}"
        )
    return (
        "last latency: reply→speech "
        f"{value('reply_to_speech_start_ms')}"
        "  request→first_audio "
        f"{value('request_to_first_audio_ms')}"
        "  first_audio→played "
        f"{value('first_audio_received_to_played_ms')}"
        "  playback "
        f"{value('first_audio_to_playback_done_ms')}"
    )


class _PartialSubtitle:
    """Live provisional ASR text drawn under the prompt while the user speaks.

    Display-only: the text never becomes a message, is never persisted and is
    replaced by the final result. It is installed as the prompt's bottom
    toolbar, which repaints in place instead of scrolling the terminal, and the
    toolbar is removed again while there is nothing to show.
    """

    __slots__ = ("_text", "_session", "_uid", "_sequence")

    def __init__(self) -> None:
        self._text = ""
        self._session = None
        self._uid: Optional[int] = None
        self._sequence = 0

    def attach(self, session) -> None:
        """Bind the prompt session; safe when ``None`` (off-TTY)."""
        self._session = session
        self._apply()

    @property
    def active_utterance_id(self) -> Optional[int]:
        return self._uid

    def set_partial(self, utterance_id: int, sequence: int, text: str) -> None:
        """Show one provisional revision, scoped to the recording turn.

        Only the active utterance may update the subtitle. An older
        utterance's late partial (or an older sequence of the current one) is
        ignored, so a new turn cannot inherit a previous turn's text.
        """
        uid = int(utterance_id)
        seq = int(sequence)
        if self._uid is not None and uid != self._uid:
            self._uid = uid
            self._sequence = seq
        elif self._uid == uid and seq < self._sequence:
            return
        else:
            self._uid = uid
            self._sequence = seq
        self.set(text)

    def clear_if_current(self, utterance_id: int) -> None:
        """Clear only if the event belongs to the turn being displayed."""
        if self._uid is not None and int(utterance_id) != self._uid:
            return
        self._uid = None
        self._sequence = 0
        self.clear()

    def set(self, text: str) -> None:
        self._text = " ".join(str(text or "").split())
        self._apply()
        self.invalidate()

    def clear(self) -> None:
        self._text = ""
        self._apply()
        self.invalidate()

    def _apply(self) -> None:
        session = self._session
        if session is None:
            return
        with contextlib.suppress(Exception):
            # ``None`` hides the toolbar entirely, so an idle prompt keeps the
            # same layout it had before voice mode.
            session.bottom_toolbar = self.render if self._text else None

    def render(self) -> str:
        return f"🎙 {self._text}" if self._text else ""

    def invalidate(self) -> None:
        session = self._session
        if session is None:
            return
        app = getattr(session, "app", None)
        if app is not None and bool(getattr(app, "is_running", False)):
            with contextlib.suppress(Exception):
                app.invalidate()


def _build_prompt_session(subtitle=None):
    """Return an interactive prompt_toolkit session, or ``None`` off-TTY.

    ``prompt_toolkit`` owns the input line for the whole read. That is what
    makes a proactive message safe to print while the person is typing: the
    editor buffer is never read or restored from another thread, and the
    prompt is redrawn by the same component that owns it.

    ``subtitle`` (voice mode only) is a :class:`_PartialSubtitle`; attaching it
    makes recognized speech appear while the person is still talking instead
    of only after the final result.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        from prompt_toolkit import PromptSession
    except Exception:
        return None
    try:
        session = PromptSession()
    except Exception:
        return None
    if subtitle is not None:
        subtitle.attach(session)
    return session


def _print_proactive_line(text: str) -> None:
    """Write one delivered proactive message on the normal assistant surface."""
    sys.stdout.write("assistant> " + text + "\n")
    sys.stdout.flush()


async def _render_proactive_event(
    event,
    session=None,
    *,
    tts_manager=None,
    coordinator=None,
    output=None,
    turn_token: int | None = None,
) -> bool:
    """Display a proactive message, then speak it through the shared TTS.

    Order is fixed: the terminal line is shown first, then the same text is
    submitted to the existing :class:`TTSManager` (never a second Router or
    conversation record). ``display_text`` is used for speech so legal voice
    labels reach Qwen; ``text`` stays the canonical persisted reply.

    Returns True when the event was a completed, non-empty proactive message
    (whether or not TTS happened to be enabled).
    """
    if event.type != "response_completed" or not (event.text or "").strip():
        return False
    # display_text keeps voice-expression labels for the surface; text remains
    # the canonical, persisted reply.
    text = (
        event.display_text
        if getattr(event, "display_text", None) is not None
        else event.text
    )
    app = getattr(session, "app", None) if session is not None else None
    if app is not None and getattr(app, "is_running", False):
        # Print inside prompt_toolkit's terminal, above the live input line.
        # The buffer and cursor stay owned by the prompt session.
        from prompt_toolkit.application import run_in_terminal

        try:
            await run_in_terminal(lambda: _print_proactive_line(text))
        except Exception:
            _print_proactive_line(text)
    else:
        _print_proactive_line(text)

    spoke = False
    if tts_manager is not None:
        speech_text = text or event.text
        if speech_text and speech_text.strip():
            try:
                spoke = bool(
                    tts_manager.submit(
                        SpeechJob(
                            response_id=event.response_id,
                            conversation_id=event.conversation_id,
                            text=speech_text,
                            generation=getattr(tts_manager, "generation", 0),
                            reply_completed_at=time.monotonic(),
                        )
                    )
                )
            except Exception:  # noqa: BLE001 - text delivery already happened
                spoke = False
        if spoke and coordinator is not None:
            try:
                coordinator.on_reply_started(
                    tts_manager.generation, token=turn_token
                )
            except TypeError:
                # Embedders/tests that implement the pre-barge-in two-argument
                # callback keep working unchanged.
                with contextlib.suppress(Exception):
                    coordinator.on_reply_started(tts_manager.generation)
            except Exception:  # noqa: BLE001 - voice state is best effort
                pass
        if spoke and output is not None:
            with contextlib.suppress(Exception):
                await output.write("[tts] Speaking...\n")
    logging.getLogger("companion.proactive").info(
        "proactive_surface_rendered conversation_id=%s source_event_id=%s"
        " message_id=%s chars=%d tts_submitted=%s",
        getattr(event, "conversation_id", None),
        getattr(event, "event_id", None),
        getattr(event, "response_id", None),
        len(text or ""),
        spoke,
    )
    return True


async def _ack_proactive_event(
    manager,
    event,
    *,
    delivered: bool,
    reason: str,
    stage: str = "text_displayed",
) -> None:
    """Report whether one proactive message really reached the surface.

    Prefers the async, database-confirmed acknowledgement so a write failure
    is not reported as a fully successful delivery. The synchronous methods
    remain as a compatibility fallback for embedders/tests.
    """
    if manager is None or event is None:
        return
    acknowledge = getattr(manager, "acknowledge_delivery", None)
    if callable(acknowledge):
        try:
            await acknowledge(
                event,
                delivered=delivered,
                reason=reason,
                stage=stage,
            )
            return
        except TypeError:
            # Older/simpler manager implementations take a shorter signature.
            pass
        except Exception:  # noqa: BLE001 - acknowledgement never breaks chat
            return
    try:
        if delivered and hasattr(manager, "note_proactive_delivered"):
            manager.note_proactive_delivered(event)
        elif hasattr(manager, "note_proactive_cancelled"):
            manager.note_proactive_cancelled(event, reason)
    except Exception:  # noqa: BLE001 - acknowledgement never breaks chat
        pass


async def _deliver_proactive_voice(
    event,
    *,
    manager,
    coordinator,
    tts_manager=None,
    session=None,
    output=None,
) -> bool:
    """Display, speak and acknowledge one proactive message in voice mode.

    Deferred (never silently dropped) while ASR is recording or Ava is
    speaking; the event is re-checked after that wait. The mic is paused for
    the proactive playback exactly like a reply turn, and listening resumes
    only after the real playback-complete event and the existing cooldown.
    A new keyboard line or voice final cancels this task.
    """
    log = logging.getLogger("companion.proactive")
    turn_token: int | None = None
    try:
        if coordinator is not None:
            await coordinator.wait_until_proactive_allowed()
            # Let an already-queued ASR speech-start event run before taking
            # the turn: a user who has just begun speaking keeps the floor.
            await asyncio.sleep(0)
            if not coordinator.should_deliver_proactive():
                await _ack_proactive_event(
                    manager,
                    event,
                    delivered=False,
                    reason="user_started_speaking",
                )
                return False
        if manager is not None:
            can_deliver = getattr(manager, "can_deliver", None)
            if callable(can_deliver) and not await can_deliver(event):
                await _ack_proactive_event(
                    manager,
                    event,
                    delivered=False,
                    reason="stale_before_delivery",
                )
                return False
        if coordinator is not None:
            turn_token = coordinator.begin_turn()
        rendered = await _render_proactive_event(
            event,
            session,
            tts_manager=tts_manager,
            coordinator=coordinator,
            output=output,
            turn_token=turn_token,
        )
        if not rendered:
            if coordinator is not None:
                coordinator.on_turn_finished(
                    {"delivered": False}, token=turn_token
                )
            await _ack_proactive_event(
                manager,
                event,
                delivered=False,
                reason="render_failed",
            )
            return False
        if coordinator is not None:
            coordinator.on_turn_finished(
                {
                    "delivered": True,
                    "status": "COMPLETED",
                    "action": "RESPOND",
                },
                token=turn_token,
            )
            # Resume only after the real playback-complete event.
            await coordinator.wait_until_proactive_allowed()
        # Telegram/terminal text delivery is not the same as a fully played
        # voice message. A barge-in or a stop halfway through playback must be
        # a cancellation (no round consumed, retry allowed), never a full
        # delivery just because the text line was printed.
        outcome = None
        if tts_manager is not None:
            getter = getattr(tts_manager, "playback_outcome_for", None)
            if callable(getter):
                with contextlib.suppress(Exception):
                    outcome = getter(getattr(event, "response_id", None))
        if outcome in ("cancelled", "failed"):
            await _ack_proactive_event(
                manager,
                event,
                delivered=False,
                reason="tts_interrupted",
                stage="tts_playback_completed",
            )
            log.info(
                "proactive_voice_interrupted conversation_id=%s"
                " source_event_id=%s message_id=%s outcome=%s",
                getattr(event, "conversation_id", None),
                getattr(event, "event_id", None),
                getattr(event, "response_id", None),
                outcome,
            )
            return False
        await _ack_proactive_event(
            manager,
            event,
            delivered=True,
            reason="delivered",
            stage=(
                "tts_playback_completed"
                if outcome == "finished"
                else "text_displayed"
            ),
        )
        log.info(
            "proactive_voice_delivered conversation_id=%s"
            " source_event_id=%s message_id=%s",
            getattr(event, "conversation_id", None),
            getattr(event, "event_id", None),
            getattr(event, "response_id", None),
        )
        return True
    except asyncio.CancelledError:
        # A keyboard line or a voice final won: never let the old proactive
        # turn keep the mic paused.
        if coordinator is not None:
            coordinator.on_turn_finished(
                {"delivered": False}, token=turn_token
            )
        await _ack_proactive_event(
            manager, event, delivered=False, reason="interrupted_by_user"
        )
        raise
    except Exception as exc:  # noqa: BLE001 - REPL stays alive
        if coordinator is not None:
            coordinator.on_turn_finished(
                {"delivered": False}, token=turn_token
            )
        await _ack_proactive_event(
            manager, event, delivered=False, reason="surface_error"
        )
        log.warning(
            "proactive_voice_failed conversation_id=%s error=%s",
            getattr(event, "conversation_id", None),
            type(exc).__name__,
        )
        return False


async def _read_line_with_prompt_session(
    manager, session, *, tts_manager=None
) -> str:
    """One async prompt; proactive messages are inserted above it."""
    if manager is None:
        return await session.prompt_async("you> ")

    input_future = asyncio.ensure_future(session.prompt_async("you> "))
    try:
        while True:
            if input_future.done():
                return input_future.result()

            event_future = asyncio.ensure_future(manager.next_event())
            done, _pending = await asyncio.wait(
                {input_future, event_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_future.done():
                # The submitted line wins every tie. Checking ``done`` rather
                # than membership in ``done`` covers the case where both
                # completed before the loop resumed: the proactive event is
                # stale, and the caller is about to invalidate it.
                event_future.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await event_future
                return input_future.result()

            event = event_future.result()
            if event is not None:
                delivered = await _render_proactive_event(
                    event, session, tts_manager=tts_manager
                )
                await _ack_proactive_event(
                    manager,
                    event,
                    delivered=delivered,
                    reason=(
                        "delivered" if delivered else "render_failed"
                    ),
                    stage="text_displayed",
                )
    finally:
        if not input_future.done():
            input_future.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await input_future


async def _read_line_with_thread(manager, *, tts_manager=None) -> str:
    """Fallback for non-TTY runs; never redraws a Readline buffer.

    ``input()`` runs in one worker thread. A proactive message is printed as
    plain output; the submitted text already belongs to that thread, so it is
    never re-read or reprinted here.
    """
    loop = asyncio.get_running_loop()
    input_future = loop.run_in_executor(None, input, "you> ")
    try:
        while True:
            if input_future.done():
                return input_future.result()

            event_future = asyncio.ensure_future(manager.next_event())
            done, _pending = await asyncio.wait(
                {input_future, event_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_future.done():
                event_future.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await event_future
                return input_future.result()

            event = event_future.result()
            if event is not None:
                delivered = await _render_proactive_event(
                    event, None, tts_manager=tts_manager
                )
                await _ack_proactive_event(
                    manager,
                    event,
                    delivered=delivered,
                    reason=(
                        "delivered" if delivered else "render_failed"
                    ),
                    stage="text_displayed",
                )
    finally:
        if not input_future.done():
            input_future.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await input_future


async def _read_line_with_proactive(rt, session=None, *, tts_manager=None) -> str:
    """Read one line, showing delivered proactive replies without losing input.

    Exactly one input owner exists per prompt: a prompt_toolkit session on a
    TTY, or a single ``input()`` thread off-TTY. Proactive output is rendered
    through that owner or as plain lines; the old Readline-buffer redraw is
    gone, so an already submitted line can never be restored to the terminal.
    """
    manager = getattr(rt, "proactive_manager", None)
    if session is not None:
        return await _read_line_with_prompt_session(
            manager, session, tts_manager=tts_manager
        )
    if manager is None:
        return await asyncio.to_thread(input, "you> ")
    return await _read_line_with_thread(manager, tts_manager=tts_manager)


class _DirectTurnOutput:
    """Legacy output: write straight to stdout (prompt is not active)."""

    async def write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    async def flush(self) -> None:
        return None


class _PromptTurnOutput:
    """Write reply output above a live prompt_toolkit input line.

    While the prompt owns the terminal, raw writes would corrupt the line.
    Text is therefore buffered into complete lines and each line is printed
    through ``run_in_terminal``, which redraws the prompt (and keeps whatever
    the user has typed) underneath. When the prompt is not running, output is
    written directly so streaming stays immediate.
    """

    def __init__(self, session) -> None:
        self._session = session
        self._buffer = ""
        self._lock = asyncio.Lock()

    def _prompt_running(self) -> bool:
        app = getattr(self._session, "app", None)
        return app is not None and bool(getattr(app, "is_running", False))

    async def write(self, text: str) -> None:
        if not text:
            return
        async with self._lock:
            if not self._prompt_running():
                _raw_stdout(text)
                return
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                await self._emit(line + "\n")

    async def flush(self) -> None:
        async with self._lock:
            if self._buffer:
                line, self._buffer = self._buffer, ""
                await self._emit(line + "\n")

    async def _emit(self, line: str) -> None:
        from prompt_toolkit.application import run_in_terminal

        try:
            await run_in_terminal(lambda: _raw_stdout(line))
        except Exception:  # noqa: BLE001 - fall back to plain output
            _raw_stdout(line)


def _raw_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


async def _run_streaming_turn(
    rt,
    event,
    *,
    debug: bool = False,
    tts_manager=None,
    submitted_at: float | None = None,
    output=None,
    on_tts_started=None,
) -> dict:
    """Render one turn from the real streaming API.

    Deltas are written and flushed as they arrive; the full text is never
    printed a second time. A failure or interruption is reported as such rather
    than as a completed reply.

    A turn that produces no response event (WAIT / IGNORE) is complete silence
    in the normal chat surface: the user asked for no reply, and no
    ``assistant>`` line or action label is invented. With ``debug=True`` the
    persisted action and reason are printed, so the same decision stays
    inspectable without leaking into ordinary conversation.
    """
    saw_started = False
    saw_delta = False
    saw_terminal = False
    completed_status = None
    completed_action = None
    # Phase B: one bidirectional TTS session per reply. ``feed``/``finish``/
    # ``cancel_stream`` never block the event loop; sender and receiver run on
    # their own threads.
    tts_stream = None
    tts_fed = False
    tts_generation = None
    # One monotonic origin for the whole turn: the moment the user's message
    # was accepted (fallback: this function's start for non-REPL callers).
    origin = submitted_at if submitted_at is not None else time.monotonic()
    first_text_at = None
    completed_at = None
    tts_on = bool(tts_manager is not None and getattr(tts_manager, "enabled", False))
    timing_printed = False
    sink = output if output is not None else _DirectTurnOutput()

    async def _write(text: str) -> None:
        await sink.write(text)

    async def _line(text: str) -> None:
        await sink.write(text + "\n")

    async def _print_timing() -> None:
        nonlocal timing_printed
        if timing_printed or completed_at is None:
            return
        timing_printed = True
        first = (
            "n/a"
            if first_text_at is None
            else f"{(first_text_at - origin):.1f}s"
        )
        done = f"{(completed_at - origin):.1f}s"
        audio = "off"
        if tts_on:
            audio = "n/a"
            if tts_generation is not None:
                audio = "pending"
                try:
                    metrics = tts_manager.metrics(tts_generation)
                except Exception:  # noqa: BLE001 - timing never breaks text
                    metrics = None
                first_audio_ms = (
                    metrics.get("user_to_first_audio_ms")
                    if metrics
                    else None
                )
                if first_audio_ms is not None:
                    audio = f"{first_audio_ms / 1000:.1f}s"
        await _line(
            f"[timing] first_text={first} | text_done={done}"
            f" | first_audio={audio}"
        )

    def _cancel_tts() -> None:
        nonlocal tts_stream
        if tts_stream is None:
            return
        if tts_manager is not None:
            tts_manager.cancel_stream(tts_stream)
        else:
            tts_stream.cancel()
        tts_stream = None

    stream = rt.orchestrator.process_utterance_stream(event)
    try:
        async for item in stream:
            if item.type == "response_started":
                saw_started = True
                await _write("assistant> ")
                if tts_manager is not None:
                    tts_stream = tts_manager.start_stream(
                        item.response_id,
                        item.conversation_id,
                        submitted_at=origin,
                        router_completed_at=getattr(
                            item, "router_completed_at", None
                        ),
                    )
                    tts_generation = (
                        tts_stream.generation
                        if tts_stream is not None
                        else None
                    )
                    if tts_generation is not None and on_tts_started is not None:
                        await on_tts_started(tts_generation)
            elif item.type == "response_delta":
                saw_delta = True
                # The display channel keeps voice labels inline; the canonical
                # ``delta`` remains the persisted text.
                delta = (
                    item.display_delta
                    if item.display_delta is not None
                    else (item.delta or "")
                )
                await _write(delta)
                if first_text_at is None and (item.delta or "").strip():
                    # The canonical channel proves this is real reply text,
                    # not a voice-expression tag split across deltas.
                    first_text_at = time.monotonic()
                    if tts_stream is not None:
                        tts_stream.note_first_text()
                if tts_stream is not None and delta:
                    # Never ``display_delta or delta``: an empty display delta
                    # is valid (e.g. a stripped header) and must not fall back
                    # to text that was already synthesised.
                    tts_stream.feed(delta)
                    tts_fed = True
            elif item.type == "response_completed":
                saw_terminal = True
                completed_at = time.monotonic()
                completed_status = item.status
                completed_action = item.action
                if not saw_started:
                    await _write("assistant> ")
                    saw_started = True
                if not saw_delta and (item.display_text or item.text):
                    # A provider without incremental chunks still produces a
                    # real reply; show it once, here, not twice.
                    await _write(item.display_text or item.text)
                if item.status == "TRUNCATED":
                    await _write("\n[truncated: token limit]\n")
                else:
                    await _write("\n")
                if tts_stream is not None:
                    if item.status == "COMPLETED":
                        if not tts_fed:
                            # No delta reached TTS (non-streaming provider):
                            # hand over the complete display text once.
                            tail = item.display_text or item.text
                            if tail and tail.strip():
                                tts_stream.feed(tail)
                        # Only the buffered tail is flushed here; deltas that
                        # were already sent are never re-sent.
                        tts_stream.finish()
                    else:
                        # TRUNCATED / any non-success end: stop speaking the
                        # unplayed remainder instead of synthesising a partial
                        # reply to the end.
                        _cancel_tts()
                elif (
                    tts_manager is not None
                    and getattr(tts_manager, "enabled", False)
                    and item.status == "COMPLETED"
                ):
                    # Phase A fallback (stream unavailable). The display
                    # channel is used verbatim so legal [tags] reach the TTS
                    # voice; VOICE=/TEXT= headers are already removed.
                    speech_text = item.display_text or item.text
                    if speech_text and speech_text.strip():
                        if tts_manager.submit(
                            SpeechJob(
                                response_id=item.response_id,
                                conversation_id=item.conversation_id,
                                text=speech_text,
                                generation=tts_manager.generation,
                                reply_completed_at=completed_at,
                            )
                        ):
                            tts_generation = tts_manager.generation
                            if on_tts_started is not None:
                                await on_tts_started(tts_generation)
                if not saw_delta and (
                    item.text or item.display_text or ""
                ).strip():
                    # No incremental deltas: the text became visible now.
                    first_text_at = completed_at
                await _print_timing()
            elif item.type == "response_failed":
                saw_terminal = True
                _cancel_tts()
                if not saw_delta and item.text:
                    await _write(item.text)
                if saw_started:
                    await _write("\n")
                await _line(f"[failed] {item.error or ''}".rstrip())
            elif item.type == "response_cancelled":
                saw_terminal = True
                _cancel_tts()
                if saw_started:
                    await _write("\n")
                await _line("[cancelled]")
    except KeyboardInterrupt:
        # Let the orchestrator persist the partial reply as CANCELLED and
        # release its conversation lock before returning to the prompt.
        _cancel_tts()
        with contextlib.suppress(Exception):
            await stream.aclose()
        if saw_started:
            await _write("\n")
        await _line("[cancelled]")
        return {"delivered": False}
    except asyncio.CancelledError:
        # A confirmed barge-in cancelled this reply. Stop and discard the old
        # speech stream, close the orchestrator generator so it persists the
        # partial reply as CANCELLED and releases the conversation lock, then
        # let the cancellation reach the scheduler. No old text is displayed
        # and the REPL keeps running for the interrupting utterance.
        _cancel_tts()
        with contextlib.suppress(Exception):
            await stream.aclose()
        if saw_started:
            with contextlib.suppress(Exception):
                await _write("\n")
        raise
    except Exception as exc:  # provider/validation failure is shown truthfully
        _cancel_tts()
        if saw_started:
            await _write("\n")
        await _line(f"[failed] {type(exc).__name__}: {exc}")
        return {"delivered": False}

    if saw_terminal:
        return {
            "delivered": True,
            "status": completed_status,
            "action": completed_action,
        }
    # No terminal event: nothing is allowed to keep speaking.
    _cancel_tts()
    # No response event was emitted: the local policy chose WAIT or IGNORE.
    # In the ordinary chat surface this is intentional silence, so nothing is
    # printed. Diagnostics remain available behind ``--debug``.
    if not debug:
        return {"delivered": False}
    action = None
    with contextlib.suppress(Exception):
        action = await rt.repos.analysis.get_action(event.event_id)
    if action is None:
        await _line("[no action]")
        return {"delivered": False}
    reason = f" ({action.reason_code.value})" if action.reason_code else ""
    await _line(f"[{action.action_type.value}{reason}]")
    return {"delivered": False}


async def handle_user_message(
    rt,
    *,
    text: str,
    source: EventSource,
    submitted_at: float,
    conversation_id: str,
    person_id: Optional[str],
    addressee: str,
    debug: bool = False,
    tts_manager=None,
    output=None,
    on_tts_started=None,
    turn_runner=None,
) -> dict:
    """The only path from accepted user text to Ava's reply.

    Keyboard (``EventSource.CLI``) and voice (``EventSource.VOICE``) both call
    this, so proactive invalidation, speaking interruption, streaming,
    persistence and the post-reply proactive schedule are identical. It never
    creates a second Router/Speaker flow.
    """
    runner = turn_runner or _run_streaming_turn
    event = make_utterance_event(
        conversation_id=conversation_id,
        text=text,
        addressee_hint=addressee,
        speaker_person_id=person_id,
        source=source,
    )
    manager = getattr(rt, "proactive_manager", None)
    if manager is not None:
        # Invalidate any pending check before the turn begins.
        manager.note_user_input(event.conversation_id, event.event_id)
    # A new ordinary chat or voice input interrupts unfinished speech from
    # the previous reply. This never cancels text generation.
    if tts_manager is not None:
        tts_manager.stop()
    delivery = await runner(
        rt,
        event,
        debug=debug,
        tts_manager=tts_manager,
        submitted_at=submitted_at,
        output=output,
        on_tts_started=on_tts_started,
    )
    if (
        manager is not None
        and delivery.get("delivered")
        and delivery.get("status") == "COMPLETED"
        and delivery.get("action") == "RESPOND"
    ):
        # Only a real-user-triggered, complete, delivered normal reply
        # schedules a check. WAIT/IGNORE, failures, cancellations and
        # truncated output never do.
        manager.on_response_delivered(
            conversation_id=event.conversation_id,
            person_id=person_id,
            source_event_id=event.event_id,
        )
    return delivery


def _brave_key_configured() -> bool:
    """Whether a Brave Search key is present — its value is never printed."""
    return bool(os.environ.get(BRAVE_KEY_ENV, "").strip())


def _valid_web_provider(name: str | None) -> str:
    """Normalise a provider choice; default auto-selects Brave when a key exists."""
    name = (name or "").strip().lower()
    if name in ("", "auto"):
        return "brave" if _brave_key_configured() else "duckduckgo"
    return name


async def cmd_web_status(args) -> int:
    db = await _connect(args.db)
    try:
        settings = make_repositories(db).settings
        enabled = await settings.get_bool("web_search.enabled", False)
        provider = (await settings.get("web_search.provider") or "").strip().lower() or "brave"
        print("web search:")
        print(f"  enabled: {str(enabled).lower()}")
        print(f"  provider: {provider}")
        print(f"  brave_api_key: {'configured' if _brave_key_configured() else 'missing'}")
        errors: list[str] = []
        if provider not in ("brave", "duckduckgo", "http", "html", "mock", "offline"):
            errors.append(f"unknown provider {provider!r}")
        if provider in ("brave", "") and not _brave_key_configured():
            errors.append("brave selected but BRAVE_SEARCH_API_KEY is missing")
        if errors:
            for error in errors:
                print(f"  config_error: {error}")
        else:
            print("  config_error: none")
        rt = build_runtime(db)
        people = await rt.registry.list()
        if not people:
            print("  allow_external_ai: (no people registered)")
        for p in people:
            consent = await rt.registry.get_consent(p.person_id)
            allowed = bool(consent and consent.allow_external_ai)
            print(f"  allow_external_ai[{p.display_name}]: {str(allowed).lower()}")
    finally:
        await db.close()
    return 0


async def cmd_web_enable(args) -> int:
    db = await _connect(args.db)
    try:
        provider = _valid_web_provider(getattr(args, "provider", None))
        if provider not in ("brave", "duckduckgo"):
            print(f"error: unsupported provider {provider!r} (brave or duckduckgo)")
            return 1
        if provider == "brave" and not _brave_key_configured():
            print(
                "error: BRAVE_SEARCH_API_KEY is not set;"
                " use --provider duckduckgo or set the key"
            )
            return 1
        settings = make_repositories(db).settings
        await settings.set("web_search.enabled", "true")
        await settings.set("web_search.provider", provider)
        print(f"web search enabled (provider={provider})")
    finally:
        await db.close()
    return 0


async def cmd_web_disable(args) -> int:
    db = await _connect(args.db)
    try:
        settings = make_repositories(db).settings
        await settings.set("web_search.enabled", "false")
        print("web search disabled")
    finally:
        await db.close()
    return 0


async def cmd_web_test(args) -> int:
    db = await _connect(args.db)
    try:
        settings = make_repositories(db).settings
        explicit = getattr(args, "provider", None)
        if explicit:
            provider_name = explicit.strip().lower()
        else:
            # No explicit provider: follow the stored config, and report a
            # precise error rather than silently switching provider.
            stored = (await settings.get("web_search.provider") or "").strip().lower() or "brave"
            if stored in ("duckduckgo", "http", "html"):
                provider_name = "duckduckgo"
            elif stored == "brave":
                provider_name = "brave"
            elif stored in ("mock", "offline"):
                print(f"error: configured provider {stored!r} cannot run a real test")
                return 1
            else:
                print(f"error: unsupported provider {stored!r}")
                return 1
        if provider_name in ("duckduckgo", "http", "html"):
            provider = HttpWebSearchProvider(enabled=True)
        elif provider_name == "brave":
            provider = BraveWebSearchProvider()
        else:
            print(f"error: unsupported provider {provider_name!r}")
            return 1
        print(f"provider: {provider.name}")
        try:
            results = await provider.search(args.query, limit=args.limit)
        except WebSearchConfigError as exc:
            print(f"error: web_search_config:{exc.code} ({exc})")
            return 1
        except WebSearchError as exc:
            print(f"error: web_search_failed:{exc.code} (retryable={exc.retryable})")
            return 1
        except Exception as exc:
            print(f"error: web_search_failed:{type(exc).__name__}: {exc}")
            return 1
        if not results:
            print("results: 0 (no results for this query)")
            return 0
        print(f"results: {len(results)}")
        for index, result in enumerate(results, 1):
            print(f"[{index}] {result.title}")
            print(f"    url: {result.url}")
            if result.snippet:
                print(f"    snippet: {result.snippet[:200]}")
    finally:
        await db.close()
    return 0


_WEB_DIAGNOSIS_HINTS = {
    "web_search_disabled": "联网未开启：python -m companion web enable",
    "permission_denied": "该说话人未开启 allow_external_ai，或联网未开启",
    "external_ai_permission_revoked": "本轮执行前该人物的外部 AI 权限已被撤回",
    "duckduckgo_anomaly": "DuckDuckGo 对本机返回反爬挑战；稍后重试、换网络或改用 Brave",
    "duckduckgo_parse_error": "搜索结果页结构无法识别（不是“没有结果”）",
    "duckduckgo_empty_response": "搜索端点返回空响应；稍后重试",
    "duckduckgo_http_error": "搜索端点返回非预期 HTTP 状态",
    "brave_api_key_missing": "缺少 BRAVE_SEARCH_API_KEY，或改用 duckduckgo",
    "result_too_large": "工具结果超过大小上限；这是实现问题，不是“没有结果”",
    "timeout": "工具超时；网络较慢或搜索端点无响应",
    "web_fetch_url_not_from_search": "web.fetch 只允许读取本轮 web.search 返回的 URL",
    "web_fetch_turn_limit": "本轮已读取页面数达到上限",
    "web_fetch_private_address": "目标解析到非公网地址，已按 SSRF 规则拒绝",
    "weather_location_not_found": "地点未匹配到任何城市；这不是天气服务故障",
}


def _web_hint(error_code: str | None) -> str:
    if not error_code:
        return ""
    for key, hint in _WEB_DIAGNOSIS_HINTS.items():
        if key in error_code:
            return hint
    return ""


async def cmd_web_diagnose(args) -> int:
    """Recent external-tool outcomes from the existing audit table."""
    db = await _connect(args.db)
    try:
        repos = make_repositories(db)
        calls = await repos.routing.recent_tool_calls(limit=args.limit)
        calls = [
            call
            for call in calls
            if call["tool_name"] in ("web.search", "web.fetch", "weather.forecast")
        ]
        if not calls:
            print("no recent web.search / weather.forecast / web.fetch calls")
            return 0
        print(f"recent external tool calls: {len(calls)}")
        for call in calls:
            arguments = json.dumps(call["arguments"], ensure_ascii=False)
            print(
                f"- {call['created_at']} {call['tool_name']}"
                f" status={call['status']} execution={call['execution_status']}"
            )
            print(f"    event: {call['event_id']} step={call['route_step_index']}")
            print(f"    args: {arguments[:160]}")
            if call["error_code"]:
                print(f"    error_code: {call['error_code']}")
                hint = _web_hint(call["error_code"])
                if hint:
                    print(f"    hint: {hint}")
            refs = (call.get("result_reference") or {}).get("source_refs") or []
            for ref in refs[:3]:
                print(f"    source: {ref}")
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
    print(f"analysis_thinking: {config.analysis_thinking}")
    print(f"response_thinking: {config.response_thinking}")
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
        # Thinking tokens share the output budget, so a tiny cap would produce
        # an empty visible reply and a false failure.
        live_tokens = 1024 if config.response_thinking == "enabled" else 8
        response = await client.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            request_type="response",
            max_tokens=live_tokens,
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


# -- CMF memory-file commands -------------------------------------------------


async def cmd_memory_file_create(args) -> int:
    try:
        info = await CmfFileService.create(
            args.path, display_name=args.name, overwrite=args.overwrite
        )
    except CmfError as exc:
        print(f"memory-file create failed: {exc}")
        return 1
    print(f"created {info.path}")
    print(f"agent_id: {info.agent_id}")
    print(f"format: {info.format_version}  schema: {info.schema_version}")
    print(f"display_name: {info.display_name}")
    return 0


async def cmd_memory_file_info(args) -> int:
    info = await CmfFileService.inspect(args.path)
    print(f"path: {info.path}")
    print(f"valid: {info.ok}")
    print(f"agent_id: {info.agent_id}")
    print(f"display_name: {info.display_name}")
    print(f"format_version: {info.format_version}")
    print(f"schema_version: {info.schema_version}")
    print(f"lineage_id: {info.lineage_id}")
    print(f"lineage_parent_agent_id: {info.lineage_parent_agent_id}")
    for name, count in sorted(info.counts.items()):
        print(f"{name}: {count}")
    for error in info.errors:
        print(f"error: {error}")
    return 0 if info.ok else 1


async def cmd_memory_file_validate(args) -> int:
    result = await CmfFileService.validate(args.path)
    print(f"valid: {result.ok}")
    for key in ("agent_id", "display_name", "format_version", "schema_version"):
        if key in result.info:
            print(f"{key}: {result.info[key]}")
    for error in result.errors:
        print(f"error: {error}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0 if result.ok else 1


async def cmd_memory_file_export(args) -> int:
    try:
        result = await CmfFileService.export(
            args.source, args.destination, overwrite=args.overwrite
        )
    except CmfError as exc:
        print(f"memory-file export failed: {exc}")
        return 1
    print(f"exported {args.source} -> {args.destination}")
    print(f"agent_id: {result.info.get('agent_id')}  valid: {result.ok}")
    return 0 if result.ok else 1


async def cmd_memory_file_clone(args) -> int:
    try:
        info = await CmfFileService.clone(
            args.source,
            args.destination,
            display_name=args.name,
            overwrite=args.overwrite,
        )
    except CmfError as exc:
        print(f"memory-file clone failed: {exc}")
        return 1
    print(f"cloned {args.source} -> {args.destination}")
    print(f"agent_id: {info.agent_id}")
    print(f"lineage_parent_agent_id: {info.lineage_parent_agent_id}")
    print(f"display_name: {info.display_name}")
    return 0 if info.ok else 1


async def cmd_memory_file_migrate(args) -> int:
    try:
        result = await CmfFileService.migrate(args.path)
    except (CmfError, RuntimeError) as exc:
        print(f"memory-file migrate failed: {exc}")
        return 1
    print(f"migrated {result['path']}")
    print(f"schema: {result['from_schema_version']} -> {result['to_schema_version']}")
    print(f"format_version: {result['format_version']}")
    print(f"agent_id: {result['agent_id']}")
    print(f"changed: {result['migrated']}")
    return 0


# -- Agent Core / relationship commands --------------------------------------


async def cmd_agent_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        data = await rt.agent_service.show()
        print(f"agent_id: {data['agent_id']}")
        print(f"format_version: {data['format_version']}")
        print(f"lineage_id: {data['lineage_id']}")
        print(f"lineage_parent_agent_id: {data['lineage_parent_agent_id']}")
        print(f"display_name: {data['display_name']}")
        print(f"pronouns: {data['pronouns']}")
        print(f"description: {data['description']}")
        print(f"identity_revision: {data['identity_revision']}")
        print(f"persona_revision: {data['persona_revision']}")
        for name, value in data["persona"].items():
            print(f"persona.{name}: {value}")
    finally:
        await db.close()
    return 0


async def cmd_agent_set_name(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        result = await rt.agent_service.set_name(
            args.name,
            pronouns=args.pronouns,
            description=args.description,
            reason=args.reason,
        )
        print(f"display_name = {result['display_name']} (revision {result['revision']})")
    except PersonalityValidationError as exc:
        print(f"invalid name: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_agent_set_persona(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        result = await rt.agent_service.set_persona(
            {args.field: args.value}, reason=args.reason
        )
        print(
            f"persona.{args.field} = {result[args.field]}"
            f" (revision {result['revision']})"
        )
    except PersonalityValidationError as exc:
        print(f"invalid persona: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_agent_revisions(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        for revision in await rt.agent_service.revision_history(args.limit):
            print(
                f"revision={revision['revision']} changed_by={revision['changed_by']}"
                f" changes={revision['changes']} reason={revision.get('reason')}"
            )
    finally:
        await db.close()
    return 0


async def cmd_agent_effective(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        config = await load_policy_config(rt.repos.settings)
        effective = await rt.agent_service.effective(args.person, policy=config)
        data = effective.as_dict()
        print(f"agent_id: {data['agent_id']}")
        print(f"display_name: {data['display_name']}")
        print(f"familiarity: {data['familiarity']}")
        print(f"response_conservatism: {data['response_conservatism']}")
        print(f"interruption_threshold: {data['interruption_threshold']}")
        print(f"curiosity_level: {data['curiosity_level']}")
        for name, value in data["traits"].items():
            print(f"effective.{name}: {value}")
    except RelationshipValidationError as exc:
        print(f"invalid person: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_agent_persona_core_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        core = await rt.agent_service.show_persona_core()
        data = core.as_prompt_dict()
        print(f"agent_id: {core.agent_id}")
        print(f"revision: {core.revision}")
        print(f"updated_by: {core.updated_by}")
        print(f"updated_at: {core.updated_at}")
        print("character_description:")
        print(data["character_description"] or "（空）")
        for field in (
            "core_values",
            "quirks",
            "contradictions",
            "conversational_boundaries",
        ):
            print(f"{field}: {json.dumps(data[field], ensure_ascii=False)}")
        for field in ("temperament", "preferences", "voice_identity"):
            print(f"{field}: {json.dumps(data[field], ensure_ascii=False, sort_keys=True)}")
    finally:
        await db.close()
    return 0


async def cmd_agent_persona_core_set(args) -> int:
    db = await _connect(args.db)
    try:
        try:
            updates = json.loads(args.core_json)
        except json.JSONDecodeError as exc:
            print(f"invalid JSON: {exc}")
            return 1
        if not isinstance(updates, dict):
            print("invalid JSON: expected an object of persona core fields")
            return 1
        # Friendly alias; persisted field names remain explicit.
        if "boundaries" in updates and "conversational_boundaries" not in updates:
            updates["conversational_boundaries"] = updates.pop("boundaries")
        rt = build_runtime(db, with_worker=False)
        core = await rt.agent_service.set_persona_core(
            updates, actor="admin", reason=args.reason
        )
        print(f"persona core updated (revision {core.revision})")
        for field in ("core_values", "temperament", "preferences", "quirks",
                      "contradictions", "conversational_boundaries", "voice_identity"):
            value = getattr(core, field)
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            print(f"{field}: {rendered}")
    except PersonalityValidationError as exc:
        print(f"invalid persona core: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_agent_style_add(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        manifest = await rt.agent_service.repo.get_manifest() or {}
        example = await rt.style_repo.add(
            agent_id=manifest.get("agent_id", ""),
            scenario=args.scenario,
            user_text=args.user_text,
            ideal_response=args.ideal_response,
            person_id=args.person,
            created_by="admin",
        )
        scope = "global" if example.get("person_id") is None else example["person_id"]
        print(f"example_id: {example['example_id']}")
        print(f"scope: {scope}")
        print(f"scenario: {example['scenario']}")
    except StyleExampleError as exc:
        print(f"invalid style example: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_agent_style_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        manifest = await rt.agent_service.repo.get_manifest() or {}
        rows = await rt.style_repo.list(
            agent_id=manifest.get("agent_id"),
            person_id=args.person,
            include_disabled=args.include_disabled,
            limit=args.limit,
        )
        for row in rows:
            scope = "global" if row.get("person_id") is None else row["person_id"]
            print(
                f"{row['example_id']} status={row['status']} scope={scope}"
                f" scenario={row['scenario']}"
            )
    finally:
        await db.close()
    return 0


async def cmd_agent_style_disable(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        row = await rt.style_repo.disable(args.example_id)
        print(f"{row['example_id']} status={row['status']}")
    except StyleExampleError as exc:
        print(f"style example error: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_relationship_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        data = await rt.agent_service.relationship(args.person_id)
        print(f"person_id: {data['person_id']}")
        print(f"familiarity: {data['familiarity']}")
        print(f"tier: {data['tier']}")
        print(f"style_overrides: {data['style_overrides']}")
    except RelationshipValidationError as exc:
        print(f"invalid person: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_relationship_set(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        result = await rt.agent_service.set_relationship(
            args.person_id, args.tier, reason=args.reason
        )
        print(f"{result['person_id']}: {result['old_tier']} -> {result['new_tier']}")
    except RelationshipValidationError as exc:
        print(f"invalid relationship: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_relationship_override(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        overrides = await rt.agent_service.set_override(
            args.person_id, args.field, args.value
        )
        print(f"{args.person_id} overrides: {overrides}")
    except (PersonalityValidationError, RelationshipValidationError) as exc:
        print(f"invalid override: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_relationship_clear_override(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        overrides = await rt.agent_service.set_override(
            args.person_id, args.field, None
        )
        print(f"{args.person_id} overrides: {overrides}")
    except (PersonalityValidationError, RelationshipValidationError) as exc:
        print(f"invalid override: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_relationship_history(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        data = await rt.agent_service.relationship(args.person_id)
        for row in data["history"]:
            print(
                f"{row['created_at']}: {row['old_tier']} -> {row['new_tier']}"
                f" by={row['actor']} reason={row.get('reason')}"
            )
    except RelationshipValidationError as exc:
        print(f"invalid person: {exc}")
        return 1
    finally:
        await db.close()
    return 0


async def cmd_goal_add(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        goal_id = await rt.repos.goals.create(
            args.title,
            description=args.description,
            priority=args.priority,
            next_action=args.next_action,
        )
        print(goal_id)
    finally:
        await db.close()
    return 0


async def cmd_goal_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        goals = await rt.repos.goals.list(status=args.status)
        for g in goals:
            print(
                f"{g['goal_id']}  {g['status']}  p={g['priority']}  {g['title']}"
                f"  next={g['next_action'] or '-'}"
            )
        if not goals:
            print("no goals")
    finally:
        await db.close()
    return 0


async def cmd_goal_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        goal = await rt.repos.goals.get(args.goal_id)
        if goal is None:
            print("goal not found")
            return 1
        print(f"goal:     {goal['goal_id']}")
        print(f"title:    {goal['title']}")
        print(f"status:   {goal['status']}")
        print(f"priority: {goal['priority']}")
        print(f"next:     {goal['next_action'] or '-'}")
        print(f"due:      {goal['due_at'] or '-'}")
        deps = await rt.repos.goals.list_dependencies(args.goal_id)
        print(f"depends:  {', '.join(deps) if deps else '-'}")
        print("checkpoints:")
        for c in await rt.repos.goals.list_checkpoints(args.goal_id):
            print(f"  {c['created_at']}: {c['status']} {c['summary'] or ''}")
    finally:
        await db.close()
    return 0


async def cmd_goal_set_status(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        if await rt.repos.goals.get(args.goal_id) is None:
            print("goal not found")
            return 1
        await rt.repos.goals.set_status(
            args.goal_id, args.status, next_action=args.next_action
        )
        print(f"{args.goal_id} -> {args.status}")
    finally:
        await db.close()
    return 0


async def cmd_goal_checkpoint(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        if await rt.repos.goals.get(args.goal_id) is None:
            print("goal not found")
            return 1
        checkpoint_id = await rt.repos.goals.add_checkpoint(
            args.goal_id, args.status, summary=args.summary
        )
        print(checkpoint_id)
    finally:
        await db.close()
    return 0


async def cmd_goal_depend(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        try:
            await rt.repos.goals.add_dependency(args.goal_id, args.depends_on)
        except ValueError as exc:
            print(f"invalid dependency: {exc}")
            return 1
        print(f"{args.goal_id} depends on {args.depends_on}")
    finally:
        await db.close()
    return 0


async def cmd_goal_ready(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        goals = await rt.repos.goals.ready()
        for g in goals:
            print(f"{g['goal_id']}  {g['title']}  next={g['next_action'] or '-'}")
        if not goals:
            print("no ready goals")
    finally:
        await db.close()
    return 0


# -- embedding index + hybrid search helpers ---------------------------------


def _build_embedding_provider(args):
    """Build a CLI embedding provider (opt-in, never silently faked).

    The real local model is the default and requires the ``[embedding]`` extra;
    if it is missing the command prints a clear error and returns ``None``.
    ``--fake-provider`` selects the deterministic offline provider.
    """
    from companion.memory.embeddings import (
        EmbeddingBackendUnavailable,
        create_embedding_provider,
    )

    backend = "fake" if getattr(args, "fake_provider", False) else "local"
    try:
        return create_embedding_provider(backend)
    except EmbeddingBackendUnavailable as exc:
        print(f"embedding error: {exc}")
        return None


def _build_embedding_runtime(db, args):
    provider = _build_embedding_provider(args)
    if provider is None:
        return None
    return build_runtime(db, embedding_provider=provider)


async def cmd_memory_index_status(args) -> int:
    db = await _connect(args.db)
    try:
        from companion.memory.embeddings import SETTING_VECTOR_BACKEND
        from companion.memory.embedding_repository import EmbeddingRepository
        from companion.memory.vector_index import VectorIndex

        repo = EmbeddingRepository(db)
        index = VectorIndex(repo)
        requested = (
            await make_repositories(db).settings.get(SETTING_VECTOR_BACKEND, "python")
            or "python"
        )
        backend = index.backend_status(requested)
        print(f"sqlite_vec_installed: {backend['sqlite_vec_installed']}")
        print(f"sqlite_vec_loadable: {backend['sqlite_vec_loadable']}")
        print(f"sqlite_vec_enabled: {backend['sqlite_vec_enabled']}")
        print(f"active_backend: {backend['active_backend']}")
        print(f"requested_backend: {backend['requested_backend']}")
        print(
            "backend: "
            + ("sqlite-vec (vec0)" if backend["sqlite_vec_enabled"] else "pure-python fallback")
        )
        models = await repo.list_models()
        if not models:
            print("no embedding model registered (run `memory index backfill`)")
            return 0
        for m in models:
            mid = m["model_id"]
            counts = await repo.counts(mid)
            print(f"model: {mid}")
            print(f"  provider: {m['provider']}  model_name: {m['model_name']}")
            print(f"  dimensions: {m['dimensions']}  metric: {m['distance_metric']}")
            print(f"  normalization: {m['normalization']}")
            print(f"  counts: {counts}")
    finally:
        await db.close()
    return 0


async def cmd_memory_index_backfill(args) -> int:
    db = await _connect(args.db)
    try:
        rt = _build_embedding_runtime(db, args)
        if rt is None:
            return 1
        results = await rt.embedding_service.backfill()
        by_status: dict[str, int] = {}
        for r in results:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        print(f"backfilled {len(results)} ACTIVE memory rows: {by_status}")
    finally:
        await db.close()
    return 0


async def cmd_memory_index_rebuild(args) -> int:
    db = await _connect(args.db)
    try:
        rt = _build_embedding_runtime(db, args)
        if rt is None:
            return 1
        result = await rt.embedding_service.rebuild()
        print(f"rebuilt model {result['model_id']}: {result['counts']}")
    finally:
        await db.close()
    return 0


# -- emotion commands --------------------------------------------------------


async def _emotion_agent_id(rt) -> str | None:
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest.get("agent_id") if manifest else None


async def cmd_emotion_show(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        agent_id = await _emotion_agent_id(rt)
        if agent_id is None:
            print("no agent identity; run `agent` first")
            return 1
        profile = await rt.emotion_service.get_profile(agent_id)
        if profile is None:
            profile = await rt.emotion_service.ensure_defaults(agent_id)
        print(f"agent_id: {agent_id}")
        print(f"baseline_valence: {profile.baseline_valence}")
        print(f"baseline_arousal: {profile.baseline_arousal}")
        print(f"default_half_life_seconds: {profile.default_half_life_seconds}")
        print(f"responsiveness: {profile.responsiveness}")
        print(f"revision: {profile.revision}")
        print(f"expression_style: {profile.expression_style}")
        cfg = rt.emotion_service.config
        print("== dynamics ==")
        print(f"acute_half_life_seconds: {cfg.acute_half_life_seconds}")
        print(f"mood_half_life_seconds: {cfg.mood_half_life_seconds}")
        print(f"tension_half_life_seconds: {cfg.tension_half_life_seconds}")
        print(f"acute_intensity_cap: {cfg.acute_intensity_cap}")
        print(f"text_expression_min_intensity: {cfg.text_expression_min_intensity}")
    finally:
        await db.close()
    return 0


async def cmd_emotion_inspect(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        agent_id = await _emotion_agent_id(rt)
        if agent_id is None:
            print("no agent identity; run `agent` first")
            return 1
        snapshots = await rt.emotion_service.list_snapshots(agent_id)
        if args.person:
            snapshots = [s for s in snapshots if s.get("person_id") == args.person]
        print(f"== state snapshots: {len(snapshots)} ==")
        for s in snapshots:
            print(
                f"  {s['person_id']}  {s['emotion']}  v={s['valence']}"
                f" a={s['arousal']} i={s['intensity']}  at={s['updated_at']}"
                f"  | mood={s.get('mood_emotion', 'NEUTRAL')}"
                f" mi={s.get('mood_intensity', 0)}"
                f"  | tension={s.get('relationship_tension', 0)}"
            )
            activations = s.get("activations_json")
            if activations and activations != "{}":
                print(f"      activations={activations}")
        events = await rt.emotion_service.list_events(agent_id, limit=args.limit)
        if args.person:
            events = [e for e in events if e.get("person_id") == args.person]
        print(f"== events: {len(events)} ==")
        for e in events:
            print(
                f"  {e['emotion_event_id']}  {e['emotion']}"
                f"  event={e.get('event_type') or '-'}"
                f" target={e.get('target') or '-'}"
                f" intent={e.get('social_intent') or '-'}"
                f" validation={e.get('validation') or '-'}"
                f"  rule={e['appraisal_rule_id'] or '-'}"
                f"  reason={e['appraisal_reason'] or '-'}"
                f"  acute={e.get('acute_before')}->{e.get('acute_after')}"
                f"  mood={e.get('mood_after')} tension={e.get('tension_after')}"
            )
        feedback = await rt.emotion_service.list_feedback(agent_id)
        print(f"== feedback: {len(feedback)} ==")
        for f in feedback:
            print(f"  {f['feedback_id']}  status={f['status']}  {f['feedback_text'][:60]}")
    finally:
        await db.close()
    return 0


async def cmd_emotion_rules_list(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        agent_id = await _emotion_agent_id(rt)
        if agent_id is None:
            print("no agent identity; run `agent` first")
            return 1
        rules = await rt.emotion_service.list_rules(agent_id)
        for r in rules:
            print(
                f"{r.rule_id}  {r.name}  {r.status}  pri={r.priority}"
                f"  {r.emotion}  v={r.valence_delta} a={r.arousal_delta}"
                f"  i={r.intensity}  source={r.source}"
            )
        if not rules:
            print("no rules")
    finally:
        await db.close()
    return 0


async def cmd_emotion_config_show(args) -> int:
    return await cmd_emotion_show(args)


async def cmd_emotion_config_set(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        agent_id = await _emotion_agent_id(rt)
        if agent_id is None:
            print("no agent identity; run `agent` first")
            return 1
        field = args.field
        if field in ("baseline_valence", "baseline_arousal", "responsiveness"):
            try:
                value = float(args.value)
            except ValueError:
                print(f"invalid value {args.value!r} for {field}")
                return 1
        else:
            try:
                value = int(args.value)
            except ValueError:
                print(f"invalid value {args.value!r} for {field}")
                return 1
        profile = await rt.emotion_service.update_profile(agent_id, **{field: value})
        print(f"{field} = {getattr(profile, field)} (revision {profile.revision})")
    finally:
        await db.close()
    return 0


async def cmd_emotion_simulate(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db, with_worker=False)
        agent_id = await _emotion_agent_id(rt)
        if agent_id is None:
            print("no agent identity; run `agent` first")
            return 1
        from datetime import datetime, timedelta

        from companion.emotion.repository import EmotionRepository
        from companion.emotion.service import EmotionService

        repo = EmotionRepository(db)
        snap = await repo.get_snapshot(agent_id, args.person)
        if snap is None:
            print("no snapshot for this person (say something to the agent first)")
            return 0
        stored_at = datetime.fromisoformat(snap["updated_at"])
        fake_now = stored_at + timedelta(seconds=args.elapsed_seconds)
        svc = EmotionService(
            repo,
            audit=None,
            logger=None,
            clock=lambda: fake_now,
            config=rt.emotion_service.config,
        )
        state = await svc.get_state(agent_id, args.person)
        print(f"person: {args.person}")
        print(f"stored_at: {snap['updated_at']}")
        print(f"simulated_now: {fake_now.isoformat()} (+{args.elapsed_seconds}s)")
        print(f"emotion: {state.emotion}")
        print(f"valence: {state.valence:.4f}")
        print(f"arousal: {state.arousal:.4f}")
        print(f"intensity: {state.intensity:.4f}")
        print(f"activations: {state.activations}")
        print(f"mood: {state.mood_emotion} i={state.mood_intensity:.4f}"
              f" v={state.mood_valence:.4f} a={state.mood_arousal:.4f}")
        print(f"relationship_tension: {state.relationship_tension:.4f}")
        from companion.emotion.expression_policy import compile_affect_context

        affect_context = compile_affect_context(state, config=svc.config)
        print(f"affect_context: {affect_context or '(none)'}")
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
