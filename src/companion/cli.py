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
import os
import sys
import tempfile
from pathlib import Path

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
from companion.runtime_context.assembler import compile_runtime_context
from companion.runtime_context.timezone import TimeZoneService, valid_zone
from companion.storage.database import Database
from companion.storage.repositories import make_repositories

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
    """Build the idle scheduler with a settings-aware (mock-first) web search.

    The real DeepSeek idle provider is constructed only when explicitly
    requested; it also re-checks ``idle.real_deepseek.enabled`` on every call.
    """
    from companion.idle import build_idle_scheduler
    from companion.providers.websearch import make_web_search

    web_search = await make_web_search(rt.repos.settings)
    deepseek = None
    if real_provider:
        from companion.providers.deepseek_client import (
            build_deepseek_client,
            load_deepseek_config,
        )
        from companion.providers.deepseek_idle import DeepSeekIdleProvider

        client = build_deepseek_client(load_deepseek_config())
        deepseek = DeepSeekIdleProvider(client, gate=rt.repos.settings)
    return build_idle_scheduler(rt, web_search=web_search, deepseek=deepseek)


async def cmd_idle_run(args) -> int:
    db = await _connect(args.db)
    try:
        rt = build_runtime(db)
        try:
            scheduler = await _build_idle_scheduler(
                rt, real_provider=bool(getattr(args, "real_provider", False))
            )
        except Exception as exc:  # missing key / unauthorised real provider
            print(f"error: {exc}")
            return 1
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
        try:
            while True:
                try:
                    line = (
                        await asyncio.to_thread(input, "you> ")
                    ).strip()
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
                await _run_streaming_turn(
                    rt, event, debug=bool(getattr(args, "debug", False))
                )
        finally:
            if worker_started:
                await rt.worker.stop()
    finally:
        await db.close()
    return 0


async def _run_streaming_turn(rt, event, *, debug: bool = False) -> None:
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
    stream = rt.orchestrator.process_utterance_stream(event)
    try:
        async for item in stream:
            if item.type == "response_started":
                saw_started = True
                sys.stdout.write("assistant> ")
                sys.stdout.flush()
            elif item.type == "response_delta":
                saw_delta = True
                sys.stdout.write(item.delta or "")
                sys.stdout.flush()
            elif item.type == "response_completed":
                saw_terminal = True
                if not saw_started:
                    sys.stdout.write("assistant> ")
                    saw_started = True
                if not saw_delta and item.text:
                    # A provider without incremental chunks still produces a
                    # real reply; show it once, here, not twice.
                    sys.stdout.write(item.text)
                if item.status == "TRUNCATED":
                    sys.stdout.write("\n[truncated: token limit]\n")
                else:
                    sys.stdout.write("\n")
                sys.stdout.flush()
            elif item.type == "response_failed":
                saw_terminal = True
                if not saw_delta and item.text:
                    sys.stdout.write(item.text)
                if saw_started:
                    sys.stdout.write("\n")
                print(f"[failed] {item.error or ''}".rstrip())
            elif item.type == "response_cancelled":
                saw_terminal = True
                if saw_started:
                    sys.stdout.write("\n")
                print("[cancelled]")
    except KeyboardInterrupt:
        # Let the orchestrator persist the partial reply as CANCELLED and
        # release its conversation lock before returning to the prompt.
        with contextlib.suppress(Exception):
            await stream.aclose()
        if saw_started:
            sys.stdout.write("\n")
        print("[cancelled]")
        return
    except Exception as exc:  # provider/validation failure is shown truthfully
        if saw_started:
            sys.stdout.write("\n")
        print(f"[failed] {type(exc).__name__}: {exc}")
        return

    if saw_terminal:
        return
    # No response event was emitted: the local policy chose WAIT or IGNORE.
    # In the ordinary chat surface this is intentional silence, so nothing is
    # printed. Diagnostics remain available behind ``--debug``.
    if not debug:
        return
    action = None
    with contextlib.suppress(Exception):
        action = await rt.repos.analysis.get_action(event.event_id)
    if action is None:
        print("[no action]")
        return
    reason = f" ({action.reason_code.value})" if action.reason_code else ""
    print(f"[{action.action_type.value}{reason}]")


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
