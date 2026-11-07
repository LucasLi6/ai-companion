"""One-time migration of the retired Person Profile candidate table.

V0.9.9.6 stopped writing ``person_profile_candidates`` and renamed the table
to ``person_profile_candidates_legacy``. This module revalidates each old row
against its original utterance using current local rules, writes only what is
still legal, and records a local audit row. It never trusts the old model
proposal on its own and never calls a model.

The command is idempotent: ``profile_candidate_migrations.candidate_id`` is
unique, and a second ``--apply`` finds every row already recorded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from companion.people.profile_models import (
    ProfileFieldStorage,
    ProfileSource,
    profile_field_spec,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ValidatedChange,
    content_supported_by_text,
    entries_equivalent,
)
from companion.people.profile_self_report import has_explicit_self_report
from companion.people.profile_service import PersonProfileService

_HYPOTHETICAL = re.compile(
    r"(如果|假如|假设|要是|开玩笑|说着玩|玩笑|据说|听说|别人说|"
    r"他说|她说|他们说|她们说|有人说|朋友说|同事说|同学说)"
)
_SENSITIVE = re.compile(
    r"(健康|疾病|病历|诊断|医院|用药|药物|抑郁|焦虑症|"
    r"银行|信用卡|贷款|工资|收入|存款|密码|"
    r"住址|家庭地址|身份证|护照号)"
)
_TRAIT_SELF_DESCRIPTION = re.compile(
    r"(我是|我比较|我做事|我做|我通常|我一般|我经常|我常常|我每天|我每周|"
    r"我习惯|我喜欢|我不喜欢|我讨厌|我倾向于|我偏向|对我来说|"
    r"我性格|我这个人|我这人|我很少|我从不|我不太|我更喜欢|"
    r"我容易|我总是|我平时)"
)
_PREFERENCE_CUE = re.compile(
    r"(不用|不要|别|请|希望|喜欢|偏好|随意|正式|简洁|简短|详细|"
    r"中文|英文|幽默|严肃|友好|专业|轻松|俏皮)"
)
_TRAIT_FIELDS = frozenset(
    {
        "personality.traits",
        "personality.behavior_patterns",
        "personality.thinking_style",
        "personality.decision_style",
        "habits.stable_habits",
        "habits.working_preferences",
    }
)
_IDENTITY_NAME_FIELDS = frozenset(
    {"identity.name", "identity.preferred_name"}
)


def _loads(value: Any, default: Any) -> Any:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return default
    return default if parsed is None else parsed


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _is_alias(value: Any) -> bool:
    text = _text(value).strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    if not text or len(text) > 30:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 .'’\-]*", text))


def _alias_from_value(value: Any) -> str:
    """Extract a reliable Latin alias from an explanatory candidate value.

    Old candidates often stored model prose such as
    ``Lucas（本人给出的英文名/别名）``. Only the leading alias token may be
    migrated; the explanation is never treated as a fact.
    """
    text = _text(value).strip()
    match = re.match(r"([A-Za-z][A-Za-z0-9 .'’\-]{0,29})", text)
    if not match:
        return ""
    alias = match.group(1).strip(" .'’\\-")
    return alias if _is_alias(alias) else ""


async def _event_texts(
    service: PersonProfileService, person_id: str, event_ids: list[str]
) -> dict[str, str]:
    ids = [item for item in dict.fromkeys(event_ids) if item]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    cur = await service.persons.db.execute(
        "SELECT event_id, text FROM utterances WHERE event_id IN"
        f" ({placeholders}) AND speaker_person_id = ?",
        tuple(ids) + (person_id,),
    )
    return {row["event_id"]: row["text"] or "" for row in await cur.fetchall()}


async def _person_event_texts(
    service: PersonProfileService, person_id: str
) -> dict[str, str]:
    """Every persisted utterance this person authored, newest first."""
    cur = await service.persons.db.execute(
        "SELECT event_id, text FROM utterances WHERE speaker_person_id = ?"
        " ORDER BY occurred_at DESC, created_at DESC",
        (person_id,),
    )
    return {row["event_id"]: row["text"] or "" for row in await cur.fetchall()}


def _content_supported(field_key: str, content: str, texts: list[str]) -> bool:
    # One concrete term that the person actually said is enough to rule out a
    # proposal built from nothing; the value still has to name a real field
    # and pass the other local checks below.
    return content_supported_by_text(content, texts)


def _reject(field_key: str, status: str, reason: str) -> dict:
    return {
        "field_key": field_key,
        "status": status,
        "reason": reason,
    }


async def _plan_one(
    service: PersonProfileService,
    person_id: str,
    row: dict,
    *,
    snapshot_entries: dict[str, list[dict]],
) -> dict:
    """Return a decision dict with an optional validated change."""
    candidate_id = row["candidate_id"]
    field_key = str(row["field_key"] or "")
    value = _loads(row.get("candidate_value_json"), None)
    content = _text(value)
    operation = str(row.get("operation") or "ADD").upper()
    events = _loads(row.get("evidence_event_ids_json"), [])
    event_ids = [str(item) for item in events if item]
    spec = profile_field_spec(field_key)
    base = {
        "candidate_id": candidate_id,
        "field_key": field_key,
        "status": "SKIPPED",
        "reason": "",
    }
    if row.get("status") not in ("PENDING", "NEEDS_REVIEW"):
        base["reason"] = "candidate_status_not_pending"
        return base
    if spec is None:
        base["reason"] = "unknown_field_key"
        return base
    if spec.storage is not ProfileFieldStorage.ENTRY:
        base["reason"] = "field_not_an_understanding_entry"
        return base
    if not content:
        base["reason"] = "missing_proposed_content"
        return base

    # -- identity: direct self-report only -------------------------------
    if spec.strict_identity or not spec.model_writable:
        if field_key in _IDENTITY_NAME_FIELDS:
            texts = await _event_texts(service, person_id, event_ids)
            if not texts:
                base["reason"] = "no_evidence"
                return base
            # The cited row may point at a later memory-check turn ("李约，
            # lucas，记住了吗"). Look for the person's own explicit statement
            # of the same value anywhere in their persisted history; only that
            # sentence authorizes a name write.
            event_id = next(
                (
                    key
                    for key, text in texts.items()
                    if has_explicit_self_report(text, field_key, content)
                ),
                None,
            )
            if event_id is None:
                event_id = next(
                    (
                        key
                        for key, text in (
                            await _person_event_texts(service, person_id)
                        ).items()
                        if has_explicit_self_report(text, field_key, content)
                    ),
                    None,
                )
            if event_id is None:
                base["reason"] = "no_explicit_self_report"
                return base
            base.update(
                {
                    "kind": "direct_self_report",
                    "value": content,
                    "source_event_id": event_id,
                }
            )
            return base
        if field_key == "identity.nicknames":
            texts = await _event_texts(service, person_id, event_ids)
            if not texts:
                base["reason"] = "no_evidence"
                return base
            alias = content if _is_alias(content) else _alias_from_value(content)
            if not alias:
                base["reason"] = "nickname_not_a_reliable_alias"
                return base
            event_id = next(
                (
                    key
                    for key, text in texts.items()
                    if alias.casefold() in text.casefold()
                ),
                None,
            )
            if event_id is None:
                event_id = next(
                    (
                        key
                        for key, text in (
                            await _person_event_texts(service, person_id)
                        ).items()
                        if alias.casefold() in text.casefold()
                    ),
                    None,
                )
            if event_id is None:
                base["reason"] = "nickname_not_in_source"
                return base
            proposal = ProfileChangeProposal(
                operation="ADD",
                field_key=field_key,
                proposed_content=alias,
                evidence_event_ids=[event_id],
                confidence=float(row.get("confidence") or 0.0),
                reason="legacy_candidate_migration",
            )
            base.update(
                {
                    "kind": "trusted_identity",
                    "change": ValidatedChange(proposal=proposal),
                    "source_event_id": event_id,
                }
            )
            return base
        base["reason"] = "identity_field_not_migratable"
        return base

    # -- ordinary understanding entries ----------------------------------
    if operation not in ("ADD", "UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE"):
        base["reason"] = "unsupported_operation"
        return base
    if _SENSITIVE.search(content):
        base["reason"] = "sensitive_attribute_not_auto_saved"
        return base
    texts = await _event_texts(service, person_id, event_ids)
    if not texts:
        base["reason"] = "evidence_event_not_accessible"
        return base
    corpus = " ".join(texts.values())
    if _HYPOTHETICAL.search(f"{content} {corpus}"):
        base["reason"] = "hypothetical_or_reported_evidence"
        return base
    if operation != "ARCHIVE" and not _content_supported(
        field_key, content, list(texts.values())
    ):
        base["reason"] = "content_not_supported_by_evidence"
        return base
    if field_key in _TRAIT_FIELDS and not _TRAIT_SELF_DESCRIPTION.search(corpus):
        base["reason"] = "trait_inferred_from_single_act"
        return base
    if field_key.startswith("communication.") and not (
        _TRAIT_SELF_DESCRIPTION.search(corpus)
        or _PREFERENCE_CUE.search(corpus)
    ):
        base["reason"] = "preference_not_explicitly_stated"
        return base

    existing = snapshot_entries.get(field_key, [])
    if operation == "ADD" and any(
        entries_equivalent(item["content"], content) for item in existing
    ):
        base["status"] = "ALREADY_PRESENT"
        base["reason"] = "already_present"
        return base
    if operation == "ADD" and spec.single_value and existing:
        base["reason"] = "unresolved_conflict_with_existing_value"
        return base

    proposal = ProfileChangeProposal(
        operation=operation,
        field_key=field_key,
        target_entry_ids=[
            str(item)
            for item in _loads(row.get("target_entry_ids_json"), [])
            if item
        ],
        proposed_content=content,
        evidence_event_ids=list(texts),
        confidence=float(row.get("confidence") or 0.0),
        reason="legacy_candidate_migration",
    )
    base.update(
        {
            "kind": "entry",
            "change": ValidatedChange(proposal=proposal),
            "source_event_id": next(iter(texts)),
        }
    )
    return base


async def migrate_profile_candidates(
    service: PersonProfileService,
    person_id: str,
    *,
    dry_run: bool = True,
) -> dict:
    """Revalidate and, when ``dry_run`` is false, apply every old candidate."""
    summary: dict = {
        "person_id": person_id,
        "dry_run": bool(dry_run),
        "scanned": 0,
        "applied": 0,
        "already_present": 0,
        "already_migrated": 0,
        "skipped": 0,
        "results": [],
    }
    snapshot = await service.get_full_snapshot(person_id)
    entries = (
        {
            key: [
                {
                    "content": item.content,
                    "entry_id": item.entry_id,
                    "source": item.source,
                }
                for item in items
            ]
            for key, items in snapshot.entries.items()
        }
        if snapshot is not None
        else {}
    )
    rows = await service.repo.list_legacy_candidates(person_id)
    summary["scanned"] = len(rows)
    for row in rows:
        candidate_id = row["candidate_id"]
        prior = await service.repo.get_candidate_migration(candidate_id)
        if prior is not None:
            summary["already_migrated"] += 1
            summary["results"].append(
                {
                    "candidate_id": candidate_id,
                    "field_key": row["field_key"],
                    "status": "ALREADY_MIGRATED",
                    "reason": prior.get("status") or "",
                }
            )
            continue
        plan = await _plan_one(
            service, person_id, row, snapshot_entries=entries
        )
        if plan.get("kind") is None:
            summary["skipped"] += 1
            summary["results"].append(plan)
            if not dry_run:
                await service.repo.record_candidate_migration(
                    candidate_id=candidate_id,
                    person_id=person_id,
                    field_key=str(row["field_key"]),
                    operation=str(row.get("operation") or "ADD"),
                    candidate_value=_loads(row.get("candidate_value_json"), None),
                    source_event_id=next(
                        iter(_loads(row.get("evidence_event_ids_json"), [])),
                        None,
                    ),
                    status="SKIPPED",
                    reason=plan["reason"],
                )
            continue

        if dry_run:
            plan["status"] = "WOULD_APPLY"
            summary["results"].append(plan)
            continue

        if plan["kind"] == "direct_self_report":
            result = await service.apply_direct_self_report(
                person_id=person_id,
                field_key=plan["field_key"],
                value=plan["value"],
                source_event_id=plan["source_event_id"],
                actor="profile_candidate_migration",
            )
            applied_status = result.status
            revision = result.revision
            entry_id = result.entry_id
            reason = result.reason
        else:
            applied = await service.apply_migrated_change(
                person_id,
                plan["change"],
                source_event_id=plan["source_event_id"],
                source=ProfileSource.INTERACTION_INFERENCE.value,
                trusted_identity=plan["kind"] == "trusted_identity",
                actor="profile_candidate_migration",
            )
            applied_status = applied.get("status", "FAILED")
            revision = applied.get("revision")
            entry_id = (applied.get("applied") or {}).get("entry_id")
            reason = applied.get("reason") or (
                ""
                if applied_status == "APPLIED"
                else str((applied.get("not_applied") or [{}])[0].get("reason", ""))
            )

        if applied_status == "APPLIED":
            status = "APPLIED"
            summary["applied"] += 1
        elif applied_status == "NO_CHANGE":
            status = "ALREADY_PRESENT"
            summary["already_present"] += 1
        else:
            status = "SKIPPED"
            summary["skipped"] += 1
        await service.repo.record_candidate_migration(
            candidate_id=candidate_id,
            person_id=person_id,
            field_key=plan["field_key"],
            operation=str(row.get("operation") or "ADD"),
            candidate_value=_loads(row.get("candidate_value_json"), None),
            source_event_id=plan.get("source_event_id"),
            status=status,
            reason=reason,
            entry_id=entry_id,
            revision=revision,
        )
        summary["results"].append(
            {
                "candidate_id": candidate_id,
                "field_key": plan["field_key"],
                "status": status,
                "reason": reason,
                "revision": revision,
            }
        )
    return summary


__all__ = ["migrate_profile_candidates"]
