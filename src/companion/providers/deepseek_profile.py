"""DeepSeek-backed Profile Refinement provider (V0.9.7).

This module reuses the existing :class:`DeepSeekClient`; it does not create a
second API client, model scheduler or persona system. The provider only turns
one review payload into a schema-shaped proposal. Every identifier is supplied
by trusted code, and anything the model invents is dropped before validation.
Prompts, completions and private profile content are never logged here.

Output handling is a single validated pipeline:

    finish_reason -> JSON parse -> shape normalisation -> Pydantic schema

If that pipeline rejects the output, the provider makes *one* dedicated
format-repair call containing the original task and the error type only (never
the raw model output). The repaired output must pass the same pipeline again.
There is no fallback to ``NO_CHANGE``: two failures raise a validation error
and let the job's normal failure/retry path handle it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import ValidationError

from companion.people.profile_refinement import (
    MAX_MODEL_CHANGES,
    ProfileChangeProposal,
    ProfileRefinementProviderError,
    ProfileRefinementValidationError,
    ProfileReviewDecision,
)
from companion.prompts.profile_refinement import (
    PROFILE_REFINEMENT_PROMPT_VERSION,
    build_profile_refinement_messages,
    build_profile_refinement_repair_message,
)

_LOGGER = logging.getLogger("companion.profile_refinement.provider")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# ``finish_reason`` values that mean the JSON was cut off mid-object.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

_PROPOSAL_KEYS = {
    "operation",
    "field_key",
    "target_entry_ids",
    "fact_id",
    "relation",
    "proposed_content",
    "evidence_event_ids",
    "evidence_quotes",
    "reason",
    "confidence",
    "requires_confirmation",
}

_FACT_KEYS = {
    "fact_id",
    "entity",
    "attribute",
    "value",
    "assertion_type",
    "temporal_scope",
    "classification",
    "evidence_event_ids",
    "evidence_quotes",
}

_MAX_CHANGES = MAX_MODEL_CHANGES


def _normalise_facts(raw: Any) -> list[dict]:
    """Keep only schema-shaped facts, never arbitrary model keys."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise _OutputProblem("invalid_facts")
    facts: list[dict] = []
    for item in raw[:MAX_MODEL_CHANGES]:
        if not isinstance(item, dict):
            raise _OutputProblem("invalid_fact_item")
        facts.append({key: item[key] for key in _FACT_KEYS if key in item})
    return facts



class _OutputProblem(Exception):
    """One rejected model output, described without any raw content.

    ``code`` is a stable, log-safe reason. ``fields`` holds only top-level
    field *names* (never values) so diagnostics stay private.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields: tuple[str, ...] = ()


def _extract_json_object(content: str) -> dict:
    text = (content or "").strip()
    text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise _OutputProblem("malformed_json") from exc
    if not isinstance(data, dict):
        raise _OutputProblem("non_object_json")
    return data


def _validation_code(exc: ValidationError) -> str:
    """A log-safe summary of a Pydantic failure: locations and types only."""
    parts: list[str] = []
    for item in exc.errors()[:6]:
        location = ".".join(str(piece) for piece in item.get("loc", ()))
        parts.append(f"{location}:{item.get('type', 'invalid')}")
    return "schema_invalid[" + ",".join(parts) + "]"


def _normalise_decision(
    data: dict, *, person_id: str, base_revision: int
) -> ProfileReviewDecision:
    """Turn one parsed object into a validated decision or a typed problem.

    The only value normalisation allowed is ``decision="CHANGE"`` with a
    non-empty ``changes`` array, which has exactly one meaning and becomes
    ``PROPOSE_CHANGES``. Everything ambiguous (missing/unknown decision,
    ``NO_CHANGE`` carrying changes, empty ``PROPOSE_CHANGES``, malformed
    changes) is rejected and goes to the repair call.
    """
    raw_decision = data.get("decision")
    if not isinstance(raw_decision, str) or not raw_decision.strip():
        raise _OutputProblem("missing_decision")
    decision = raw_decision.strip().upper()

    raw_changes = data.get("changes")
    if decision == "CHANGE":
        if not isinstance(raw_changes, list) or not raw_changes:
            raise _OutputProblem("change_without_changes")
        decision = "PROPOSE_CHANGES"

    if decision not in ("NO_CHANGE", "PROPOSE_CHANGES"):
        raise _OutputProblem("invalid_decision")

    if decision == "NO_CHANGE":
        # ``NO_CHANGE`` plus changes is contradictory: never silently discard
        # real proposals and never turn an invalid output into a no-op.
        if raw_changes not in (None, []):
            raise _OutputProblem("no_change_with_changes")
        changes: list[dict] = []
    else:
        if not isinstance(raw_changes, list) or not raw_changes:
            raise _OutputProblem("missing_changes")
        changes = []
        for item in raw_changes[:_MAX_CHANGES]:
            if not isinstance(item, dict):
                raise _OutputProblem("invalid_change_item")
            changes.append(
                {key: item[key] for key in _PROPOSAL_KEYS if key in item}
            )
        if len(raw_changes) > _MAX_CHANGES:
            # Record the omission by operation/field only — never content.
            omitted = [
                item
                for item in raw_changes[_MAX_CHANGES:]
                if isinstance(item, dict)
            ]
            _LOGGER.warning(
                "profile_review_changes_truncated total=%s kept=%s omitted=%s keys=%s",
                len(raw_changes),
                len(changes),
                len(omitted),
                ",".join(
                    f"{item.get('operation')}:{item.get('field_key')}"
                    for item in omitted
                ),
            )

    candidate = {
        # Trusted identifiers always come from the local payload.
        "person_id": person_id,
        "base_revision": base_revision,
        "decision": decision,
        "reason": str(data.get("reason") or "")[:500],
        "facts": _normalise_facts(data.get("facts")),
        "changes": changes,
    }
    try:
        return ProfileReviewDecision.model_validate(candidate)
    except ValidationError as exc:
        raise _OutputProblem(_validation_code(exc)) from exc


class DeepSeekProfileRefinementProvider:
    name = "deepseek-profile-refinement"
    prompt_version = PROFILE_REFINEMENT_PROMPT_VERSION

    def __init__(
        self,
        client,
        *,
        max_tokens: int = 1600,
        logger=None,
        event_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.logger = logger
        self.event_id = event_id
        self.last_model = ""

    async def review(self, payload: dict) -> ProfileReviewDecision:
        person_id = str(payload.get("person_id") or "")
        base_revision = int(payload.get("profile_revision") or 0)
        messages = build_profile_refinement_messages(payload)

        response = await self._complete(messages)
        try:
            decision, fields = self._decode(response, person_id, base_revision)
        except _OutputProblem as first:
            self._diagnose(response, first, repaired=False, ok=False)
            # Exactly one format-repair attempt: original task + error type.
            repair_messages = messages + [
                build_profile_refinement_repair_message(first.code)
            ]
            repaired_response = await self._complete(repair_messages)
            try:
                decision, fields = self._decode(
                    repaired_response, person_id, base_revision
                )
            except _OutputProblem as second:
                self._diagnose(
                    repaired_response, second, repaired=True, ok=False
                )
                raise ProfileRefinementValidationError(
                    "profile refinement output invalid after repair:"
                    f" {second.code}"
                ) from second
            self._diagnose(
                repaired_response, None, fields=fields, repaired=True, ok=True
            )
            return decision

        self._diagnose(response, None, fields=fields, repaired=False, ok=True)
        return decision

    async def _complete(self, messages: list[dict]):
        try:
            response = await self.client.complete(
                messages,
                request_type="analysis",
                json_mode=True,
                max_tokens=self.max_tokens,
                event_id=self.event_id,
            )
        except Exception as exc:
            raise ProfileRefinementProviderError(
                f"profile refinement call failed: {type(exc).__name__}"
            ) from exc
        self.last_model = getattr(response, "model", "") or self.last_model
        return response

    def _decode(
        self, response, person_id: str, base_revision: int
    ) -> tuple[ProfileReviewDecision, tuple[str, ...]]:
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason in _TRUNCATED_FINISH_REASONS:
            raise _OutputProblem("truncated")
        try:
            data = _extract_json_object(response.content)
        except _OutputProblem as problem:
            problem.fields = ()
            raise
        problem_fields = tuple(sorted(str(key) for key in data.keys()))
        try:
            decision = _normalise_decision(
                data, person_id=person_id, base_revision=base_revision
            )
        except _OutputProblem as problem:
            problem.fields = problem_fields
            raise
        return decision, problem_fields

    def _diagnose(
        self,
        response,
        problem: Optional[_OutputProblem],
        *,
        fields: tuple[str, ...] = (),
        repaired: bool,
        ok: bool,
    ) -> None:
        """Log routing metadata only: no content, no values, no profile text."""
        logger = self.logger or _LOGGER
        if problem is not None:
            fields = problem.fields
        logger.info(
            "profile_refinement_output model=%s prompt_version=%s"
            " finish_reason=%s fields=%s error=%s repaired=%s ok=%s",
            getattr(response, "model", "") or self.last_model or "unknown",
            self.prompt_version,
            getattr(response, "finish_reason", None) or "unknown",
            ",".join(fields) or "-",
            problem.code if problem is not None else "none",
            repaired,
            ok,
        )


class MockProfileRefinementProvider:
    """Offline default: no change is the honest result without a model."""

    name = "mock-profile-refinement"
    prompt_version = PROFILE_REFINEMENT_PROMPT_VERSION

    async def review(self, payload: dict) -> ProfileReviewDecision:
        return ProfileReviewDecision(
            person_id=str(payload.get("person_id") or ""),
            base_revision=int(payload.get("profile_revision") or 0),
            decision="NO_CHANGE",
            reason="mock provider",
            changes=[],
        )


class ScriptedProfileRefinementProvider:
    """Test helper: replay a fixed decision or queue of decisions."""

    name = "scripted-profile-refinement"
    prompt_version = PROFILE_REFINEMENT_PROMPT_VERSION

    def __init__(self, decisions: Any = None, *, error: Optional[Exception] = None):
        if decisions is None:
            decisions = []
        if isinstance(decisions, (dict, ProfileReviewDecision)):
            decisions = [decisions]
        self.decisions = list(decisions)
        self.error = error
        self.payloads: list[dict] = []

    async def review(self, payload: dict) -> Any:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        if not self.decisions:
            return ProfileReviewDecision(
                person_id=str(payload.get("person_id") or ""),
                base_revision=int(payload.get("profile_revision") or 0),
                decision="NO_CHANGE",
                reason="script exhausted",
                changes=[],
            )
        item = self.decisions.pop(0)
        if callable(item):
            return item(payload)
        return item


__all__ = [
    "DeepSeekProfileRefinementProvider",
    "MockProfileRefinementProvider",
    "ScriptedProfileRefinementProvider",
    "ProfileChangeProposal",
]
