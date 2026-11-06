"""DeepSeek-backed Profile Refinement provider (V0.9.7).

This module reuses the existing :class:`DeepSeekClient`; it does not create a
second API client, model scheduler or persona system. The provider only turns
one review payload into a schema-shaped proposal. Every identifier is supplied
by trusted code, and anything the model invents is dropped before validation.
Prompts and completions are never logged here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementProviderError,
    ProfileRefinementValidationError,
    ProfileReviewDecision,
)
from companion.prompts.profile_refinement import (
    PROFILE_REFINEMENT_PROMPT_VERSION,
    build_profile_refinement_messages,
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_PROPOSAL_KEYS = {
    "operation",
    "field_key",
    "target_entry_ids",
    "proposed_content",
    "evidence_event_ids",
    "evidence_quotes",
    "reason",
    "confidence",
    "requires_confirmation",
}


def _extract_json(content: str) -> dict:
    text = (content or "").strip()
    text = _FENCE.sub("", text).strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ProfileRefinementProviderError(
            "profile refinement returned malformed JSON"
        ) from exc
    if not isinstance(data, dict):
        raise ProfileRefinementProviderError(
            "profile refinement returned a non-object JSON value"
        )
    return data


def _sanitise_decision(data: dict) -> dict:
    """Keep only fields the trusted local schema knows about.

    Unknown keys are discarded rather than interpreted. A model cannot add a
    person id, operation or permission that the local policy did not ask for.
    """
    decision = str(data.get("decision", "")).strip().upper()
    if decision not in ("NO_CHANGE", "PROPOSE_CHANGES"):
        raise ProfileRefinementValidationError(
            f"invalid decision {data.get('decision')!r}"
        )
    changes: list[dict] = []
    raw_changes = data.get("changes")
    if decision == "NO_CHANGE":
        raw_changes = []
    if isinstance(raw_changes, list):
        for item in raw_changes[:8]:
            if not isinstance(item, dict):
                continue
            clean = {key: item[key] for key in _PROPOSAL_KEYS if key in item}
            changes.append(clean)
    return {
        "person_id": str(data.get("person_id") or ""),
        "base_revision": data.get("base_revision", 0),
        "decision": decision,
        "reason": str(data.get("reason") or "")[:500],
        "changes": changes,
    }


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
        messages = build_profile_refinement_messages(payload)
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
        self.last_model = getattr(response, "model", "") or ""
        data = _sanitise_decision(_extract_json(response.content))
        try:
            return ProfileReviewDecision.model_validate(data)
        except Exception as exc:
            raise ProfileRefinementValidationError(str(exc)) from exc


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
