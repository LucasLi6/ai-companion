"""Profile-aware continuous refinement (V0.9.7).

The old local learning pass looked at *one new utterance* and decided which
field to fill. This module implements the opposite order:

1. read the person's complete existing profile,
2. read the current turn plus the minimal authorized context,
3. ask a model whether the understanding should change at all,
4. let trusted local code validate and apply only the necessary changes.

A model never writes the database. Its output is a *proposal*; every person id,
revision, entry id and evidence id is supplied or checked locally. ``NO_CHANGE``
is a first-class result and does not bump the profile revision.

This module contains the schemas and the engine. Persistence and the atomic
commit live in :class:`~companion.people.profile_service.PersonProfileService`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from companion.people.profile_models import (
    ProfileFieldSpec,
    ProfileFieldStorage,
    profile_field_spec,
)

MAX_CHANGES_PER_REVIEW = 5
MAX_ENTRY_CHARS = 1200
MAX_QUOTE_CHARS = 500
DEFAULT_AUTO_APPLY_CONFIDENCE = 0.7
DEFAULT_RECENT_TURNS = 8
DEFAULT_RECENT_CHARS = 2000

_OPERATIONS = ("ADD", "UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE")


class ProfileRefinementError(Exception):
    """Base class for refinement failures."""


class ProfileRefinementValidationError(ProfileRefinementError):
    """The model output is not valid JSON or fails the strict schema."""


class ProfileRefinementProviderError(ProfileRefinementError):
    """The provider call failed; the review may be retried."""


class ProfileEvidenceBundle(BaseModel):
    """Authorized evidence for exactly one person and one turn."""

    person_id: str
    source_event_id: str
    current_turn: str = ""
    recent_context: list[dict] = Field(default_factory=list)
    allowed_event_ids: list[str] = Field(default_factory=list)
    allowed_memory_ids: list[str] = Field(default_factory=list)
    authorized: bool = True
    reason: str = ""

    def quote_texts(self) -> list[str]:
        texts = [self.current_turn]
        texts.extend(
            str(item.get("text") or "") for item in self.recent_context
        )
        return [text for text in texts if text]


class SnapshotEntry(BaseModel):
    entry_id: str
    field_key: str
    content: str
    status: str
    source: str
    confidence: float = 0.5
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    updated_at: Optional[str] = None

    def prompt_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
        }


class FullProfileSnapshot(BaseModel):
    """Complete, consistent view of one person's profile at one revision."""

    person_id: str
    revision: int
    display_name: str = "unknown"
    identity: dict[str, Any] = Field(default_factory=dict)
    relationship: dict[str, Any] = Field(default_factory=dict)
    communication: dict[str, Any] = Field(default_factory=dict)
    personality: dict[str, Any] = Field(default_factory=dict)
    interests: dict[str, Any] = Field(default_factory=dict)
    knowledge: dict[str, Any] = Field(default_factory=dict)
    background: dict[str, Any] = Field(default_factory=dict)
    projects: dict[str, Any] = Field(default_factory=dict)
    goals: dict[str, Any] = Field(default_factory=dict)
    habits: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    interaction: dict[str, Any] = Field(default_factory=dict)
    entries: dict[str, list[SnapshotEntry]] = Field(default_factory=dict)
    legacy_fields: dict[str, Any] = Field(default_factory=dict)
    pending_candidates: list[dict] = Field(default_factory=list)
    compact_summary: str = ""
    updated_at: Optional[str] = None
    last_reviewed_at: Optional[str] = None

    def entry_ids(self) -> set[str]:
        return {
            entry.entry_id
            for items in self.entries.values()
            for entry in items
        }

    def model_input(self) -> dict:
        """The exact payload the refinement model may see.

        IDs and the revision are supplied by the caller, never by the model.
        Entries keep their entry id so a proposal can target one precisely.
        """
        categories = {
            key: value
            for key, value in (
                ("communication", self.communication),
                ("personality", self.personality),
                ("interests", self.interests),
                ("knowledge", self.knowledge),
                ("background", self.background),
                ("projects", self.projects),
                ("goals", self.goals),
                ("habits", self.habits),
                ("values", self.values),
                ("interaction", self.interaction),
            )
        }
        return {
            "person_id": self.person_id,
            "profile_revision": self.revision,
            "existing_profile": {
                "display_name_is_registration_label": True,
                "identity": self.identity,
                "relationship": self.relationship,
                **categories,
            },
            "entry_ids": {
                key: [entry.entry_id for entry in items]
                for key, items in self.entries.items()
            },
        }


class ProfileChangeProposal(BaseModel):
    """One minimal change proposed by the model.

    The schema forbids unknown keys so a model cannot smuggle an id or
    permission the local rules never asked for.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "ADD", "UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE"
    ]
    field_key: str = Field(min_length=1, max_length=120)
    target_entry_ids: list[str] = Field(default_factory=list)
    proposed_content: Optional[str] = Field(default=None, max_length=MAX_ENTRY_CHARS)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_confirmation: bool = False

    @field_validator("target_entry_ids", "evidence_event_ids")
    @classmethod
    def _bounded_ids(cls, value: list[str]) -> list[str]:
        return [str(item)[:128] for item in value if str(item or "").strip()][:32]

    @field_validator("evidence_quotes")
    @classmethod
    def _bounded_quotes(cls, value: list[str]) -> list[str]:
        return [
            " ".join(str(item).split())[:MAX_QUOTE_CHARS]
            for item in value
            if str(item or "").strip()
        ][:8]


class ProfileReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str = Field(min_length=1, max_length=128)
    base_revision: int = Field(ge=0)
    decision: Literal["NO_CHANGE", "PROPOSE_CHANGES"]
    reason: str = Field(default="", max_length=500)
    changes: list[ProfileChangeProposal] = Field(default_factory=list)

    @field_validator("changes")
    @classmethod
    def _no_change_has_no_changes(
        cls, value: list[ProfileChangeProposal]
    ) -> list[ProfileChangeProposal]:
        return value


class ValidatedChange(BaseModel):
    proposal: ProfileChangeProposal
    accepted: bool = True
    auto_apply: bool = True
    issue: Optional[str] = None
    pending_reason: Optional[str] = None


class ProfileReviewResult(BaseModel):
    person_id: str
    source_event_id: str
    base_revision: int
    decision: Literal["NO_CHANGE", "PROPOSE_CHANGES"] = "NO_CHANGE"
    reason: str = ""
    provider: str = ""
    model: str = ""
    changes: list[ValidatedChange] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    status: str = "NO_CHANGE"

    def accepted_changes(self) -> list[ValidatedChange]:
        return [
            change
            for change in self.changes
            if change.accepted and change.auto_apply
        ]

    def pending_changes(self) -> list[ValidatedChange]:
        return [
            change
            for change in self.changes
            if change.accepted and not change.auto_apply
        ]

    def has_changes(self) -> bool:
        return bool(self.accepted_changes())


class ProfileApplyResult(BaseModel):
    person_id: str
    source_event_id: Optional[str] = None
    status: str = "NO_CHANGE"
    base_revision: int = 0
    new_revision: Optional[int] = None
    applied: list[dict] = Field(default_factory=list)
    pending: list[dict] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    reason: str = ""


# -- local validation --------------------------------------------------------


_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}")
_DIGIT_TERM = re.compile(r"\d+(?:\.\d+)?")
_CJK_RUN = re.compile(r"[一-鿿]{2,}")


def _distinctive_terms(text: str) -> set[str]:
    value = str(text or "")
    terms = {match.lower() for match in _LATIN_TERM.findall(value)}
    terms.update(_DIGIT_TERM.findall(value))
    for run in _CJK_RUN.findall(value):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return {term for term in terms if len(term) >= 2}


def _coverage(targets: list[str], proposed: str) -> float:
    """How much of the old specific vocabulary the new text keeps.

    This is deliberately a local, conservative check for *lossy* merges and
    refinements. It is not a semantic judge: it only catches the obvious failure
    where STM32 / RTOS / robot control becomes "喜欢科技".
    """
    terms: set[str] = set()
    for target in targets:
        terms |= _distinctive_terms(target)
    if not terms:
        return 1.0
    proposed_terms = _distinctive_terms(proposed)
    return len(terms & proposed_terms) / len(terms)


_FILLER = re.compile(r"^(喜欢|爱好|兴趣|擅长|正在|在|做|搞|研究|开发)+$")
_SENSITIVE = re.compile(
    r"(健康|疾病|病历|诊断|医院|用药|药物|抑郁|焦虑症|"
    r"银行|信用卡|贷款|工资|收入|存款|密码|"
    r"住址|家庭地址|身份证|护照号)"
)


def _effective_content(snapshot: FullProfileSnapshot, entry_ids: list[str]) -> list[str]:
    wanted = set(entry_ids)
    return [
        entry.content
        for items in snapshot.entries.values()
        for entry in items
        if entry.entry_id in wanted
    ]


class ProfileFieldRegistry:
    """Small local policy layer over the storage field registry."""

    def spec(self, field_key: str) -> Optional[ProfileFieldSpec]:
        return profile_field_spec(field_key)

    def is_strict(self, field_key: str) -> bool:
        spec = self.spec(field_key)
        return bool(spec and spec.strict_identity)

    def is_model_writable(self, field_key: str) -> bool:
        spec = self.spec(field_key)
        return bool(spec and spec.model_writable)

    def is_entry(self, field_key: str) -> bool:
        spec = self.spec(field_key)
        return bool(spec and spec.storage is ProfileFieldStorage.ENTRY)

    @property
    def max_changes(self) -> int:
        return MAX_CHANGES_PER_REVIEW


class ProfileRefinementEngine:
    """Read existing profile -> read evidence -> propose -> validate.

    The exact order matters. ``existing`` is loaded *before* any model output
    exists, and the model never receives or edits anything else.
    """

    def __init__(
        self,
        profiles,
        repository,
        provider,
        context_source,
        registry: Optional[ProfileFieldRegistry] = None,
        logger: Optional[logging.Logger] = None,
        *,
        auto_apply_confidence: float = DEFAULT_AUTO_APPLY_CONFIDENCE,
    ) -> None:
        self.profiles = profiles
        self.repository = repository
        self.provider = provider
        self.context_source = context_source
        self.registry = registry or ProfileFieldRegistry()
        self.logger = logger or logging.getLogger("companion.profile_refinement")
        self.auto_apply_confidence = max(0.0, min(1.0, float(auto_apply_confidence)))

    async def review(self, person_id: str, source_event_id: str) -> ProfileReviewResult:
        # (1) Existing profile first. A review without a profile is not a
        # reason to create or guess one.
        existing = await self.profiles.get_full_snapshot(person_id)
        if existing is None:
            return ProfileReviewResult(
                person_id=person_id,
                source_event_id=source_event_id,
                base_revision=0,
                status="NO_PROFILE",
                reason="no authorized profile",
            )
        # (2) Authorized evidence for this person only.
        evidence = await self.context_source.load(
            person_id=person_id, source_event_id=source_event_id
        )
        if not evidence.authorized:
            return ProfileReviewResult(
                person_id=person_id,
                source_event_id=source_event_id,
                base_revision=existing.revision,
                status="UNAUTHORIZED",
                reason=evidence.reason or "evidence_not_authorized",
            )
        # (3)/(4)/(5) The provider decides whether the whole understanding
        # should change and proposes minimal edits. It never sees a different
        # revision than the one above.
        payload = {
            "person_id": person_id,
            "profile_revision": existing.revision,
            "existing_profile": existing.model_input()["existing_profile"],
            "entry_ids": existing.model_input()["entry_ids"],
            "new_evidence": {
                "source_event_id": source_event_id,
                "current_turn": evidence.current_turn,
                "recent_context": evidence.recent_context,
                "allowed_event_ids": evidence.allowed_event_ids,
            },
        }
        try:
            raw = await self.provider.review(payload)
        except ProfileRefinementError:
            raise
        except Exception as exc:  # provider failure is retryable
            raise ProfileRefinementProviderError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
        decision = self._coerce_decision(raw, person_id, existing.revision)
        return self.validate_decision(existing, evidence, decision)

    # -- decision coercion / validation -----------------------------------

    @staticmethod
    def _coerce_decision(
        raw: Any, person_id: str, base_revision: int
    ) -> ProfileReviewDecision:
        if isinstance(raw, ProfileReviewDecision):
            data = raw.model_dump()
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            raise ProfileRefinementValidationError(
                "provider returned an unsupported decision object"
            )
        # Program-owned fields are authoritative.
        data["person_id"] = person_id
        data["base_revision"] = base_revision
        if str(data.get("decision", "")).upper() == "NO_CHANGE":
            data["decision"] = "NO_CHANGE"
            data["changes"] = []
        try:
            return ProfileReviewDecision.model_validate(data)
        except ValidationError as exc:
            raise ProfileRefinementValidationError(str(exc)) from exc

    def validate_decision(
        self,
        existing: FullProfileSnapshot,
        evidence: ProfileEvidenceBundle,
        decision: ProfileReviewDecision,
    ) -> ProfileReviewResult:
        result = ProfileReviewResult(
            person_id=existing.person_id,
            source_event_id=evidence.source_event_id,
            base_revision=existing.revision,
            decision=decision.decision,
            reason=decision.reason,
        )
        if decision.decision == "NO_CHANGE":
            result.status = "NO_CHANGE"
            return result

        allowed_entry_ids = existing.entry_ids()
        allowed_event_ids = set(evidence.allowed_event_ids)
        quote_texts = evidence.quote_texts()
        validated: list[ValidatedChange] = []

        for index, proposal in enumerate(decision.changes):
            issue, pending_reason = self._check_proposal(
                proposal, existing, allowed_entry_ids, allowed_event_ids, quote_texts
            )
            if issue is None and index >= self.registry.max_changes:
                issue = "change_limit_exceeded"
            if issue is not None:
                result.rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": issue,
                    }
                )
                continue
            validated.append(
                ValidatedChange(
                    proposal=proposal,
                    accepted=True,
                    auto_apply=pending_reason is None,
                    pending_reason=pending_reason,
                )
            )

        result.changes = validated
        result.status = "PROPOSED" if validated else "REJECTED"
        return result

    def _check_proposal(
        self,
        proposal: ProfileChangeProposal,
        existing: FullProfileSnapshot,
        allowed_entry_ids: set[str],
        allowed_event_ids: set[str],
        quote_texts: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Return ``(issue, pending_reason)``.

        ``issue`` rejects the proposal entirely. ``pending_reason`` keeps it as
        a durable pending candidate that a person can confirm later.
        """
        spec = self.registry.spec(proposal.field_key)
        if spec is None:
            return "unknown_field_key", None
        if not self.registry.is_entry(proposal.field_key):
            return "field_not_an_understanding_entry", None
        if not proposal.proposed_content and proposal.operation != "ARCHIVE":
            return "missing_proposed_content", None
        if proposal.operation in ("UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE"):
            if not proposal.target_entry_ids:
                return "missing_target_entry_ids", None
            if not set(proposal.target_entry_ids) <= allowed_entry_ids:
                return "target_entry_not_in_this_profile", None
            targets = _effective_content(existing, proposal.target_entry_ids)
            if len(targets) != len(set(proposal.target_entry_ids)):
                return "target_entry_not_effective", None
            if proposal.operation == "MERGE" and len(set(proposal.target_entry_ids)) < 2:
                return "merge_needs_two_targets", None
        if proposal.operation == "ADD":
            if set(proposal.target_entry_ids):
                return "add_must_not_target_entries", None
            if not self._add_is_new(existing, proposal):
                return "already_present", None
        if proposal.operation in ("MERGE", "REFINE"):
            targets = _effective_content(existing, proposal.target_entry_ids)
            content = proposal.proposed_content or ""
            # Only merging/condensing is checked for lost specifics. A genuine
            # CORRECT may replace a fact (university -> high school); an UPDATE
            # may change a project's state. Those are judged on evidence.
            if _coverage(targets, content) < 0.45:
                return "lossy_change_rejects_existing_facts", None
            # A merge/refine may shorten wording, not delete every specific.
            if content and _FILLER.match(content.strip()):
                return "lossy_change_rejects_existing_facts", None
        if proposal.operation == "ARCHIVE" and (
            proposal.field_key.startswith("identity.")
            or proposal.field_key.startswith("relationship.")
        ):
            return "strict_field_cannot_be_archived", None
        if proposal.evidence_event_ids and not set(
            proposal.evidence_event_ids
        ) <= allowed_event_ids:
            return "evidence_event_not_authorized", None
        if proposal.evidence_quotes:
            for quote in proposal.evidence_quotes:
                if not any(quote and quote in text for text in quote_texts):
                    return "evidence_quote_not_found", None
        if not proposal.evidence_event_ids and not proposal.evidence_quotes:
            # New stable understanding needs some evidence. A no-evidence
            # proposal is kept for a person to confirm rather than applied.
            return None, "no_evidence"
        if _SENSITIVE.search(proposal.proposed_content or ""):
            return None, "sensitive_field_requires_review"
        if self.registry.is_strict(proposal.field_key):
            return None, "confirmed_identity_requires_confirmation"
        if proposal.requires_confirmation:
            return None, "model_requested_confirmation"
        if proposal.confidence < self.auto_apply_confidence:
            return None, "low_confidence"
        return None, None

    @staticmethod
    def _add_is_new(
        existing: FullProfileSnapshot, proposal: ProfileChangeProposal
    ) -> bool:
        content = " ".join((proposal.proposed_content or "").split()).casefold()
        if not content:
            return False
        for entry in existing.entries.get(proposal.field_key, []):
            if " ".join(entry.content.split()).casefold() == content:
                return False
        return True

    # -- apply -------------------------------------------------------------

    async def apply_proposals(self, review: ProfileReviewResult) -> ProfileApplyResult:
        return await self.profiles.apply_refinement(
            review.person_id,
            review,
            source_event_id=review.source_event_id,
        )


class ProfileReviewContextSource:
    """Load only this person's authorized turn and recent context.

    It never includes another speaker's utterance, another person's profile or
    an assistant reply that was not addressed to this person's own turn.
    """

    def __init__(
        self,
        repos,
        *,
        privacy_gateway=None,
        recent_turns: int = DEFAULT_RECENT_TURNS,
        max_chars: int = DEFAULT_RECENT_CHARS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.privacy_gateway = privacy_gateway
        self.recent_turns = max(1, int(recent_turns))
        self.max_chars = max(200, int(max_chars))
        self.logger = logger or logging.getLogger(
            "companion.profile_refinement.context"
        )

    async def load(self, *, person_id: str, source_event_id: str) -> ProfileEvidenceBundle:
        consent = await self.repos.person.get_consent(person_id)
        if consent is None or not (
            consent.allow_processing
            and consent.allow_persistence
            and consent.allow_memory_learning
            and consent.allow_external_ai
        ):
            return ProfileEvidenceBundle(
                person_id=person_id,
                source_event_id=source_event_id,
                authorized=False,
                reason="consent_missing_or_withdrawn",
            )
        event = await self.repos.utterance.get(source_event_id)
        if event is None or event.speaker_person_id != person_id:
            return ProfileEvidenceBundle(
                person_id=person_id,
                source_event_id=source_event_id,
                authorized=False,
                reason="source_event_not_owned_by_person",
            )

        history = await self.repos.utterance.recent_context(
            event.conversation_id,
            exclude_event_id=event.event_id,
            limit=self.recent_turns * 2,
        )
        history = [
            item for item in history if item.speaker_person_id == person_id
        ]
        if self.privacy_gateway is not None:
            history = await self.privacy_gateway.filter_utterances(history)
        history = history[-self.recent_turns :]

        actions = await self.repos.analysis.get_actions_for_events(
            [item.event_id for item in history]
        )
        recent: list[dict] = []
        allowed_events = [event.event_id]
        for item in history:
            recent.append({"role": "human", "text": item.text})
            allowed_events.append(item.event_id)
            action = actions.get(item.event_id)
            if (
                action is not None
                and getattr(
                    getattr(action, "action_type", None), "value", None
                )
                == "RESPOND"
                and getattr(action, "status", None) in ("COMPLETED", "TRUNCATED")
                and getattr(action, "response_text", None)
            ):
                recent.append({"role": "assistant", "text": action.response_text})

        # Character budget: drop oldest complete units, never a partial line.
        total = sum(len(item["text"]) for item in recent)
        while total > self.max_chars and recent:
            dropped = recent.pop(0)
            total -= len(dropped["text"])
            if recent and recent[0]["role"] == "assistant":
                total -= len(recent.pop(0)["text"])

        return ProfileEvidenceBundle(
            person_id=person_id,
            source_event_id=source_event_id,
            current_turn=event.text,
            recent_context=recent,
            allowed_event_ids=allowed_events,
            allowed_memory_ids=[],
        )


__all__ = [
    "DEFAULT_AUTO_APPLY_CONFIDENCE",
    "FullProfileSnapshot",
    "MAX_CHANGES_PER_REVIEW",
    "ProfileApplyResult",
    "ProfileChangeProposal",
    "ProfileEvidenceBundle",
    "ProfileFieldRegistry",
    "ProfileRefinementEngine",
    "ProfileRefinementError",
    "ProfileRefinementProviderError",
    "ProfileRefinementValidationError",
    "ProfileReviewContextSource",
    "ProfileReviewDecision",
    "ProfileReviewResult",
    "SnapshotEntry",
    "ValidatedChange",
]
