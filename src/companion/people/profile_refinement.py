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
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from companion.people.profile_facts import (
    AssertionType,
    ExtractedPersonFact,
    FactClass,
    FactRelation,
    FactRelationDecision,
    RELATION_OPERATIONS,
    TemporalScope,
    classify_fact,
)
from companion.people.profile_models import (
    ENTRY_FIELD_KEYS,
    ProfileFieldSpec,
    ProfileFieldStorage,
    ProfileSource,
    profile_field_spec,
)

MAX_CHANGES_PER_REVIEW = 10
MAX_ENTRY_CHARS = 1200
MAX_QUOTE_CHARS = 500
DEFAULT_RECENT_TURNS = 8
DEFAULT_RECENT_CHARS = 2000
MAX_SNAPSHOT_LEGACY_FIELDS = 40
MAX_SNAPSHOT_TEXT_CHARS = 4000
MAX_FACT_INDEX_ROWS = 80

_OPERATIONS = ("ADD", "UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE", "SAME")

# Cross-field movement is not a general capability. Only an explicit project
# state transition may target an entry that lives under a different field key,
# and only through UPDATE. Any other cross-field write is a target-safety bug.
ALLOWED_FIELD_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("projects.active", "projects.paused"),
        ("projects.paused", "projects.active"),
        ("projects.active", "projects.completed"),
        ("projects.paused", "projects.completed"),
    }
)
_STRICT_FIELD_PREFIXES = ("identity.", "relationship.")


def is_strict_field(field_key: str) -> bool:
    """Whether a field is confirmed identity/relationship or registry-marked."""
    if str(field_key or "").startswith(_STRICT_FIELD_PREFIXES):
        return True
    spec = profile_field_spec(field_key)
    return bool(spec and spec.strict_identity)


def validate_change_targets(
    operation: str,
    target_field_key: str,
    target_entries: list,
) -> None:
    """Enforce that an operation only touches entries it is allowed to touch.

    ``target_entries`` are the *actual* stored rows (dicts with ``entry_id``
    and ``field_key``), never the model's claim about them. The function raises
    ``ValueError`` with a stable reason code; both the engine's pre-validation
    and the final write transaction call it, so a caller cannot bypass it by
    constructing a ``ValidatedChange`` directly.
    """
    operation = str(operation or "").upper()
    entries = list(target_entries or [])
    fields = [str(item.get("field_key") or "") for item in entries]

    if operation == "ADD":
        if entries:
            raise ValueError("add_must_not_target_entries")
        return
    if operation == "SAME":
        # Evidence-only: the fact is already stored, possibly under another
        # field. It may target zero or more entries, but it never rewrites one.
        ids = {item.get("entry_id") for item in entries}
        if len(ids) < len(entries):
            raise ValueError("same_targets_must_be_unique")
        return
    if operation not in ("UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE"):
        raise ValueError("unsupported_operation")

    if not entries:
        raise ValueError("missing_target_entry_ids")

    if operation in ("MERGE", "REFINE"):
        if any(field != target_field_key for field in fields):
            raise ValueError(
                "merge_targets_differ_in_field"
                if operation == "MERGE"
                else "refine_target_field_mismatch"
            )
        if operation == "MERGE":
            ids = {item.get("entry_id") for item in entries}
            if len(entries) < 2 or len(ids) < 2:
                raise ValueError("merge_needs_two_targets")
        elif len(entries) != 1:
            raise ValueError("refine_needs_one_target")
        return

    if operation in ("UPDATE", "CORRECT"):
        if len(entries) != 1:
            raise ValueError("update_needs_one_target")
        source_field = fields[0]
        if source_field != target_field_key:
            if operation != "UPDATE":
                raise ValueError("cross_field_change_not_allowed")
            if (source_field, target_field_key) not in ALLOWED_FIELD_TRANSITIONS:
                raise ValueError("field_transition_not_allowed")
            if is_strict_field(source_field) or is_strict_field(target_field_key):
                raise ValueError("strict_field_transition_not_allowed")
        return

    # ARCHIVE
    if any(field != target_field_key for field in fields):
        raise ValueError("archive_target_field_mismatch")
    for field in set(fields) | {target_field_key}:
        if is_strict_field(field):
            raise ValueError("strict_field_cannot_be_archived")


class ProfileRefinementError(Exception):
    """Base class for refinement failures."""


class ProfileRefinementValidationError(ProfileRefinementError):
    """The model output is not valid JSON or fails the strict schema."""


class ProfileRefinementProviderError(ProfileRefinementError):
    """The provider call failed; the review may be retried."""


class ProfileEvidenceBundle(BaseModel):
    """Authorized evidence for exactly one person and one turn.

    ``recent_context`` is the conversational context the model may read: it
    contains the person's own turns and Ava's visible replies. Only
    ``eligible_user_evidence`` is *evidence* about the person. Ava's reply is
    context, never a direct self-report, so a proposal may not cite or quote it
    as proof of a personal fact.
    """

    person_id: str
    source_event_id: str
    current_turn: str = ""
    recent_context: list[dict] = Field(default_factory=list)
    eligible_user_evidence: list[dict] = Field(default_factory=list)
    allowed_event_ids: list[str] = Field(default_factory=list)
    allowed_memory_ids: list[str] = Field(default_factory=list)
    authorized: bool = True
    reason: str = ""

    def user_event_texts(self) -> dict[str, str]:
        """``event_id -> text`` for this person's own messages only."""
        texts: dict[str, str] = {}
        for item in self.eligible_user_evidence:
            event_id = str(item.get("event_id") or "")
            text = str(item.get("text") or "")
            if event_id and text:
                texts[event_id] = text
        if self.current_turn:
            texts.setdefault(self.source_event_id, self.current_turn)
        return texts

    def quote_texts(self) -> list[str]:
        """Texts a quote may be located in: the person's own words only."""
        return [text for text in self.user_event_texts().values() if text]


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
        legacy_names = sorted(self.legacy_fields)
        legacy = {
            key: self.legacy_fields[key]
            for key in legacy_names[:MAX_SNAPSHOT_LEGACY_FIELDS]
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
            "legacy_unconverted": legacy,
            "compact_summary": self.compact_summary[:MAX_SNAPSHOT_TEXT_CHARS],
            "snapshot_limits": {
                "legacy_total": len(legacy_names),
                "legacy_included": len(legacy),
                "truncated": (
                    len(legacy_names) > len(legacy)
                    or len(self.compact_summary) > MAX_SNAPSHOT_TEXT_CHARS
                ),
            },
        }


class ProfileChangeProposal(BaseModel):
    """One minimal change proposed by the model.

    The schema forbids unknown keys so a model cannot smuggle an id or
    permission the local rules never asked for.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "ADD", "UPDATE", "CORRECT", "MERGE", "REFINE", "ARCHIVE", "SAME"
    ]
    field_key: str = Field(min_length=1, max_length=120)
    target_entry_ids: list[str] = Field(default_factory=list)
    fact_id: Optional[str] = Field(default=None, max_length=64)
    # Resolved by the engine from ``facts`` and attached locally. It is never
    # read from model output; the provider drops unknown proposal keys.
    fact: Optional[ExtractedPersonFact] = None
    relation: Optional[
        Literal[
            "SAME", "EXTENDS", "STATE_CHANGE", "CORRECTS", "DISTINCT",
            "UNCERTAIN",
        ]
    ] = None
    proposed_content: Optional[str] = Field(default=None, max_length=MAX_ENTRY_CHARS)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Legacy model output key. It is accepted for compatibility but is *never*
    # consulted: a model cannot block a locally legal write by setting it.
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
    facts: list[ExtractedPersonFact] = Field(default_factory=list)
    changes: list[ProfileChangeProposal] = Field(default_factory=list)

    @field_validator("changes")
    @classmethod
    def _no_change_has_no_changes(
        cls, value: list[ProfileChangeProposal]
    ) -> list[ProfileChangeProposal]:
        return value


class ValidatedChange(BaseModel):
    """One locally validated proposal.

    Validation is terminal: a change is either accepted for the write
    transaction or rejected with a stable reason. There is no "maybe" state and
    no proposal is ever parked for a later confirmation.
    """

    proposal: ProfileChangeProposal
    accepted: bool = True
    issue: Optional[str] = None


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
        return [change for change in self.changes if change.accepted]

    def has_changes(self) -> bool:
        return bool(self.accepted_changes())


class ProfileApplyResult(BaseModel):
    person_id: str
    source_event_id: Optional[str] = None
    status: str = "NO_CHANGE"
    base_revision: int = 0
    new_revision: Optional[int] = None
    applied: list[dict] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    reason: str = ""

    @property
    def applied_count(self) -> int:
        return len(self.applied)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def receipt(self) -> dict:
        """The stable, content-free result a review receipt records."""
        return {
            "status": self.status,
            "applied_count": self.applied_count,
            "rejected_count": self.rejected_count,
            "new_revision": self.new_revision,
            "base_revision": self.base_revision,
        }


class DirectSelfReportResult(BaseModel):
    """Outcome of one trusted, explicit first-person profile statement."""

    person_id: str
    field_key: str
    value: Any = None
    status: str = "NO_CHANGE"
    revision: Optional[int] = None
    reason: str = ""
    entry_id: Optional[str] = None



@dataclass(frozen=True)
class ProfileReviewExecution:
    """Trusted identity of one background review execution.

    Constructed by the worker handler from values only trusted code knows: the
    claimed receipt, the source event and the person-level lease owner. It
    carries no permission the model could claim. The service re-checks all
    four values inside the final write transaction, so a worker that lost its
    lease (or whose receipt was taken over) can never commit a profile change.
    """

    person_id: str
    review_id: str
    lease_owner_id: str
    source_event_id: str


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
# A joke, hypothetical or reported speech is not a stable self-report. This is
# only a conservative textual flag: it sends the proposal to pending review,
# it does not claim to understand the sentence.
_HYPOTHETICAL = re.compile(
    r"(如果|假如|假设|要是|开玩笑|说着玩|玩笑|据说|听说|别人说|他说|她说|"
    r"他们说|她们说|有人说|朋友说|同事说|同学说)"
)
_SENSITIVE = re.compile(
    r"(健康|疾病|病历|诊断|医院|用药|药物|抑郁|焦虑症|"
    r"银行|信用卡|贷款|工资|收入|存款|密码|"
    r"住址|家庭地址|身份证|护照号)"
)

# A trait/habit must be described by the person, not inferred from one act.
# These markers are required inside the supporting quote for inference-only
# fields (personality, stable habits, working preferences).
_TRAIT_SELF_DESCRIPTION = re.compile(
    r"(我是|我比较|我做事|我做|我通常|我一般|我经常|我常常|我每天|我每周|"
    r"我习惯|我喜欢|我不喜欢|我讨厌|我倾向于|我偏向|对我来说|"
    r"我性格|我这个人|我这人|我很少|我从不|我不太|我更喜欢|"
    r"我容易|我总是|我平时)"
)

# Negation changes meaning. "喜欢正式回答" and "不喜欢正式回答" must never
# deduplicate into one stored item.
_NEGATION = re.compile(
    r"(不喜欢|不想要|不需要|不爱|讨厌|不想|不再|不常|很少|从不|"
    r"没有|不是|不用|不太|别|勿)"
)

# A single observed behaviour is not a stable trait. These fields may still be
# filled from a direct self-description, but a model inference without a
# verbatim user quote stays pending instead of being written as fact.
INFERENCE_ONLY_FIELDS = frozenset(
    {
        "personality.traits",
        "personality.behavior_patterns",
        "personality.thinking_style",
        "personality.decision_style",
        "habits.stable_habits",
        "habits.working_preferences",
    }
)

_ENTRY_PREFIX = re.compile(
    r"^(?:用户|本人|我|目前|最近|现在|正在|一直|平时|主要|"
    r"喜欢|不喜欢|不太喜欢|爱好|兴趣|擅长|开始|"
    r"在做|在学|在|做|学|研究)+"
)
_ENTRY_PUNCT = re.compile(r"[\s，。！？、,.!?;；:：·\-—~～\"'“”‘’()（）\[\]【】]+")


def _entry_core(text: Any) -> str:
    """Content with leading subject/filler words removed, for de-duplication."""
    value = _ENTRY_PUNCT.sub("", _normalise_text(text).casefold())
    previous = None
    while value and value != previous:
        previous = value
        value = _ENTRY_PREFIX.sub("", value, count=1)
    return value


def _entry_terms(core: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9_+\-]{2,}", core))
    for run in re.findall(r"[一-鿿]{2,}", core):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return {term for term in terms if len(term) >= 2}


def _polarity(text: Any) -> str:
    """Whether a statement is affirmative or negated.

    This is intentionally shallow and local: it exists only so de-duplication
    never collapses opposite statements.
    """
    return "NEG" if _NEGATION.search(_normalise_text(text)) else "POS"


def entries_equivalent(left: Any, right: Any) -> bool:
    """Conservative local "same item" check (no model call).

    Exact duplicates, one text containing the other's core entity, and very
    high bigram overlap are treated as the same stored item. Unrelated
    projects ("机器狗" / "奶茶机") stay distinct; genuinely new details belong
    in an UPDATE, not a second ADD. Opposite polarity ("喜欢" / "不喜欢") is
    never equivalent.
    """
    if _polarity(left) != _polarity(right):
        return False
    a, b = _entry_core(left), _entry_core(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    # Containment means "same item with a slightly different prefix/filler",
    # not "the same entity plus several new facts". The latter is new detail
    # that belongs in an UPDATE and must not be discarded as a duplicate.
    if len(shorter) >= 3 and shorter in longer and len(longer) - len(shorter) <= 3:
        return True
    terms_a, terms_b = _entry_terms(a), _entry_terms(b)
    if not terms_a or not terms_b:
        return False
    return len(terms_a & terms_b) / len(terms_a | terms_b) >= 0.75


def content_supported_by_text(content: Any, texts: list[str]) -> bool:
    """Whether a proposed statement has concrete support in the source text.

    This is a conservative local anti-fabrication guard, not a semantic judge:
    at least one distinctive term of the proposal must occur in the person's
    own cited words. A model may paraphrase, but may not attach a real quote to
    an unrelated invention.
    """
    # A trailing state marker such as "（已完成）" describes the change the
    # UPDATE is making; it is not a new fact that the source text must contain.
    core = re.sub(r"[（(][^）)]*[）)]", "", str(content or ""))
    terms = _distinctive_terms(_entry_core(core)) or _distinctive_terms(core)
    if not terms:
        return False
    corpus: set[str] = set()
    for text in texts or []:
        corpus |= _distinctive_terms(text)
    if not corpus:
        return False
    # At least one distinctive term of the proposal must occur in the
    # person's own cited words. This is deliberately not a semantic judge:
    # the point is to catch a proposal built from nothing, not to penalise
    # legitimate paraphrase ("期末考试" -> "考试结束了").
    return bool(terms & corpus)


def requires_direct_quote(field_key: str) -> bool:
    """Whether an auto-applied claim from this field needs a verbatim quote."""
    return str(field_key or "") in INFERENCE_ONLY_FIELDS


def is_human_confirmed_source(source: Any) -> bool:
    """Entries written through the explicit confirmation path."""
    return str(source or "").upper() == ProfileSource.USER_CORRECTION.value


def _effective_content(snapshot: FullProfileSnapshot, entry_ids: list[str]) -> list[str]:
    wanted = set(entry_ids)
    return [
        entry.content
        for items in snapshot.entries.values()
        for entry in items
        if entry.entry_id in wanted
    ]


def _entries_by_id(snapshot: FullProfileSnapshot) -> dict[str, SnapshotEntry]:
    return {
        entry.entry_id: entry
        for items in snapshot.entries.values()
        for entry in items
    }


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _unanswered_fact_reason(fact: ExtractedPersonFact) -> str:
    """Explain why an extracted fact produced no profile change.

    Every fact in model output must appear in the review result. When the model
    gave no relation for it, the reason still distinguishes a missing decision
    from a fact that never belonged to the long-term Profile.
    """
    if fact.assertion_type == AssertionType.UNCERTAIN:
        return "fact_uncertain_requires_more_evidence"
    if fact.classification is not None:
        return (
            "fact_has_no_relation_decision"
            if fact.classification == FactClass.PROFILE
            else "fact_not_profile_relevant"
        )
    if fact.temporal_scope in (
        TemporalScope.EVENT,
        TemporalScope.TEMPORARY,
    ) or fact.assertion_type in (
        AssertionType.HYPOTHETICAL,
        AssertionType.THIRD_PARTY,
    ):
        return "fact_not_profile_relevant"
    return "fact_has_no_relation_decision"


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
    ) -> None:
        self.profiles = profiles
        self.repository = repository
        self.provider = provider
        self.context_source = context_source
        self.registry = registry or ProfileFieldRegistry()
        self.logger = logger or logging.getLogger("companion.profile_refinement")

    async def _read_authorized_snapshot(
        self, person_id: str
    ) -> Optional[FullProfileSnapshot]:
        """Read the snapshot, repairing a legally missing profile row once.

        ``get_full_snapshot`` returns ``None`` for two very different states:
        no permission at all, and permission granted but the row was never
        created (accounts registered before the ensure-on-register hook). Only
        the second is repaired, and only for a person who currently grants
        processing, persistence, learning and external-AI use.
        """
        can_review = getattr(self.profiles, "can_review", None)
        if can_review is None or not await can_review(person_id):
            return None
        existing = await self.profiles.get_full_snapshot(person_id)
        if existing is not None:
            return existing
        created = await self.profiles.ensure_profile(
            person_id, actor="profile_review_repair"
        )
        existing = await self.profiles.get_full_snapshot(person_id)
        if existing is None:
            self.logger.warning(
                "profile_review_profile_repair_failed person_id=%s created=%s",
                person_id,
                bool(created),
            )
            raise ProfileRefinementError("authorized_profile_initialization_failed")
        self.logger.info(
            "profile_review_profile_repaired person_id=%s created=%s revision=%s",
            person_id,
            bool(created),
            existing.revision,
        )
        return existing

    async def review(self, person_id: str, source_event_id: str) -> ProfileReviewResult:
        # (1) Existing profile first. A legally missing row is created for an
        # authorized person; no permission means no profile and no review. A
        # repair that still cannot be read raises — never a fake NO_CHANGE.
        existing = await self._read_authorized_snapshot(person_id)
        if existing is None:
            return ProfileReviewResult(
                person_id=person_id,
                source_event_id=source_event_id,
                base_revision=0,
                status="UNAUTHORIZED",
                reason="missing_consent",
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
        snapshot = existing.model_input()
        # Derived semantic index: entity / attribute / time scope / canonical
        # value per entry. It is advisory recall material only; stored entries
        # remain authoritative and local validation re-reads them.
        list_fact_index = getattr(self.repository, "list_fact_index", None)
        fact_index = (
            await list_fact_index(person_id) if list_fact_index is not None else []
        )
        # Strict identity/relationship fields are written only by the trusted
        # direct self-report path. Hiding them from the model keeps the prompt
        # honest and the token budget small; local validation still rejects any
        # crafted output that names one.
        allowed_field_keys = sorted(
            key
            for key in ENTRY_FIELD_KEYS
            if (spec := profile_field_spec(key)) is not None
            and spec.model_writable
            and not spec.strict_identity
        )
        payload = {
            "person_id": person_id,
            "profile_revision": existing.revision,
            "existing_profile": snapshot["existing_profile"],
            "entry_ids": snapshot["entry_ids"],
            "fact_index": fact_index,
            "allowed_field_keys": allowed_field_keys,
            "legacy_unconverted": snapshot["legacy_unconverted"],
            "compact_summary": snapshot["compact_summary"],
            "snapshot_limits": snapshot["snapshot_limits"],
            "new_evidence": {
                "source_event_id": source_event_id,
                "current_turn": evidence.current_turn,
                "recent_context": evidence.recent_context,
                "eligible_user_evidence": evidence.eligible_user_evidence,
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

    @staticmethod
    def _assign_fact_ids(
        facts: list[ExtractedPersonFact],
    ) -> list[ExtractedPersonFact]:
        """Give every extracted fact a stable local id.

        The model may omit ``fact_id``; local ids are positional and never
        derived from content. A duplicate id is a schema error because it would
        let one fact authorize two different changes.
        """
        assigned: list[ExtractedPersonFact] = []
        seen: set[str] = set()
        for index, fact in enumerate(facts):
            fact_id = str(fact.fact_id or "").strip() or f"f{index + 1}"
            if fact_id in seen:
                raise ProfileRefinementValidationError(
                    "duplicate_fact_id_in_model_output"
                )
            seen.add(fact_id)
            assigned.append(fact.model_copy(update={"fact_id": fact_id}))
        return assigned

    def validate_decision(
        self,
        existing: FullProfileSnapshot,
        evidence: ProfileEvidenceBundle,
        decision: ProfileReviewDecision,
    ) -> ProfileReviewResult:
        """Validate every proposal locally.

        The result has exactly three terminal outcomes:

        - ``NO_CHANGE``: nothing legal to write (including an exact duplicate);
        - ``ACCEPTED``: one or more changes may be written atomically;
        - ``REJECTED``: proposals exist but none is legal.

        A proposal is never parked in a candidate or pending queue.
        """
        result = ProfileReviewResult(
            person_id=existing.person_id,
            source_event_id=evidence.source_event_id,
            base_revision=existing.revision,
            decision=decision.decision,
            reason=decision.reason,
        )
        if decision.decision == "NO_CHANGE":
            result.status = "NO_CHANGE"
            # A fact can never disappear because the model chose NO_CHANGE.
            for fact in self._assign_fact_ids(decision.facts):
                result.rejected.append(
                    {
                        "operation": "OBSERVE",
                        "field_key": "",
                        "fact_id": fact.fact_id,
                        "reason": _unanswered_fact_reason(fact),
                        "status": "REJECTED",
                    }
                )
            return result

        entries_by_id = _entries_by_id(existing)
        user_texts = evidence.user_event_texts()
        validated: list[ValidatedChange] = []

        facts = self._assign_fact_ids(decision.facts)
        facts_by_id = {fact.fact_id: fact for fact in facts}

        for index, raw_proposal in enumerate(decision.changes):
            proposal, issue = self._normalise_relation(
                raw_proposal, facts_by_id
            )
            if issue is None:
                issue = self._check_proposal(
                    proposal,
                    existing,
                    entries_by_id,
                    user_texts,
                    facts_by_id=facts_by_id,
                )
            if issue is None and index >= self.registry.max_changes:
                issue = "change_limit_exceeded"
            if issue is not None:
                result.rejected.append(
                    {
                        "operation": raw_proposal.operation,
                        "field_key": raw_proposal.field_key,
                        "fact_id": raw_proposal.fact_id,
                        "reason": issue,
                        "status": "REJECTED",
                    }
                )
                continue
            validated.append(ValidatedChange(proposal=proposal, accepted=True))

        # Every extracted fact must end up with a result. A relation the model
        # itself could not decide is never a write, and a fact the model left
        # without any change is recorded too: nothing is silently dropped.
        answered = {
            change.proposal.fact_id
            for change in validated
            if change.proposal.fact_id
        } | {
            item["fact_id"] for item in result.rejected if item.get("fact_id")
        }
        for fact in facts:
            if fact.fact_id in answered:
                continue
            result.rejected.append(
                {
                    "operation": "OBSERVE",
                    "field_key": "",
                    "fact_id": fact.fact_id,
                    "reason": _unanswered_fact_reason(fact),
                    "status": "REJECTED",
                }
            )


        result.changes = validated
        if validated:
            result.status = "ACCEPTED"
        elif result.rejected and all(
            item["reason"] == "already_present" for item in result.rejected
        ):
            # Identical information is not a failure and must not bump the
            # revision; it is simply nothing new.
            result.status = "NO_CHANGE"
        else:
            result.status = "REJECTED"
        return result

    @staticmethod
    def _normalise_relation(
        proposal: ProfileChangeProposal,
        facts_by_id: dict[str, ExtractedPersonFact],
    ) -> tuple[ProfileChangeProposal, Optional[str]]:
        """Turn an optional semantic relation into one mechanical operation.

        The relation is the semantic decision and therefore wins over the
        model's ``operation`` field. Legacy output without ``relation`` is left
        untouched: old providers and tests keep working, and every new write
        still passes the same local evidence and target checks.
        """
        if proposal.relation is None and proposal.fact_id is None:
            return proposal, None

        fact: Optional[ExtractedPersonFact] = None
        if proposal.fact_id:
            fact = facts_by_id.get(str(proposal.fact_id))
            if fact is None:
                return proposal, "unknown_fact_id"
        if proposal.relation is None:
            return proposal, "missing_relation_decision"
        relation = str(proposal.relation)
        if relation not in RELATION_OPERATIONS:
            return proposal, "unknown_relation"
        if relation == FactRelation.UNCERTAIN:
            return proposal, "relation_uncertain_requires_more_evidence"

        update: dict[str, Any] = {
            "operation": RELATION_OPERATIONS[relation],
        }
        if fact is not None:
            # Attach the validated fact locally so the write transaction can
            # persist its semantic identity without trusting the model again.
            update["fact"] = fact
            # The extracted fact is itself the evidence carrier; a well-formed
            # change may omit duplicates. Never let the change cite *different*
            # evidence than its fact.
            if not proposal.evidence_event_ids and not proposal.evidence_quotes:
                update["evidence_event_ids"] = list(fact.evidence_event_ids)
                update["evidence_quotes"] = list(fact.evidence_quotes)
            elif not proposal.evidence_event_ids:
                update["evidence_event_ids"] = list(fact.evidence_event_ids)
            if not proposal.proposed_content and relation != FactRelation.SAME:
                update["proposed_content"] = fact.value

        if relation == FactRelation.SAME:
            if not proposal.target_entry_ids:
                return proposal, "same_relation_requires_target"
        elif relation == FactRelation.DISTINCT:
            if proposal.target_entry_ids:
                return proposal, "distinct_must_not_target_entries"
        else:
            if not proposal.target_entry_ids:
                return proposal, "relation_requires_target_entry"
            if len(dict.fromkeys(proposal.target_entry_ids)) != 1:
                return proposal, "relation_requires_one_target"
        return proposal.model_copy(update=update), None

    def _check_fact_evidence(
        self,
        fact: ExtractedPersonFact,
        user_texts: dict[str, str],
    ) -> Optional[str]:
        """Validate the evidence carried by one extracted fact."""
        event_ids = list(dict.fromkeys(fact.evidence_event_ids))
        if event_ids and not set(event_ids) <= set(user_texts):
            return "fact_evidence_event_not_authorized"
        if not event_ids and not fact.evidence_quotes:
            return "fact_no_evidence"
        cited = [
            _normalise_text(user_texts[event_id])
            for event_id in event_ids
            if event_id in user_texts
        ]
        for quote in fact.evidence_quotes:
            normalised = _normalise_text(quote)
            if not any(normalised and normalised in text for text in cited):
                return "fact_evidence_quote_not_found"
        corpus = " ".join(cited) if cited else " ".join(
            _normalise_text(text) for text in user_texts.values()
        )
        if _HYPOTHETICAL.search(f"{fact.value} {corpus}"):
            return "hypothetical_or_reported_evidence"
        if not content_supported_by_text(fact.value, cited or list(user_texts.values())):
            return "fact_content_not_supported_by_evidence"
        return None

    def _check_proposal(
        self,
        proposal: ProfileChangeProposal,
        existing: FullProfileSnapshot,
        entries_by_id: dict[str, SnapshotEntry],
        user_texts: dict[str, str],
        *,
        facts_by_id: Optional[dict[str, ExtractedPersonFact]] = None,
    ) -> Optional[str]:
        """Return a stable rejection reason, or ``None`` when the change is legal.

        Target checks use the *stored* entry rows, never the model's label for
        them. Every rejection is final; there is no confirmation branch.
        """
        spec = self.registry.spec(proposal.field_key)
        if spec is None:
            return "unknown_field_key"
        if not self.registry.is_entry(proposal.field_key):
            return "field_not_an_understanding_entry"

        fact: Optional[ExtractedPersonFact] = None
        if proposal.fact_id:
            fact = (facts_by_id or {}).get(str(proposal.fact_id))
            if fact is None:
                return "unknown_fact_id"
            issue = self._check_fact_evidence(fact, user_texts)
            if issue is not None:
                return issue
            classification = classify_fact(fact, proposal.field_key)
            if classification != FactClass.PROFILE:
                return f"fact_classified_as_{classification.lower()}"
            if (
                fact.classification is not None
                and fact.classification != FactClass.PROFILE
            ):
                return f"fact_classified_as_{fact.classification.lower()}"

        # Shared experiences and one-off events belong to Scene/Memory. They
        # used to be written here verbatim and were the main source of Profile
        # growth without long-term value.
        if proposal.field_key == "interaction.shared_context":
            return "shared_context_belongs_to_scene_memory"

        if (
            not proposal.proposed_content
            and proposal.operation not in ("ARCHIVE", "SAME")
        ):
            return "missing_proposed_content"

        evidence_only = proposal.operation == "SAME"
        if proposal.operation == "SAME":
            if not proposal.target_entry_ids:
                return "missing_target_entry_ids"
            unique_ids = list(dict.fromkeys(proposal.target_entry_ids))
            if any(entry_id not in entries_by_id for entry_id in unique_ids):
                return "target_entry_not_in_this_profile"
            target_entries = [
                entries_by_id[entry_id].model_dump() for entry_id in unique_ids
            ]
            try:
                validate_change_targets(
                    proposal.operation, proposal.field_key, target_entries
                )
            except ValueError as exc:
                return str(exc) or "invalid_change_target"
        elif proposal.operation == "ADD":
            if proposal.target_entry_ids:
                return "add_must_not_target_entries"
            if not self._add_is_new(existing, proposal):
                return "already_present"
            target_entries: list[dict] = []
        else:
            if not proposal.target_entry_ids:
                return "missing_target_entry_ids"
            unique_ids = list(dict.fromkeys(proposal.target_entry_ids))
            if any(entry_id not in entries_by_id for entry_id in unique_ids):
                return "target_entry_not_in_this_profile"
            target_entries = [
                entries_by_id[entry_id].model_dump() for entry_id in unique_ids
            ]
            try:
                validate_change_targets(
                    proposal.operation, proposal.field_key, target_entries
                )
            except ValueError as exc:
                return str(exc) or "invalid_change_target"

        if proposal.operation in ("MERGE", "REFINE"):
            targets = [item["content"] for item in target_entries]
            content = proposal.proposed_content or ""
            # Only merging/condensing is checked for lost specifics. A genuine
            # CORRECT may replace a fact (university -> high school); an UPDATE
            # may change a project's state. Those are judged on evidence.
            if _coverage(targets, content) < 0.45:
                return "lossy_change_rejects_existing_facts"
            if content and _FILLER.match(content.strip()):
                return "lossy_change_rejects_existing_facts"

        # Strict identity and relationship values are written only through the
        # trusted direct self-report path, never from model output.
        if self.registry.is_strict(proposal.field_key):
            return "strict_identity_requires_explicit_self_report"
        if not self.registry.is_model_writable(proposal.field_key):
            return "field_not_model_writable"
        if not evidence_only and proposal.operation != "ADD" and any(
            is_human_confirmed_source(item.get("source")) for item in target_entries
        ):
            # A fact the person explicitly corrected or confirmed outranks a
            # model rewrite. Only a new explicit correction may replace it.
            return "human_confirmed_target_requires_review"
        if not evidence_only and _SENSITIVE.search(
            proposal.proposed_content or ""
        ):
            return "sensitive_attribute_not_auto_saved"

        allowed_event_ids = set(user_texts)
        if proposal.evidence_event_ids and not set(
            proposal.evidence_event_ids
        ) <= allowed_event_ids:
            return "evidence_event_not_authorized"
        cited_texts: list[str] = []
        if proposal.evidence_quotes:
            cited = list(dict.fromkeys(proposal.evidence_event_ids))
            if not cited:
                return "evidence_quote_without_event"
            cited_texts = [
                _normalise_text(user_texts[event_id])
                for event_id in cited
                if event_id in user_texts
            ]
            for quote in proposal.evidence_quotes:
                normalised = _normalise_text(quote)
                if not any(
                    normalised and normalised in text for text in cited_texts
                ):
                    return "evidence_quote_not_found"
        else:
            cited_texts = [
                _normalise_text(user_texts[event_id])
                for event_id in dict.fromkeys(proposal.evidence_event_ids)
                if event_id in user_texts
            ]
        if not proposal.evidence_event_ids and not proposal.evidence_quotes:
            # No source event means no legal write. A model statement alone is
            # never evidence.
            return "no_evidence"

        quoted = " ".join(proposal.evidence_quotes)
        evidence_corpus = " ".join(cited_texts)
        if _HYPOTHETICAL.search(
            f"{proposal.proposed_content or ''} {quoted} {evidence_corpus}"
        ):
            # Jokes, hypotheticals and third-party reports are not stable
            # self-report and are never auto-written.
            return "hypothetical_or_reported_evidence"
        if requires_direct_quote(proposal.field_key):
            observed_pattern = bool(
                fact is not None
                and fact.assertion_type == AssertionType.OBSERVED
                and fact.temporal_scope == TemporalScope.STABLE
                and len(set(fact.evidence_event_ids)) >= 2
                and classify_fact(fact, proposal.field_key)
                == FactClass.PROFILE
            )
            if not observed_pattern:
                if not proposal.evidence_quotes:
                    return "trait_inference_requires_self_description"
                if not any(
                    _TRAIT_SELF_DESCRIPTION.search(_normalise_text(quote))
                    for quote in proposal.evidence_quotes
                ):
                    # A single observed act is not a stable personality trait
                    # or habit, even when the act itself is quoted.
                    return "trait_inferred_from_single_act"
        if (
            not evidence_only
            and proposal.operation in ("ADD", "UPDATE", "CORRECT")
            and not (
                content_supported_by_text(
                    proposal.proposed_content or "",
                    cited_texts or list(user_texts.values()),
                )
            )
        ):
            return "content_not_supported_by_evidence"
        if (
            not evidence_only
            and proposal.operation == "ADD"
            and spec.single_value
            and existing.entries.get(proposal.field_key)
        ):
            return "unresolved_conflict_with_existing_value"
        return None

    @staticmethod
    def _add_is_new(
        existing: FullProfileSnapshot, proposal: ProfileChangeProposal
    ) -> bool:
        content = proposal.proposed_content or ""
        if not _normalise_text(content):
            return False
        category = proposal.field_key.split(".", 1)[0]
        for field_key, entries in existing.entries.items():
            # Cross-field de-duplication is limited to the same category
            # ("interests.topics" vs "interests.likes"). Merging across
            # categories would collide an interest with a skill ("摄影" vs
            # "喜欢摄影"), which the spec explicitly forbids. Cross-category
            # semantic duplicates are caught by the model's SAME relation.
            if field_key != proposal.field_key and field_key.split(
                ".", 1
            )[0] != category:
                continue
            for entry in entries:
                if entries_equivalent(entry.content, content):
                    return False
        return True

    # -- apply -------------------------------------------------------------

    async def apply_proposals(
        self,
        review: ProfileReviewResult,
        execution: Optional[ProfileReviewExecution] = None,
    ) -> ProfileApplyResult:
        return await self.profiles.apply_refinement(
            review.person_id,
            review,
            source_event_id=review.source_event_id,
            execution=execution,
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

        # A background review may run minutes later. Only messages persisted
        # *before* this event, in this conversation, are context — a later turn
        # can never leak backwards into an earlier review. ``exclude_event_id``
        # alone would not provide that boundary.
        history = await self.repos.utterance.context_before_event(
            event.conversation_id,
            event.event_id,
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
        eligible_user_evidence: list[dict] = [
            {
                "event_id": event.event_id,
                "speaker_person_id": person_id,
                "text": event.text,
                "role": "user",
            }
        ]
        for item in history:
            recent.append({"role": "human", "text": item.text})
            eligible_user_evidence.append(
                {
                    "event_id": item.event_id,
                    "speaker_person_id": person_id,
                    "text": item.text,
                    "role": "user",
                }
            )
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
            eligible_user_evidence=eligible_user_evidence,
            allowed_event_ids=[
                item["event_id"] for item in eligible_user_evidence
            ],
            allowed_memory_ids=[],
        )


__all__ = [
    "ALLOWED_FIELD_TRANSITIONS",
    "DirectSelfReportResult",
    "FullProfileSnapshot",
    "INFERENCE_ONLY_FIELDS",
    "MAX_SNAPSHOT_LEGACY_FIELDS",
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
    "ProfileReviewExecution",
    "ProfileReviewResult",
    "SnapshotEntry",
    "ValidatedChange",
    "content_supported_by_text",
    "entries_equivalent",
    "is_human_confirmed_source",
    "is_strict_field",
    "requires_direct_quote",
    "validate_change_targets",
]
