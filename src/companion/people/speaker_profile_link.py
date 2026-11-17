"""Explicit Speaker -> Person -> Profile linking (V0.9.9.31.1 §7).

A voice print only says *which voice cluster* the audio matched. A person
binding (``person_speaker_bindings``) is the only bridge from a speaker id to a
person, and it exists only because a human confirmed it locally. This module
sits on top of that bridge and adds the final hop to a person's profile:

    Speaker ID  ->  Person ID  ->  PersonInteractionProfile

It deliberately adds **no new binding table**. The speaker -> person hop keeps
using :class:`SpeakerBindingService`; the person -> profile hop reads the
existing profile repository. Nothing here copies profile content into the
speaker store, writes a Speaker ID into ``PersonInteractionProfile.person_id``,
infers a link from similarity, or lets a model choose a person.

Consent stays independent: linking a voice does not grant voiceprint storage,
does not turn on memory learning, and does not make profile content readable.
:meth:`SpeakerProfileLinkService.resolve_profile` reports the person as soon as
the binding resolves, even when that person has no profile yet — a missing
profile is not an unknown person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from companion.domain.enums import RegistrationStatus
from companion.people.speaker_bindings import (
    SpeakerBindingError,
    SpeakerBindingService,
)
from companion.storage.repositories import AuditRepository, PersonRepository

# Stable outcome codes (logged and asserted in tests; no free-form text).
LINK_OK = "ok"
LINK_NO_PERSON = "no_person_selected"
LINK_LOOKUP_FAILED = "link_lookup_failed"
LINK_UNKNOWN_SPEAKER = "unknown_speaker"
LINK_ALIAS_UNRESOLVED = "alias_unresolved"
LINK_BINDING_CONFLICT = "binding_conflict"
LINK_PERSON_MISSING = "person_missing"
LINK_PERSON_INACTIVE = "person_inactive"

PROFILE_READY = "profile_ready"
PROFILE_MISSING = "profile_missing"
PROFILE_NO_BINDING = "no_binding"
PROFILE_UNKNOWN_SPEAKER = "unknown_speaker"
PROFILE_ALIAS_UNRESOLVED = "alias_unresolved"
PROFILE_BINDING_CONFLICT = "binding_conflict"
PROFILE_PERSON_MISSING = "person_missing"
PROFILE_PERSON_INACTIVE = "person_inactive"
PROFILE_CONSENT_MISSING = "consent_missing"

# Consent statuses mirrored from the speaker store / enrollment policy.
CONSENT_GRANTED = "GRANTED"
CONSENT_REVOKED = "REVOKED"
CONSENT_LEGACY_READ_ONLY = "LEGACY_READ_ONLY"


class SpeakerLinkError(Exception):
    """A link request was refused; ``code`` is a stable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class LinkPreview:
    """What a human is about to do, computed without writing anything."""

    requested_speaker_id: str
    canonical_speaker_id: Optional[str]
    speaker_ids: tuple[str, ...]
    alias_complete: bool
    speaker_status: Optional[str]
    speaker_samples: int
    speaker_consent: Optional[str]
    person_id: Optional[str]
    display_name: Optional[str]
    person_active: bool
    binding: Optional[dict]
    conflicting_person_ids: tuple[str, ...] = ()
    profile_present: bool = False
    profile_revision: Optional[int] = None
    consent_flags: dict[str, bool] = field(default_factory=dict)
    can_link: bool = False
    reason: str = LINK_OK
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkResult:
    """The committed binding plus the preview that authorised it."""

    binding: dict
    preview: LinkPreview
    created: bool


@dataclass(frozen=True)
class ProfileResolution:
    """Speaker -> person -> profile, with explicit incompleteness states.

    ``person_id`` being set means the person is known even when ``profile`` is
    ``None``. Only an unresolved speaker/binding leaves ``person_id`` empty.
    """

    speaker_id: Optional[str]
    canonical_speaker_id: Optional[str]
    person_id: Optional[str]
    display_name: Optional[str]
    state: str
    reason: str
    profile: Any = None
    consent_flags: dict[str, bool] = field(default_factory=dict)


class SpeakerProfileLinkService:
    """Human-confirmed links from a voice cluster to a person's profile."""

    def __init__(
        self,
        bindings: SpeakerBindingService,
        persons: PersonRepository,
        speaker_store: Any,
        profile_service: Any,
        *,
        audit: Optional[AuditRepository] = None,
    ) -> None:
        self.bindings = bindings
        self.persons = persons
        self.speaker_store = speaker_store
        self.profile_service = profile_service
        self.audit = audit

    # ------------------------------------------------------------- helpers
    def _closure(self, speaker_id: str):
        resolver = getattr(self.speaker_store, "resolve_alias_closure", None)
        if resolver is None:
            # A store that cannot prove the alias set is complete may not be
            # used to choose a person.
            raise SpeakerLinkError(LINK_ALIAS_UNRESOLVED)
        return resolver(str(speaker_id or "").strip())

    def _speaker_row(self, speaker_id: str) -> Optional[dict]:
        getter = getattr(self.speaker_store, "get_profile", None)
        if getter is None:
            return None
        try:
            return getter(speaker_id)
        except Exception:  # noqa: BLE001 - preview only
            return None

    def _samples(self, speaker_id: str) -> int:
        lister = getattr(self.speaker_store, "list_speakers", None)
        if lister is None:
            return 0
        try:
            for row in lister():
                if row.get("speaker_id") == speaker_id:
                    return int(row.get("samples", 0) or 0)
        except Exception:  # noqa: BLE001 - preview only
            return 0
        return 0

    def _speaker_consent(self, speaker_id: str) -> Optional[str]:
        getter = getattr(self.speaker_store, "enrollment_consent", None)
        if getter is None:
            return None
        try:
            row = getter(speaker_id)
        except Exception:  # noqa: BLE001 - preview only
            return None
        if row is None:
            return None
        return str(row.get("status") or "").upper() or None

    @staticmethod
    def _consent_flags(consent: Any) -> dict[str, bool]:
        return {
            "processing": bool(getattr(consent, "allow_processing", False)),
            "persistence": bool(getattr(consent, "allow_persistence", False)),
            "memory_learning": bool(
                getattr(consent, "allow_memory_learning", False)
            ),
            "external_ai": bool(
                getattr(consent, "allow_external_ai", False)
            ),
        }

    async def _profile_row(self, person_id: str) -> Optional[dict]:
        repo = getattr(self.profile_service, "repo", None)
        if repo is None:
            return None
        try:
            return await repo.get(person_id)
        except Exception:  # noqa: BLE001 - preview only
            return None

    async def _person_view(
        self, person_id: Optional[str]
    ) -> tuple[Optional[Any], bool, Optional[Any]]:
        if not person_id:
            return None, False, None
        try:
            person = await self.persons.get(person_id)
        except Exception:  # noqa: BLE001 - preview only
            return None, False, None
        active = (
            person is not None
            and person.registration_status is RegistrationStatus.ACTIVE
        )
        try:
            consent = await self.persons.get_consent(person_id)
        except Exception:  # noqa: BLE001 - preview only
            consent = None
        return person, active, consent

    # ------------------------------------------------------------- preview
    async def preview_link(
        self, speaker_id: str, person_id: Optional[str] = None
    ) -> LinkPreview:
        """Describe a prospective link. Never writes or creates anything."""
        requested = str(speaker_id or "").strip()
        target = str(person_id or "").strip() or None
        closure = self._closure(requested)
        row = (
            self._speaker_row(closure.canonical)
            if closure.canonical
            else None
        )
        warnings: list[str] = []

        complete = bool(closure.complete)
        if not complete:
            reason = (
                LINK_UNKNOWN_SPEAKER
                if closure.reason == "unknown_id"
                else LINK_ALIAS_UNRESOLVED
            )
            warnings.append("alias chain cannot be verified; refusing to link")
            return LinkPreview(
                requested_speaker_id=requested,
                canonical_speaker_id=closure.canonical,
                speaker_ids=tuple(closure.speaker_ids),
                alias_complete=False,
                speaker_status=None,
                speaker_samples=0,
                speaker_consent=None,
                person_id=target,
                display_name=None,
                person_active=False,
                binding=None,
                can_link=False,
                reason=reason,
                warnings=tuple(warnings),
            )

        canonical = str(closure.canonical)
        speaker_ids = list(closure.speaker_ids)
        if canonical not in speaker_ids:
            speaker_ids.insert(0, canonical)
        if row is None:
            return LinkPreview(
                requested_speaker_id=requested,
                canonical_speaker_id=canonical,
                speaker_ids=tuple(speaker_ids),
                alias_complete=True,
                speaker_status=None,
                speaker_samples=0,
                speaker_consent=self._speaker_consent(canonical),
                person_id=target,
                display_name=None,
                person_active=False,
                binding=None,
                can_link=False,
                reason=LINK_UNKNOWN_SPEAKER,
            )

        rows = await self.bindings.bindings_for_speaker_ids(speaker_ids)
        bound_person_ids = tuple(
            sorted(
                {
                    str(item.get("person_id") or "").strip()
                    for item in rows
                    if str(item.get("person_id") or "").strip()
                }
            )
        )
        binding = await self.bindings.get_binding(canonical)
        effective_person = target or (
            bound_person_ids[0] if len(bound_person_ids) == 1 else None
        )

        person, active, consent = await self._person_view(effective_person)
        profile_row = (
            await self._profile_row(effective_person)
            if effective_person
            else None
        )
        if bound_person_ids and target and target not in bound_person_ids:
            warnings.append(
                "this voice is already bound elsewhere; unbind explicitly"
                " first"
            )
        if len(bound_person_ids) > 1:
            warnings.append("aliases are bound to different people")

        speaker_consent = self._speaker_consent(canonical)
        if speaker_consent in (CONSENT_REVOKED, None):
            warnings.append(
                "voiceprint storage is not authorised; recognition remains"
                " read-only"
            )
        if consent is not None:
            flags = self._consent_flags(consent)
            if not flags["processing"] or not flags["persistence"]:
                warnings.append(
                    "person consent does not allow processing and"
                    " persistence; voice turns will stay UNKNOWN"
                )
        else:
            flags = {}
            warnings.append("person consent is missing")

        reason = LINK_OK
        can_link = bool(target)
        if not target:
            can_link = False
            reason = LINK_NO_PERSON
        elif len(bound_person_ids) > 1:
            can_link = False
            reason = LINK_BINDING_CONFLICT
        elif bound_person_ids and target not in bound_person_ids:
            can_link = False
            reason = LINK_BINDING_CONFLICT
        elif person is None:
            can_link = False
            reason = LINK_PERSON_MISSING
        elif not active:
            can_link = False
            reason = LINK_PERSON_INACTIVE

        return LinkPreview(
            requested_speaker_id=requested,
            canonical_speaker_id=canonical,
            speaker_ids=tuple(speaker_ids),
            alias_complete=True,
            speaker_status=str(row.get("status") or "") or None,
            speaker_samples=self._samples(canonical),
            speaker_consent=speaker_consent,
            person_id=effective_person,
            display_name=(
                str(getattr(person, "display_name", "") or "") or None
            ),
            person_active=active,
            binding=dict(binding) if binding is not None else None,
            conflicting_person_ids=bound_person_ids,
            profile_present=profile_row is not None,
            profile_revision=(
                int(profile_row.get("revision") or 1)
                if profile_row is not None
                else None
            ),
            consent_flags=flags,
            can_link=can_link,
            reason=reason,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # ------------------------------------------------------------- confirm
    async def confirm_link(
        self,
        speaker_id: str,
        person_id: str,
        *,
        verified_by: str = "local_confirmation",
        actor: str = "local_cli",
    ) -> LinkResult:
        """Commit a human-confirmed link. Requires an explicit person id.

        Refuses moves between people (``binding_conflict``), unknown/broken
        voice ids, missing/inactive people and alias sets that cannot be
        verified. The delegated binding service re-checks the person, so this
        path can never bind a Speaker Result, a similarity score or a model
        suggestion.
        """
        target = str(person_id or "").strip()
        verifier = str(verified_by or "").strip()
        if not target or not verifier:
            raise SpeakerLinkError(LINK_PERSON_MISSING)
        preview = await self.preview_link(speaker_id, target)
        if not preview.can_link:
            raise SpeakerLinkError(preview.reason or LINK_PERSON_MISSING)
        assert preview.canonical_speaker_id is not None
        existing = preview.binding
        try:
            binding = await self.bindings.bind_speaker(
                target,
                preview.canonical_speaker_id,
                verifier,
                related_speaker_ids=list(preview.speaker_ids),
            )
        except SpeakerBindingError as exc:
            # Surface the binding service's stable code; never translate it
            # into a permissive success.
            raise SpeakerLinkError(exc.code) from exc
        created = existing is None
        if self.audit is not None:
            await self.audit.record(
                actor=actor,
                action="link_speaker_profile",
                entity_type="speaker",
                entity_id=preview.canonical_speaker_id,
                detail=(
                    f"person_id={target} verified_by={verifier}"
                    f" created={str(created).lower()}"
                ),
            )
        return LinkResult(binding=binding, preview=preview, created=created)

    # ------------------------------------------------------------- resolve
    async def resolve_profile(self, speaker_id: str) -> ProfileResolution:
        """Speaker id -> person id -> profile, without guessing.

        A known, active person with no profile (or without profile consent) is
        reported as known: ``person_id`` stays set and ``state`` says why the
        profile itself is unavailable. Only an unresolved speaker/binding
        leaves ``person_id`` empty.
        """
        requested = str(speaker_id or "").strip()
        try:
            closure = self._closure(requested)
        except SpeakerLinkError as exc:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=None,
                person_id=None,
                display_name=None,
                state=PROFILE_ALIAS_UNRESOLVED,
                reason=exc.code,
            )
        if closure.canonical is None:
            state = (
                PROFILE_UNKNOWN_SPEAKER
                if closure.reason == "unknown_id"
                else PROFILE_ALIAS_UNRESOLVED
            )
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=None,
                person_id=None,
                display_name=None,
                state=state,
                reason=closure.reason or PROFILE_UNKNOWN_SPEAKER,
            )
        if not closure.complete:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=closure.canonical,
                person_id=None,
                display_name=None,
                state=PROFILE_ALIAS_UNRESOLVED,
                reason=closure.reason or PROFILE_ALIAS_UNRESOLVED,
            )

        canonical = str(closure.canonical)
        speaker_ids = list(closure.speaker_ids)
        if canonical not in speaker_ids:
            speaker_ids.insert(0, canonical)
        rows = await self.bindings.bindings_for_speaker_ids(speaker_ids)
        person_ids = {
            str(item.get("person_id") or "").strip()
            for item in rows
            if str(item.get("person_id") or "").strip()
        }
        if not person_ids:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=None,
                display_name=None,
                state=PROFILE_NO_BINDING,
                reason=PROFILE_NO_BINDING,
            )
        if len(person_ids) > 1:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=None,
                display_name=None,
                state=PROFILE_BINDING_CONFLICT,
                reason=PROFILE_BINDING_CONFLICT,
            )
        person_id = next(iter(person_ids))
        person, active, consent = await self._person_view(person_id)
        flags = self._consent_flags(consent) if consent is not None else {}
        name = str(getattr(person, "display_name", "") or "") or None
        if person is None:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=person_id,
                display_name=None,
                state=PROFILE_PERSON_MISSING,
                reason=PROFILE_PERSON_MISSING,
                consent_flags=flags,
            )
        if not active:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=person_id,
                display_name=name,
                state=PROFILE_PERSON_INACTIVE,
                reason=PROFILE_PERSON_INACTIVE,
                consent_flags=flags,
            )
        if (
            consent is None
            or not flags.get("processing")
            or not flags.get("persistence")
        ):
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=person_id,
                display_name=name,
                state=PROFILE_CONSENT_MISSING,
                reason=PROFILE_CONSENT_MISSING,
                consent_flags=flags,
            )
        profile = None
        try:
            profile = await self.profile_service.get(person_id)
        except Exception:  # noqa: BLE001 - profile failures stay non-fatal
            profile = None
        if profile is None:
            return ProfileResolution(
                speaker_id=requested or None,
                canonical_speaker_id=canonical,
                person_id=person_id,
                display_name=name,
                state=PROFILE_MISSING,
                reason=PROFILE_MISSING,
                consent_flags=flags,
            )
        return ProfileResolution(
            speaker_id=requested or None,
            canonical_speaker_id=canonical,
            person_id=person_id,
            display_name=name,
            state=PROFILE_READY,
            reason=PROFILE_READY,
            profile=profile,
            consent_flags=flags,
        )

    # -------------------------------------------------------------- unlink
    async def unlink(
        self, speaker_id: str, *, actor: str = "local_cli"
    ) -> bool:
        """Remove the human binding for this voice and its aliases.

        Only the binding is removed. Voice samples, speaker profiles and the
        person's profile are untouched; consent is unchanged.
        """
        closure = self._closure(speaker_id)
        if closure.canonical is None or not closure.complete:
            raise SpeakerLinkError(
                LINK_UNKNOWN_SPEAKER
                if closure.canonical is None
                and closure.reason == "unknown_id"
                else LINK_ALIAS_UNRESOLVED
            )
        canonical = str(closure.canonical)
        related = list(closure.speaker_ids)
        if canonical not in related:
            related.insert(0, canonical)
        removed = await self.bindings.unbind_speaker(
            canonical, related_speaker_ids=related
        )
        if removed and self.audit is not None:
            await self.audit.record(
                actor=actor,
                action="unlink_speaker_profile",
                entity_type="speaker",
                entity_id=canonical,
                detail="local_confirmation",
            )
        return bool(removed)


__all__ = [
    "LINK_ALIAS_UNRESOLVED",
    "LINK_BINDING_CONFLICT",
    "LINK_OK",
    "LINK_PERSON_INACTIVE",
    "LINK_PERSON_MISSING",
    "LINK_UNKNOWN_SPEAKER",
    "LinkPreview",
    "LinkResult",
    "PROFILE_ALIAS_UNRESOLVED",
    "PROFILE_BINDING_CONFLICT",
    "PROFILE_CONSENT_MISSING",
    "PROFILE_MISSING",
    "PROFILE_NO_BINDING",
    "PROFILE_PERSON_INACTIVE",
    "PROFILE_PERSON_MISSING",
    "PROFILE_READY",
    "PROFILE_UNKNOWN_SPEAKER",
    "ProfileResolution",
    "SpeakerLinkError",
    "SpeakerProfileLinkService",
]
