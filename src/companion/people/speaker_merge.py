"""Explicit, local-only review before joining two voice clusters.

A similarity score is a diagnostic, not identity proof. This service never
automatically merges clusters, binds a person, or modifies a person's profile.
The actual merge must be confirmed by the human at the REPL, with mic off.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

from companion.people.speaker_bindings import SpeakerBindingService
from companion.speaker_id.config import load_speaker_config
from companion.speaker_id.voice_profiles import cosine


@dataclass(frozen=True)
class MergePreview:
    canonical: str
    candidate: str
    canonical_samples: int
    candidate_samples: int
    median_best_similarity: Optional[float]
    can_merge: bool
    reason: str
    bound_person_id: Optional[str] = None


class SpeakerMergeError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SpeakerMergeService:
    """Compare and human-confirm a TEMP candidate into a stable S identity."""

    def __init__(
        self, store: Any, bindings: SpeakerBindingService, *, audit: Any = None
    ) -> None:
        self.store = store
        self.bindings = bindings
        self.audit = audit

    async def preview(self, canonical: str, candidate: str) -> MergePreview:
        target = str(canonical or "").strip()
        source = str(candidate or "").strip()

        def refused(reason: str) -> MergePreview:
            return MergePreview(target, source, 0, 0, None, False, reason)

        if not target or not source or target == source:
            return refused("distinct_ids_required")
        try:
            left = self.store.resolve_alias_closure(target)
            right = self.store.resolve_alias_closure(source)
        except Exception:
            return refused("alias_lookup_failed")
        if (
            not left.complete or not right.complete
            or left.canonical != target or right.canonical != source
        ):
            return refused("unresolved_or_noncanonical_id")
        try:
            stable = self.store.get_profile(target)
            temporary = self.store.get_profile(source)
        except Exception:
            return refused("profile_lookup_failed")
        if not stable or not temporary:
            return refused("speaker_not_found")
        if stable["status"] != "KNOWN" or temporary["status"] != "CANDIDATE":
            return refused("merge_requires_known_target_and_temp_source")
        if (
            stable["model_id"] != temporary["model_id"]
            or stable["embedding_version"] != temporary["embedding_version"]
            or not stable["model_id"]
        ):
            return refused("incompatible_embeddings")
        try:
            rows = await self.bindings.bindings_for_speaker_ids(
                list(dict.fromkeys([
                    *left.speaker_ids, *right.speaker_ids
                ]))
            )
        except Exception:
            return refused("binding_lookup_failed")
        persons = {row["person_id"] for row in rows}
        if len(persons) > 1:
            return refused("binding_conflict")
        owner = next(iter(persons)) if persons else None

        try:
            speakers, _ = self.store.load_candidates(
                model_id=stable["model_id"],
                embedding_version=stable["embedding_version"],
            )
            profiles = {entry.speaker_id: entry for entry in speakers}
            a = profiles.get(target)
            b = profiles.get(source)
            if a is None or b is None or not a.vectors or not b.vectors:
                return refused("no_compatible_samples")
            # This score has no interpretation as an authentication
            # probability. Use symmetric nearest-neighbour coverage so one
            # coincidentally similar sample cannot mask dissimilar clusters.
            best_a = [
                max(cosine(vec, other) for other in b.vectors)
                for vec in a.vectors
            ]
            best_b = [
                max(cosine(vec, other) for other in a.vectors)
                for vec in b.vectors
            ]
            score = float(min(median(best_a), median(best_b)))
        except Exception:
            return refused("sample_comparison_failed")
        threshold = load_speaker_config(env={}).candidate_match_threshold
        return MergePreview(
            target, source, len(a.vectors), len(b.vectors), score,
            score >= threshold,
            "ready_for_human_confirmation"
            if score >= threshold else "clusters_not_similar_enough",
            owner,
        )

    async def confirm(self, canonical: str, candidate: str) -> MergePreview:
        """Re-check all guards immediately before the irreversible merge."""
        preview = await self.preview(canonical, candidate)
        if not preview.can_merge:
            raise SpeakerMergeError(preview.reason)
        if not self.store.merge_speakers(preview.canonical, preview.candidate):
            raise SpeakerMergeError("voice_store_merge_failed")
        closure = self.store.resolve_alias_closure(preview.canonical)
        if closure.canonical is None or not closure.complete:
            raise SpeakerMergeError("merged_alias_incomplete")
        # SQLite stores are separate. Should main-DB cleanup fail, the alias
        # remains resolvable, and the identity resolver refuses conflicts.
        await self.bindings.canonicalize_alias_bindings(
            preview.canonical, list(closure.speaker_ids)
        )
        if self.audit is not None:
            await self.audit.record(
                actor="local_cli",
                action="merge_speaker_profiles",
                entity_type="speaker",
                entity_id=preview.canonical,
                detail=(
                    f"merged={preview.candidate}"
                    " verified_by=local_cli_confirmation"
                ),
            )
        return preview


__all__ = [
    "MergePreview",
    "SpeakerMergeError",
    "SpeakerMergeService",
]
