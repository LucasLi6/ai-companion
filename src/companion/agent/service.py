"""Trusted Agent Core service.

Only the trusted application calls this: a model can *propose* a persona, but
nothing here accepts a model proposal as authoritative. Every change is
validated, recorded in ``persona_revisions`` or ``relationship_history`` and
audited. Identity and personality never grant permissions: consent and policy
remain separate, and a relationship tier is descriptive, never authorisation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional

from companion.agent.constants import (
    DEFAULT_FAMILIARITY,
    DEFAULT_PRESET_DELTAS,
    PERSONA_TRAITS,
    TRAIT_MAX,
    TRAIT_MIN,
    PersonalityValidationError,
    RelationshipValidationError,
    clamp,
    validate_tier,
    validate_trait,
)
from companion.agent.models import EffectiveAgentContext
from companion.agent.persona_core import PersonaCore, PersonaCoreRepository
from companion.agent.repository import AgentRepository
from companion.agent.stance import (
    AgentStance,
    AgentStanceRepository,
    extract_stance_candidates,
)
from companion.core.policy import PolicyConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from companion.people.registry import PersonRegistry
    from companion.storage.repositories import AuditRepository, Repositories


class AgentService:
    def __init__(
        self,
        *,
        repos: "Repositories",
        registry: "PersonRegistry | None" = None,
        audit: "AuditRepository | None" = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.audit = audit
        self.logger = logger or logging.getLogger("companion.agent")
        self.repo = AgentRepository(repos.db)
        self.persona_core_repo = PersonaCoreRepository(repos.db)
        self.stance_repo = AgentStanceRepository(repos.db)

    async def ensure_defaults(self) -> None:
        await self.repo.ensure_defaults()

    async def _audit(self, actor: str, action: str, entity_type: str, entity_id: str, detail: str) -> None:
        if self.audit is not None:
            await self.audit.record(
                actor=actor,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                detail=detail,
            )

    # -- read --------------------------------------------------------------

    async def show(self) -> dict[str, Any]:
        await self.repo.ensure_defaults()
        manifest = await self.repo.get_manifest() or {}
        identity = await self.repo.get_identity() or {}
        persona = await self.repo.get_persona() or {}
        return {
            "agent_id": manifest.get("agent_id"),
            "format_version": manifest.get("format_version"),
            "lineage_id": manifest.get("lineage_id"),
            "lineage_parent_agent_id": manifest.get("lineage_parent_agent_id"),
            "display_name": identity.get("display_name"),
            "pronouns": identity.get("pronouns"),
            "description": identity.get("description"),
            "identity_revision": identity.get("revision"),
            "persona": {name: persona.get(name) for name in PERSONA_TRAITS},
            "persona_revision": persona.get("revision"),
        }

    async def revision_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self.repo.list_persona_revisions(limit)

    async def relationship(self, person_id: str) -> dict[str, Any]:
        await self._require_person(person_id)
        profile = await self.repo.get_profile(person_id)
        history = await self.repo.list_relationship_history(person_id)
        return {
            "person_id": person_id,
            "familiarity": profile["familiarity"],
            "tier": profile.get("relationship_tier") or profile["familiarity"],
            "style_overrides": profile["style_overrides"],
            "history": history,
        }

    # -- identity ----------------------------------------------------------

    async def set_name(
        self,
        display_name: str,
        *,
        actor: str = "admin",
        reason: str | None = None,
        pronouns: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        name = " ".join(str(display_name).split()).strip()
        if not name or len(name) > 80:
            raise PersonalityValidationError(
                "display name must be 1..80 characters"
            )
        if pronouns is not None and len(pronouns) > 40:
            raise PersonalityValidationError("pronouns must be at most 40 characters")
        if description is not None and len(description) > 1000:
            raise PersonalityValidationError("description must be at most 1000 characters")
        await self.repo.ensure_defaults()
        result = await self.repo.set_identity(
            display_name=name,
            pronouns=pronouns,
            description=description,
            changed_by=actor,
            reason=reason,
        )
        await self._audit(actor, "set_agent_name", "agent", result.get("display_name", ""), f"revision={result['revision']}")
        self.logger.info("agent_name_set name=%s by=%s", name, actor)
        return result

    # -- persona -----------------------------------------------------------

    async def set_persona(
        self,
        updates: Mapping[str, object],
        *,
        actor: str = "admin",
        reason: str | None = None,
    ) -> dict[str, Any]:
        parsed: dict[str, int] = {}
        for name, value in updates.items():
            parsed[name] = validate_trait(str(name), value)
        if not parsed:
            raise PersonalityValidationError("no personality fields supplied")
        await self.repo.ensure_defaults()
        result = await self.repo.update_persona(
            parsed, changed_by=actor, reason=reason
        )
        await self._audit(
            actor,
            "set_agent_persona",
            "agent",
            "persona",
            ",".join(f"{k}={v}" for k, v in sorted(parsed.items())),
        )
        self.logger.info(
            "agent_persona_set fields=%s by=%s", ",".join(sorted(parsed)), actor
        )
        return result

    # -- relationships -----------------------------------------------------

    async def set_relationship(
        self,
        person_id: str,
        tier: str,
        *,
        actor: str = "admin",
        reason: str | None = None,
    ) -> dict[str, Any]:
        await self._require_person(person_id)
        parsed = validate_tier(tier)
        old, new = await self.repo.set_familiarity(
            person_id, parsed, actor=actor, reason=reason
        )
        await self._audit(
            actor,
            "set_relationship",
            "person",
            person_id,
            f"{old}->{new}",
        )
        # A relationship tier is not consent and is not authorisation. Log that
        # explicitly so an operator can never mistake one for the other.
        self.logger.info(
            "relationship_set person_id=%s old=%s new=%s by=%s",
            person_id,
            old,
            new,
            actor,
        )
        return {"person_id": person_id, "old_tier": old, "new_tier": new}

    async def set_override(
        self,
        person_id: str,
        field: str,
        value: int | None,
        *,
        actor: str = "admin",
        reason: str | None = None,
    ) -> dict[str, int]:
        await self._require_person(person_id)
        updates: dict[str, int | None] = {field: value}
        if value is not None:
            validate_trait(field, value)
        elif field not in PERSONA_TRAITS:
            raise PersonalityValidationError(
                f"unknown personality trait {field!r}"
            )
        result = await self.repo.set_overrides(
            person_id, updates, actor=actor, reason=reason
        )
        await self._audit(
            actor,
            "override_relationship_style",
            "person",
            person_id,
            f"{field}={value}",
        )
        return result

    # -- effective context -------------------------------------------------

    async def effective(
        self,
        person_id: str | None,
        *,
        policy: PolicyConfig | None = None,
        temporary: Mapping[str, int] | None = None,
    ) -> EffectiveAgentContext:
        await self.repo.ensure_defaults()
        identity = await self.repo.get_identity() or {}
        persona = await self.repo.get_persona() or {}
        policy = policy or PolicyConfig()
        profile = (
            await self.repo.get_profile(person_id)
            if person_id
            else {
                "familiarity": DEFAULT_FAMILIARITY,
                "relationship_tier": DEFAULT_FAMILIARITY,
                "style_overrides": {},
            }
        )
        tier = profile.get("relationship_tier") or profile.get("familiarity") or DEFAULT_FAMILIARITY
        presets = await self.repo.get_presets()
        deltas = presets.get(tier, DEFAULT_PRESET_DELTAS.get(tier, {}))

        traits: dict[str, int] = {}
        for name in PERSONA_TRAITS:
            base = persona.get(name)
            traits[name] = clamp(int(base)) if isinstance(base, int) else clamp(50)
        for name, delta in deltas.items():
            if name in traits:
                traits[name] = clamp(traits[name] + int(delta))
        for name, value in profile.get("style_overrides", {}).items():
            if name in traits:
                traits[name] = max(TRAIT_MIN, min(TRAIT_MAX, int(value)))
        for name, value in (temporary or {}).items():
            if name in traits:
                traits[name] = max(TRAIT_MIN, min(TRAIT_MAX, int(value)))

        manifest = await self.repo.get_manifest() or {}
        agent_id = manifest.get("agent_id", "")
        core = (
            await self.persona_core_repo.get_or_default(agent_id)
            if agent_id
            else None
        )
        return EffectiveAgentContext(
            agent_id=agent_id,
            display_name=identity.get("display_name") or "Companion",
            pronouns=identity.get("pronouns"),
            description=identity.get("description"),
            person_id=person_id,
            familiarity=tier,
            traits=traits,
            overrides=dict(profile.get("style_overrides", {})),
            persona_core=core.as_prompt_dict() if core else {},
            response_conservatism=policy.response_conservatism,
            interruption_threshold=policy.interruption_threshold,
            curiosity_level=policy.curiosity_level,
            revision=int(persona.get("revision") or 1),
        )

    # -- persona core ------------------------------------------------------

    async def show_persona_core(self) -> PersonaCore:
        manifest = await self.repo.get_manifest() or {}
        return await self.persona_core_repo.get_or_default(
            manifest.get("agent_id", "")
        )

    async def set_persona_core(
        self,
        updates: Mapping[str, object],
        *,
        actor: str = "admin",
        reason: str | None = None,
    ) -> PersonaCore:
        """Admin-only update of the stable character layer (audited)."""
        manifest = await self.repo.get_manifest() or {}
        agent_id = manifest.get("agent_id", "")
        current = await self.persona_core_repo.get_or_default(agent_id)
        allowed = {
            "core_values",
            "temperament",
            "preferences",
            "quirks",
            "contradictions",
            "conversational_boundaries",
            "voice_identity",
            "character_description",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise PersonalityValidationError(
                f"unknown persona core fields: {', '.join(sorted(unknown))}"
            )
        candidate = PersonaCore(
            agent_id=agent_id,
            core_values=updates.get("core_values", current.core_values),
            temperament=updates.get("temperament", current.temperament),
            preferences=updates.get("preferences", current.preferences),
            quirks=updates.get("quirks", current.quirks),
            contradictions=updates.get(
                "contradictions", current.contradictions
            ),
            conversational_boundaries=updates.get(
                "conversational_boundaries", current.conversational_boundaries
            ),
            voice_identity=updates.get(
                "voice_identity", current.voice_identity
            ),
            character_description=updates.get(
                "character_description", current.character_description
            ),
            revision=current.revision,
            updated_at=current.updated_at,
            updated_by=actor,
        )
        written = await self.persona_core_repo.update(
            candidate,
            expected_revision=current.revision,
            changed_by=actor,
        )
        if not written:
            raise PersonalityValidationError(
                "persona core changed concurrently; reload and retry"
            )
        updated = await self.persona_core_repo.get(agent_id) or candidate
        await self._audit(
            actor,
            "set_persona_core",
            "agent",
            agent_id,
            f"revision={updated.revision}"
            + (f" reason={reason}" if reason else ""),
        )
        self.logger.info(
            "persona_core_updated agent_id=%s revision=%d by=%s",
            agent_id,
            updated.revision,
            actor,
        )
        return updated

    async def import_persona_description(
        self,
        description: str,
        *,
        actor: str = "admin",
        reason: str | None = None,
    ) -> PersonaCore:
        """Import an admin-authored full character description.

        This is the reviewed import interface: the description is stored as
        trusted configuration and rendered as character context. It is not
        parsed into facts, and a model never calls this. A future importer may
        extract structured fields from the same text after human review.
        """
        return await self.set_persona_core(
            {"character_description": description},
            actor=actor,
            reason=reason or "import_persona_description",
        )

    # -- stance continuity -------------------------------------------------

    async def list_stances(
        self,
        agent_id: str,
        *,
        person_id: str | None = None,
        limit: int = 6,
    ) -> list[AgentStance]:
        if not agent_id:
            return []
        return await self.stance_repo.list_recent(
            agent_id, person_id=person_id, limit=limit
        )

    async def record_response_stances(
        self,
        agent_id: str,
        *,
        person_id: str | None,
        conversation_id: str | None,
        source_event_id: str | None,
        response_text: str,
        topic: str = "UNKNOWN",
    ) -> list[str]:
        """Persist stances the companion actually emitted. Never fails a turn."""
        if not agent_id or not response_text:
            return []
        candidates = extract_stance_candidates(response_text)
        if not candidates:
            return []
        for stance in candidates:
            await self.stance_repo.insert(
                agent_id=agent_id,
                person_id=person_id,
                conversation_id=conversation_id,
                topic=topic,
                stance=stance,
                source="RESPONSE",
                source_event_id=source_event_id,
            )
        self.logger.info(
            "stance_recorded agent_id=%s source_event_id=%s count=%d",
            agent_id,
            source_event_id,
            len(candidates),
        )
        return candidates

    async def add_stance(
        self,
        stance: str,
        *,
        topic: str = "UNKNOWN",
        person_id: str | None = None,
        actor: str = "admin",
    ) -> str | None:
        """Admin-configured stance (audited, trusted)."""
        manifest = await self.repo.get_manifest() or {}
        agent_id = manifest.get("agent_id", "")
        if not agent_id:
            return None
        stance_id = await self.stance_repo.insert(
            agent_id=agent_id,
            person_id=person_id,
            conversation_id=None,
            topic=topic,
            stance=stance,
            source="ADMIN",
        )
        if stance_id is not None:
            await self._audit(
                actor, "add_agent_stance", "agent", agent_id, topic
            )
        return stance_id

    async def retire_stance(
        self, stance_id: str, *, actor: str = "admin"
    ) -> bool:
        manifest = await self.repo.get_manifest() or {}
        agent_id = manifest.get("agent_id", "")
        if not agent_id:
            return False
        changed = await self.stance_repo.retire(agent_id, stance_id)
        if changed:
            await self._audit(
                actor, "retire_agent_stance", "agent", agent_id, stance_id
            )
        return changed

    # -- helpers -----------------------------------------------------------

    async def _require_person(self, person_id: str) -> None:
        if self.registry is None:
            person = await self.repos.person.get(person_id)
        else:
            person = await self.registry.get(person_id)
        if person is None:
            raise RelationshipValidationError(f"person not found: {person_id}")
