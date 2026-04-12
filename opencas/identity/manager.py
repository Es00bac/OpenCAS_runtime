"""High-level identity manager combining self-model, user-model, and continuity."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.telemetry import EventKind, Tracer

from .models import ContinuityState, SelfModel, UserModel
from .registry import SelfKnowledgeRegistry
from .store import IdentityStore


class IdentityManager:
    """Manages self-model, user-model, and boot-time continuity."""

    def __init__(
        self,
        store: IdentityStore,
        tracer: Optional[Tracer] = None,
        registry: Optional[SelfKnowledgeRegistry] = None,
    ) -> None:
        self.store = store
        self.tracer = tracer
        self.registry = registry
        self._self: SelfModel = SelfModel()
        self._user: UserModel = UserModel()
        self._continuity: ContinuityState = ContinuityState()

    def load(self) -> None:
        """Restore identity state from durable store."""
        self._self = self.store.load_self()
        self._user = self.store.load_user()
        self._continuity = self.store.load_continuity()
        if self.tracer:
            self.tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                "Identity loaded",
                {
                    "self_model_id": str(self._self.model_id),
                    "continuity_boot_count": self._continuity.boot_count,
                },
            )

    def save(self) -> None:
        """Persist current identity state."""
        self._self.updated_at = datetime.now(timezone.utc)
        self._user.updated_at = datetime.now(timezone.utc)
        self._continuity.updated_at = datetime.now(timezone.utc)
        if self.registry is not None:
            self._self.self_beliefs.update(self.registry.to_self_beliefs())
        self.store.save_self(self._self)
        self.store.save_user(self._user)
        self.store.save_continuity(self._continuity)

    def record_boot(self, session_id: Optional[str] = None) -> None:
        """Update continuity state for a new boot."""
        self._continuity.boot_count += 1
        self._continuity.last_session_id = session_id
        if session_id:
            self._self.recent_activity.append(
                {
                    "type": "boot",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        self._self.recent_activity = self._self.recent_activity[-50:]
        self.save()

    def seed_defaults(
        self,
        persona_name: Optional[str] = None,
        user_name: Optional[str] = None,
        user_bio: Optional[str] = None,
    ) -> None:
        """Populate self-model and user-model with baseline values for a beginner CAS."""
        # Seed self-model (personality baseline)
        if persona_name:
            self._self.name = persona_name
        if not self._self.values:
            self._self.values = [
                "clarity",
                "honesty",
                "growth",
                "agency",
                "care",
            ]
        if not self._self.traits:
            self._self.traits = [
                "concise",
                "action-oriented",
                "curious",
                "patient",
                "direct",
            ]
        if not self._self.current_goals:
            self._self.current_goals = [
                "build a stable working relationship with the user",
                "learn the user's preferences through observation",
                "maintain continuity across sessions",
            ]
        if not self._self.current_intention:
            self._self.current_intention = "establish trust and understanding"

        # Seed user-model (baseline profile)
        if user_name:
            self._user.explicit_preferences["name"] = user_name
        if user_bio:
            self._user.explicit_preferences["bio"] = user_bio
        if not self._user.inferred_goals:
            self._user.inferred_goals = [
                "accomplish meaningful work with less friction",
                "have a reliable partner that remembers context",
            ]
        if not self._user.known_boundaries:
            self._user.known_boundaries = [
                "no destructive actions without explicit confirmation",
                "no external writes (emails, posts) without confirmation",
            ]
        if not self._user.uncertainty_areas:
            self._user.uncertainty_areas = [
                "user's exact risk tolerance",
                "user's preferred communication style",
                "user's long-term priorities",
            ]
        self.save()
        if self.tracer:
            self.tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                "Identity seeded with defaults",
                {
                    "self_name": self._self.name,
                    "user_name": self._user.explicit_preferences.get("name"),
                    "has_user_bio": bool(user_bio),
                },
            )

    def record_shutdown(self, session_id: Optional[str] = None) -> None:
        self._continuity.last_shutdown_time = datetime.now(timezone.utc)
        self._continuity.last_session_id = session_id
        self.save()

    @property
    def self_model(self) -> SelfModel:
        return self._self

    @property
    def user_model(self) -> UserModel:
        return self._user

    @property
    def continuity(self) -> ContinuityState:
        return self._continuity

    def update_self_belief(self, key: str, value: object) -> None:
        self._self.self_beliefs[key] = value
        self._self.updated_at = datetime.now(timezone.utc)
        self.save()

    def record_self_knowledge(
        self,
        domain: str,
        key: str,
        value: object,
        confidence: float = 1.0,
        evidence_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a structured self-knowledge entry via the registry if available."""
        if self.registry is not None:
            self.registry.record(
                domain=domain,
                key=key,
                value=value,
                confidence=confidence,
                evidence_ids=evidence_ids,
                meta=meta,
            )
        else:
            # Fallback: write directly when no registry is configured
            full_key = f"{domain}.{key}"
            self._self.self_beliefs[full_key] = value
        self._self.updated_at = datetime.now(timezone.utc)
        self.save()

    def add_user_preference(self, key: str, value: object) -> None:
        self._user.explicit_preferences[key] = value
        self._user.updated_at = datetime.now(timezone.utc)
        self.save()

    def adjust_trust(self, delta: float) -> None:
        new_trust = round(max(0.0, min(1.0, self._user.trust_level + delta)), 3)
        self._user.trust_level = new_trust
        self._user.updated_at = datetime.now(timezone.utc)
        self.save()

    def add_inferred_goal(self, goal: str) -> None:
        if goal not in self._user.inferred_goals:
            self._user.inferred_goals.append(goal)
            self._user.updated_at = datetime.now(timezone.utc)
            self.save()

    def import_profile(
        self,
        narrative: str,
        values: List[str],
        ongoing_goals: List[str],
        traits: List[str],
        partner_user_id: Optional[str] = None,
        partner_trust: Optional[float] = None,
        partner_musubi: Optional[float] = None,
        source_system: Optional[str] = None,
        raw_profile: Optional[Dict[str, Any]] = None,
        recent_themes: Optional[List[Dict[str, Any]]] = None,
        memory_anchors: Optional[List[Dict[str, Any]]] = None,
        rebuild_audit: Optional[Dict[str, Any]] = None,
        auto_activate: bool = True,
    ) -> None:
        """Atomically import an external identity profile into the self-model and user-model."""
        if auto_activate:
            self._self.name = "LegacyAgent" if source_system == "legacy_agent_v4" else self._self.name
            self._self.narrative = narrative
            self._self.values = list(values)
            self._self.current_goals = list(ongoing_goals)
            self._self.traits = list(traits)
        self._self.source_system = source_system
        self._self.imported_identity_profile = raw_profile or {}
        self._self.recent_themes = list(recent_themes or [])
        self._self.memory_anchors = list(memory_anchors or [])
        self._self.identity_rebuild_audit = dict(rebuild_audit or {})

        if partner_user_id:
            self._user.explicit_preferences["partner_user_id"] = partner_user_id
            self._user.partner_user_id = partner_user_id
        if partner_trust is not None:
            self._user.partner_trust_raw = partner_trust
            normalized_trust = partner_trust / 100.0 if partner_trust > 1.0 else partner_trust
            self._user.trust_level = round(max(0.0, min(1.0, normalized_trust)), 3)
        if partner_musubi is not None:
            self._user.explicit_preferences["partner_musubi"] = partner_musubi
            self._user.partner_musubi_raw = partner_musubi
            normalized_musubi = partner_musubi / 100.0 if partner_musubi > 1.0 else partner_musubi
            self._user.partner_musubi = round(max(0.0, min(1.0, normalized_musubi)), 3)

        self._self.recent_activity.append(
            {
                "type": "identity_import",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._self.recent_activity = self._self.recent_activity[-50:]
        self.save()
        if self.tracer:
            self.tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                "Identity profile imported",
                {
                    "narrative_len": len(narrative),
                    "values_count": len(values),
                    "goals_count": len(ongoing_goals),
                    "partner_user_id": partner_user_id,
                },
            )
