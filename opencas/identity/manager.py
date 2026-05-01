"""High-level identity manager combining self-model, user-model, and continuity."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.provenance_adapter import append_provenance_record
from opencas.telemetry import EventKind, Tracer
from opencas.identity.text_hygiene import has_recursive_identity_loop, sanitize_identity_structure, sanitize_identity_text

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
        self._continuity_breadcrumb_limit = 25

    def load(self) -> None:
        """Restore identity state from durable store."""
        self._self = self.store.load_self()
        self._user = self.store.load_user()
        self._continuity = self.store.load_continuity()
        self._sanitize_loaded_identity_state()
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

    def _sanitize_loaded_identity_state(self) -> None:
        """Normalize in-memory identity and continuity state without external side effects."""
        self._self.narrative = sanitize_identity_text(self._self.narrative)
        self._self.current_intention = sanitize_identity_text(self._self.current_intention)
        self._self.values = self._sanitize_string_list(self._self.values)
        self._self.traits = self._sanitize_string_list(self._self.traits)
        self._self.current_goals = self._sanitize_string_list(self._self.current_goals)
        self._self.imported_identity_profile = self._sanitize_imported_identity_profile(
            self._self.imported_identity_profile
        )
        self._self.memory_anchors = self._sanitize_memory_anchors(self._self.memory_anchors)
        self._self.recent_themes = self._sanitize_recent_themes(self._self.recent_themes)
        self._self.self_beliefs = self._sanitize_self_beliefs(self._self.self_beliefs)
        self._continuity.last_continuity_monologue = sanitize_identity_text(
            self._continuity.last_continuity_monologue
        )

    def _sanitize_string_list(self, values: Optional[List[str]]) -> List[str]:
        cleaned: List[str] = []
        for value in values or []:
            if not isinstance(value, str):
                continue
            sanitized = sanitize_identity_text(value)
            if sanitized:
                cleaned.append(sanitized)
        return cleaned

    @staticmethod
    def _has_recursive_text(value: Any) -> bool:
        if isinstance(value, str):
            return has_recursive_identity_loop(value)
        if isinstance(value, dict):
            return any(IdentityManager._has_recursive_text(v) for v in value.values())
        if isinstance(value, list):
            return any(IdentityManager._has_recursive_text(item) for item in value)
        return False

    @classmethod
    def _sanitize_focus_items(cls, focus: Dict[str, Any]) -> Dict[str, Any]:
        """Drop recursive loop items and enforce valid focus status values."""
        if not isinstance(focus, dict):
            return {}
        clean_focus = dict(focus)
        items = focus.get("items")
        if not isinstance(items, list):
            return clean_focus

        cleaned_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if isinstance(label, str) and cls._is_recursive_focus_label(label):
                continue

            entry = dict(item)
            if isinstance(entry.get("label"), str):
                entry["label"] = sanitize_identity_text(entry["label"])

            status = str(entry.get("status", "")).strip().lower()
            if status not in {"closed", "archived"}:
                entry["status"] = "closed"
            else:
                entry["status"] = status
            cleaned_items.append(entry)

        clean_focus["items"] = cleaned_items
        return clean_focus

    @staticmethod
    def _is_recursive_focus_label(label: str) -> bool:
        return has_recursive_identity_loop(label)

    @classmethod
    def _sanitize_self_beliefs(cls, beliefs: Any) -> Dict[str, Any]:
        if not isinstance(beliefs, dict):
            return {}
        sanitized = sanitize_identity_structure(beliefs)
        if not isinstance(sanitized, dict):
            return {}

        daydream = sanitized.get("daydream")
        if isinstance(daydream, dict):
            bulma_focus = daydream.get("bulma_current_focus")
            if isinstance(bulma_focus, dict):
                daydream["bulma_current_focus"] = cls._sanitize_focus_items(bulma_focus)
            else:
                daydream["bulma_current_focus"] = {"items": []}
            sanitized["daydream"] = daydream
        return sanitized

    @staticmethod
    def _sanitize_imported_identity_profile(raw_profile: Any) -> Dict[str, Any]:
        if not isinstance(raw_profile, dict):
            return {}
        sanitized = sanitize_identity_structure(raw_profile)
        if isinstance(sanitized, dict):
            return sanitized
        return {}

    @classmethod
    def _sanitize_memory_anchors(cls, memory_anchors: Any) -> List[Dict[str, Any]]:
        if not isinstance(memory_anchors, list):
            return []

        sanitized: List[Dict[str, Any]] = []
        for anchor in memory_anchors:
            if not isinstance(anchor, dict):
                continue
            if cls._has_recursive_text(anchor):
                continue
            cleaned = sanitize_identity_structure(anchor)
            if not isinstance(cleaned, dict):
                continue
            sanitized.append(cleaned)
        return sanitized

    @staticmethod
    def _sanitize_recent_themes(recent_themes: Any) -> List[Dict[str, Any]]:
        if not isinstance(recent_themes, list):
            return []
        themes = []
        for item in recent_themes:
            if not isinstance(item, dict):
                continue
            term = sanitize_identity_text(item.get("term"))
            if not term:
                continue
            count = item.get("count")
            try:
                count_value = int(count)
            except (TypeError, ValueError):
                continue
            if count_value < 0:
                count_value = 0
            themes.append({"term": term, "count": count_value})
        return themes

    def record_boot(self, session_id: Optional[str] = None) -> None:
        """Update continuity state for a new boot."""
        self._continuity.boot_count += 1
        self._continuity.last_session_id = session_id
        if session_id:
            self._self.recent_activity.append(
                append_provenance_record(
                    {
                        "type": "boot",
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    session_id=session_id,
                    artifact="identity|continuity|boot",
                    action="CREATE",
                    why="bootstrap continuity restored",
                    risk="LOW",
                    source_trace={
                        "event": "record_boot",
                        "boot_count": self._continuity.boot_count,
                        "session_id": session_id,
                    },
                )
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
        self._self.recent_activity.append(
            append_provenance_record(
                {
                    "type": "shutdown",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                session_id=session_id or str(self.store.base_path.parent.name or "identity"),
                artifact="identity|continuity|shutdown",
                action="COMMIT",
                why="shutdown continuity persisted",
                risk="LOW",
                source_trace={
                    "event": "record_shutdown",
                    "session_id": session_id,
                },
            )
        )
        self._self.recent_activity = self._self.recent_activity[-50:]
        self.save()

    def decay_continuous_present(self, sleep_hours: float) -> float:
        """Decay the continuous_present_score based on hours of inactivity.

        Each hour multiplies the score by ``continuity_decay_rate`` (default 0.95).
        Returns the new score.
        """
        if sleep_hours <= 0:
            return self._continuity.continuous_present_score
        rate = self._continuity.continuity_decay_rate
        import math
        decay_factor = math.pow(rate, sleep_hours)
        self._continuity.continuous_present_score = max(
            0.0,
            min(1.0, self._continuity.continuous_present_score * decay_factor),
        )
        self.save()
        return self._continuity.continuous_present_score

    def recover_continuous_present(self) -> float:
        """Recover the continuous_present_score by one turn increment.

        Returns the new score.
        """
        recovery = self._continuity.continuity_recovery_rate
        self._continuity.continuous_present_score = min(
            1.0,
            self._continuity.continuous_present_score + recovery,
        )
        self.save()
        return self._continuity.continuous_present_score

    def set_continuity_monologue(self, monologue: str) -> None:
        """Store the latest boot-time continuity monologue."""
        self._continuity.last_continuity_monologue = sanitize_identity_text(monologue)
        self.save()

    def record_continuity_breadcrumb(
        self,
        intent: str,
        decision: str,
        next_step: str,
        *,
        note: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Store a normalized, timestamped continuity breadcrumb.

        The canonical breadcrumb format is:
        "{date-time} | intent: ... | decision: ... | note: ... | next_step: ..."
        """
        if not intent or not intent.strip():
            raise ValueError("continuity_breadcrumb requires a non-empty intent")
        if not decision or not decision.strip():
            raise ValueError("continuity_breadcrumb requires a non-empty decision")
        if note is not None and not note.strip():
            raise ValueError("continuity_breadcrumb note, when provided, must be non-empty")
        if not next_step or not next_step.strip():
            raise ValueError("continuity_breadcrumb requires a non-empty next_step")

        intent = sanitize_identity_text(intent.strip())
        decision = sanitize_identity_text(decision.strip())
        if note is not None:
            note = sanitize_identity_text(note)
        next_step = sanitize_identity_text(next_step.strip())

        stamp = (timestamp or datetime.now(timezone.utc)).isoformat()
        payload = f"{stamp} | intent: {intent} | decision: {decision}"
        if note:
            payload = f"{payload} | note: {note.strip()}"
        payload = f"{payload} | next_step: {next_step}"
        self._continuity.continuity_breadcrumb = payload
        self._continuity.continuity_breadcrumbs.append(payload)
        self._continuity.continuity_breadcrumbs = self._continuity.continuity_breadcrumbs[
            -self._continuity_breadcrumb_limit :
        ]
        self.save()
        return payload

    def record_compaction(self, session_id: Optional[str] = None) -> None:
        """Track a compaction event in the continuity state."""
        self._continuity.compaction_count += 1
        self._continuity.last_session_id = session_id
        self._self.recent_activity.append(
            append_provenance_record(
                {
                    "type": "compaction",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                session_id=session_id or str(self.store.base_path.parent.name or "identity"),
                artifact="identity|continuity|compaction",
                action="CONSOLIDATE",
                why="continuity compaction recorded",
                risk="MEDIUM",
                source_trace={
                    "event": "record_compaction",
                    "compaction_count": self._continuity.compaction_count,
                    "session_id": session_id,
                },
            )
        )
        self._self.recent_activity = self._self.recent_activity[-50:]
        self.save()

    def apply_memory_mutation(
        self,
        content: str,
        source_type: str = "memory",
        confidence: float = 0.7,
    ) -> None:
        """Apply an identity mutation triggered by a mutagenic memory/episode.

        Records a new self-belief and updates the identity core narrative.
        Throttled to max 1 mutation per call — callers should enforce
        per-turn limits externally.
        """
        # Extract a short predicate from the content
        predicate = content.strip().lower()[:80]
        self.record_self_knowledge(
            domain="identity_mutation",
            key=f"mutagen_{hash(predicate) % 10000:04d}",
            value={
                "predicate": predicate,
                "source_type": source_type,
                "confidence": confidence,
                "mutated_at": datetime.now(timezone.utc).isoformat(),
            },
            confidence=confidence,
        )
        if self.tracer:
            self.tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                "Identity mutation applied from memory",
                {
                    "predicate": predicate[:60],
                    "source_type": source_type,
                    "confidence": confidence,
                },
            )
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

    def set_recent_themes(self, themes: List[Dict[str, Any]]) -> None:
        """Replace the recent themes list with sanitized theme records."""
        cleaned: List[Dict[str, Any]] = []
        for theme in themes:
            if not isinstance(theme, dict):
                continue
            term = str(theme.get("term", "")).strip().lower()
            if not term:
                continue
            count = theme.get("count")
            try:
                count_value = int(count)
            except (TypeError, ValueError):
                continue
            if count_value < 0:
                count_value = 0

            cleaned.append(
                {
                    "term": term,
                    "count": count_value,
                }
            )
        self._self.recent_themes = cleaned[:40]
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
            self._self.name = "the OpenCAS agent" if source_system == "openbulma-v4" else self._self.name
            self._self.narrative = sanitize_identity_text(narrative)
            self._self.values = self._sanitize_string_list(values)
            self._self.current_goals = self._sanitize_string_list(ongoing_goals)
            self._self.traits = self._sanitize_string_list(traits)
        self._self.source_system = source_system
        self._self.imported_identity_profile = self._sanitize_imported_identity_profile(raw_profile)
        if memory_anchors is None and isinstance(raw_profile, dict):
            raw_memory_anchors = raw_profile.get("memoryAnchors", [])
            if isinstance(raw_memory_anchors, list):
                memory_anchors = raw_memory_anchors
        self._self.recent_themes = self._sanitize_recent_themes(recent_themes)
        self._self.memory_anchors = self._sanitize_memory_anchors(memory_anchors)
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
            append_provenance_record(
                {
                    "type": "identity_import",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                session_id=str(self.store.base_path.parent.name or "identity"),
                artifact="identity|profile|import",
                action="UPDATE",
                why="identity profile imported",
                risk="MEDIUM",
                source_trace={
                    "event": "import_profile",
                    "source_system": source_system,
                    "has_raw_profile": raw_profile is not None,
                },
            )
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
