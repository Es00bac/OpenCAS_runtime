"""Theory of Mind engine for OpenCAS."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.identity import IdentityManager
from opencas.telemetry import EventKind, Tracer

from .models import Belief, BeliefSubject, Intention, IntentionStatus, MetacognitiveResult
from .store import TomStore


class ToMEngine:
    """Tracks beliefs, intentions, and runs metacognitive consistency checks."""

    def __init__(
        self,
        identity: IdentityManager,
        tracer: Optional[Tracer] = None,
        store: Optional[TomStore] = None,
    ) -> None:
        self.identity = identity
        self.tracer = tracer
        self.store = store
        self._beliefs: List[Belief] = []
        self._intentions: List[Intention] = []

    async def load(self) -> None:
        """Hydrate in-memory lists from durable store (capped to 1000 each)."""
        if self.store is None:
            return
        beliefs = await self.store.list_beliefs(limit=1000)
        self._beliefs = list(reversed(beliefs))
        intentions = await self.store.list_intentions(limit=1000)
        self._intentions = list(reversed(intentions))

    async def record_belief(
        self,
        subject: BeliefSubject,
        predicate: str,
        confidence: float = 1.0,
        evidence_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Belief:
        """Record a new belief, surfacing key beliefs to the identity model."""
        belief = Belief(
            subject=subject,
            predicate=predicate.strip().lower(),
            confidence=max(0.0, min(1.0, confidence)),
            evidence_ids=evidence_ids or [],
            meta=meta or {},
        )
        self._beliefs.append(belief)
        if len(self._beliefs) > 1000:
            self._beliefs = self._beliefs[-1000:]
        if self.store is not None:
            await self.store.save_belief(belief)
        self._sync_to_identity(subject, predicate, confidence)
        self._trace("belief_recorded", {
            "belief_id": str(belief.belief_id),
            "subject": subject.value,
            "predicate": belief.predicate,
            "confidence": belief.confidence,
            "belief_count": len(self._beliefs),
        })
        return belief

    async def record_intention(
        self,
        actor: BeliefSubject,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Intention:
        """Record a new active intention."""
        intention = Intention(
            actor=actor,
            content=content.strip().lower(),
            status=IntentionStatus.ACTIVE,
            meta=meta or {},
        )
        self._intentions.append(intention)
        if len(self._intentions) > 1000:
            self._intentions = self._intentions[-1000:]
        if self.store is not None:
            await self.store.save_intention(intention)
        if actor == BeliefSubject.SELF:
            self.identity.self_model.current_intention = content
            self.identity.save()
        self._trace("intention_recorded", {
            "intention_id": str(intention.intention_id),
            "actor": actor.value,
            "content": intention.content,
            "active_intention_count": len([i for i in self._intentions if i.status == IntentionStatus.ACTIVE]),
        })
        return intention

    async def resolve_intention(
        self,
        content: str,
        status: IntentionStatus = IntentionStatus.COMPLETED,
    ) -> bool:
        """Mark the most recent matching active intention as resolved."""
        target = content.strip().lower()
        for intention in reversed(self._intentions):
            if intention.status == IntentionStatus.ACTIVE and intention.content == target:
                intention.status = status
                intention.resolved_at = datetime.now(timezone.utc)
                if self.store is not None:
                    await self.store.resolve_intention(
                        str(intention.intention_id),
                        status,
                        intention.resolved_at,
                    )
                self._trace("intention_resolved", {
                    "intention_id": str(intention.intention_id),
                    "status": status.value,
                })
                return True
        return False

    def list_beliefs(
        self,
        subject: Optional[BeliefSubject] = None,
        predicate: Optional[str] = None,
    ) -> List[Belief]:
        results = self._beliefs
        if subject:
            results = [b for b in results if b.subject == subject]
        if predicate:
            target = predicate.strip().lower()
            results = [b for b in results if b.predicate == target]
        return results

    def list_intentions(
        self,
        actor: Optional[BeliefSubject] = None,
        status: Optional[IntentionStatus] = None,
    ) -> List[Intention]:
        results = self._intentions
        if actor:
            results = [i for i in results if i.actor == actor]
        if status:
            results = [i for i in results if i.status == status]
        return results

    def check_consistency(self) -> MetacognitiveResult:
        """Run a metacognitive check for contradictions between beliefs and intentions."""
        contradictions: List[str] = []
        warnings: List[str] = []

        # Load boundaries and known preferences as user-model beliefs
        boundaries = set(self.identity.user_model.known_boundaries)
        user_prefs = self.identity.user_model.explicit_preferences

        for intention in self._intentions:
            if intention.status != IntentionStatus.ACTIVE:
                continue

            # Boundary contradiction: active self-intention violates a known user boundary
            if intention.actor == BeliefSubject.SELF:
                for boundary in boundaries:
                    b_lower = boundary.lower()
                    if b_lower in intention.content:
                        contradictions.append(
                            f"Self-intention '{intention.content}' hits known boundary '{boundary}'"
                        )
                        continue
                    # Detect negated boundaries (e.g., "no email" vs "send email")
                    for neg in ("no ", "don't ", "never ", "avoid ", "stop "):
                        if b_lower.startswith(neg):
                            core = b_lower[len(neg):].strip()
                            if core in intention.content:
                                contradictions.append(
                                    f"Self-intention '{intention.content}' hits known boundary '{boundary}'"
                                )
                            break

            # Preference contradiction: user intention conflicts with explicit preference
            if intention.actor == BeliefSubject.USER:
                for key, value in user_prefs.items():
                    if key.lower() in intention.content and str(value).lower() in ("false", "no", "0"):
                        contradictions.append(
                            f"User intention '{intention.content}' conflicts with preference '{key}={value}'"
                        )

        # Confidence-based warning: very low-confidence beliefs about the user
        for belief in self._beliefs:
            if belief.subject == BeliefSubject.USER and belief.confidence < 0.3:
                warnings.append(
                    f"Very low confidence in user belief '{belief.predicate}' ({belief.confidence:.2f})"
                )

        # Self-belief contradiction: two beliefs with opposite predicates at high confidence
        self_beliefs = [b for b in self._beliefs if b.subject == BeliefSubject.SELF and b.confidence > 0.7]
        predicates = {b.predicate for b in self_beliefs}
        opposites = {
            ("tired", "rested"),
            ("busy", "idle"),
            ("confident", "uncertain"),
            ("focused", "distracted"),
            ("available", "unavailable"),
        }
        for a, b in opposites:
            if a in predicates and b in predicates:
                contradictions.append(f"Self-beliefs '{a}' and '{b}' both held with high confidence")

        result = MetacognitiveResult(
            contradictions=contradictions,
            warnings=warnings,
            belief_count=len(self._beliefs),
            intention_count=len([i for i in self._intentions if i.status == IntentionStatus.ACTIVE]),
        )
        self._trace("consistency_check", {
            "contradictions": result.contradictions,
            "warnings": result.warnings,
            "belief_count": result.belief_count,
            "active_intention_count": result.intention_count,
        })
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "beliefs": [
                {
                    "subject": b.subject.value,
                    "predicate": b.predicate,
                    "confidence": b.confidence,
                }
                for b in self._beliefs[-10:]
            ],
            "intentions": [
                {
                    "actor": i.actor.value,
                    "content": i.content,
                    "status": i.status.value,
                }
                for i in self._intentions[-10:]
            ],
        }

    def _sync_to_identity(self, subject: BeliefSubject, predicate: str, confidence: float) -> None:
        """Mirror high-confidence self-beliefs into the identity self-model via registry."""
        if subject == BeliefSubject.SELF and confidence >= 0.7:
            key = f"belief_{predicate.replace(' ', '_')[:40]}"
            self.identity.record_self_knowledge(
                domain="tom",
                key=key,
                value={
                    "predicate": predicate,
                    "confidence": confidence,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                confidence=confidence,
            )

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOM_EVAL,
                f"ToMEngine: {event}",
                payload,
            )
