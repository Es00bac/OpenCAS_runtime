"""Theory of Mind engine for OpenCAS."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.identity import IdentityManager
from opencas.telemetry import EventKind, Tracer

from .models import (
    Belief,
    BeliefSubject,
    Intention,
    IntentionStatus,
    MetacognitiveResult,
    PromiseFollowthroughSignal,
)
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
        self._beliefs = [_apply_inferred_belief_frame(belief) for belief in reversed(beliefs)]
        intentions = await self.store.list_intentions(limit=1000)
        self._intentions = list(reversed(intentions))

    async def record_belief(
        self,
        subject: BeliefSubject,
        predicate: str,
        confidence: float = 1.0,
        evidence_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        relation: Optional[str] = None,
        object: Optional[str] = None,
        source_kind: Optional[str] = None,
        source_strength: Optional[float] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        decay_rate: float = 0.0,
    ) -> Belief:
        """Record a new belief or reinforce an existing one with the same subject+predicate."""
        normalized = predicate.strip().lower()
        frame = _derive_belief_frame(
            subject=subject,
            predicate=normalized,
            meta=meta,
            relation=relation,
            object_value=object,
            source_kind=source_kind,
            source_strength=source_strength,
            valid_from=valid_from,
            valid_until=valid_until,
            decay_rate=decay_rate,
        )
        if _is_audit_only_meta(meta):
            belief = Belief(
                subject=subject,
                predicate=normalized,
                confidence=max(0.0, min(1.0, confidence)),
                evidence_ids=evidence_ids or [],
                meta=meta or {},
                reinforcement_count=1,
                last_reinforced=datetime.now(timezone.utc),
                **frame,
            )
            self._trace(
                "belief_suppressed",
                {
                    "reason": "audit_only",
                    "subject": subject.value,
                    "predicate": normalized,
                },
            )
            return belief
        now = datetime.now(timezone.utc)
        entity_id = ""
        if subject == BeliefSubject.AGENT and meta:
            entity_id = str(meta.get("entity_id") or "").strip().lower()

        # Check for existing belief with same subject+predicate
        existing = await self._find_existing_belief(subject, normalized, entity_id=entity_id or None)

        if existing is not None:
            # Reinforce: merge confidence upward, increment count
            old_conf = existing.confidence
            reinforcement_gain = 0.05 + (0.15 * float(frame["source_strength"]))
            new_confidence = old_conf + (1.0 - old_conf) * reinforcement_gain
            new_confidence = max(0.0, min(1.0, new_confidence))
            existing.confidence = new_confidence
            existing.reinforcement_count += 1
            existing.last_reinforced = now
            # Merge evidence IDs (dedup)
            merged_ids = list(set(existing.evidence_ids + (evidence_ids or [])))
            existing.evidence_ids = merged_ids
            if not existing.relation and frame["relation"]:
                existing.relation = frame["relation"]
            if not existing.object and frame["object"]:
                existing.object = frame["object"]
            existing.source_kind = frame["source_kind"] or existing.source_kind
            existing.source_strength = max(existing.source_strength, frame["source_strength"])
            if frame["valid_from"] and (
                existing.valid_from is None or frame["valid_from"] < existing.valid_from
            ):
                existing.valid_from = frame["valid_from"]
            if frame["valid_until"] and (
                existing.valid_until is None or frame["valid_until"] > existing.valid_until
            ):
                existing.valid_until = frame["valid_until"]
            existing.decay_rate = max(existing.decay_rate, frame["decay_rate"])
            if meta:
                existing.meta.update(meta)

            # Update in-memory list (replace the old entry)
            replaced = False
            for i, b in enumerate(self._beliefs):
                if b.belief_id == existing.belief_id:
                    self._beliefs[i] = existing
                    replaced = True
                    break
            if not replaced:
                self._beliefs.append(existing)
                if len(self._beliefs) > 1000:
                    self._beliefs.pop(0)

            if self.store is not None:
                await self.store.save_belief(existing)
            self._sync_to_identity(subject, normalized, existing.confidence)
            self._trace("belief_reinforced", {
                "belief_id": str(existing.belief_id),
                "subject": subject.value,
                "predicate": normalized,
                "old_confidence": old_conf,
                "new_confidence": new_confidence,
                "reinforcement_count": existing.reinforcement_count,
                "belief_count": len(self._beliefs),
            })
            return existing

        # New belief — no existing match
        belief = Belief(
            subject=subject,
            predicate=normalized,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_ids=evidence_ids or [],
            meta=meta or {},
            reinforcement_count=1,
            last_reinforced=now,
            **frame,
        )
        self._beliefs.append(belief)
        if len(self._beliefs) > 1000:
            self._beliefs = self._beliefs[-1000:]
        if self.store is not None:
            await self.store.save_belief(belief)
        self._sync_to_identity(subject, normalized, belief.confidence)
        self._trace("belief_recorded", {
            "belief_id": str(belief.belief_id),
            "subject": subject.value,
            "predicate": normalized,
            "confidence": belief.confidence,
            "belief_count": len(self._beliefs),
        })
        return belief

    async def _find_existing_belief(
        self,
        subject: BeliefSubject,
        predicate: str,
        *,
        entity_id: str | None = None,
    ) -> Optional[Belief]:
        """Find an existing belief by subject+predicate. Checks store if available."""
        # Check in-memory first
        for b in reversed(self._beliefs):
            if b.subject != subject or b.predicate != predicate:
                continue
            if entity_id is not None and str((b.meta or {}).get("entity_id") or "").lower() != entity_id:
                continue
            return b
        # Fallback to store query
        if self.store is not None:
            return await self.store.get_belief_by_predicate(subject, predicate, entity_id=entity_id)
        return None

    async def record_agent_belief(
        self,
        entity_id: str,
        predicate: str,
        *,
        entity_label: str = "",
        confidence: float = 0.7,
        evidence_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Belief:
        """Record a belief about a third-party AI/agent entity."""

        clean_entity = _normalize_entity_id(entity_id or entity_label)
        merged_meta = dict(meta or {})
        merged_meta.update({
            "entity_id": clean_entity,
            "entity_label": entity_label.strip() or clean_entity,
            "entity_kind": "ai_agent",
        })
        return await self.record_belief(
            BeliefSubject.AGENT,
            predicate,
            confidence=confidence,
            evidence_ids=evidence_ids,
            meta=merged_meta,
        )

    async def record_intention(
        self,
        actor: BeliefSubject,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Intention:
        """Record a new active intention."""
        if _is_audit_only_meta(meta):
            intention = Intention(
                actor=actor,
                content=content.strip().lower(),
                status=IntentionStatus.ACTIVE,
                meta=meta or {},
            )
            self._trace(
                "intention_suppressed",
                {
                    "reason": "audit_only",
                    "actor": actor.value,
                    "content": intention.content,
                },
            )
            return intention
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

    async def record_agent_intention(
        self,
        entity_id: str,
        content: str,
        *,
        entity_label: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Intention:
        """Record an inferred intention for a third-party AI/agent entity."""

        clean_entity = _normalize_entity_id(entity_id or entity_label)
        merged_meta = dict(meta or {})
        merged_meta.update({
            "entity_id": clean_entity,
            "entity_label": entity_label.strip() or clean_entity,
            "entity_kind": "ai_agent",
        })
        return await self.record_intention(BeliefSubject.AGENT, content, meta=merged_meta)

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

    def list_agent_models(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Return compact third-party agent models assembled from ToM records."""

        models: Dict[str, Dict[str, Any]] = {}
        for belief in self._beliefs:
            if belief.subject != BeliefSubject.AGENT:
                continue
            entity_id = str((belief.meta or {}).get("entity_id") or "").strip()
            if not entity_id:
                continue
            model = models.setdefault(
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_label": str((belief.meta or {}).get("entity_label") or entity_id),
                    "entity_kind": str((belief.meta or {}).get("entity_kind") or "ai_agent"),
                    "beliefs": [],
                    "intentions": [],
                },
            )
            model["beliefs"].append(
                {
                    "predicate": belief.predicate,
                    "confidence": belief.confidence,
                    "evidence_ids": list(belief.evidence_ids),
                }
            )
        for intention in self._intentions:
            if intention.actor != BeliefSubject.AGENT:
                continue
            entity_id = str((intention.meta or {}).get("entity_id") or "").strip()
            if not entity_id:
                continue
            model = models.setdefault(
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_label": str((intention.meta or {}).get("entity_label") or entity_id),
                    "entity_kind": str((intention.meta or {}).get("entity_kind") or "ai_agent"),
                    "beliefs": [],
                    "intentions": [],
                },
            )
            model["intentions"].append(
                {
                    "content": intention.content,
                    "status": intention.status.value,
                    "timestamp": intention.timestamp.isoformat(),
                }
            )
        ranked = sorted(
            models.values(),
            key=lambda item: (len(item["beliefs"]) + len(item["intentions"]), item["entity_id"]),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def salient_user_model(self, user_input: str = "", *, limit: int = 6) -> List[Dict[str, Any]]:
        """Return evidence-grounded user-model items relevant enough to guide behavior.

        This surfaces ToM records as weighted evidence, not as certain facts or
        scripted relationship claims.
        """

        query_terms = _query_terms(user_input)
        now = datetime.now(timezone.utc)
        scored: List[tuple[float, str, Dict[str, Any]]] = []
        for belief in self._beliefs:
            if belief.subject != BeliefSubject.USER:
                continue
            _apply_inferred_belief_frame(belief)
            frame = _belief_frame_for(belief)
            effective_confidence = _effective_belief_confidence(belief, now=now)
            if effective_confidence < 0.34:
                continue
            predicate = str(belief.predicate or "").strip()
            if not predicate:
                continue
            relation = str(frame["relation"] or "asserts")
            object_value = str(frame["object"] or "")
            haystack = " ".join((predicate, relation, object_value)).lower()
            overlap = len(query_terms.intersection(_query_terms(haystack))) if query_terms else 0
            relation_bonus = _behavior_relation_bonus(relation, predicate)
            recency_bonus = _belief_recency_bonus(belief, now=now)
            score = (
                effective_confidence * 3.0
                + float(frame["source_strength"]) * 1.25
                + relation_bonus
                + min(2.0, float(overlap) * 0.5)
                + recency_bonus
            )
            if query_terms and overlap == 0 and relation_bonus < 0.9:
                continue
            if not query_terms and relation_bonus <= 0.0 and effective_confidence < 0.72:
                continue
            timestamp = belief.timestamp.isoformat() if belief.timestamp is not None else ""
            scored.append(
                (
                    score,
                    timestamp,
                    {
                        "belief_id": str(belief.belief_id),
                        "timestamp": timestamp,
                        "subject": belief.subject.value,
                        "predicate": predicate,
                        "relation": relation,
                        "object": object_value,
                        "confidence": belief.confidence,
                        "effective_confidence": effective_confidence,
                        "source_kind": frame["source_kind"],
                        "source_strength": frame["source_strength"],
                        "evidence_ids": list(belief.evidence_ids),
                        "reinforcement_count": belief.reinforcement_count,
                        "score": round(score, 4),
                    },
                )
            )
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item for _score, _timestamp, item in scored[: max(1, int(limit))]]

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

        now = datetime.now(timezone.utc)

        # Confidence-based warning: very low-confidence beliefs about the user
        for belief in self._beliefs:
            _apply_inferred_belief_frame(belief)
            frame = _belief_frame_for(belief)
            effective_confidence = _effective_belief_confidence(belief, now=now)
            if belief.valid_until is not None and belief.valid_until < now and belief.confidence >= 0.5:
                warnings.append(
                    f"expired belief '{belief.predicate}' should be refreshed before use"
                )
            if belief.confidence >= 0.8 and frame["source_strength"] < 0.45:
                warnings.append(
                    f"High-confidence belief '{belief.predicate}' has weak source grounding ({frame['source_strength']:.2f})"
                )
            if belief.subject == BeliefSubject.USER and effective_confidence < 0.3:
                warnings.append(
                    f"Very low confidence in user belief '{belief.predicate}' ({effective_confidence:.2f})"
                )

        # Self-belief contradiction: two beliefs with opposite predicates at high confidence
        self_beliefs = [
            b
            for b in self._beliefs
            if b.subject == BeliefSubject.SELF and _effective_belief_confidence(b, now=now) > 0.7
        ]
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

        user_locations: Dict[str, Belief] = {}
        for belief in self._beliefs:
            _apply_inferred_belief_frame(belief)
            frame = _belief_frame_for(belief)
            if (
                belief.subject == BeliefSubject.USER
                and frame["relation"] in {"lives_in", "located_in"}
                and frame["object"]
                and _effective_belief_confidence(belief, now=now) > 0.7
            ):
                user_locations[frame["object"]] = belief
        if len(user_locations) > 1:
            locations = ", ".join(sorted(user_locations))
            contradictions.append(f"User location beliefs conflict: {locations}")

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

    def evaluate_promise_followthrough(
        self,
        somatic_state: Optional[Any] = None,
        relational_engine: Optional[Any] = None,
        metacognitive_result: Optional[MetacognitiveResult] = None,
    ) -> PromiseFollowthroughSignal:
        """Interpret unresolved self-commitments for follow-through behavior."""
        pending = [
            intention
            for intention in self._intentions
            if intention.status == IntentionStatus.ACTIVE
            and intention.actor == BeliefSubject.SELF
            and str(intention.meta.get("source", "")).lower() == "self_commitment_capture"
        ]
        if not pending:
            return PromiseFollowthroughSignal()

        check = metacognitive_result or self.check_consistency()
        contradictions = check.contradictions
        fatigue = float(getattr(somatic_state, "fatigue", 0.0) or 0.0)
        tension = float(getattr(somatic_state, "tension", 0.0) or 0.0)
        certainty = float(getattr(somatic_state, "certainty", 1.0) or 1.0)

        trust = 0.0
        resonance = 0.0
        if relational_engine is not None:
            try:
                relational_state = relational_engine.state
            except AssertionError:
                relational_state = None
            if relational_state is not None:
                trust = float(relational_state.dimensions.get("trust", 0.0))
                resonance = float(relational_state.dimensions.get("resonance", 0.0))

        should_acknowledge_delay = (
            fatigue > 0.58
            or tension > 0.64
            or certainty < 0.40
            or bool(contradictions)
        )
        should_repair_trust = should_acknowledge_delay and (
            trust < -0.10
            or resonance < -0.10
            or any(
                "boundary" in item.lower() or "preference" in item.lower()
                for item in contradictions
            )
        )
        should_resume_now = not should_acknowledge_delay and fatigue < 0.72 and tension < 0.75

        pending_contents: List[str] = []
        for intention in pending:
            if intention.content not in pending_contents:
                pending_contents.append(intention.content)

        return PromiseFollowthroughSignal(
            pending_count=len(pending_contents),
            pending_contents=pending_contents[:5],
            should_acknowledge_delay=should_acknowledge_delay,
            should_repair_trust=should_repair_trust,
            should_resume_now=should_resume_now,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "beliefs": [
                _belief_snapshot_payload(b)
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
            "agent_models": self.list_agent_models(limit=5),
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


def _normalize_entity_id(value: str) -> str:
    normalized = "_".join(str(value or "").strip().lower().split())
    return normalized[:80] or "unknown_agent"


def _is_audit_only_meta(meta: Optional[Dict[str, Any]]) -> bool:
    return bool((meta or {}).get("audit_only"))


def _derive_belief_frame(
    *,
    subject: BeliefSubject,
    predicate: str,
    meta: Optional[Dict[str, Any]],
    relation: Optional[str],
    object_value: Optional[str],
    source_kind: Optional[str],
    source_strength: Optional[float],
    valid_from: Optional[datetime],
    valid_until: Optional[datetime],
    decay_rate: float,
) -> Dict[str, Any]:
    inferred_relation, inferred_object = _infer_relation_object(subject, predicate)
    resolved_source = (source_kind or str((meta or {}).get("source") or "unknown")).strip().lower()
    return {
        "relation": (relation or inferred_relation or "").strip().lower(),
        "object": (object_value or inferred_object or "").strip().lower(),
        "source_kind": resolved_source or "unknown",
        "source_strength": _resolve_source_strength(
            source_kind=resolved_source,
            explicit=source_strength if source_strength is not None else (meta or {}).get("source_strength"),
            meta=meta,
        ),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "decay_rate": max(0.0, min(1.0, float(decay_rate or (meta or {}).get("decay_rate", 0.0) or 0.0))),
    }


def _belief_frame_for(belief: Belief) -> Dict[str, Any]:
    raw_source_kind = str(getattr(belief, "source_kind", "") or "").strip().lower()
    meta = dict(getattr(belief, "meta", {}) or {})
    source_kind = raw_source_kind if raw_source_kind not in {"", "unknown"} else None
    explicit_strength: Optional[float]
    if source_kind is None and "source_strength" not in meta:
        explicit_strength = None
    else:
        explicit_strength = getattr(belief, "source_strength", None)
    return _derive_belief_frame(
        subject=belief.subject,
        predicate=belief.predicate,
        meta=meta,
        relation=getattr(belief, "relation", "") or None,
        object_value=getattr(belief, "object", "") or None,
        source_kind=source_kind,
        source_strength=explicit_strength,
        valid_from=getattr(belief, "valid_from", None),
        valid_until=getattr(belief, "valid_until", None),
        decay_rate=float(getattr(belief, "decay_rate", 0.0) or 0.0),
    )


def _apply_inferred_belief_frame(belief: Belief) -> Belief:
    frame = _belief_frame_for(belief)
    if not belief.relation and frame["relation"]:
        belief.relation = frame["relation"]
    if not belief.object and frame["object"]:
        belief.object = frame["object"]
    if (not belief.source_kind or belief.source_kind == "unknown") and frame["source_kind"]:
        belief.source_kind = frame["source_kind"]
    if belief.source_strength == 0.5 and frame["source_strength"] != 0.5:
        belief.source_strength = frame["source_strength"]
    return belief


def _belief_snapshot_payload(belief: Belief) -> Dict[str, Any]:
    frame = _belief_frame_for(_apply_inferred_belief_frame(belief))
    return {
        "subject": belief.subject.value,
        "predicate": belief.predicate,
        "confidence": belief.confidence,
        "relation": frame["relation"],
        "object": frame["object"],
        "source_kind": frame["source_kind"],
        "effective_confidence": _effective_belief_confidence(
            belief,
            now=datetime.now(timezone.utc),
        ),
    }


def _infer_relation_object(subject: BeliefSubject, predicate: str) -> tuple[str, str]:
    text = " ".join(str(predicate or "").strip().lower().split())
    prefix_relations = {
        "asked:": "asked",
        "said:": "said",
        "wants:": "wants",
        "needs:": "needs",
    }
    for prefix, relation in prefix_relations.items():
        if text.startswith(prefix):
            return relation, text[len(prefix):].strip()
    plain_prefix_relations = (
        ("prefers ", "prefers"),
        ("wants ", "wants"),
        ("needs ", "needs"),
        ("likes ", "likes"),
        ("dislikes ", "dislikes"),
        ("expects ", "expects"),
        ("values ", "values"),
        ("does not want ", "does_not_want"),
    )
    for prefix, relation in plain_prefix_relations:
        if text.startswith(prefix):
            return relation, text[len(prefix):].strip()
    location_patterns = (
        r"^(?:i|user|the user|jarrod)\s+(?:live|lives)\s+in\s+(.+)$",
        r"^(?:i|user|the user|jarrod)\s+(?:am|is)\s+in\s+(.+)$",
        r"^(?:my|user'?s|the user'?s)\s+location\s+is\s+(.+)$",
    )
    if subject == BeliefSubject.USER:
        for pattern in location_patterns:
            match = re.match(pattern, text)
            if match:
                return "lives_in", match.group(1).strip(" .")
    favorite = re.match(r"^(?:my|user'?s|the user'?s)\s+favorite\s+(.+?)\s+is\s+(.+)$", text)
    if favorite:
        return f"favorite_{favorite.group(1).strip().replace(' ', '_')}", favorite.group(2).strip(" .")
    likes = re.match(r"^(?:likes|like)\s+(.+)$", text)
    if likes:
        return "likes", likes.group(1).strip(" .")
    state = re.match(r"^(?:is|am)\s+(.+)$", text)
    if state:
        return "state", state.group(1).strip(" .")
    return "asserts", text


def _resolve_source_strength(
    *,
    source_kind: str,
    explicit: Any,
    meta: Optional[Dict[str, Any]],
) -> float:
    if explicit is not None:
        try:
            return max(0.0, min(1.0, float(explicit)))
        except Exception:
            pass
    extractor = str((meta or {}).get("extractor") or "").strip().lower()
    if extractor == "legacy_said":
        return 0.45
    if source_kind in {"tool_result", "artifact_lookup", "fs_read_file", "system_state", "runtime"}:
        return 0.9
    if source_kind in {"self_commitment_capture", "active_work_dispatch"}:
        return 0.85
    if source_kind == "conversation_turn":
        return 0.75
    if source_kind in {"recall_autobiography", "autobiography", "memory_recall"}:
        return 0.62
    if source_kind == "cognitive_social_model_record":
        return 0.7
    if source_kind in {"inference", "unknown", ""}:
        return 0.5
    return 0.6


def _query_terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "and",
        "are",
        "can",
        "continue",
        "does",
        "for",
        "from",
        "have",
        "into",
        "need",
        "needs",
        "now",
        "please",
        "that",
        "the",
        "this",
        "want",
        "wants",
        "what",
        "when",
        "where",
        "with",
        "you",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(text or "").lower())
        if token not in stopwords
    }


def _behavior_relation_bonus(relation: str, predicate: str) -> float:
    relation_l = str(relation or "").lower()
    predicate_l = str(predicate or "").lower()
    if relation_l in {"prefers", "needs", "wants", "asked", "likes", "dislikes", "expects", "values", "does_not_want"}:
        return 1.4
    if relation_l.startswith("favorite_"):
        return 1.1
    if any(
        token in predicate_l
        for token in (
            "prefer",
            "do not want",
            "don't want",
            "need",
            "want",
            "like",
            "dislike",
            "expect",
            "value",
            "approval",
            "ordinary action",
            "voice update",
            "progress report",
            "autonomous",
            "proactive",
        )
    ):
        return 1.0
    return 0.0


def _belief_recency_bonus(belief: Belief, *, now: datetime) -> float:
    timestamp = belief.last_reinforced or belief.timestamp
    if timestamp is None:
        return 0.0
    try:
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    except Exception:
        return 0.0
    if age_days <= 1.0:
        return 0.6
    if age_days <= 7.0:
        return 0.35
    if age_days <= 30.0:
        return 0.15
    return 0.0


def _effective_belief_confidence(belief: Belief, *, now: datetime) -> float:
    confidence = float(belief.confidence)
    if belief.valid_until is not None and belief.valid_until < now:
        confidence *= 0.25
    if belief.decay_rate > 0.0:
        anchor = belief.last_reinforced or belief.timestamp
        age_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
        confidence *= max(0.0, 1.0 - (belief.decay_rate * age_days))
    return max(0.0, min(1.0, confidence))
