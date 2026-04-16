"""Relational resonance (musubi) engine for OpenCAS."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.memory.models import Episode
from opencas.telemetry import EventKind, Tracer

from .models import MusubiRecord, MusubiState, ResonanceDimension
from .store import MusubiStore


class RelationalEngine:
    """Tracks and evolves the relational field between agent and user."""

    def __init__(
        self,
        store: MusubiStore,
        tracer: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.tracer = tracer
        self._state: Optional[MusubiState] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> "RelationalEngine":
        await self.store.connect()
        self._state = await self.store.load_state()
        if self._state is None:
            self._state = MusubiState()
            await self.store.save_state(self._state)
        return self

    async def close(self) -> None:
        await self.store.close()

    @property
    def state(self) -> MusubiState:
        assert self._state is not None
        return self._state

    async def import_partner_state(
        self,
        user_id: str,
        trust: float,
        musubi: float,
        warmth: Optional[float] = None,
    ) -> MusubiState:
        """Seed relational state from an imported partner profile."""
        # Derive dimensions from the partner trust/musubi scores
        resonance = warmth if warmth is not None else max(0.0, musubi * 0.8)
        presence = max(0.0, musubi * 0.6)
        attunement = max(0.0, trust * 0.7)
        return await self.initialize(
            trust=trust,
            resonance=resonance,
            presence=presence,
            attunement=attunement,
            note=f"imported partner state for {user_id}",
        )

    async def initialize(
        self,
        trust: float = 0.0,
        resonance: float = 0.0,
        presence: float = 0.0,
        attunement: float = 0.0,
        note: Optional[str] = None,
    ) -> MusubiState:
        """Set the baseline musubi state."""
        prev = self._derive_musubi_from_dimensions()
        self._state = MusubiState(
            dimensions={
                ResonanceDimension.TRUST.value: max(-1.0, min(1.0, trust)),
                ResonanceDimension.RESONANCE.value: max(-1.0, min(1.0, resonance)),
                ResonanceDimension.PRESENCE.value: max(-1.0, min(1.0, presence)),
                ResonanceDimension.ATTUNEMENT.value: max(-1.0, min(1.0, attunement)),
            },
            source_tag="initialize",
        )
        self._state.musubi = self._derive_musubi_from_dimensions(self._state.dimensions)
        await self.store.save_state(self._state)
        record = MusubiRecord(
            musubi_before=prev,
            musubi_after=self._state.musubi,
            delta=round(self._state.musubi - prev, 4),
            dimension_deltas=dict(self._state.dimensions),
            trigger_event="initialize",
            note=note,
        )
        await self.store.append_record(record)
        self._trace("relational_initialized", {
            "musubi": self._state.musubi,
            "dimensions": self._state.dimensions,
        })
        return self._state

    async def heartbeat(self, session_active: bool = False) -> MusubiState:
        """Adjust presence based on whether contact is currently happening."""
        assert self._state is not None
        prev_musubi = self._state.musubi
        delta: Dict[str, float] = {}
        if session_active:
            delta[ResonanceDimension.PRESENCE.value] = 0.05
        else:
            delta[ResonanceDimension.PRESENCE.value] = -0.03
        await self._apply_deltas(delta, "heartbeat", f"session_active={session_active}")
        self._trace("heartbeat", {"session_active": session_active, "musubi": self._state.musubi})
        return self._state

    async def record_interaction(
        self,
        episode: Episode,
        outcome: str = "neutral",
    ) -> MusubiState:
        """Evaluate an episode for musubi impact."""
        assert self._state is not None
        delta: Dict[str, float] = {}
        dims = self._state.dimensions

        # Base presence bump from any interaction
        delta[ResonanceDimension.PRESENCE.value] = 0.05

        # Attunement shift from episode valence / somatic_tag interpretation
        # If the episode has a positive somatic_tag, attunement rises slightly.
        attunement_shift = 0.0
        if episode.somatic_tag:
            positive_tags = {"joy", "anticipation", "trust", "calm", "excited"}
            negative_tags = {"anger", "sadness", "fear", "disgust", "boredom"}
            tag_lower = episode.somatic_tag.lower()
            if any(t in tag_lower for t in positive_tags):
                attunement_shift = 0.04
            elif any(t in tag_lower for t in negative_tags):
                attunement_shift = -0.04
        delta[ResonanceDimension.ATTUNEMENT.value] = attunement_shift

        # Resonance from outcome
        resonance_shift = 0.0
        if outcome == "positive":
            resonance_shift = 0.05
        elif outcome == "negative":
            resonance_shift = -0.05
        elif outcome == "creative_collab":
            resonance_shift = 0.08
        delta[ResonanceDimension.RESONANCE.value] = resonance_shift

        # Trust is only nudged by explicit boundary or reliability signals
        trust_shift = 0.0
        if outcome == "boundary_respected":
            trust_shift = 0.06
        elif outcome == "boundary_violated":
            trust_shift = -0.10
        delta[ResonanceDimension.TRUST.value] = trust_shift

        note = f"episode_kind={episode.kind.value}, outcome={outcome}"
        await self._apply_deltas(
            delta,
            "interaction",
            note=note,
            episode_id=str(episode.episode_id),
        )
        self._trace("record_interaction", {
            "episode_id": str(episode.episode_id),
            "outcome": outcome,
            "musubi": self._state.musubi,
        })
        return self._state

    async def record_creative_collab(self, success: bool = True) -> MusubiState:
        """Record the impact of creative collaboration."""
        assert self._state is not None
        delta = {
            ResonanceDimension.RESONANCE.value: 0.10 if success else -0.05,
            ResonanceDimension.ATTUNEMENT.value: 0.05 if success else -0.02,
        }
        await self._apply_deltas(
            delta,
            "creative_collab",
            note=f"success={success}",
        )
        self._trace("creative_collab", {"success": success, "musubi": self._state.musubi})
        return self._state

    async def record_boundary_respected(self, respected: bool = True) -> MusubiState:
        """Update trust based on whether a boundary was respected or violated."""
        assert self._state is not None
        delta = {
            ResonanceDimension.TRUST.value: 0.08 if respected else -0.12,
        }
        await self._apply_deltas(
            delta,
            "boundary_respected" if respected else "boundary_violated",
        )
        self._trace("boundary_respected", {"respected": respected, "musubi": self._state.musubi})
        return self._state

    def to_memory_salience_modifier(self, has_user_collab_tag: bool = False) -> float:
        """Return a modifier to apply to memory salience based on musubi.

        High musubi boosts collaborative memories; low musubi demotes unrelated ones.
        """
        assert self._state is not None
        musubi = self._state.musubi
        if has_user_collab_tag and musubi > 0.3:
            return min(0.35, musubi * 0.4)
        if musubi < -0.3:
            return max(-0.35, musubi * 0.25)
        return 0.0

    def to_creative_boost(self, aligns_with_shared_goals: bool = False) -> float:
        """Return a creative-ladder promotion boost based on musubi."""
        assert self._state is not None
        musubi = self._state.musubi
        if aligns_with_shared_goals and musubi > 0.2:
            return min(0.30, musubi * 0.35)
        if musubi > 0.5:
            return 0.10
        return 0.0

    def to_approval_risk_modifier(self) -> float:
        """Return a risk-appetite modifier for self-approval.

        Positive musubi expands the agent's comfort zone;
        very negative musubi increases caution.
        """
        assert self._state is not None
        musubi = self._state.musubi
        if musubi > 0.6:
            return min(0.25, musubi * 0.3)
        if musubi < -0.5:
            return max(-0.25, musubi * 0.3)
        return 0.0

    def to_promise_priority_boost(self, user_facing: bool = False) -> float:
        """Return a bounded priority nudge for promise-follow-through.

        High trust, attunement, and musubi should slightly increase how strongly
        explicit user-facing promises compete against otherwise similar work.
        """
        assert self._state is not None
        musubi = self._state.musubi
        trust = self._state.dimensions.get(ResonanceDimension.TRUST.value, 0.0)
        attunement = self._state.dimensions.get(ResonanceDimension.ATTUNEMENT.value, 0.0)

        if not user_facing:
            return round(max(-0.05, min(0.05, musubi * 0.05)), 4)

        boost = (musubi * 0.08) + (trust * 0.08) + (attunement * 0.04)
        return round(max(-0.12, min(0.18, boost)), 4)

    async def _apply_deltas(
        self,
        deltas: Dict[str, float],
        trigger: str,
        note: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            assert self._state is not None
            prev_musubi = self._state.musubi
            applied: Dict[str, float] = {}
            for key, val in deltas.items():
                current = self._state.dimensions.get(key, 0.0)
                new_val = max(-1.0, min(1.0, current + val))
                self._state.dimensions[key] = new_val
                applied[key] = round(new_val - current, 4)
            self._state.musubi = self._derive_musubi_from_dimensions(self._state.dimensions)
            self._state.updated_at = datetime.now(timezone.utc)
            self._state.source_tag = trigger
            await self.store.save_state(self._state)
            record = MusubiRecord(
                musubi_before=prev_musubi,
                musubi_after=self._state.musubi,
                delta=round(self._state.musubi - prev_musubi, 4),
                dimension_deltas=applied,
                trigger_event=trigger,
                note=note,
                episode_id=episode_id,
            )
            await self.store.append_record(record)

    @staticmethod
    def _derive_musubi_from_dimensions(dimensions: Optional[Dict[str, float]] = None) -> float:
        dims = dimensions or {}
        trust = dims.get(ResonanceDimension.TRUST.value, 0.0)
        resonance = dims.get(ResonanceDimension.RESONANCE.value, 0.0)
        presence = dims.get(ResonanceDimension.PRESENCE.value, 0.0)
        attunement = dims.get(ResonanceDimension.ATTUNEMENT.value, 0.0)
        raw = 0.30 * trust + 0.25 * resonance + 0.25 * presence + 0.20 * attunement
        return round(max(-1.0, min(1.0, raw)), 4)

    @staticmethod
    def _clamp_delta(value: float) -> float:
        return round(max(-1.0, min(1.0, value)), 4)

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer and hasattr(self.tracer, "log"):
            self.tracer.log(
                EventKind.TOM_EVAL,
                f"RelationalEngine: {event}",
                payload,
            )
