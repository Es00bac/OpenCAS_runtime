"""Somatic state manager for OpenCAS."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Sequence, Tuple

from opencas.embeddings import EmbeddingService

from .appraisal import AppraisalEventType, SomaticAppraisalEvent
from .models import AffectState, PrimaryEmotion, SomaticSnapshot, SomaticState, SocialTarget
from .store import SomaticStore

# Configurable thresholds for somatic reconciliation
_MASKING_TENSION_THRESHOLD = 0.5
_WARM_VALENCE_THRESHOLD = 0.3
_WARM_AROUSAL_CEILING = 0.6
_NUDGE_MAGNITUDE = 0.04


class SomaticManager:
    """Manages the agent's somatic state with durable persistence."""

    def __init__(
        self,
        state_path: Path | str,
        store: Optional[SomaticStore] = None,
        embeddings: Optional[EmbeddingService] = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.embeddings = embeddings
        self._state = SomaticState()
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self._state = SomaticState.model_validate_json(
                    self.state_path.read_text(encoding="utf-8")
                )
            except (ValueError, OSError):
                self._state = SomaticState()

    def save(self) -> None:
        self._state.updated_at = datetime.now(timezone.utc)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(self._state.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    @property
    def state(self) -> SomaticState:
        return self._state

    async def emit_appraisal_event(
        self,
        event_type: AppraisalEventType,
        source_text: str = "",
        trigger_event_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        snapshot: bool = True,
    ) -> SomaticAppraisalEvent:
        """Emit a typed appraisal event, optionally nudging state and recording a snapshot."""
        affect = self.appraise(source_text) if source_text else None
        event = SomaticAppraisalEvent(
            event_type=event_type,
            source_text=source_text,
            affect_state=affect,
            trigger_event_id=trigger_event_id,
            meta=meta or {},
        )
        if affect is not None:
            self.nudge_from_appraisal(affect)
        if snapshot and self.store is not None:
            await self.record_snapshot(
                source=event_type.value,
                trigger_event_id=str(event.event_id),
            )
        return event

    async def appraise_generated(self, content: str) -> AffectState:
        """Appraise the assistant's own generated response using keyword matching."""
        return self.appraise(content)

    async def reconcile(
        self,
        pre_state: SomaticState,
        expressed_affect: AffectState,
        tom_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compare pre-generation somatic state with post-generation expressed affect.

        Returns a dict describing what was detected and any adjustments made.
        """
        adjustments: Dict[str, Any] = {"masking_detected": False, "adjustments": []}
        felt_tension = pre_state.tension
        felt_certainty = pre_state.certainty
        expressed_valence = expressed_affect.valence
        expressed_arousal = expressed_affect.arousal
        expressed_certainty = expressed_affect.certainty

        # Masking: high internal tension but expressed text claims calm
        if felt_tension > _MASKING_TENSION_THRESHOLD and expressed_valence > _WARM_VALENCE_THRESHOLD and expressed_arousal < _WARM_AROUSAL_CEILING:
            self._state.valence = round(
                max(-1.0, min(1.0, self._state.valence - _NUDGE_MAGNITUDE)), 3
            )
            adjustments["masking_detected"] = True
            adjustments["adjustments"].append("valence_down_masking")
            if tom_engine is not None:
                from opencas.tom.models import BeliefSubject
                await tom_engine.record_belief(
                    BeliefSubject.SELF, "masking anxiety", confidence=0.6
                )

        # Warm expressed affect → tension relief
        if expressed_valence > _WARM_VALENCE_THRESHOLD and expressed_affect.primary_emotion in (
            PrimaryEmotion.CARING, PrimaryEmotion.TRUST, PrimaryEmotion.JOY
        ):
            self._state.tension = round(
                max(0.0, self._state.tension - _NUDGE_MAGNITUDE), 3
            )
            adjustments["adjustments"].append("tension_down_warm")

        # Expressed uncertainty/apology while felt confident → certainty drops
        if felt_certainty > 0.6 and expressed_affect.primary_emotion in (
            PrimaryEmotion.APOLOGETIC, PrimaryEmotion.CONCERNED
        ):
            self._state.certainty = round(
                max(0.0, self._state.certainty - _NUDGE_MAGNITUDE), 3
            )
            adjustments["adjustments"].append("certainty_down_apologetic")

        self.save()

        # Store post-reconciliation snapshot durably
        if self.store is not None:
            await self.record_snapshot(source="somatic_reconciliation")

        return adjustments

    def set_arousal(self, value: float) -> None:
        self._state.arousal = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_fatigue(self, value: float) -> None:
        self._state.fatigue = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_tension(self, value: float) -> None:
        self._state.tension = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_valence(self, value: float) -> None:
        self._state.valence = round(max(-1.0, min(1.0, value)), 3)
        self.save()

    def set_focus(self, value: float) -> None:
        self._state.focus = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_energy(self, value: float) -> None:
        self._state.energy = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_certainty(self, value: float) -> None:
        self._state.certainty = round(max(0.0, min(1.0, value)), 3)
        self.save()

    def set_tag(self, tag: Optional[str]) -> None:
        self._state.somatic_tag = tag
        self.save()

    def decay(self, fatigue_delta: float = 0.02, tension_delta: float = -0.01) -> None:
        """Apply natural decay: fatigue rises under load, recovers at rest; tension relaxes."""
        # Recover fatigue when the agent is idle (low arousal, low tension)
        s = self._state
        if s.arousal < 0.3 and s.tension < 0.3:
            effective_fatigue_delta = -fatigue_delta * 2  # rest recovers twice as fast as load builds
        else:
            effective_fatigue_delta = fatigue_delta
        self._state.fatigue = round(max(0.0, min(1.0, s.fatigue + effective_fatigue_delta)), 3)
        self._state.tension = round(max(0.0, min(1.0, s.tension + tension_delta)), 3)
        self.save()

    def bump_from_work(self, intensity: float = 0.1, success: bool = True) -> None:
        """Update somatic state based on completed work."""
        self._state.fatigue = round(max(0.0, min(1.0, self._state.fatigue + intensity * 0.3)), 3)
        self._state.arousal = round(max(0.0, min(1.0, self._state.arousal + intensity * 0.1)), 3)
        if success:
            self._state.valence = round(max(-1.0, min(1.0, self._state.valence + 0.05)), 3)
            self._state.tension = round(max(0.0, self._state.tension - 0.05), 3)
        else:
            self._state.valence = round(max(-1.0, min(1.0, self._state.valence - 0.05)), 3)
            self._state.tension = round(max(0.0, min(1.0, self._state.tension + 0.1)), 3)
        self.save()

    def nudge_from_appraisal(self, affect: AffectState, intensity_scale: float = 0.25) -> None:
        """Gently nudge live somatic state toward an appraised affect."""
        if affect is None:
            return
        a = intensity_scale
        # Blend valence and arousal toward appraisal
        self._state.valence = round(
            max(-1.0, min(1.0, self._state.valence * (1 - a) + affect.valence * a)), 3
        )
        self._state.arousal = round(
            max(0.0, min(1.0, self._state.arousal * (1 - a) + affect.arousal * a)), 3
        )
        # Tension rises on negative high-arousal emotions, falls on calm positive
        if affect.valence < -0.3 and affect.arousal > 0.5:
            self._state.tension = round(min(1.0, self._state.tension + 0.04), 3)
        elif affect.valence > 0.2 and affect.arousal < 0.6:
            self._state.tension = round(max(0.0, self._state.tension - 0.03), 3)
        # Energy drains slightly on high arousal, recovers on calm positive
        if affect.arousal > 0.7:
            self._state.energy = round(max(0.0, self._state.energy - 0.02), 3)
        elif affect.valence > 0.2 and affect.arousal < 0.5:
            self._state.energy = round(min(1.0, self._state.energy + 0.02), 3)
        # Certainty tracks appraisal certainty
        self._state.certainty = round(
            max(0.0, min(1.0, self._state.certainty * (1 - a) + affect.certainty * a)), 3
        )
        # Tag tracks primary emotion when it's salient
        if affect.intensity > 0.4 and affect.primary_emotion != PrimaryEmotion.NEUTRAL:
            self._state.somatic_tag = affect.primary_emotion.value
        self.save()

    def appraise(self, text: str, outcome: str = "neutral") -> AffectState:
        """Heuristic affect appraisal from text with negation handling."""
        text_lower = text.lower()
        tokens = text_lower.split()

        keyword_map: dict[PrimaryEmotion, set[str]] = {
            PrimaryEmotion.JOY: {
                "happy", "joy", "great", "wonderful", "excellent", "love", "fantastic",
                "delighted", "cheerful", "glad", "pleased", "elated", "bliss", "content",
            },
            PrimaryEmotion.SADNESS: {
                "sad", "sorry", "unfortunate", "depressed", "grief", "melancholy",
                "gloomy", "miserable", "heartbroken", "disappointed", "hopeless",
            },
            PrimaryEmotion.ANGER: {
                "angry", "furious", "rage", "annoyed", "frustrated", "irritated",
                "livid", "outraged", "hostile", "bitter", "resentful",
            },
            PrimaryEmotion.FEAR: {
                "afraid", "scared", "terrified", "anxious", "worried", "nervous",
                "panic", "dread", "horrified", "uneasy", "tense",
            },
            PrimaryEmotion.SURPRISE: {
                "surprised", "shocked", "amazed", "unexpected", "astonished",
                "stunned", "bewildered", "startled",
            },
            PrimaryEmotion.TRUST: {
                "trust", "reliable", "confident", "believe", "faith", "loyal",
                "honest", "sincere", "dependable", "safe",
            },
            PrimaryEmotion.ANTICIPATION: {
                "excited", "looking forward", "eager", "soon", "await", "expect",
                "enthusiastic", "hopeful", "keen", "prepared",
            },
            PrimaryEmotion.DISGUST: {
                "disgusted", "repulsed", "horrible", "awful", "revolted", "nauseated",
                "appalled", "detest", "loathe",
            },
            PrimaryEmotion.CURIOUS: {
                "curious", "wonder", "interested", "question", "intrigued", "fascinated",
                "inquisitive", "explore", "investigate",
            },
            PrimaryEmotion.CONCERNED: {
                "concerned", "worried", "cautious", "uncertain", "apprehensive",
                "doubtful", "hesitant", "uneasy", "alarmed",
            },
            PrimaryEmotion.PROUD: {
                "proud", "accomplished", "achievement", "honored", "triumphant",
                "successful", "confident", "worthy",
            },
            PrimaryEmotion.TIRED: {
                "tired", "exhausted", "weary", "sleepy", "drained", "fatigued",
                "burned out", "spent", "lethargic",
            },
            PrimaryEmotion.DETERMINED: {
                "determined", "committed", "resolve", "will do", "dedicated",
                "persistent", "steadfast", "ambitious", "driven",
            },
            PrimaryEmotion.PLAYFUL: {
                "playful", "fun", "silly", "whimsical", "lighthearted", "teasing",
            },
            PrimaryEmotion.CARING: {
                "caring", "kind", "compassionate", "gentle", "supportive", "nurturing",
            },
            PrimaryEmotion.APOLOGETIC: {
                "apologize", "sorry", "regret", "repentant", "remorseful", "forgive",
            },
            PrimaryEmotion.ANNOYED: {
                "irritated", "bothered", "irked", "peeved", "aggravated", "impatient",
            },
        }

        negation_words = {"not", "no", "never", "n't", "none", "nothing", "nobody", "neither", "nowhere", "hardly", "scarcely", "barely", "don", "doesn", "didn", "wasn", "weren", "won", "wouldn", "couldn", "shouldn", "isn", "aren", "haven", "hasn", "hadn"}

        def _is_negated(token_index: int) -> bool:
            window = tokens[max(0, token_index - 4):token_index]
            return any(t in negation_words for t in window)

        scores: dict[PrimaryEmotion, float] = {emotion: 0.0 for emotion in PrimaryEmotion}
        for emotion, keywords in keyword_map.items():
            for kw in keywords:
                idx = text_lower.find(kw)
                while idx != -1:
                    # Map character index to token index roughly
                    prefix = text_lower[:idx]
                    token_index = len(prefix.split())
                    if _is_negated(token_index):
                        scores[emotion] -= 0.5
                    else:
                        scores[emotion] += 1.0
                    idx = text_lower.find(kw, idx + len(kw))

        primary = max(scores, key=scores.get)
        if scores[primary] <= 0:
            primary = PrimaryEmotion.NEUTRAL

        # Compute blended valence and arousal from all matched emotions weighted by score
        valence_map: dict[PrimaryEmotion, float] = {
            PrimaryEmotion.JOY: 0.8,
            PrimaryEmotion.TRUST: 0.6,
            PrimaryEmotion.ANTICIPATION: 0.4,
            PrimaryEmotion.SURPRISE: 0.2,
            PrimaryEmotion.SADNESS: -0.7,
            PrimaryEmotion.FEAR: -0.6,
            PrimaryEmotion.ANGER: -0.7,
            PrimaryEmotion.DISGUST: -0.6,
            PrimaryEmotion.NEUTRAL: 0.0,
            PrimaryEmotion.EXCITED: 0.7,
            PrimaryEmotion.PLAYFUL: 0.5,
            PrimaryEmotion.CURIOUS: 0.3,
            PrimaryEmotion.FOCUSED: 0.2,
            PrimaryEmotion.THOUGHTFUL: 0.1,
            PrimaryEmotion.CONCERNED: -0.3,
            PrimaryEmotion.CARING: 0.4,
            PrimaryEmotion.APOLOGETIC: -0.1,
            PrimaryEmotion.ANNOYED: -0.4,
            PrimaryEmotion.PROUD: 0.7,
            PrimaryEmotion.TIRED: -0.2,
            PrimaryEmotion.DETERMINED: 0.3,
        }
        arousal_map: dict[PrimaryEmotion, float] = {
            PrimaryEmotion.JOY: 0.7,
            PrimaryEmotion.TRUST: 0.4,
            PrimaryEmotion.ANTICIPATION: 0.6,
            PrimaryEmotion.SURPRISE: 0.8,
            PrimaryEmotion.SADNESS: 0.2,
            PrimaryEmotion.FEAR: 0.8,
            PrimaryEmotion.ANGER: 0.9,
            PrimaryEmotion.DISGUST: 0.5,
            PrimaryEmotion.NEUTRAL: 0.3,
            PrimaryEmotion.EXCITED: 0.9,
            PrimaryEmotion.PLAYFUL: 0.6,
            PrimaryEmotion.CURIOUS: 0.5,
            PrimaryEmotion.FOCUSED: 0.5,
            PrimaryEmotion.THOUGHTFUL: 0.3,
            PrimaryEmotion.CONCERNED: 0.4,
            PrimaryEmotion.CARING: 0.4,
            PrimaryEmotion.APOLOGETIC: 0.2,
            PrimaryEmotion.ANNOYED: 0.5,
            PrimaryEmotion.PROUD: 0.6,
            PrimaryEmotion.TIRED: 0.1,
            PrimaryEmotion.DETERMINED: 0.6,
        }

        total_score = sum(max(0.0, s) for s in scores.values())
        if total_score > 0:
            valence = sum(
                max(0.0, scores[e]) * valence_map.get(e, 0.0) for e in PrimaryEmotion
            ) / total_score
            arousal = sum(
                max(0.0, scores[e]) * arousal_map.get(e, 0.0) for e in PrimaryEmotion
            ) / total_score
            intensity = min(1.0, 0.2 + (scores[primary] / total_score) * 0.6 + (total_score * 0.05))
        else:
            valence = 0.0
            arousal = 0.5
            intensity = 0.0

        certainty = 0.5 + (0.3 if total_score > 2 else 0.0)
        certainty = min(1.0, certainty)

        if outcome == "positive":
            valence = min(1.0, valence + 0.2)
            arousal = min(1.0, arousal + 0.05)
        elif outcome == "negative":
            valence = max(-1.0, valence - 0.2)
            arousal = min(1.0, arousal + 0.1)

        return AffectState(
            primary_emotion=primary,
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            certainty=round(certainty, 3),
            intensity=round(intensity, 3),
            social_target=SocialTarget.USER,
            emotion_tags=[primary.value] if primary != PrimaryEmotion.NEUTRAL else [],
        )

    # ------------------------------------------------------------------
    # Snapshot history and embedding integration
    # ------------------------------------------------------------------

    def _snapshot_from_state(
        self,
        source: str = "unknown",
        trigger_event_id: Optional[str] = None,
        musubi: Optional[float] = None,
    ) -> SomaticSnapshot:
        """Build a snapshot from the current live state with nuanced emotion inference."""
        s = self._state
        primary = PrimaryEmotion.NEUTRAL

        # Tension-dominant states
        if s.tension > 0.6:
            if s.arousal > 0.7:
                primary = PrimaryEmotion.ANGER
            elif s.arousal > 0.4:
                primary = PrimaryEmotion.FEAR
            else:
                primary = PrimaryEmotion.CONCERNED
        # Fatigue-dominant states
        elif s.fatigue > 0.7:
            primary = PrimaryEmotion.TIRED
        # Positive valence quadrant
        elif s.valence > 0.3:
            if s.arousal > 0.7:
                primary = PrimaryEmotion.EXCITED
            elif s.arousal > 0.5:
                primary = PrimaryEmotion.JOY
            elif s.focus > 0.6 and s.energy > 0.5:
                primary = PrimaryEmotion.DETERMINED
            else:
                primary = PrimaryEmotion.TRUST
        # Negative valence quadrant
        elif s.valence < -0.3:
            if s.arousal > 0.6:
                primary = PrimaryEmotion.ANGER
            else:
                primary = PrimaryEmotion.SADNESS
        # High arousal, neutral valence
        elif s.arousal > 0.7:
            primary = PrimaryEmotion.ANTICIPATION if s.focus > 0.5 else PrimaryEmotion.SURPRISE
        # Focus/energy patterns when valence is near-neutral
        elif s.focus > 0.7 and s.energy > 0.5:
            primary = PrimaryEmotion.FOCUSED
        elif s.focus > 0.5 and s.energy <= 0.4:
            primary = PrimaryEmotion.THOUGHTFUL
        elif s.certainty < 0.3:
            primary = PrimaryEmotion.CONCERNED

        return SomaticSnapshot(
            arousal=s.arousal,
            fatigue=s.fatigue,
            tension=s.tension,
            valence=s.valence,
            focus=s.focus,
            energy=s.energy,
            certainty=s.certainty,
            musubi=musubi,
            primary_emotion=primary,
            somatic_tag=s.somatic_tag,
            source=source,
            trigger_event_id=trigger_event_id,
        )

    async def record_snapshot(
        self,
        source: str = "unknown",
        trigger_event_id: Optional[str] = None,
        musubi: Optional[float] = None,
    ) -> Optional[SomaticSnapshot]:
        """Persist current state as a snapshot, embedding it if wired."""
        if self.store is None:
            return None

        snapshot = self._snapshot_from_state(source, trigger_event_id, musubi=musubi)

        # Simple dedup: if the latest snapshot is within 30s and identical core dims, skip
        latest = await self.store.get_latest()
        if latest is not None:
            age_seconds = (snapshot.recorded_at - latest.recorded_at).total_seconds()
            if age_seconds < 30:
                if (
                    round(snapshot.arousal, 3) == round(latest.arousal, 3)
                    and round(snapshot.fatigue, 3) == round(latest.fatigue, 3)
                    and round(snapshot.tension, 3) == round(latest.tension, 3)
                    and round(snapshot.valence, 3) == round(latest.valence, 3)
                ):
                    return latest

        if self.embeddings is not None:
            snapshot = await self.embed_snapshot(snapshot)

        await self.store.save(snapshot)
        return snapshot

    async def embed_snapshot(self, snapshot: SomaticSnapshot) -> SomaticSnapshot:
        """Embed the snapshot's canonical prose and update embedding_id."""
        if self.embeddings is None:
            return snapshot
        record = await self.embeddings.embed(
            snapshot.to_canonical_text(),
            meta={
                "source": "somatic_snapshot",
                "snapshot_id": str(snapshot.snapshot_id),
                "primary_emotion": snapshot.primary_emotion.value,
            },
            task_type="somatic_snapshot",
        )
        snapshot.embedding_id = record.source_hash
        return snapshot

    async def find_similar_periods(
        self,
        limit: int = 5,
    ) -> List[Tuple[SomaticSnapshot, float]]:
        """Find past snapshots with similar affective embeddings."""
        if self.store is None or self.embeddings is None:
            return []

        latest = await self.store.get_latest()
        if latest is None:
            return []

        query_embed = await self.embeddings.embed(
            latest.to_canonical_text(),
            task_type="somatic_query",
        )
        candidates = await self.embeddings.cache.search_similar(
            query_embed.vector, limit=limit * 3
        )
        if not candidates:
            return []

        # Filter to only records tagged as somatic_snapshot and resolve via store
        hash_to_sim = {record.source_hash: sim for record, sim in candidates}
        snapshots = await self.store.list_recent(limit=limit * 4)
        results: List[Tuple[SomaticSnapshot, float]] = []
        for snap in snapshots:
            if snap.embedding_id and snap.embedding_id in hash_to_sim:
                # Exclude exact self-match by ID
                if str(snap.snapshot_id) != str(latest.snapshot_id):
                    results.append((snap, hash_to_sim[snap.embedding_id]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
