"""Hybrid memory retrieval: multi-signal fusion with MMR and temporal decay."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from opencas.telemetry import EventKind
from opencas.embeddings import EmbeddingService
from opencas.memory import MemoryStore
from opencas.somatic.models import AffectState, PrimaryEmotion

from .cognitive_retrieval import apply_cognitive_state_to_candidates
from .models import RetrievalResult
from .retrieval_candidates import (
    build_candidate_map,
    expand_candidate_graph,
    fuse_candidates,
    normalize_candidate_signals,
    resolve_fusion_weights,
    seed_relational_scores,
)
from .retrieval_mmr import rerank_mmr
from .retrieval_query import (
    KEYWORD_STOPWORDS,
    detect_personal_recall_intent,
    detect_temporal_intent,
    extract_anchor_terms,
    keyword_queries_for,
)
from .retrieval_ranking import apply_diversity_penalty, apply_temporal_decay
from .retrieval_search import (
    emotion_boost,
    exact_handle_search,
    reciprocal_rank_fusion,
)
from .retrieval_search import (
    expand_graph as expand_graph_search,
)
from .retrieval_search import (
    keyword_search as keyword_search_impl,
)
from .retrieval_search import (
    semantic_search as semantic_search_impl,
)

if TYPE_CHECKING:
    from opencas.memory.fabric.graph import EpisodeGraph
    from opencas.relational import RelationalEngine
    from opencas.somatic import SomaticManager


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _retrieval_context_metadata(result: RetrievalResult) -> Dict[str, Any]:
    episode = getattr(result, "episode", None)
    if episode is not None:
        payload = dict(getattr(episode, "payload", {}) or {})
        kind = _enum_value(getattr(episode, "kind", ""))
        role = str(payload.get("role") or "").lower()
        origin = " ".join(
            str(payload.get(key) or "").lower()
            for key in ("origin", "source", "activation_source", "proposal_kind", "context_material")
        )
        lane = str(
            payload.get("source_lane")
            or payload.get("origin_context_lane")
            or payload.get("context_lane")
            or ""
        ).lower()
        inferred = False
        lane_source = "payload"
        if not lane:
            inferred = True
            lane_source = "kind_role_heuristic"
            if any(token in origin for token in ("daydream", "reflect", "proposal", "association")):
                lane = "reflective"
            elif kind in {"artifact", "compaction", "consolidation", "procedural"}:
                lane = "reflective"
            elif kind in {"turn", "action", "observation"} or role:
                lane = "executive"
            else:
                lane = "unknown"
        material = str(payload.get("context_material") or "").strip()
        if not material:
            material = "artifact" if kind == "artifact" else f"episode_{kind or 'record'}"
        authority = str(payload.get("context_authority") or payload.get("authority") or "").strip()
        if not authority:
            authority = "interpretation" if lane == "reflective" else "live_observation"
        return {
            "source_lane": lane,
            "source_lane_inferred": inferred,
            "lane_source": lane_source,
            "authority": authority,
            "context_material": material,
        }
    if getattr(result, "memory", None) is not None:
        return {
            "source_lane": "database",
            "source_lane_inferred": True,
            "lane_source": "memory_material",
            "authority": "retrieved_memory",
            "context_material": "distilled_memory",
        }
    return {}


class MemoryRetriever:
    """Retrieve relevant context by fusing semantic, keyword, recency, salience, and graph signals."""

    DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
        "semantic_score": 0.30,
        "keyword_score": 0.20,
        "exact_handle_score": 0.45,
        "recency_score": 0.15,
        "salience_score": 0.10,
        "graph_score": 0.10,
        "emotional_resonance": 0.08,
        "temporal_echo": 0.04,
        "reliability": 0.03,
        "relational_score": 0.08,
        "affective_pressure_score": 0.04,
        "cognitive_focus_score": 0.05,
        "cognitive_working_memory_score": 0.05,
        "cognitive_prospective_score": 0.04,
        "cognitive_event_score": 0.04,
    }
    RECALL_FUSION_WEIGHTS: Dict[str, float] = {
        "semantic_score": 0.20,
        "keyword_score": 0.34,
        "exact_handle_score": 0.45,
        "recency_score": 0.12,
        "salience_score": 0.08,
        "graph_score": 0.08,
        "emotional_resonance": 0.08,
        "temporal_echo": 0.04,
        "reliability": 0.03,
        "relational_score": 0.08,
        "affective_pressure_score": 0.04,
        "cognitive_focus_score": 0.06,
        "cognitive_working_memory_score": 0.06,
        "cognitive_prospective_score": 0.05,
        "cognitive_event_score": 0.05,
    }
    def __init__(
        self,
        memory: MemoryStore,
        embeddings: EmbeddingService,
        rrf_k: int = 60,
        episode_graph: Optional["EpisodeGraph"] = None,
        somatic_manager: Optional["SomaticManager"] = None,
        relational_engine: Optional["RelationalEngine"] = None,
        affective_examinations: Optional[Any] = None,
        cognitive_state_store: Optional[Any] = None,
        tracer: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.episode_graph = episode_graph
        self.somatic_manager = somatic_manager
        self.relational_engine = relational_engine
        self.affective_examinations = affective_examinations
        self.cognitive_state_store = cognitive_state_store
        self.tracer = tracer

    @staticmethod
    def extract_anchor_terms(query: str) -> List[str]:
        return extract_anchor_terms(query)

    @staticmethod
    def detect_personal_recall_intent(query: str) -> bool:
        return detect_personal_recall_intent(query)

    @staticmethod
    def detect_temporal_intent(query: str) -> Optional[str]:
        return detect_temporal_intent(query)

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        expand_graph: bool = True,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
        emotion_boost_tag: Optional[str] = None,
        emotion_boost_value: float = 0.0,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
        identity: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Return top-k relevant memories/episodes using multi-signal fusion.

        If ``identity`` is provided, checks for mutagenic memories in the
        results and triggers at most one identity mutation per call.
        """
        inspection = await self.inspect(
            query=query,
            session_id=session_id,
            limit=limit,
            offset=offset,
            expand_graph=expand_graph,
            affect_query=affect_query,
            affect_weight=affect_weight,
            emotion_boost_tag=emotion_boost_tag,
            emotion_boost_value=emotion_boost_value,
            min_confidence=min_confidence,
            lambda_param=lambda_param,
        )
        results = inspection["results"]

        # Phase 6: Throttled identity mutation from mutagenic memories (max 1 per call)
        if identity is not None:
            for result in results:
                if hasattr(result, "payload") and isinstance(result.payload, dict):
                    if result.payload.get("identity_mutagen"):
                        identity.apply_memory_mutation(
                            content=result.text[:200],
                            source_type=result.payload.get("source_type", "memory"),
                            confidence=result.score if hasattr(result, "score") else 0.7,
                        )
                        break  # Only one mutation per retrieval call

        return results

    async def inspect(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        expand_graph: bool = True,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
        emotion_boost_tag: Optional[str] = None,
        emotion_boost_value: float = 0.0,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Return retrieval results plus score breakdowns for operator inspection."""
        inspect_started = time.perf_counter()
        phase_started = time.perf_counter()
        semantic_results = await self._semantic_search(
            query, limit=limit * 3, affect_query=affect_query, affect_weight=affect_weight
        )
        self._trace_retrieval_phase(
            "semantic_search",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            result_count=len(semantic_results),
            limit=limit * 3,
        )
        phase_started = time.perf_counter()
        keyword_results = await self._keyword_search(query, limit=limit * 3)
        self._trace_retrieval_phase(
            "keyword_search",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            result_count=len(keyword_results),
            limit=limit * 3,
        )
        phase_started = time.perf_counter()
        exact_handle_results = await self._exact_handle_search(query, limit=limit * 3)
        self._trace_retrieval_phase(
            "exact_handle_search",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            result_count=len(exact_handle_results),
            limit=limit * 3,
        )

        now = datetime.now(timezone.utc)

        # Resolve live somatic state for query-time adjustments
        adj = None
        query_affect = affect_query
        if self.somatic_manager is not None:
            from opencas.somatic.modulators import SomaticModulators
            modulators = SomaticModulators(self.somatic_manager.state)
            adj = modulators.to_retrieval_adjustment()
            if query_affect is None:
                query_affect = AffectState(
                    primary_emotion=modulators._infer_primary_emotion(),
                    valence=self.somatic_manager.state.valence,
                    arousal=self.somatic_manager.state.arousal,
                    intensity=self.somatic_manager.state.certainty,
                )
        elif query_affect is None:
            query_affect = await self._infer_query_affect_from_recent_episodes(session_id)

        phase_started = time.perf_counter()
        candidate_map = build_candidate_map(
            semantic_results,
            keyword_results,
            exact_handle_results,
            now=now,
            query_affect=query_affect,
        )
        seed_relational_scores(self, candidate_map)
        keys = normalize_candidate_signals(candidate_map)
        self._trace_retrieval_phase(
            "candidate_build",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            candidate_count=len(candidate_map),
        )

        if expand_graph:
            phase_started = time.perf_counter()
            keys = await expand_candidate_graph(self, candidate_map, now=now)
            self._trace_retrieval_phase(
                "graph_expand",
                phase_started,
                session_id=session_id,
                cumulative_started=inspect_started,
                candidate_count=len(candidate_map),
                key_count=len(keys),
            )

        phase_started = time.perf_counter()
        await self._apply_affective_pressure_to_candidates(
            candidate_map,
            session_id=session_id,
        )
        self._trace_retrieval_phase(
            "affective_pressure",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            candidate_count=len(candidate_map),
        )
        phase_started = time.perf_counter()
        await apply_cognitive_state_to_candidates(
            self.cognitive_state_store,
            candidate_map,
            query=query,
            session_id=session_id,
        )
        self._trace_retrieval_phase(
            "cognitive_state",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            candidate_count=len(candidate_map),
        )

        phase_started = time.perf_counter()
        resolved_weights = dict(self.DEFAULT_FUSION_WEIGHTS)
        if self.detect_personal_recall_intent(query):
            resolved_weights = resolve_fusion_weights(
                resolved_weights,
                self.RECALL_FUSION_WEIGHTS,
            )
        resolved_weights = resolve_fusion_weights(resolved_weights, weights)
        fused, candidate_debug = fuse_candidates(
            self,
            candidate_map,
            keys=keys,
            now=now,
            weights=resolved_weights,
            adjustment=adj,
            min_confidence=min_confidence,
        )
        self._trace_retrieval_phase(
            "candidate_fusion",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            fused_count=len(fused),
            candidate_count=len(candidate_map),
        )

        fused.sort(key=lambda r: r.score, reverse=True)

        if emotion_boost_tag and emotion_boost_value:
            fused = self._apply_emotion_boost(
                fused, emotion_boost_tag, emotion_boost_value, query_affect
            )

        # Diversity penalty
        phase_started = time.perf_counter()
        fused = apply_diversity_penalty(fused, window=5, penalty=0.05)

        # MMR rerank
        fused = await self._mmr_rerank(fused, lambda_param=lambda_param, limit=(offset + limit) * 2)
        self._trace_retrieval_phase(
            "diversity_mmr",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            fused_count=len(fused),
            limit=(offset + limit) * 2,
        )
        total_count = len(fused)
        fused = fused[offset : offset + limit]
        selected_ids = {(item.source_type, item.source_id) for item in fused}
        for item in candidate_debug:
            item["selected"] = (item["source_type"], item["source_id"]) in selected_ids

        phase_started = time.perf_counter()
        await self._record_retrieval_examinations(
            fused,
            session_id=session_id,
        )
        await self._record_memory_activation_events(
            fused,
            query=query,
            session_id=session_id,
        )
        self._trace_retrieval_phase(
            "retrieval_side_effects",
            phase_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            selected_count=len(fused),
        )

        candidate_debug.sort(key=lambda item: item["final_score"], reverse=True)
        self._trace_retrieval_phase(
            "retrieval_total",
            inspect_started,
            session_id=session_id,
            cumulative_started=inspect_started,
            selected_count=len(fused),
            total_count=total_count,
        )
        return {
            "results": fused,
            "candidates": candidate_debug,
            "weights": resolved_weights,
            "meta": {
                "query": query,
                "expand_graph": expand_graph,
                "limit": limit,
                "offset": offset,
                "total_count": total_count,
                "min_confidence": min_confidence,
                "lambda_param": lambda_param,
                "emotion_boost_tag": emotion_boost_tag,
                "emotion_boost_value": emotion_boost_value,
                "semantic_seed_count": len(semantic_results),
                "keyword_seed_count": len(keyword_results),
                "exact_handle_seed_count": len(exact_handle_results),
            },
        }

    def _trace_retrieval_phase(
        self,
        phase: str,
        started_at: float,
        *,
        session_id: Optional[str],
        cumulative_started: Optional[float] = None,
        **extra: Any,
    ) -> None:
        log = getattr(self.tracer, "log", None)
        if not callable(log):
            return
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "subsystem": "memory_retrieval",
            "phase": phase,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        }
        if cumulative_started is not None:
            payload["cumulative_ms"] = int((time.perf_counter() - cumulative_started) * 1000)
        payload.update(extra)
        try:
            log(EventKind.TOM_EVAL, f"MemoryRetriever phase: {phase}", payload)
        except Exception:
            pass

    async def _record_memory_activation_events(
        self,
        results: List[RetrievalResult],
        *,
        query: str,
        session_id: Optional[str],
        limit: int = 12,
    ) -> None:
        """Emit live atlas activity for memories that fire during retrieval."""

        recorder = getattr(self.tracer, "activate_memory_node", None)
        if not callable(recorder):
            return
        for rank, result in enumerate(results[:limit], start=1):
            source_type = str(getattr(result, "source_type", "") or "")
            source_id = str(getattr(result, "source_id", "") or "")
            if source_type not in {"episode", "memory"} or not source_id:
                continue
            node_id = f"{source_type}:{source_id}"
            content = " ".join(str(getattr(result, "content", "") or "").split())
            if len(content) > 180:
                content = content[:177].rstrip() + "..."
            context_meta = _retrieval_context_metadata(result)
            try:
                maybe = recorder(
                    node_id=node_id,
                    source_type=source_type,
                    source_id=source_id,
                    activation_source="retriever",
                    query=query,
                    score=float(getattr(result, "score", 0.0) or 0.0),
                    rank=rank,
                    session_id=session_id,
                    content_preview=content,
                    extra=context_meta,
                )
                if hasattr(maybe, "__await__"):
                    await maybe
            except Exception:
                continue

    async def _apply_affective_pressure_to_candidates(
        self,
        candidate_map: Dict[Tuple[str, str], Dict[str, Any]],
        *,
        session_id: Optional[str],
    ) -> None:
        loader = getattr(self.affective_examinations, "list_unresolved_pressures", None)
        if not callable(loader):
            return
        try:
            records = await loader(session_id=session_id, limit=50)
        except Exception:
            return
        if not records:
            return

        for key, candidate in candidate_map.items():
            source_type, source_id = key
            match = self._matching_affective_pressure(records, source_type, source_id)
            if match is None:
                continue
            action_pressure = getattr(match, "action_pressure", "")
            pressure = str(getattr(action_pressure, "value", action_pressure))
            if pressure in {"", "archive_only", "continue"}:
                continue
            confidence = float(getattr(match, "confidence", 0.0) or 0.0)
            intensity = float(getattr(match, "intensity", 0.0) or 0.0)
            candidate["affective_pressure_score"] = min(1.0, max(0.0, (confidence + intensity) / 2.0))
            candidate["affective_action_pressure"] = pressure
            candidate["affective_pressure_reason"] = str(getattr(match, "bounded_reason", "") or "")

    @staticmethod
    def _matching_affective_pressure(
        records: List[Any],
        source_type: str,
        source_id: str,
    ) -> Optional[Any]:
        scoped_id = f"{source_type}:{source_id}"
        for record in records:
            record_source_id = str(getattr(record, "source_id", "") or "")
            meta = getattr(record, "meta", {}) if isinstance(getattr(record, "meta", None), dict) else {}
            if record_source_id in {source_id, scoped_id}:
                return record
            if str(meta.get("memory_source_id") or "") == source_id:
                record_type = str(meta.get("memory_source_type") or "")
                if not record_type or record_type == source_type:
                    return record
        return None

    async def _record_retrieval_examinations(
        self,
        results: List[RetrievalResult],
        *,
        session_id: Optional[str],
        limit: int = 3,
    ) -> None:
        recorder = getattr(self.affective_examinations, "examine_memory_retrieval", None)
        if not callable(recorder):
            return
        for result in results[:limit]:
            episode = getattr(result, "episode", None)
            affect = getattr(episode, "affect", None)
            if affect is None:
                continue
            if getattr(affect, "primary_emotion", PrimaryEmotion.NEUTRAL) == PrimaryEmotion.NEUTRAL:
                continue
            if float(getattr(affect, "intensity", 0.0) or 0.0) < 0.6:
                continue
            try:
                await recorder(
                    session_id=session_id,
                    source_type=result.source_type,
                    source_id=result.source_id,
                    content=result.content,
                    affect=affect,
                    meta={"retrieval_score": result.score},
                )
            except Exception:
                continue

    async def _infer_query_affect_from_recent_episodes(
        self,
        session_id: Optional[str],
    ) -> Optional[AffectState]:
        """Infer a durable emotional anchor from recently stored episodes."""
        episodes = await self.memory.list_recent_episodes(session_id=session_id, limit=20)
        if not episodes and session_id is not None:
            episodes = await self.memory.list_recent_episodes(limit=20)

        for episode in episodes:
            affect = getattr(episode, "affect", None)
            if affect is None:
                continue
            if affect.primary_emotion == PrimaryEmotion.NEUTRAL:
                continue
            return affect
        return None

    @staticmethod
    def apply_temporal_decay(
        score: float, age_days: float, half_life_days: float = 60.0
    ) -> float:
        return apply_temporal_decay(score, age_days, half_life_days=half_life_days)

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
    ) -> List[RetrievalResult]:
        return await semantic_search_impl(
            self,
            query,
            limit=limit,
            affect_query=affect_query,
            affect_weight=affect_weight,
        )

    async def _keyword_search(
        self,
        query: str,
        limit: int,
    ) -> List[RetrievalResult]:
        return await keyword_search_impl(self, query, limit=limit)

    async def _exact_handle_search(
        self,
        query: str,
        limit: int,
    ) -> List[RetrievalResult]:
        return await exact_handle_search(self, query, limit=limit)

    def _keyword_queries_for(self, query: str, recall_intent: bool) -> List[str]:
        """Generate useful FTS queries instead of only searching the raw sentence."""
        return keyword_queries_for(query, recall_intent=recall_intent, stopwords=KEYWORD_STOPWORDS)

    async def _expand_graph(
        self,
        seed_results: List[RetrievalResult],
        decay: float = 0.8,
        edge_limit: int = 12,
    ) -> List[RetrievalResult]:
        return await expand_graph_search(
            self,
            seed_results,
            decay=decay,
            edge_limit=edge_limit,
        )

    async def _mmr_rerank(
        self,
        results: List[RetrievalResult],
        lambda_param: float = 0.5,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Rerank results using Maximal Marginal Relevance."""
        return await rerank_mmr(
            self.embeddings,
            results,
            lambda_param=lambda_param,
            limit=limit,
        )

    def _apply_diversity_penalty(
        self,
        results: List[RetrievalResult],
        window: int = 5,
        penalty: float = 0.05,
    ) -> List[RetrievalResult]:
        return apply_diversity_penalty(results, window=window, penalty=penalty)

    def _apply_emotion_boost(
        self,
        results: List[RetrievalResult],
        tag: str,
        boost: float,
        query_affect: Optional[AffectState] = None,
    ) -> List[RetrievalResult]:
        return emotion_boost(results, tag, boost, query_affect=query_affect)

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[Tuple[str, List[RetrievalResult]]],
    ) -> List[RetrievalResult]:
        return reciprocal_rank_fusion(self.rrf_k, result_lists)
