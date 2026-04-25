"""Episode signal ranker for promoting strong short-term signals to long-term memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from opencas.identity import IdentityManager
from opencas.memory import Episode, MemoryStore


@dataclass
class SignalScore:
    """Detailed scoring result for a single episode."""

    episode_id: str
    signal_score: float
    salience_score: float
    recency_score: float
    affect_score: float
    identity_score: float
    centrality_score: float
    richness_score: float


class EpisodeSignalRanker:
    """Ranks episodes by composite signal strength for long-term memory promotion."""

    DEFAULT_WEIGHTS = {
        "salience": 0.25,
        "recency": 0.20,
        "affect": 0.15,
        "identity": 0.20,
        "centrality": 0.15,
        "richness": 0.05,
    }

    def __init__(
        self,
        memory: MemoryStore,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.memory = memory
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    async def rank_episodes(
        self,
        episodes: List[Episode],
        identity: Optional[IdentityManager] = None,
        now: Optional[datetime] = None,
    ) -> List[SignalScore]:
        """Return a SignalScore for each episode, sorted by signal_score descending."""
        if now is None:
            now = datetime.now(timezone.utc)

        identity_tokens: Set[str] = set()
        if identity:
            sm = identity.self_model
            sources = [
                *(sm.current_goals or []),
                *(sm.values or []),
                *(sm.traits or []),
                sm.current_intention or "",
                sm.narrative or "",
            ]
            identity_tokens = self._tokenize(" ".join(sources))

        results: List[SignalScore] = []
        for ep in episodes:
            scores = self._score_episode(ep, now, identity_tokens)
            results.append(scores)

        results.sort(key=lambda s: s.signal_score, reverse=True)
        return results

    def _score_episode(
        self,
        episode: Episode,
        now: datetime,
        identity_tokens: Set[str],
    ) -> SignalScore:
        # Salience: already normalized 0-10 in the model, scale to 0-1
        salience_score = min(1.0, episode.salience / 10.0)

        # Recency: exponential decay over hours, 24h half-life
        hours_old = max(0.0, (now - episode.created_at).total_seconds() / 3600.0)
        recency_score = math.exp(-hours_old / 24.0)

        # Affect intensity: arousal + abs(valence), capped at 1.0
        affect_score = 0.0
        if episode.affect:
            affect_score = min(
                1.0,
                (episode.affect.arousal + abs(episode.affect.valence)) / 2.0,
            )

        # Identity relevance: Jaccard-ish token overlap
        identity_score = 0.0
        if identity_tokens:
            ep_tokens = self._tokenize(episode.content)
            if ep_tokens:
                overlap = len(ep_tokens & identity_tokens)
                identity_score = min(1.0, overlap / max(1, len(identity_tokens) * 0.1))

        # Graph centrality is supplied by rank_episodes_with_degrees when callers
        # need edge-aware scoring without making this synchronous helper do I/O.
        # We defer async edge lookup to the caller and pass degrees via a mapping.
        # Since this method is synchronous, centrality is scored heuristically
        # based on payload hints if present, otherwise defaults to neutral.
        centrality_score = 0.5
        degree_hint = episode.payload.get("edge_degree")
        if isinstance(degree_hint, (int, float)):
            centrality_score = min(1.0, float(degree_hint) / 10.0)

        # Conceptual richness: unique word density vs length
        richness_score = 0.0
        if episode.content:
            words = episode.content.lower().split()
            if words:
                unique_ratio = len(set(words)) / len(words)
                # Penalize very short utterances, reward moderate density
                length_factor = min(1.0, len(words) / 20.0)
                richness_score = unique_ratio * length_factor

        signal_score = round(
            salience_score * self.weights["salience"]
            + recency_score * self.weights["recency"]
            + affect_score * self.weights["affect"]
            + identity_score * self.weights["identity"]
            + centrality_score * self.weights["centrality"]
            + richness_score * self.weights["richness"],
            4,
        )

        return SignalScore(
            episode_id=str(episode.episode_id),
            signal_score=signal_score,
            salience_score=round(salience_score, 4),
            recency_score=round(recency_score, 4),
            affect_score=round(affect_score, 4),
            identity_score=round(identity_score, 4),
            centrality_score=round(centrality_score, 4),
            richness_score=round(richness_score, 4),
        )

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple whitespace tokenization with punctuation stripped."""
        return {
            word.strip(".,!?;:-\"'()[]{}")
            for word in text.lower().split()
            if len(word.strip(".,!?;:-\"'()[]{}") ) > 2
        }

    async def rank_episodes_with_degrees(
        self,
        episodes: List[Episode],
        identity: Optional[IdentityManager] = None,
        now: Optional[datetime] = None,
    ) -> List[SignalScore]:
        """Async variant that fetches edge degrees from MemoryStore before scoring."""
        if now is None:
            now = datetime.now(timezone.utc)

        # Pre-fetch edge degrees
        degree_map: Dict[str, int] = {}
        for ep in episodes:
            ep_id = str(ep.episode_id)
            edges = await self.memory.get_edges_for(ep_id, min_confidence=0.0, limit=100)
            degree_map[ep_id] = len(edges)

        # Inject degrees into payloads temporarily
        original_degrees: Dict[str, Any] = {}
        for ep in episodes:
            ep_id = str(ep.episode_id)
            original_degrees[ep_id] = ep.payload.get("edge_degree")
            ep.payload["edge_degree"] = degree_map.get(ep_id, 0)

        try:
            results = await self.rank_episodes(episodes, identity, now)
        finally:
            # Restore original payloads
            for ep in episodes:
                ep_id = str(ep.episode_id)
                if original_degrees[ep_id] is not None:
                    ep.payload["edge_degree"] = original_degrees[ep_id]
                else:
                    ep.payload.pop("edge_degree", None)

        return results
