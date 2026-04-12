"""Episode graph traversal API for memory fabric."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from opencas.memory import EdgeKind, EpisodeEdge, MemoryStore


class EpisodeGraph:
    """Structured graph traversal over typed episode edges."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def get_neighbors(
        self,
        episode_id: str,
        kind: Optional[EdgeKind] = None,
        min_confidence: float = 0.0,
        limit: int = 24,
    ) -> List[EpisodeEdge]:
        """Return edges connected to an episode, optionally filtered by kind."""
        return await self.store.get_edges_for(
            episode_id,
            min_confidence=min_confidence,
            limit=limit,
            kind=kind,
        )

    async def get_neighbors_batch(
        self,
        episode_ids: List[str],
        kind: Optional[EdgeKind] = None,
        min_confidence: float = 0.0,
    ) -> List[EpisodeEdge]:
        """Return edges connected to a set of episodes, optionally filtered by kind."""
        return await self.store.get_edges_for_batch(
            episode_ids,
            min_confidence=min_confidence,
            kind=kind,
        )

    async def walk(
        self,
        start_id: str,
        steps: int,
        kind_filter: Optional[EdgeKind] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, int]:
        """BFS walk up to *steps* hops.

        Returns a map of reachable episode_id -> hop distance.
        """
        visited: Dict[str, int] = {start_id: 0}
        frontier: Set[str] = {start_id}
        for step in range(1, steps + 1):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            edges = await self.get_neighbors_batch(
                list(frontier), kind=kind_filter, min_confidence=min_confidence
            )
            for edge in edges:
                if edge.source_id in frontier:
                    neighbor = edge.target_id
                else:
                    neighbor = edge.source_id
                    
                if neighbor not in visited:
                    visited[neighbor] = step
                    next_frontier.add(neighbor)
            frontier = next_frontier
        return visited

    async def subgraph(
        self,
        episode_ids: List[str],
        kind_filter: Optional[EdgeKind] = None,
        min_confidence: float = 0.0,
    ) -> List[EpisodeEdge]:
        """Return all internal edges among the given episode IDs."""
        ids_set = set(episode_ids)
        edges: List[EpisodeEdge] = []
        seen: Set[str] = set()
        for ep_id in episode_ids:
            for edge in await self.get_neighbors(
                ep_id, kind=kind_filter, min_confidence=min_confidence, limit=200
            ):
                if str(edge.edge_id) in seen:
                    continue
                seen.add(str(edge.edge_id))
                neighbor = (
                    edge.target_id
                    if edge.source_id == ep_id
                    else edge.source_id
                )
                if neighbor in ids_set:
                    edges.append(edge)
        return edges
