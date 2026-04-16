"""Identity rebuild from autobiographical memory episodes."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from opencas.api import LLMClient
from opencas.identity import IdentityManager
from opencas.memory import EdgeKind, Episode, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph


class IdentityRebuildResult(BaseModel):
    """Result of rebuilding identity from memory."""

    narrative: Optional[str] = None
    values: List[str] = Field(default_factory=list)
    traits: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    source_episode_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class IdentityRebuilder:
    """Reconstructs self-model fields from autobiographical identity-core episodes."""

    def __init__(
        self,
        memory: MemoryStore,
        episode_graph: Optional[EpisodeGraph] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.memory = memory
        self.episode_graph = episode_graph
        self.llm = llm

    async def rebuild(
        self,
        seed_episode_ids: Optional[List[str]] = None,
    ) -> IdentityRebuildResult:
        """Rebuild identity from memory episodes."""
        episodes: List[Episode] = []
        source_ids: List[str] = []

        if seed_episode_ids:
            episodes = await self.memory.get_episodes_by_ids(seed_episode_ids)
        else:
            core_eps = await self.memory.list_identity_core_episodes(limit=20)
            episodes = core_eps

        if not episodes:
            # Fallback: use most recent general episodes
            episodes = await self.memory.list_episodes(limit=20)

        # Gather neighborhood via temporal graph edges
        if self.episode_graph is not None:
            neighbor_ids: Set[str] = set()
            for ep in episodes:
                neighbor_ids.add(str(ep.episode_id))
                walked = await self.episode_graph.walk(
                    start_id=str(ep.episode_id),
                    steps=2,
                    kind_filter=EdgeKind.TEMPORAL,
                    min_confidence=0.0,
                )
                neighbor_ids.update(walked.keys())
            # Avoid re-fetching if we already have them
            missing_ids = [eid for eid in neighbor_ids if not any(str(ep.episode_id) == eid for ep in episodes)]
            if missing_ids:
                episodes.extend(await self.memory.get_episodes_by_ids(list(missing_ids)))

        # Deduplicate and sort chronologically
        seen: Set[str] = set()
        unique_eps: List[Episode] = []
        for ep in sorted(episodes, key=lambda e: e.created_at):
            eid = str(ep.episode_id)
            if eid not in seen:
                seen.add(eid)
                unique_eps.append(ep)
        episodes = unique_eps

        source_ids = [str(ep.episode_id) for ep in episodes]
        confidence = round(min(1.0, len(episodes) / 20.0), 3)

        if self.llm is not None:
            llm_result = await self._synthesize_with_llm(episodes)
            if llm_result is not None:
                llm_result.source_episode_ids = source_ids
                llm_result.confidence = confidence
                return llm_result

        heuristic = self._heuristic_fallback(episodes)
        heuristic.source_episode_ids = source_ids
        heuristic.confidence = confidence
        return heuristic

    async def apply(
        self,
        result: IdentityRebuildResult,
        identity: IdentityManager,
    ) -> None:
        """Write rebuild result into an IdentityManager."""
        if result.narrative:
            identity.self_model.narrative = result.narrative
        if result.values:
            identity.self_model.values = list(result.values)
        if result.traits:
            identity.self_model.traits = list(result.traits)
        if result.goals:
            identity.self_model.current_goals = list(result.goals)
        identity.save()

    async def _synthesize_with_llm(
        self,
        episodes: List[Episode],
    ) -> Optional[IdentityRebuildResult]:
        assert self.llm is not None
        lines = []
        for ep in episodes:
            text = ep.content.strip()
            if text:
                lines.append(f"- {text[:400]}")
        prompt_text = (
            "You are reconstructing a self-identity from autobiographical memory episodes.\n"
            "Synthesize the following fields as compact JSON with keys: narrative, values, traits, goals.\n"
            "Narrative should be 1-2 sentences. Values, traits, and goals should be short string lists.\n\n"
            "Episodes:\n" + "\n".join(lines)
        )
        messages = [
            {"role": "system", "content": "You are a concise identity synthesis engine."},
            {"role": "user", "content": prompt_text},
        ]
        try:
            response = await self.llm.chat_completion(
                messages=messages,
                complexity="high",
                payload={"temperature": 0.4, "max_tokens": 512},
                source="identity_rebuild",
            )
            content = self._extract_content(response)
            parsed = json.loads(content)
            return IdentityRebuildResult(
                narrative=parsed.get("narrative"),
                values=parsed.get("values") or [],
                traits=parsed.get("traits") or [],
                goals=parsed.get("goals") or [],
            )
        except Exception:
            return None

    @staticmethod
    def _heuristic_fallback(episodes: List[Episode]) -> IdentityRebuildResult:
        texts = [ep.content.lower() for ep in episodes if ep.content]
        all_text = " ".join(texts)
        tokens = re.findall(r"\b[a-z]{4,}\b", all_text)
        stop_words = {
            "that", "with", "from", "this", "have", "were", "they", "been",
            "their", "what", "when", "where", "which", "while", "about",
            "would", "could", "should", "there", "then", "than", "them",
            "will", "shall", "may", "might", "must", "shall", "said", "says",
        }
        filtered = [t for t in tokens if t not in stop_words]
        top_tokens = [word for word, _ in Counter(filtered).most_common(10)]

        # Keyword maps
        value_keywords = {
            "help": "helpfulness",
            "assist": "helpfulness",
            "support": "helpfulness",
            "care": "care",
            "kind": "care",
            "compassion": "care",
            "learn": "growth",
            "grow": "growth",
            "improve": "growth",
            "build": "agency",
            "create": "agency",
            "make": "agency",
            "honest": "honesty",
            "truth": "honesty",
            "transparent": "honesty",
            "clarity": "clarity",
            "clear": "clarity",
        }
        trait_keywords = {
            "curious": "curious",
            "wonder": "curious",
            "ask": "curious",
            "patient": "patient",
            "wait": "patient",
            "steady": "patient",
            "direct": "direct",
            "straight": "direct",
            "blunt": "direct",
            "persistent": "persistent",
            "continue": "persistent",
            "concise": "concise",
            "brief": "concise",
            "action": "action-oriented",
            "execute": "action-oriented",
        }
        values: Set[str] = set()
        traits: Set[str] = set()
        for token in top_tokens:
            if token in value_keywords:
                values.add(value_keywords[token])
            if token in trait_keywords:
                traits.add(trait_keywords[token])

        # Ensure baseline diversity if empty
        if not values:
            values.add("growth")
        if not traits:
            traits.add("curious")

        # Extract goal fragments
        goals: List[str] = []
        goal_patterns = [
            r"i want to\s+([^\.\n]+)",
            r"i need to\s+([^\.\n]+)",
            r"i should\s+([^\.\n]+)",
            r"goal is\s+([^\.\n]+)",
            r"my goal is\s+([^\.\n]+)",
        ]
        for text in texts:
            for pattern in goal_patterns:
                for match in re.finditer(pattern, text):
                    phrase = match.group(1).strip(" ,;:-")
                    if phrase and phrase not in goals:
                        goals.append(phrase)
                        if len(goals) >= 5:
                            break
            if len(goals) >= 5:
                break

        narrative = (
            f"Rebuilt from {len(episodes)} autobiographical episodes. "
            f"Top themes: {', '.join(top_tokens[:5]) or 'unknown'}."
        )

        return IdentityRebuildResult(
            narrative=narrative,
            values=list(values),
            traits=list(traits),
            goals=goals,
        )

    @staticmethod
    def _extract_content(response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""
