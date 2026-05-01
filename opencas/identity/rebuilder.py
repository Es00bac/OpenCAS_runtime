"""Identity rebuild from autobiographical memory episodes."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from opencas.api import LLMClient
from opencas.identity import IdentityManager
from opencas.identity.text_hygiene import (
    FORBIDDEN_TERM_PATTERNS as _FORBIDDEN_TERM_PATTERNS,
    FORBIDDEN_TERM_REPLACEMENTS as _FORBIDDEN_TERM_REPLACEMENTS,
    sanitize_identity_text,
)
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


DEFAULT_TERM_LIMITS: Dict[str, int] = {
    "returning": 1,
    "thread": 1,
    "drifted": 1,
}

# Kept for backward compatibility with existing module-local callers/tests.
# The canonical maps are imported from `identity.text_hygiene` above.


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
        *,
        min_created_at: Optional[datetime] = None,
        expand_graph: bool = True,
        term_limits: Optional[Dict[str, int]] = None,
    ) -> IdentityRebuildResult:
        """Rebuild identity from memory episodes."""
        episodes: List[Episode] = []
        source_ids: List[str] = []

        if seed_episode_ids:
            episodes = await self.memory.get_episodes_by_ids(seed_episode_ids)
        else:
            core_eps = await self.memory.list_identity_core_episodes(limit=20)
            episodes = core_eps
        episodes = self._filter_min_created_at(episodes, min_created_at)

        if not episodes:
            # Fallback: use most recent general episodes
            fallback_limit = 200 if min_created_at is not None else 20
            episodes = self._filter_min_created_at(
                await self.memory.list_episodes(limit=fallback_limit),
                min_created_at,
            )[:20]

        # Gather neighborhood via temporal graph edges
        if self.episode_graph is not None and expand_graph:
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
            episodes = self._filter_min_created_at(episodes, min_created_at)

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
                llm_result = self._sanitize_result(llm_result)
                validation = self.validate_result(llm_result, term_limits=term_limits)
                if validation["ok"]:
                    llm_result.source_episode_ids = source_ids
                    llm_result.confidence = confidence
                    return llm_result

            # If LLM output violates term limits (or fails to parse), recover with
            # heuristic output that has explicit term hygiene.
            heuristic = self._heuristic_fallback(episodes)
            heuristic = self._sanitize_result(heuristic)
            heuristic.source_episode_ids = source_ids
            heuristic.confidence = confidence
            return heuristic


        heuristic = self._heuristic_fallback(episodes)
        heuristic = self._sanitize_result(heuristic)
        heuristic.source_episode_ids = source_ids
        heuristic.confidence = confidence
        return heuristic

    @staticmethod
    def _filter_min_created_at(
        episodes: List[Episode],
        min_created_at: Optional[datetime],
    ) -> List[Episode]:
        if min_created_at is None:
            return episodes
        return [ep for ep in episodes if ep.created_at >= min_created_at]

    async def apply(
        self,
        result: IdentityRebuildResult,
        identity: IdentityManager,
        *,
        term_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        """Write rebuild result into an IdentityManager."""
        if term_limits is not None:
            validation = self.validate_result(result, term_limits=term_limits)
            if not validation["ok"]:
                raise ValueError(
                    "Identity rebuild result failed term limits: "
                    f"{validation['violations']}"
                )
        if result.narrative:
            identity.self_model.narrative = result.narrative
        if result.values:
            identity.self_model.values = list(result.values)
        if result.traits:
            identity.self_model.traits = list(result.traits)
        if result.goals:
            identity.self_model.current_goals = list(result.goals)
        identity.save()

    @staticmethod
    def validate_result(
        result: IdentityRebuildResult,
        *,
        term_limits: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        limits = term_limits if term_limits is not None else DEFAULT_TERM_LIMITS
        text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False).lower()
        counts = {
            term: len(
                re.findall(
                    _FORBIDDEN_TERM_PATTERNS.get(term, rf"\b{re.escape(term)}\b"),
                    text,
                    flags=re.IGNORECASE,
                )
            )
            for term in limits
        }
        violations = {
            term: {"count": count, "limit": limits[term]}
            for term, count in counts.items()
            if count > limits[term]
        }
        return {
            "ok": not violations,
            "term_counts": counts,
            "violations": violations,
        }

    async def _synthesize_with_llm(
        self,
        episodes: List[Episode],
    ) -> Optional[IdentityRebuildResult]:
        assert self.llm is not None
        lines = []
        for ep in episodes:
            text = self._sanitize_text(ep.content.strip())
            if text:
                lines.append(f"- {text[:400]}")
        prompt_text = (
            "You are reconstructing a self-identity from autobiographical memory episodes.\n"
            "Synthesize the following fields as compact JSON with keys: narrative, values, traits, goals.\n"
            "Narrative should be 1-2 sentences. Values, traits, and goals should be short string lists.\n\n"
            "Use plain language and avoid recursive wording, especially words like returning, "
            "thread, and drifted.\n\n"
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

    @classmethod
    def _sanitize_result(cls, result: IdentityRebuildResult) -> IdentityRebuildResult:
        """Apply identity-safe normalization to reconstructed fields."""
        narrative = cls._sanitize_text(result.narrative) if result.narrative else None
        values = [cls._sanitize_text(value) for value in result.values if cls._sanitize_text(value)]
        traits = [cls._sanitize_text(value) for value in result.traits if cls._sanitize_text(value)]
        goals = [cls._sanitize_text(value) for value in result.goals if cls._sanitize_text(value)]

        return IdentityRebuildResult(
            narrative=narrative,
            values=values,
            traits=traits,
            goals=goals,
            source_episode_ids=result.source_episode_ids,
            confidence=result.confidence,
        )

    @staticmethod
    def _sanitize_text(value: Optional[str]) -> str:
        return sanitize_identity_text(value)

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

        top_tokens = [
            word for word in top_tokens
            if word not in _FORBIDDEN_TERM_REPLACEMENTS
        ]

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
                        goals.append(IdentityRebuilder._sanitize_text(phrase))
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
