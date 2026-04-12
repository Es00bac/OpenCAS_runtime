"""Daydream generator for OpenCAS."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from opencas.api import LLMClient
from opencas.autonomy import WorkObject, WorkStage
from opencas.daydream import DaydreamReflection, DaydreamStore
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.identity import IdentityManager
from opencas.memory import MemoryStore
from opencas.relational import RelationalEngine
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, Tracer
from opencas.somatic.models import PrimaryEmotion


class DaydreamGenerator:
    """Generates imaginative sparks from memory, goals, and somatic tension."""

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryStore,
        tracer: Optional[Tracer] = None,
        identity: Optional[IdentityManager] = None,
        somatic: Optional[SomaticManager] = None,
        relational: Optional[RelationalEngine] = None,
        daydream_store: Optional[DaydreamStore] = None,
        spark_evaluator: Optional[SparkEvaluator] = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tracer = tracer
        self.identity = identity
        self.somatic = somatic
        self.relational = relational
        self.daydream_store = daydream_store
        self.spark_evaluator = spark_evaluator

    async def generate(
        self,
        goals: Optional[List[str]] = None,
        tension: float = 0.0,
        limit: int = 5,
    ) -> Tuple[List[WorkObject], List[DaydreamReflection]]:
        """Produce daydream sparks and reflection drafts."""
        recent = await self.memory.list_episodes(limit=limit)
        memory_snippets = [e.content for e in recent if e.content]

        context = await self._build_prompt(memory_snippets, goals or [], tension)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a creative daydream engine for an autonomous agent. "
                    "Return a JSON object with keys: sparks (array of strings), "
                    "recollection, interpretation, synthesis, open_question, "
                    "changed_self_view, tension_hints (array of strings). "
                    "Each spark should be a concise idea, question, or association."
                ),
            },
            {"role": "user", "content": context},
        ]

        try:
            response = await self.llm.chat_completion(messages)
            content = self._extract_content(response)
            reflections = self._parse_structured(content)
        except Exception as exc:
            self._trace("generation_failed", {"error": str(exc)})
            reflections = []

        work_objects: List[WorkObject] = []
        for reflection in reflections:
            if reflection.spark_content.strip():
                meta: dict = {
                    "origin": "daydream",
                    "tension": tension,
                    "alignment_score": reflection.alignment_score,
                    "novelty_score": reflection.novelty_score,
                    "keeper": reflection.keeper,
                }
                if self.somatic:
                    s = self.somatic.state
                    meta["valence"] = s.valence
                    meta["arousal"] = s.arousal
                if self.relational:
                    meta["musubi"] = self.relational.state.musubi
                if self.identity:
                    meta["intention"] = self.identity.self_model.current_intention
                work_objects.append(
                    WorkObject(
                        content=reflection.spark_content.strip(),
                        stage=WorkStage.SPARK,
                        meta=meta,
                    )
                )

        # Structured novelty filter: promote only sparks above novelty floor
        if self.spark_evaluator is not None:
            work_objects = await self.spark_evaluator.filter_sparks(work_objects)

        self._trace(
            "generated",
            {"count": len(work_objects), "tension": tension, "evaluated": self.spark_evaluator is not None},
        )
        return work_objects, reflections

    async def _build_prompt(
        self,
        memory_snippets: List[str],
        goals: List[str],
        tension: float,
    ) -> str:
        parts: List[str] = []

        # Identity fragment
        if self.identity:
            sm = self.identity.self_model
            um = self.identity.user_model
            id_parts: List[str] = []
            if sm.current_goals:
                id_parts.append("My current goals: " + ", ".join(sm.current_goals))
            if sm.values:
                id_parts.append("My values: " + ", ".join(sm.values))
            if sm.traits:
                id_parts.append("My traits: " + ", ".join(sm.traits))
            if sm.current_intention:
                id_parts.append(f"My current intention: {sm.current_intention}")
            if um.inferred_goals:
                id_parts.append("Inferred user goals: " + ", ".join(um.inferred_goals[:3]))
            if id_parts:
                parts.append("Identity\n" + "\n".join(f"- {p}" for p in id_parts))

        # Somatic coloring
        if self.somatic:
            s = self.somatic.state
            coloring: List[str] = []
            coloring.append(f"Overall state: valence={s.valence:.2f}, arousal={s.arousal:.2f}, fatigue={s.fatigue:.2f}, tension={s.tension:.2f}")
            if s.fatigue > 0.65:
                coloring.append("I am fatigued. Keep thoughts brief and concrete.")
            if s.tension > 0.4:
                coloring.append("There is unresolved tension. Let thoughts drift toward what feels stabilizing.")
            if s.valence > 0.3 and s.arousal > 0.5:
                coloring.append("Energy is positive and elevated. Follow curiosity further than usual.")
            if s.somatic_tag:
                coloring.append(f"Somatic tag: {s.somatic_tag}")
            if coloring:
                parts.append("Somatic coloring\n" + "\n".join(f"- {c}" for c in coloring))

        # Musubi fragment
        if self.relational:
            r = self.relational.state
            parts.append(
                f"Relational state (musubi): {r.musubi:.2f}. "
                f"Dimensions: trust={r.dimensions.get('trust', 0):.2f}, "
                f"resonance={r.dimensions.get('resonance', 0):.2f}, "
                f"presence={r.dimensions.get('presence', 0):.2f}, "
                f"attunement={r.dimensions.get('attunement', 0):.2f}."
            )
            if r.musubi < 0.3:
                parts.append("Musubi is low. Let the thought touch something relational — a memory of shared work, a wish to reconnect.")
            elif r.musubi > 0.7:
                parts.append("Musubi is high. Build on shared meaning rather than solo exploration.")

        # Recent daydreams (anti-repetition)
        if self.daydream_store:
            recent_refs = await self.daydream_store.list_recent(limit=3)
            if recent_refs:
                parts.append(
                    "Recent private thoughts to avoid repeating verbatim:\n"
                    + "\n".join(f"- {r.spark_content[:120]}" for r in recent_refs)
                )

        # Memory seeds with identity-core episodes and graph neighbors
        identity_core_snippets: List[str] = []
        neighbor_snippets: List[str] = []
        if self.memory:
            # Find identity-core episodes among recent
            for ep in await self.memory.list_episodes(limit=20):
                if ep.identity_core and ep.content:
                    identity_core_snippets.append(ep.content)
                if len(identity_core_snippets) >= 2:
                    break
            # Graph neighbors from most recent episode
            recent_eps = await self.memory.list_recent_episodes(limit=1)
            if recent_eps:
                edges = await self.memory.get_edges_for(str(recent_eps[0].episode_id), limit=4)
                for edge in edges:
                    nid = edge.target_id if edge.source_id == str(recent_eps[0].episode_id) else edge.source_id
                    nep = await self.memory.get_episode(nid)
                    if nep and nep.content:
                        neighbor_snippets.append(nep.content)
                    if len(neighbor_snippets) >= 2:
                        break

        if identity_core_snippets:
            parts.append(
                "Identity-core memories:\n"
                + "\n".join(f"- {s[:120]}" for s in identity_core_snippets)
            )
        if neighbor_snippets:
            parts.append(
                "Related memory neighbors:\n"
                + "\n".join(f"- {s[:120]}" for s in neighbor_snippets)
            )

        # Active conflicts
        # We cannot import ConflictStore here to avoid circular deps;
        # caller (AgentRuntime) will inject active conflicts via a future extension.
        # For now, we rely on the model to detect tensions from the prompt context.

        if goals:
            parts.append("Current goals:\n" + "\n".join(f"- {g}" for g in goals))
        if memory_snippets:
            parts.append("Recent memories:\n" + "\n".join(f"- {s}" for s in memory_snippets))
        parts.append(f"Somatic tension: {tension:.2f}")
        parts.append(
            "Generate 1-3 short imaginative sparks (ideas, questions, or associations) "
            "that might grow into useful work. Return as a JSON object with keys: "
            "sparks, recollection, interpretation, synthesis, open_question, changed_self_view, tension_hints."
        )
        return "\n\n".join(parts)

    def _extract_content(self, response: dict) -> str:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""

    def _parse_structured(self, content: str) -> List[DaydreamReflection]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                sparks = parsed.get("sparks", [])
                if not isinstance(sparks, list):
                    sparks = []
                reflections: List[DaydreamReflection] = []
                for spark in sparks:
                    reflections.append(
                        DaydreamReflection(
                            spark_content=str(spark),
                            recollection=parsed.get("recollection", ""),
                            interpretation=parsed.get("interpretation", ""),
                            synthesis=parsed.get("synthesis", ""),
                            open_question=parsed.get("open_question"),
                            changed_self_view=parsed.get("changed_self_view", ""),
                            tension_hints=parsed.get("tension_hints", []),
                        )
                    )
                return reflections
        except json.JSONDecodeError:
            pass
        # Fallback: treat entire content as a single spark
        return [DaydreamReflection(spark_content=text)]

    def _trace(self, event: str, payload: dict) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.CREATIVE_PROMOTION,
                f"DaydreamGenerator: {event}",
                payload,
            )
