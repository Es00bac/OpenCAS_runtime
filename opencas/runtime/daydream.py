"""Daydream generator for OpenCAS."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from opencas.api import LLMClient
from opencas.autonomy import WorkObject, WorkStage
from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.daydream import (
    DaydreamDialogueTurn,
    DaydreamReflection,
    DaydreamStore,
    DaydreamThought,
)
from opencas.daydream.mirror import strip_legacy_compassion_prefix
from opencas.daydream.models import DaydreamThoughtKind, DaydreamThoughtRoute
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.generation.policy import GenerationDomain, GenerationPhase, GenerationPolicyRequest
from opencas.identity import IdentityManager
from opencas.identity.text_hygiene import sanitize_identity_text
from opencas.memory import MemoryStore
from opencas.relational import RelationalEngine
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, Tracer


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
        working_set: Optional[List[dict[str, Any]]] = None,
    ) -> Tuple[List[WorkObject], List[DaydreamReflection]]:
        """Produce daydream sparks and reflection drafts."""
        recent = await self.memory.list_episodes(limit=limit)
        memory_snippets = [e.content for e in recent if e.content]

        context = await self._build_prompt(
            memory_snippets,
            goals or [],
            tension,
            working_set=working_set or [],
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a creative daydream engine for an autonomous agent. "
                    "Return a JSON object with a thoughts array. Each thought must "
                    "include kind, route, summary, confidence, grounding, and any "
                    "question, hypothesis, or possible_experiment that applies. "
                    "Use sparks only as a backward-compatible fallback."
                ),
            },
            {"role": "user", "content": context},
        ]

        variation_seed = f"daydream:{uuid4().hex}"
        generation_request = GenerationPolicyRequest(
            phase=GenerationPhase.DAYDREAM,
            domain=GenerationDomain.AUTONOMY,
            novelty_pressure=min(1.0, 0.4 + float(tension or 0.0)),
            continuity_pressure=0.4 if working_set else 0.2,
            source="daydream_generation",
            somatic=getattr(self.somatic, "state", None),
            memory_focus=[
                "reflective_memory",
                "association_memory",
                "project_memory",
                "active_work_context",
                "generation_sampling_eval",
            ],
            variation_seed=variation_seed,
        )
        generation_policy_trace: dict[str, Any] = {
            "generation_phase": GenerationPhase.DAYDREAM.value,
            "generation_authority": "reflective_proposal_only",
            "sampling_variation_seed": variation_seed,
        }
        try:
            response = await self.llm.chat_completion(
                messages,
                complexity="light",
                source="daydream_generation",
                generation_request=generation_request,
            )
            if isinstance(response, dict) and isinstance(
                response.get("_opencas_generation_policy"), dict
            ):
                generation_policy_trace = dict(response["_opencas_generation_policy"])
            content = self._extract_content(response)
            reflections = self._parse_structured(content)
            reflections = self._attach_working_set_grounding(
                reflections,
                working_set or [],
            )
        except Exception as exc:
            self._trace("generation_failed", {"error": str(exc)})
            reflections = []
        reflections = [self._sanitize_reflection(reflection) for reflection in reflections]
        for reflection in reflections:
            reflection.experience_context = {
                **(reflection.experience_context or {}),
                "generation_policy": generation_policy_trace,
            }
        if not reflections:
            reflections = self._fallback_reflections_from_context(
                working_set=working_set or [],
                memory_snippets=memory_snippets,
                goals=goals or [],
                tension=tension,
            )
            for reflection in reflections:
                reflection.experience_context = {
                    **(reflection.experience_context or {}),
                    "generation_policy": generation_policy_trace,
                }
            if reflections:
                self._trace(
                    "fallback_reflections_generated",
                    {
                        "count": len(reflections),
                        "working_set_count": len(working_set or []),
                        "memory_snippet_count": len(memory_snippets),
                    },
                )

        work_objects: List[WorkObject] = []
        for reflection in reflections:
            if reflection.spark_content.strip():
                meta: dict = {
                    "origin": "daydream",
                    "tension": tension,
                    "alignment_score": reflection.alignment_score,
                    "novelty_score": reflection.novelty_score,
                    "keeper": reflection.keeper,
                    "generation_policy": generation_policy_trace,
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
        working_set: Optional[List[dict[str, Any]]] = None,
    ) -> str:
        parts: List[str] = []

        # Identity fragment
        if self.identity:
            sm = self.identity.self_model
            um = self.identity.user_model
            id_parts: List[str] = []
            if sm.current_goals:
                id_parts.append(
                    "My current goals: " + ", ".join(self._sanitize_text(g) for g in sm.current_goals if self._sanitize_text(g))
                )
            if sm.values:
                id_parts.append("My values: " + ", ".join(self._sanitize_text(v) for v in sm.values if self._sanitize_text(v)))
            if sm.traits:
                id_parts.append("My traits: " + ", ".join(self._sanitize_text(t) for t in sm.traits if self._sanitize_text(t)))
            if sm.current_intention:
                id_parts.append(f"My current intention: {self._sanitize_text(sm.current_intention)}")
            if um.inferred_goals:
                inferred_goal_texts = [
                    text
                    for text in (
                        self._inferred_goal_text(goal)
                        for goal in list(um.inferred_goals or [])[:3]
                    )
                    if text
                ]
                if inferred_goal_texts:
                    id_parts.append(
                        "Inferred user goals: "
                        + ", ".join(inferred_goal_texts)
                    )
            if id_parts:
                parts.append("Identity\n" + "\n".join(f"- {p}" for p in id_parts))

        working_set_lines = self._format_working_set(working_set or [])
        if working_set_lines:
            coverage_lines = self._format_working_set_coverage(working_set or [])
            if coverage_lines:
                parts.append(
                    "Daydream intake coverage\n"
                    + "\n".join(coverage_lines)
                    + "\nWhen two or more source families are present, synthesize across "
                    "at least two source families rather than reflecting on only the "
                    "loudest item."
                )
            parts.append(
                "Active grounded working set\n"
                + "\n".join(working_set_lines)
                + "\nUse this as current lived material. Do not treat collaborator-agent "
                "turns as owner preferences unless the item explicitly says it is "
                "operator-grounded."
            )

        # Somatic coloring
        if self.somatic:
            s = self.somatic.state
            coloring: List[str] = []
            coloring.append(
                "Overall state: "
                f"valence={s.valence:.2f}, arousal={s.arousal:.2f}, "
                f"fatigue={s.fatigue:.2f}, tension={s.tension:.2f}"
            )
            if s.fatigue > 0.65:
                coloring.append("Fatigue is high. Keep thoughts brief and concrete.")
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
                    + "\n".join(
                        f"- {self._sanitize_text(r.spark_content)[:120]}" for r in recent_refs
                    )
                )

        # Memory seeds with identity-core episodes and graph neighbors
        identity_core_snippets: List[str] = []
        neighbor_snippets: List[str] = []
        if self.memory:
            # Find identity-core episodes among recent
            for ep in await self.memory.list_episodes(limit=20):
                if ep.identity_core and ep.content:
                    identity_core_snippets.append(self._sanitize_text(ep.content))
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
                        neighbor_snippets.append(self._sanitize_text(nep.content))
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
            parts.append(
                "Recent memories:\n"
                + "\n".join(
                    f"- {self._sanitize_text(s)}" for s in memory_snippets if self._sanitize_text(s)
                )
            )
        parts.append(f"Somatic tension: {tension:.2f}")
        parts.append(
            "Generate 1-3 grounded thought records that might grow into useful work. "
            "At least one useful thought should connect to the active grounded working "
            "set when it contains active work, recent dialogue, observations, or "
            "external headlines. "
            "When the intake coverage is diverse, synthesize across at least two "
            "source families and name the evidence trail in grounding rather than "
            "collapsing everything into a generic mood. "
            "Daydreaming should include a compact self-dialogue: for each thought, "
            "include inner_dialogue with 2-5 short turns where reflective stances "
            "ask questions, pressure-test the idea, answer one another, and choose "
            "what should remain only reflective. "
            "Return JSON with key thoughts. Each thought should include: kind "
            "(noticing, association, question, hypothesis, experiment, story_seed, "
            "system_insight, relationship_insight), route (act_now, deep_think, "
            "ask_user, research, incubate, discard), summary, question, hypothesis, "
            "possible_experiment, imaginative_branch, practical_branch, bridge, "
            "inner_dialogue, suggested_handler, contact_posture, self_work_intent, risk, usefulness, "
            "novelty, confidence, and grounding. suggested_handler may be self_note, "
            "self_experiment, self_prototype, thread, research, work_candidate, "
            "ask_user, compost, or blank. contact_posture may be silent, ask_first, "
            "share_after_artifact, share_failure, or urgent. "
            "Grounding entries should include kind, source, claim, confidence, and "
            "evidence_ids when available. Also include recollection, interpretation, "
            "synthesis, open_question, changed_self_view, and tension_hints when useful."
        )
        return "\n\n".join(parts)

    def _format_working_set(self, working_set: List[dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        for item in working_set[:14]:
            if not isinstance(item, dict):
                continue
            kind = self._sanitize_text(str(item.get("kind") or "observation"))[:64]
            source = self._sanitize_text(str(item.get("source") or "runtime"))[:96]
            label = self._sanitize_text(str(item.get("label") or kind))[:90]
            text = self._sanitize_text(str(item.get("text") or ""))
            if not text:
                continue
            created_at = self._sanitize_text(str(item.get("created_at") or ""))[:48]
            observed_at = self._sanitize_text(str(item.get("observed_at") or ""))[:48]
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            authority = self._sanitize_text(str(meta.get("authority") or ""))[:48] if isinstance(meta, dict) else ""
            source_family = self._sanitize_text(str(meta.get("source_family") or ""))[:48] if isinstance(meta, dict) else ""
            novelty_pressure = meta.get("novelty_pressure") if isinstance(meta, dict) else None
            evidence_ids = item.get("evidence_ids") or []
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            evidence = ", ".join(
                self._sanitize_text(str(eid))[:48] for eid in evidence_ids[:3] if self._sanitize_text(str(eid))
            )
            prefix = f"[{kind} / {source}"
            if created_at:
                prefix += f" @ {created_at}"
            if observed_at:
                prefix += f" observed @ {observed_at}"
            prefix += "]"
            suffix_parts: List[str] = []
            if evidence:
                suffix_parts.append(f"evidence: {evidence}")
            if authority:
                suffix_parts.append(f"authority: {authority}")
            if source_family:
                suffix_parts.append(f"family: {source_family}")
            if novelty_pressure is not None:
                suffix_parts.append(f"novelty_pressure: {novelty_pressure}")
            suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
            lines.append(f"- {prefix} {label}: {text[:220]}{suffix}")
        return lines

    def _format_working_set_coverage(self, working_set: List[dict[str, Any]]) -> List[str]:
        counts: dict[str, int] = {}
        for item in working_set:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            family = self._sanitize_text(str(meta.get("source_family") or ""))[:64]
            if not family:
                family = self._fallback_source_family(
                    str(item.get("kind") or ""),
                    str(item.get("source") or ""),
                )
            counts[family] = counts.get(family, 0) + 1
        return [f"- {family}: {count}" for family, count in sorted(counts.items()) if family]

    def _fallback_source_family(self, kind: str, source: str) -> str:
        kind_l = kind.lower()
        source_l = source.lower()
        if kind_l == "active_turn" or source_l == "runtime.current_cognition":
            return "active_turn"
        if kind_l == "recent_dialogue" or source_l == "context_store":
            return "conversation"
        if kind_l in {"active_goal", "active_commitment", "active_plan", "pending_task"}:
            return "active_work"
        if kind_l == "tom_user_belief" or source_l.startswith("tom."):
            return "tom_user_model"
        if kind_l in {"personal_curiosity", "user_interest_or_preference", "inferred_user_goal"}:
            return "identity_interest"
        if kind_l == "tool_observation" or source_l == "telemetry.tool_call":
            return "tool_observation"
        if kind_l == "external_observation" or source_l == "telemetry.external":
            return "external_observation"
        if kind_l in {"self_work_scaffold", "self_work_receipt"} or source_l == "daydream.self_work":
            return "self_work"
        if source_l.startswith("cognitive_state"):
            return "cognitive_state"
        if source_l == "memory":
            return "memory"
        if source_l.startswith("telemetry"):
            return "telemetry"
        return "runtime" if source_l.startswith("runtime") else "other"

    def _attach_working_set_grounding(
        self,
        reflections: List[DaydreamReflection],
        working_set: List[dict[str, Any]],
    ) -> List[DaydreamReflection]:
        intake_grounding = self._working_set_grounding(working_set)
        if not intake_grounding:
            return reflections
        for reflection in reflections:
            for thought in reflection.thoughts:
                existing = list(thought.grounding or [])
                has_specific_evidence = any(
                    grounding.evidence_ids
                    and not (
                        grounding.kind == GroundingKind.GENERATED_SYNTHESIS
                        and grounding.source == GroundingSource.DAYDREAM
                    )
                    for grounding in existing
                )
                if has_specific_evidence:
                    continue
                seen = {
                    (grounding.kind.value, grounding.source.value, tuple(grounding.evidence_ids), grounding.claim)
                    for grounding in existing
                }
                for grounding in intake_grounding:
                    key = (
                        grounding.kind.value,
                        grounding.source.value,
                        tuple(grounding.evidence_ids),
                        grounding.claim,
                    )
                    if key not in seen:
                        existing.append(grounding)
                        seen.add(key)
                    if len(existing) >= 4:
                        break
                thought.grounding = existing
        return reflections

    def _working_set_grounding(self, working_set: List[dict[str, Any]]) -> List[CognitionGrounding]:
        grounding: List[CognitionGrounding] = []
        for item in working_set[:8]:
            if not isinstance(item, dict):
                continue
            text = self._sanitize_text(str(item.get("text") or ""))
            if not text:
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            family = self._sanitize_text(str(meta.get("source_family") or ""))[:64]
            if not family:
                family = self._fallback_source_family(
                    str(item.get("kind") or ""),
                    str(item.get("source") or ""),
                )
            label = self._sanitize_text(str(item.get("label") or family or "working set"))[:90]
            evidence_ids = item.get("evidence_ids") or []
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            clean_evidence = [
                self._sanitize_text(str(eid))[:120]
                for eid in evidence_ids[:4]
                if self._sanitize_text(str(eid))
            ]
            try:
                weight = max(0.0, min(1.0, float(item.get("weight", 0.5) or 0.5)))
            except (TypeError, ValueError):
                weight = 0.5
            grounding.append(
                CognitionGrounding(
                    kind=GroundingKind.OBSERVED,
                    source=self._grounding_source_for_working_set_item(item, family, meta),
                    claim=f"{family} working-set item: {label}: {text[:150]}",
                    confidence=round(min(0.85, 0.5 + (weight * 0.3)), 3),
                    evidence_ids=clean_evidence,
                    meta={
                        "working_set_kind": self._sanitize_text(str(item.get("kind") or ""))[:64],
                        "working_set_source": self._sanitize_text(str(item.get("source") or ""))[:96],
                        "source_family": family,
                        "authority": self._sanitize_text(str(meta.get("authority") or ""))[:48],
                    },
                )
            )
            if len(grounding) >= 3:
                break
        return grounding

    def _grounding_source_for_working_set_item(
        self,
        item: dict[str, Any],
        family: str,
        meta: dict[str, Any],
    ) -> GroundingSource:
        source = str(item.get("source") or "").lower()
        authority = str(meta.get("authority") or "").lower()
        if source == "memory":
            return GroundingSource.MEMORY
        if authority == "operator" and family in {"active_turn", "conversation"}:
            return GroundingSource.USER
        if source.startswith("identity"):
            return GroundingSource.IDENTITY
        return GroundingSource.RUNTIME

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
                thought_reflections = self._parse_thought_reflections(parsed)
                if thought_reflections:
                    return thought_reflections
                sparks = parsed.get("sparks", [])
                if not isinstance(sparks, list):
                    sparks = []
                reflections: List[DaydreamReflection] = []
                for spark in sparks:
                    tension_hints = self._coerce_tension_hints(parsed.get("tension_hints", []))
                    spark_text = self._sanitize_text(str(spark))
                    reflections.append(
                        DaydreamReflection(
                            spark_content=spark_text,
                            recollection=self._sanitize_text(parsed.get("recollection", "")),
                            interpretation=self._sanitize_text(parsed.get("interpretation", "")),
                            synthesis=self._sanitize_text(parsed.get("synthesis", "")),
                            open_question=self._sanitize_text(parsed.get("open_question")),
                            changed_self_view=self._sanitize_text(parsed.get("changed_self_view", "")),
                            tension_hints=tension_hints,
                            thoughts=[self._fallback_thought(spark_text)],
                        )
                    )
                return reflections
        except json.JSONDecodeError:
            pass
        # Fallback: treat entire content as a single spark
        spark_text = self._sanitize_text(text)
        return [
            DaydreamReflection(
                spark_content=spark_text,
                thoughts=[self._fallback_thought(spark_text)],
            )
        ]

    def _trace(self, event: str, payload: dict) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.CREATIVE_PROMOTION,
                f"DaydreamGenerator: {event}",
                payload,
            )

    def _sanitize_text(self, value: str) -> str:
        return sanitize_identity_text(value)

    def _inferred_goal_text(self, value: object) -> str:
        if isinstance(value, dict):
            return self._sanitize_text(str(value.get("text", "")))
        return self._sanitize_text(str(value or ""))

    def _sanitize_reflection(self, reflection: DaydreamReflection) -> DaydreamReflection:
        reflection.spark_content = self._sanitize_text(
            strip_legacy_compassion_prefix(reflection.spark_content)
        )
        reflection.recollection = self._sanitize_text(reflection.recollection)
        reflection.interpretation = self._sanitize_text(reflection.interpretation)
        reflection.synthesis = self._sanitize_text(reflection.synthesis)
        reflection.open_question = self._sanitize_text(reflection.open_question)
        if not reflection.open_question:
            reflection.open_question = None
        reflection.changed_self_view = self._sanitize_text(reflection.changed_self_view)
        reflection.tension_hints = self._coerce_tension_hints(reflection.tension_hints)
        reflection.thoughts = [self._sanitize_thought(thought) for thought in reflection.thoughts]
        return reflection

    def _parse_thought_reflections(self, parsed: dict) -> List[DaydreamReflection]:
        raw_thoughts = parsed.get("thoughts", [])
        if not isinstance(raw_thoughts, list):
            return []

        reflections: List[DaydreamReflection] = []
        for raw in raw_thoughts:
            thought = self._coerce_thought(raw)
            if thought is None:
                continue
            spark_content = (
                thought.summary
                or thought.question
                or thought.hypothesis
                or thought.possible_experiment
            )
            spark_content = self._sanitize_text(spark_content)
            if not spark_content:
                continue
            reflections.append(
                DaydreamReflection(
                    spark_content=spark_content,
                    recollection=self._sanitize_text(parsed.get("recollection", "")),
                    interpretation=self._sanitize_text(parsed.get("interpretation", "")),
                    synthesis=self._sanitize_text(parsed.get("synthesis", "")),
                    open_question=self._sanitize_text(
                        parsed.get("open_question") or thought.question
                    ),
                    changed_self_view=self._sanitize_text(
                        parsed.get("changed_self_view", "")
                    ),
                    tension_hints=self._coerce_tension_hints(
                        parsed.get("tension_hints", [])
                    ),
                    thoughts=[thought],
                )
            )
        return reflections

    def _coerce_thought(self, raw: object) -> DaydreamThought | None:
        if not isinstance(raw, dict):
            return None
        payload = dict(raw)
        payload["kind"] = self._normalize_enum_text(payload.get("kind"))
        payload["route"] = self._normalize_enum_text(payload.get("route"))
        payload["summary"] = self._sanitize_text(payload.get("summary", ""))
        payload["question"] = self._sanitize_text(payload.get("question", ""))
        payload["hypothesis"] = self._sanitize_text(payload.get("hypothesis", ""))
        payload["possible_experiment"] = self._sanitize_text(
            payload.get("possible_experiment", "")
        )
        payload["imaginative_branch"] = self._sanitize_text(
            payload.get("imaginative_branch", "")
        )
        payload["practical_branch"] = self._sanitize_text(
            payload.get("practical_branch", "")
        )
        payload["bridge"] = self._sanitize_text(payload.get("bridge", ""))
        payload["suggested_handler"] = self._sanitize_text(
            payload.get("suggested_handler", "")
        )
        payload["contact_posture"] = self._sanitize_text(
            payload.get("contact_posture", "")
        )
        payload["self_work_intent"] = self._sanitize_text(
            payload.get("self_work_intent", "")
        )
        payload["grounding"] = self._coerce_grounding(payload.get("grounding", []))
        payload["inner_dialogue"] = self._coerce_inner_dialogue(
            payload.get("inner_dialogue")
            or payload.get("self_dialogue")
            or payload.get("dialogue")
            or []
        )
        try:
            return DaydreamThought.model_validate(payload)
        except Exception:
            fallback_text = self._first_non_empty(
                payload.get("summary"),
                payload.get("question"),
                payload.get("hypothesis"),
                payload.get("possible_experiment"),
                payload.get("bridge"),
                payload.get("practical_branch"),
                payload.get("imaginative_branch"),
            )
            if not fallback_text:
                return None
            fallback_payload = {
                "summary": str(payload.get("summary") or fallback_text),
                "question": str(payload.get("question") or ""),
                "hypothesis": str(payload.get("hypothesis") or ""),
                "possible_experiment": str(payload.get("possible_experiment") or ""),
                "imaginative_branch": str(payload.get("imaginative_branch") or ""),
                "practical_branch": str(payload.get("practical_branch") or ""),
                "bridge": str(payload.get("bridge") or ""),
                "suggested_handler": str(payload.get("suggested_handler") or ""),
                "contact_posture": str(payload.get("contact_posture") or ""),
                "self_work_intent": str(payload.get("self_work_intent") or ""),
                "usefulness": self._bounded_float(payload.get("usefulness"), 0.5),
                "novelty": self._bounded_float(payload.get("novelty"), 0.5),
                "confidence": self._bounded_float(payload.get("confidence"), 0.5),
                "risk": self._bounded_float(payload.get("risk"), 0.0),
                "grounding": payload.get("grounding") or [
                    CognitionGrounding(
                        kind=GroundingKind.GENERATED_SYNTHESIS,
                        source=GroundingSource.DAYDREAM,
                        claim="Structured daydream thought was salvaged from partially invalid model output.",
                        confidence=0.35,
                    )
                ],
                "inner_dialogue": payload.get("inner_dialogue") or [],
            }
            kind = self._enum_value_or_none(DaydreamThoughtKind, payload.get("kind"))
            route = self._enum_value_or_none(DaydreamThoughtRoute, payload.get("route"))
            if kind is not None:
                fallback_payload["kind"] = kind
            if route is not None:
                fallback_payload["route"] = route
            return DaydreamThought.model_validate(fallback_payload)

    def _sanitize_thought(self, thought: DaydreamThought) -> DaydreamThought:
        return thought.model_copy(
            update={
                "summary": self._sanitize_text(thought.summary),
                "question": self._sanitize_text(thought.question),
                "hypothesis": self._sanitize_text(thought.hypothesis),
                "possible_experiment": self._sanitize_text(thought.possible_experiment),
                "imaginative_branch": self._sanitize_text(thought.imaginative_branch),
                "practical_branch": self._sanitize_text(thought.practical_branch),
                "bridge": self._sanitize_text(thought.bridge),
                "suggested_handler": self._sanitize_text(thought.suggested_handler),
                "contact_posture": self._sanitize_text(thought.contact_posture),
                "self_work_intent": self._sanitize_text(thought.self_work_intent),
                "inner_dialogue": self._sanitize_dialogue(list(thought.inner_dialogue or [])),
            }
        )

    def _sanitize_dialogue_turn(self, turn: DaydreamDialogueTurn) -> DaydreamDialogueTurn:
        return turn.model_copy(
            update={
                "voice": self._sanitize_text(turn.voice),
                "stance": self._sanitize_text(turn.stance),
                "text": self._sanitize_text(turn.text),
            }
        )

    def _sanitize_dialogue(self, turns: List[DaydreamDialogueTurn]) -> List[DaydreamDialogueTurn]:
        sanitized: List[DaydreamDialogueTurn] = []
        for turn in turns[:5]:
            clean = self._sanitize_dialogue_turn(turn)
            if clean.text:
                sanitized.append(clean)
        return sanitized

    def _fallback_thought(self, spark_text: str) -> DaydreamThought:
        return DaydreamThought(
            summary=spark_text,
            grounding=[
                CognitionGrounding(
                    kind=GroundingKind.GENERATED_SYNTHESIS,
                    source=GroundingSource.DAYDREAM,
                    claim="Legacy daydream spark parsed without structured grounding.",
                    confidence=0.4,
                )
            ],
        )

    @staticmethod
    def _coerce_grounding(value: object) -> List[CognitionGrounding]:
        if not isinstance(value, list):
            return []
        grounding: List[CognitionGrounding] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                grounding.append(CognitionGrounding.model_validate(item))
            except Exception:
                continue
        return grounding

    def _coerce_inner_dialogue(self, value: object) -> List[DaydreamDialogueTurn]:
        if not isinstance(value, list):
            return []
        turns: List[DaydreamDialogueTurn] = []
        for item in value[:5]:
            if isinstance(item, str):
                payload = {"text": item}
            elif isinstance(item, dict):
                payload = {
                    "voice": self._sanitize_text(item.get("voice", item.get("speaker", ""))),
                    "stance": self._sanitize_text(item.get("stance", item.get("role", ""))),
                    "text": self._sanitize_text(item.get("text", item.get("content", ""))),
                }
            else:
                continue
            try:
                turn = DaydreamDialogueTurn.model_validate(payload)
            except Exception:
                continue
            if turn.text.strip():
                turns.append(turn)
        return turns

    def _fallback_reflections_from_context(
        self,
        *,
        working_set: List[dict[str, Any]],
        memory_snippets: List[str],
        goals: List[str],
        tension: float,
    ) -> List[DaydreamReflection]:
        seeds: List[dict[str, Any]] = []
        for item in working_set:
            if not isinstance(item, dict):
                continue
            text = self._sanitize_text(str(item.get("text") or ""))
            label = self._sanitize_text(str(item.get("label") or item.get("kind") or "working set"))
            if not text:
                continue
            seeds.append(
                {
                    "label": label[:90] or "working set",
                    "text": text[:240],
                    "source": self._sanitize_text(str(item.get("source") or "runtime"))[:90],
                    "evidence_ids": [
                        self._sanitize_text(str(eid))[:80]
                        for eid in (item.get("evidence_ids") or [])[:3]
                        if self._sanitize_text(str(eid))
                    ],
                    "grounding_source": GroundingSource.RUNTIME,
                }
            )
            if len(seeds) >= 2:
                break
        if not seeds:
            for snippet in memory_snippets[:2]:
                text = self._sanitize_text(snippet)
                if text:
                    seeds.append(
                        {
                            "label": "recent memory",
                            "text": text[:240],
                            "source": "memory",
                            "evidence_ids": [],
                            "grounding_source": GroundingSource.MEMORY,
                        }
                    )
        if not seeds:
            for goal in goals[:2]:
                text = self._sanitize_text(goal)
                if text:
                    seeds.append(
                        {
                            "label": "active goal",
                            "text": text[:240],
                            "source": "executive",
                            "evidence_ids": [],
                            "grounding_source": GroundingSource.RUNTIME,
                        }
                    )
        reflections: List[DaydreamReflection] = []
        for seed in seeds[:2]:
            label = str(seed["label"])
            text = str(seed["text"])
            source = str(seed["source"])
            summary = f"Reflective fallback on {label}: {text}"
            question = f"What useful pattern or next thought is hidden in {label}?"
            thought = DaydreamThought(
                kind=DaydreamThoughtKind.NOTICING,
                route=DaydreamThoughtRoute.INCUBATE,
                summary=summary,
                question=question,
                hypothesis=(
                    "If the daydream model returns no usable structured thoughts, "
                    "the grounded working set is still meaningful reflective material."
                ),
                practical_branch="Keep this as reflective context until a concrete action is justified.",
                bridge="This is a fallback reflection, not authorization to execute.",
                suggested_handler="self_note",
                contact_posture="silent",
                self_work_intent="preserve grounded material for later reflective processing",
                usefulness=0.45,
                novelty=0.35,
                confidence=0.4,
                risk=min(0.3, max(0.05, float(tension or 0.0) * 0.2)),
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.OBSERVED,
                        source=seed["grounding_source"],
                        claim=f"{source} supplied grounded material for fallback daydream incubation.",
                        confidence=0.55,
                        evidence_ids=list(seed.get("evidence_ids") or []),
                    )
                ],
                inner_dialogue=[
                    DaydreamDialogueTurn(
                        voice="reflective",
                        stance="salvage",
                        text="The model returned no usable thought, but this grounded material should not vanish.",
                    ),
                    DaydreamDialogueTurn(
                        voice="practical",
                        stance="boundary",
                        text="Hold it as an incubated note; do not promote it without stronger evidence.",
                    ),
                ],
            )
            reflections.append(
                DaydreamReflection(
                    spark_content=summary,
                    recollection=f"Grounded fallback seed from {source}.",
                    interpretation=(
                        "The daydream generator produced no usable structured reflections; "
                        "runtime evidence was preserved as reflective material instead."
                    ),
                    synthesis="Fallback daydream preserved grounded context without creating an executable task.",
                    open_question=question,
                    tension_hints=["empty_generation_fallback"],
                    thoughts=[thought],
                    experience_context={
                        "trigger": "daydream_empty_generation_fallback",
                        "source": source,
                        "authority": "reflective",
                    },
                )
            )
        return reflections

    @staticmethod
    def _coerce_tension_hints(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        hints: List[str] = []
        for hint in value:
            if not isinstance(hint, str):
                continue
            sanitized_hint = sanitize_identity_text(hint)
            if sanitized_hint:
                hints.append(sanitized_hint)
        return hints

    @staticmethod
    def _normalize_enum_text(value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _bounded_float(value: object, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))

    @staticmethod
    def _enum_value_or_none(enum_type: type, value: object) -> object | None:
        if value in getattr(enum_type, "_value2member_map_", {}):
            return value
        return None

    @staticmethod
    def _first_non_empty(*values: object) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""
