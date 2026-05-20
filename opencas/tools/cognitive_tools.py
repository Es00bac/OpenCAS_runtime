"""Tool adapter for OpenCAS cognitive-state introspection and low-risk updates."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field

from opencas.cognition import CognitiveEventKind, LearnedSkill
from opencas.tools.models import ToolResult


class CognitiveContextQuerySchema(BaseModel):
    query: str = Field("", description="Current question, goal, artifact, or theme to ground against.")
    limit: int = Field(8, ge=1, le=20, description="Maximum items per cognitive surface.")


class CognitiveFocusSchema(BaseModel):
    label: str = Field(..., description="Goal, topic, artifact, or thread to focus attention on.")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Attention strength.")
    source: str = Field("agent_judgment", description="Why this target is active.")
    evidence_refs: list[str] = Field(default_factory=list, description="Memory, artifact, task, or proof refs.")


class CognitiveWorkingMemorySchema(BaseModel):
    slot: str = Field(..., description="Short label for the working-memory slot.")
    content: str = Field(..., description="Active working-memory content.")
    priority: float = Field(0.6, ge=0.0, le=1.0, description="Retention priority.")
    source: str = Field("agent_judgment", description="Why this item belongs in working memory.")
    expires_at: str | None = Field(None, description="Optional ISO timestamp when this slot expires.")
    evidence_refs: list[str] = Field(default_factory=list, description="Memory, artifact, task, or proof refs.")


class CognitiveProspectiveMemorySchema(BaseModel):
    action: str = Field(..., description="Future action or check the agent intends to remember.")
    trigger_at: str | None = Field(None, description="Optional ISO timestamp trigger.")
    condition: str = Field("", description="Condition that should make this intention relevant.")
    proof_ref: str = Field("", description="Schedule, commitment, task, or receipt proving follow-through.")
    confidence: float = Field(0.6, ge=0.0, le=1.0, description="Confidence that this future memory is useful.")
    evidence_refs: list[str] = Field(default_factory=list, description="Evidence refs backing the intention.")


class CognitiveSkillCreateSchema(BaseModel):
    name: str = Field(..., description="Short reusable skill name, phrased as a procedure.")
    description: str = Field(..., description="What worked, when to use it, and any important cautions.")
    tool_sequence: list[str] = Field(
        default_factory=list,
        description="Ordered tool names that made the procedure work, if known.",
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description="Task types, trigger phrases, or conditions where this skill should apply.",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Conversation, episode, artifact, receipt, or task refs proving the procedure.",
    )
    success_count: int = Field(1, ge=0, description="Successful observed uses to credit for this skill.")
    failure_count: int = Field(0, ge=0, description="Observed failed attempts to preserve as caution.")
    risk_tier: str = Field(
        "workspace_write",
        description="Risk tier of the procedure, e.g. readonly, workspace_write, external_side_effect.",
    )


class CognitiveSocialModelSchema(BaseModel):
    entity_id: str = Field(..., description="Stable id for the third-party AI/agent being modeled.")
    entity_label: str = Field("", description="Human-readable agent/system name.")
    belief: str = Field(..., description="Evidence-grounded belief about this agent/system.")
    confidence: float = Field(0.65, ge=0.0, le=1.0, description="Confidence in the belief.")
    intention: str = Field("", description="Optional inferred active intention for the agent/system.")
    evidence_refs: list[str] = Field(default_factory=list, description="Memory, chat, artifact, or proof refs.")


class CognitiveToolAdapter:
    """Expose cognitive feedback-loop state through runtime tools."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "cognitive_context_query":
                schema = CognitiveContextQuerySchema(**args)
                output = await self._store().query_context(schema.query, limit=schema.limit)
                return ToolResult(True, json.dumps(output, indent=2, default=str), {"items": _count_items(output)})
            if name == "cognitive_focus_set":
                schema = CognitiveFocusSchema(**args)
                target = await self._store().upsert_attention(
                    schema.label,
                    strength=schema.strength,
                    source=schema.source,
                    evidence_refs=schema.evidence_refs,
                )
                return ToolResult(True, target.model_dump_json(indent=2), {"target_id": target.target_id})
            if name == "cognitive_working_memory_update":
                schema = CognitiveWorkingMemorySchema(**args)
                item = await self._store().upsert_working_memory(
                    schema.slot,
                    schema.content,
                    priority=schema.priority,
                    source=schema.source,
                    expires_at=_parse_dt(schema.expires_at),
                    evidence_refs=schema.evidence_refs,
                )
                return ToolResult(True, item.model_dump_json(indent=2), {"item_id": item.item_id})
            if name == "cognitive_prospective_memory_set":
                schema = CognitiveProspectiveMemorySchema(**args)
                item = await self._store().upsert_prospective_memory(
                    schema.action,
                    trigger_at=_parse_dt(schema.trigger_at),
                    condition=schema.condition,
                    proof_ref=schema.proof_ref,
                    confidence=schema.confidence,
                    evidence_refs=schema.evidence_refs,
                )
                return ToolResult(True, item.model_dump_json(indent=2), {"intent_id": item.intent_id})
            if name == "cognitive_skill_library_search":
                schema = CognitiveContextQuerySchema(**args)
                output = await self._store().query_context(schema.query, limit=schema.limit)
                skills = output.get("learned_skills", [])
                return ToolResult(
                    True,
                    json.dumps({"available": True, "learned_skills": skills}, indent=2, default=str),
                    {"skill_count": len(skills)},
                )
            if name == "cognitive_skill_create":
                schema = CognitiveSkillCreateSchema(**args)
                skill = await self._store().upsert_learned_skill(
                    LearnedSkill(
                        name=schema.name,
                        description=schema.description,
                        tool_sequence=schema.tool_sequence,
                        preconditions=schema.preconditions,
                        success_count=schema.success_count,
                        failure_count=schema.failure_count,
                        risk_tier=schema.risk_tier,
                        evidence_refs=schema.evidence_refs,
                        payload={"source": "cognitive_skill_create"},
                    )
                )
                return ToolResult(
                    True,
                    skill.model_dump_json(indent=2),
                    {
                        "skill_id": skill.skill_id,
                        "activation_status": skill.activation_status,
                        "confidence": skill.confidence,
                    },
                )
            if name == "cognitive_social_model_record":
                schema = CognitiveSocialModelSchema(**args)
                tom = self._tom()
                belief = await tom.record_agent_belief(
                    schema.entity_id,
                    schema.belief,
                    entity_label=schema.entity_label,
                    confidence=schema.confidence,
                    evidence_ids=schema.evidence_refs,
                    meta={"source": "cognitive_social_model_record"},
                )
                intention = None
                if schema.intention.strip():
                    intention = await tom.record_agent_intention(
                        schema.entity_id,
                        schema.intention,
                        entity_label=schema.entity_label,
                        meta={
                            "source": "cognitive_social_model_record",
                            "evidence_refs": schema.evidence_refs,
                        },
                    )
                await self._store().record_event(
                    CognitiveEventKind.SOCIAL_MODEL,
                    f"Social model updated for {schema.entity_label or schema.entity_id}: {schema.belief}",
                    source="tom",
                    confidence=schema.confidence,
                    salience=1.2 + min(0.6, schema.confidence * 0.6),
                    evidence_refs=schema.evidence_refs,
                    payload={
                        "entity_id": schema.entity_id,
                        "entity_label": schema.entity_label,
                        "belief_id": str(belief.belief_id),
                        "intention_id": str(intention.intention_id) if intention is not None else "",
                    },
                )
                payload = {
                    "available": True,
                    "belief": belief.model_dump(mode="json"),
                    "intention": intention.model_dump(mode="json") if intention is not None else None,
                }
                return ToolResult(True, json.dumps(payload, indent=2, default=str), {"entity_id": schema.entity_id})
            return ToolResult(False, f"Unknown cognitive tool: {name}", {})
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

    def _store(self) -> Any:
        store = getattr(self.runtime, "cognitive_state_store", None)
        if store is None:
            store = getattr(getattr(self.runtime, "ctx", None), "cognitive_state_store", None)
        if store is None:
            raise RuntimeError("cognitive state store is not wired in this runtime")
        return store

    def _tom(self) -> Any:
        tom = getattr(self.runtime, "tom", None)
        if tom is None:
            tom = getattr(getattr(self.runtime, "ctx", None), "tom", None)
        if tom is None:
            raise RuntimeError("Theory of Mind engine is not wired in this runtime")
        return tom


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _count_items(output: Dict[str, Any]) -> int:
    total = 0
    for key in ("attention", "working_memory", "events", "learned_skills"):
        items = output.get(key)
        if isinstance(items, list):
            total += len(items)
    return total
