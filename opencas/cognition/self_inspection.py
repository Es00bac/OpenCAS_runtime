"""Self-inspection records for response drift, pressure source, and promise gaps."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .grounding import CognitionGrounding, GroundingKind, GroundingSource


class SelfInspectionPhase(str, Enum):
    """Turn phases where OpenCAS can inspect its own assumptions and output."""

    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"


class CommitmentGapStatus(str, Enum):
    """Operator-facing lifecycle state for a commitment gap observation."""

    OPEN = "open"
    WATCH = "watch"
    RESOLVED = "resolved"


class ResponseShapeSignature(BaseModel):
    """Compact structural signature for a generated response."""

    structure: str = "empty"
    word_count: int = 0
    paragraph_count: int = 0
    bullet_count: int = 0
    question_count: int = 0
    promise_count: int = 0
    uncertainty_count: int = 0
    source_claim_count: int = 0
    warmth_count: int = 0
    shape_tags: list[str] = Field(default_factory=list)

    @field_validator("shape_tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        seen: set[str] = set()
        tags: list[str] = []
        for item in value:
            tag = "_".join(str(item or "").strip().lower().split())
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags


class DriftObservation(BaseModel):
    """A non-punitive observation that a response shape may be repeating."""

    reason: str
    severity: str = "notice"
    route: str = "observe"
    addressed_by_outcome_id: str = ""
    addressed_at: datetime | None = None
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason", "severity", "route", "addressed_by_outcome_id")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class ValenceSourceTag(BaseModel):
    """Evidence-backed candidate cause for a somatic or control-state shift."""

    source: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "reason")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, parsed))


class ToolUseInspection(BaseModel):
    """A searchable record of why a tool was selected for a turn."""

    tool_name: str
    call_id: str = ""
    reason: str
    objective: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name", "call_id", "reason", "objective")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("evidence_used", mode="before")
    @classmethod
    def _normalize_evidence_used(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item or "").strip()]


class ToolCallTransit(BaseModel):
    """Mechanical transit evidence for one tool call inside a chain."""

    chain_id: str
    call_id: str
    tool_name: str
    entry_intent: str
    trust_context: str = "unknown"
    success: bool = False
    duration_ms: int = Field(default=0, ge=0)
    result_shape: dict[str, Any] = Field(default_factory=dict)
    certainty_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    somatic_delta: dict[str, float] = Field(default_factory=dict)
    task_mutation: bool = False
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chain_id", "call_id", "tool_name", "entry_intent", "trust_context")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("result_shape", "somatic_delta", "meta", mode="before")
    @classmethod
    def _normalize_mapping(cls, value: Any) -> dict[str, Any]:
        return dict(value or {}) if isinstance(value, dict) else {}


class ToolChainTransitSummary(BaseModel):
    """Derived contour for a tool chain, anchored to concrete call IDs."""

    chain_id: str
    objective: str = ""
    entry_intent: str = ""
    trust_context: str = "unknown"
    call_ids: list[str] = Field(default_factory=list)
    call_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    total_duration_ms: int = Field(default=0, ge=0)
    result_shape_counts: dict[str, int] = Field(default_factory=dict)
    certainty_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    somatic_delta: dict[str, float] = Field(default_factory=dict)
    load_type: str = "none"
    intent_drift_score: float = Field(default=0.0, ge=0.0, le=1.0)
    meaning_scope: str = "private_operational_evidence"
    undercoupled_somatic: bool = False
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chain_id", "objective", "entry_intent", "trust_context", "load_type", "meaning_scope")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("call_ids", mode="before")
    @classmethod
    def _normalize_call_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item or "").strip()]

    @field_validator("result_shape_counts", "somatic_delta", "meta", mode="before")
    @classmethod
    def _normalize_mapping(cls, value: Any) -> dict[str, Any]:
        return dict(value or {}) if isinstance(value, dict) else {}


class CommitmentGap(BaseModel):
    """A compact record of a promised/done/not-yet-linked gap."""

    commitment_id: str
    promised: str
    gap_type: str
    status: CommitmentGapStatus = CommitmentGapStatus.WATCH
    likely_cause: str = ""
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("commitment_id", "promised", "gap_type", "likely_cause")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class SelfInspectionRecord(BaseModel):
    """A durable pre/post turn self-inspection packet."""

    record_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    phase: SelfInspectionPhase
    response_signature: ResponseShapeSignature | None = None
    drift_observations: list[DriftObservation] = Field(default_factory=list)
    valence_sources: list[ValenceSourceTag] = Field(default_factory=list)
    tool_use_inspections: list[ToolUseInspection] = Field(default_factory=list)
    tool_call_transits: list[ToolCallTransit] = Field(default_factory=list)
    tool_chain_summary: ToolChainTransitSummary | None = None
    commitment_gaps: list[CommitmentGap] = Field(default_factory=list)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def _normalize_session_id(cls, value: str) -> str:
        return " ".join(str(value or "").split()) or "default"


_PROMISE_PHRASES = (
    "i will",
    "i'll",
    "i am going to",
    "i'm going to",
    "i promise",
    "i plan to",
    "i need to",
)
_UNCERTAINTY_PHRASES = (
    "i do not know",
    "i don't know",
    "not sure",
    "uncertain",
    "i cannot tell",
    "i can't tell",
    "memory gap",
)
_SOURCE_PHRASES = (
    "evidence",
    "source",
    "memory",
    "record",
    "stored",
    "retrieved",
    "verified",
)
_WARMTH_PHRASES = (
    "i care",
    "with you",
    "i understand",
    "i hear you",
    "that matters",
)


def build_response_shape_signature(text: str) -> ResponseShapeSignature:
    """Return a compact response-shape signature without rewriting the response."""

    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return ResponseShapeSignature()

    lines = [line.strip() for line in raw.splitlines()]
    nonempty_lines = [line for line in lines if line]
    paragraphs = [part for part in raw.split("\n\n") if part.strip()]
    lowered = stripped.lower()
    words = stripped.split()
    bullet_count = sum(1 for line in nonempty_lines if line.startswith(("- ", "* ")))
    question_count = stripped.count("?")
    promise_count = _phrase_count(lowered, _PROMISE_PHRASES)
    uncertainty_count = _phrase_count(lowered, _UNCERTAINTY_PHRASES)
    source_claim_count = _phrase_count(lowered, _SOURCE_PHRASES)
    warmth_count = _phrase_count(lowered, _WARMTH_PHRASES)

    if bullet_count:
        structure = "bullet_list"
    elif len(paragraphs) > 1:
        structure = "multi_paragraph"
    else:
        structure = "paragraph"

    tags: list[str] = [structure]
    if promise_count:
        tags.append("promise_language")
    if uncertainty_count:
        tags.append("uncertainty_language")
    if source_claim_count:
        tags.append("source_grounding")
    if warmth_count:
        tags.append("warmth_language")
    if question_count:
        tags.append("asks_question")

    return ResponseShapeSignature(
        structure=structure,
        word_count=len(words),
        paragraph_count=len(paragraphs),
        bullet_count=bullet_count,
        question_count=question_count,
        promise_count=promise_count,
        uncertainty_count=uncertainty_count,
        source_claim_count=source_claim_count,
        warmth_count=warmth_count,
        shape_tags=tags,
    )


def build_pre_turn_self_inspection_record(
    *,
    session_id: str,
    user_input: str,
    active_commitments: list[Any] | None = None,
) -> SelfInspectionRecord:
    """Capture assumptions and promise pressure before the response is generated."""

    commitments = list(active_commitments or [])
    commitment_ids = [str(getattr(item, "commitment_id", "")) for item in commitments]
    grounding: list[CognitionGrounding] = [
        CognitionGrounding(
            kind=GroundingKind.OBSERVED,
            source=GroundingSource.USER,
            claim="Current user turn received for response-time self-inspection.",
            subject="conversation_turn",
            evidence_ids=[],
            confidence=1.0,
        )
    ]
    if commitment_ids:
        grounding.append(
            CognitionGrounding(
                kind=GroundingKind.OBSERVED,
                source=GroundingSource.RUNTIME,
                claim="Active commitments were present before responding.",
                subject="commitment_pressure",
                evidence_ids=commitment_ids[:10],
                confidence=0.8,
            )
        )
    return SelfInspectionRecord(
        session_id=session_id,
        phase=SelfInspectionPhase.PRE_TURN,
        grounding=grounding,
        meta={
            "user_input_excerpt": _excerpt(user_input, 240),
            "active_commitment_count": len(commitments),
            "active_commitment_ids": commitment_ids[:10],
        },
    )


def build_post_turn_self_inspection_record(
    *,
    session_id: str,
    user_input: str,
    assistant_output: str,
    captured_commitments: list[Any] | None = None,
    integrity_review: dict[str, Any] | None = None,
    tool_use_inspections: list[ToolUseInspection] | None = None,
    tool_call_transits: list[ToolCallTransit] | None = None,
    tool_chain_summary: ToolChainTransitSummary | None = None,
    pre_somatic_state: Any | None = None,
    post_somatic_state: Any | None = None,
    recent_records: list[SelfInspectionRecord] | None = None,
) -> SelfInspectionRecord:
    """Capture response shape, pressure tags, and promise gaps after a turn."""

    signature = build_response_shape_signature(assistant_output)
    captured = list(captured_commitments or [])
    review = dict(integrity_review or {})
    drift = _build_drift_observations(
        signature=signature,
        user_input=user_input,
        assistant_output=assistant_output,
        integrity_review=review,
        recent_records=list(recent_records or []),
    )
    gaps = [_commitment_to_gap(item) for item in captured if _commitment_has_gap(item)]
    valence_sources = _build_valence_sources(
        user_input=user_input,
        assistant_output=assistant_output,
        captured_commitments=captured,
        integrity_review=review,
        pre_somatic_state=pre_somatic_state,
        post_somatic_state=post_somatic_state,
    )
    return SelfInspectionRecord(
        session_id=session_id,
        phase=SelfInspectionPhase.POST_TURN,
        response_signature=signature,
        drift_observations=drift,
        valence_sources=valence_sources,
        tool_use_inspections=list(tool_use_inspections or []),
        tool_call_transits=list(tool_call_transits or []),
        tool_chain_summary=tool_chain_summary,
        commitment_gaps=gaps,
        grounding=[
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                claim="Post-turn self-inspection record derived from generated response and runtime state.",
                subject="conversation_turn",
                confidence=0.82,
            )
        ],
        meta={
            "user_input_excerpt": _excerpt(user_input, 240),
            "assistant_output_excerpt": _excerpt(assistant_output, 320),
            "integrity_review": review,
            "captured_commitment_count": len(captured),
        },
    )


def _build_drift_observations(
    *,
    signature: ResponseShapeSignature,
    user_input: str,
    assistant_output: str,
    integrity_review: dict[str, Any],
    recent_records: list[SelfInspectionRecord],
) -> list[DriftObservation]:
    observations: list[DriftObservation] = []
    reasons = integrity_review.get("reasons") if isinstance(integrity_review, dict) else None
    if isinstance(reasons, list) and reasons:
        observations.append(
            DriftObservation(
                reason="Response-integrity review noted: " + "; ".join(str(item) for item in reasons[:3]),
                severity="revision" if integrity_review.get("revised") else "notice",
                route="revise" if integrity_review.get("revised") else "observe",
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.DERIVED,
                        source=GroundingSource.RUNTIME,
                        claim="Response-integrity review contributed drift evidence.",
                        confidence=0.75,
                    )
                ],
            )
        )

    interface_observation = _borrowed_operator_interface_observation(
        user_input=user_input,
        assistant_output=assistant_output,
    )
    if interface_observation is not None:
        observations.append(interface_observation)

    prior_signatures = [
        record.response_signature
        for record in recent_records
        if record.phase == SelfInspectionPhase.POST_TURN and record.response_signature is not None
    ]
    similar_structures = sum(1 for item in prior_signatures if item.structure == signature.structure)
    similar_tags = sum(
        1
        for item in prior_signatures
        if len(set(item.shape_tags).intersection(signature.shape_tags)) >= 2
    )
    if prior_signatures and (similar_structures >= 3 or similar_tags >= 3):
        observations.append(
            DriftObservation(
                reason="Recent assistant turns used a similar response shape; verify it is still evidence-fit.",
                severity="notice",
                route="observe",
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.DERIVED,
                        source=GroundingSource.RUNTIME,
                        claim="Stored response-shape signatures repeated across recent turns.",
                        confidence=0.64,
                        meta={
                            "current_structure": signature.structure,
                            "similar_structures": similar_structures,
                            "similar_tags": similar_tags,
                        },
                    )
                ],
            )
        )
    return observations


def build_tool_use_inspections(
    *,
    objective: str,
    tool_calls: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> list[ToolUseInspection]:
    """Build searchable tool-use intent records from a tool-loop result."""

    calls = list(tool_calls or [])
    if not calls:
        return []
    assistant_reasons = _assistant_tool_call_reasons(messages or [])
    inspections: list[ToolUseInspection] = []
    for call in calls:
        name = str(call.get("name") or call.get("function", {}).get("name") or "").strip()
        if not name:
            continue
        call_id = str(call.get("id") or "")
        reason = assistant_reasons.get(call_id) or assistant_reasons.get(name) or "Model selected this tool for the current objective."
        inspections.append(
            ToolUseInspection(
                tool_name=name,
                call_id=call_id,
                reason=reason,
                objective=_excerpt(objective, 240),
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.OBSERVED,
                        source=GroundingSource.RUNTIME,
                        claim="Tool call selected during the conversation tool loop.",
                        evidence_ids=[call_id] if call_id else [],
                        confidence=0.74 if reason else 0.55,
                    )
                ],
                meta={"arguments": call.get("args", {})},
            )
        )
    return inspections


def build_tool_chain_transit_summary(
    *,
    objective: str,
    chain_id: str,
    call_transits: list[ToolCallTransit] | None = None,
) -> ToolChainTransitSummary:
    """Derive a chain contour from concrete tool-call transit records."""

    calls = [call for call in list(call_transits or []) if call.chain_id == chain_id]
    call_ids = [call.call_id for call in calls]
    success_count = sum(1 for call in calls if call.success)
    failure_count = len(calls) - success_count
    certainty_delta = round(max(-1.0, min(1.0, sum(call.certainty_delta for call in calls))), 3)
    result_shape_counts = _count_result_shape_tags(calls)
    somatic_delta = _sum_transit_somatic_delta(calls)
    entry_intents = [call.entry_intent for call in calls if call.entry_intent]
    entry_intent = entry_intents[0] if entry_intents else _excerpt(objective, 240)
    trust_context = _most_common([call.trust_context for call in calls if call.trust_context]) or "unknown"
    intent_drift_score = _intent_drift_score(entry_intents)
    undercoupled_somatic = _is_undercoupled_somatic(
        call_count=len(calls),
        certainty_delta=certainty_delta,
        somatic_delta=somatic_delta,
    )
    return ToolChainTransitSummary(
        chain_id=chain_id,
        objective=_excerpt(objective, 240),
        entry_intent=entry_intent,
        trust_context=trust_context,
        call_ids=call_ids,
        call_count=len(calls),
        success_count=success_count,
        failure_count=failure_count,
        total_duration_ms=sum(call.duration_ms for call in calls),
        result_shape_counts=result_shape_counts,
        certainty_delta=certainty_delta,
        somatic_delta=somatic_delta,
        load_type=_infer_transit_load_type(
            result_shape_counts=result_shape_counts,
            certainty_delta=certainty_delta,
            failure_count=failure_count,
        ),
        intent_drift_score=intent_drift_score,
        meaning_scope=_infer_meaning_scope(trust_context),
        undercoupled_somatic=undercoupled_somatic,
        grounding=[
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                claim="Tool chain transit summary derived from concrete tool-call transit records.",
                subject="tool_chain_transit",
                evidence_ids=call_ids,
                confidence=0.84 if call_ids else 0.35,
            )
        ],
        meta={"source": "tool_call_transits"},
    )


def _borrowed_operator_interface_observation(
    *,
    user_input: str,
    assistant_output: str,
) -> DriftObservation | None:
    user_lower = str(user_input or "").lower()
    output_lower = str(assistant_output or "").lower()
    operator_interface_terms = ("keyboard", "mouse", "arrow keys", "click", "hands")
    agent_interface_terms = ("tool call", "context", "memory", "retrieval", "output generation")
    borrowed_terms = [term for term in operator_interface_terms if term in output_lower]
    if not borrowed_terms:
        return None
    if any(term in output_lower for term in agent_interface_terms):
        return None
    if not any(term in user_lower for term in operator_interface_terms):
        return None
    return DriftObservation(
        reason="Response appears to borrow the operator interface instead of grounding in OpenCAS's actual tool/context/memory interface.",
        severity="notice",
        route="inspect",
        grounding=[
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                claim="Assistant output reused human interface terms from the user turn without agent-interface grounding.",
                confidence=0.68,
                meta={"borrowed_terms": borrowed_terms},
            )
        ],
    )


def _assistant_tool_call_reasons(messages: list[dict[str, Any]]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        content = _excerpt(str(message.get("content") or ""), 260)
        if not content:
            continue
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            call_id = str(raw_call.get("id") or "")
            function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
            name = str(raw_call.get("name") or function.get("name") or "")
            if call_id:
                reasons[call_id] = content
            if name:
                reasons[name] = content
    return reasons


def _build_valence_sources(
    *,
    user_input: str,
    assistant_output: str,
    captured_commitments: list[Any],
    integrity_review: dict[str, Any],
    pre_somatic_state: Any | None,
    post_somatic_state: Any | None,
) -> list[ValenceSourceTag]:
    sources: list[ValenceSourceTag] = []
    if captured_commitments:
        ids = [str(getattr(item, "commitment_id", "")) for item in captured_commitments]
        sources.append(
            ValenceSourceTag(
                source="commitment_pressure",
                confidence=0.72,
                reason="The response created or preserved a durable commitment.",
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.OBSERVED,
                        source=GroundingSource.RUNTIME,
                        claim="Self-commitments were captured from the assistant response.",
                        evidence_ids=ids,
                        confidence=0.85,
                    )
                ],
            )
        )
    if integrity_review.get("revised") or integrity_review.get("review_error"):
        sources.append(
            ValenceSourceTag(
                source="response_integrity_pressure",
                confidence=0.7,
                reason="Response-integrity review intervened or failed during the turn.",
            )
        )
    lowered_user = str(user_input or "").lower()
    if any(token in lowered_user for token in ("wrong", "still", "doesn't", "does not", "fix", "broken")):
        sources.append(
            ValenceSourceTag(
                source="operator_correction",
                confidence=0.58,
                reason="The user turn appears to carry correction or repair pressure.",
            )
        )
    if str(assistant_output or "").lstrip().startswith("[Error generating response:"):
        sources.append(
            ValenceSourceTag(
                source="runtime_failure",
                confidence=0.9,
                reason="The assistant output came from the conversation error path.",
            )
        )
    somatic_delta = _somatic_delta(pre_somatic_state, post_somatic_state)
    if somatic_delta:
        sources.append(
            ValenceSourceTag(
                source="somatic_shift",
                confidence=0.55,
                reason="Somatic state changed while generating or reconciling the response.",
                meta={"delta": somatic_delta},
            )
        )
    if not sources:
        sources.append(
            ValenceSourceTag(
                source="conversation_turn",
                confidence=0.35,
                reason="No specific pressure source dominated this turn.",
            )
        )
    return sources


def _commitment_to_gap(commitment: Any) -> CommitmentGap:
    commitment_id = str(getattr(commitment, "commitment_id", ""))
    promised = str(getattr(commitment, "content", "") or "")
    return CommitmentGap(
        commitment_id=commitment_id,
        promised=promised,
        gap_type="captured_without_execution_link",
        status=CommitmentGapStatus.OPEN,
        likely_cause="new_conversation_commitment",
        grounding=[
            CognitionGrounding(
                kind=GroundingKind.OBSERVED,
                source=GroundingSource.RUNTIME,
                claim="A commitment was captured before any work or task link existed.",
                evidence_ids=[commitment_id] if commitment_id else [],
                confidence=0.82,
            )
        ],
    )


def _commitment_has_gap(commitment: Any) -> bool:
    linked_work = list(getattr(commitment, "linked_work_ids", []) or [])
    linked_tasks = list(getattr(commitment, "linked_task_ids", []) or [])
    return not linked_work and not linked_tasks


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(text.count(phrase) for phrase in phrases)


def _somatic_delta(pre_state: Any | None, post_state: Any | None) -> dict[str, float]:
    if pre_state is None or post_state is None:
        return {}
    delta: dict[str, float] = {}
    for key in ("valence", "arousal", "fatigue", "tension", "focus", "energy"):
        try:
            before = float(getattr(pre_state, key))
            after = float(getattr(post_state, key))
        except (TypeError, ValueError):
            continue
        change = round(after - before, 3)
        if abs(change) >= 0.01:
            delta[key] = change
    return delta


def _count_result_shape_tags(calls: list[ToolCallTransit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in calls:
        tags = call.result_shape.get("tags") if isinstance(call.result_shape, dict) else None
        if not isinstance(tags, list):
            continue
        for tag in tags:
            normalized = "_".join(str(tag or "").strip().lower().split())
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _sum_transit_somatic_delta(calls: list[ToolCallTransit]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for call in calls:
        for key, value in call.somatic_delta.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            totals[key] = round(totals.get(key, 0.0) + parsed, 3)
    return {key: value for key, value in totals.items() if abs(value) >= 0.01}


def _most_common(items: list[str]) -> str:
    if not items:
        return ""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return max(counts.items(), key=lambda pair: pair[1])[0]


def _intent_drift_score(entry_intents: list[str]) -> float:
    unique = {" ".join(item.lower().split()) for item in entry_intents if item}
    if len(unique) <= 1:
        return 0.0
    return round(min(1.0, (len(unique) - 1) / max(1, len(entry_intents))), 3)


def _is_undercoupled_somatic(
    *,
    call_count: int,
    certainty_delta: float,
    somatic_delta: dict[str, float],
) -> bool:
    if call_count < 2 or abs(certainty_delta) < 0.03:
        return False
    tension = abs(float(somatic_delta.get("tension", 0.0) or 0.0))
    fatigue = abs(float(somatic_delta.get("fatigue", 0.0) or 0.0))
    return tension < 0.01 and fatigue < 0.01


def _infer_transit_load_type(
    *,
    result_shape_counts: dict[str, int],
    certainty_delta: float,
    failure_count: int,
) -> str:
    if failure_count or result_shape_counts.get("blocked") or result_shape_counts.get("retry"):
        return "operational"
    if certainty_delta < -0.02 or result_shape_counts.get("ambiguous"):
        return "cognitive"
    return "cognitive" if result_shape_counts else "none"


def _infer_meaning_scope(trust_context: str) -> str:
    if trust_context == "operator_requested":
        return "potential_shared_meaning"
    return "private_operational_evidence"


def _excerpt(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
