"""Autonomous consumer for ShadowRegistry feedback clusters."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from opencas.cognition import CognitiveEventKind, LearnedSkill, recommended_counterfactual
from opencas.governance import BlockedIntention, ClusterTriageStatus
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind
from opencas.telemetry import EventKind

MIN_DISMISSAL_AGE = timedelta(hours=24)
TIME_DECAY_AGE = timedelta(days=7)
LESSON_CLUSTER_MIN = 5
DEFAULT_MAX_DISMISSALS = 20


@dataclass(frozen=True)
class ShadowClusterCandidate:
    fingerprint: str
    entries: list[BlockedIntention]
    latest: BlockedIntention
    age: timedelta
    tools: tuple[str, ...]
    primary_tool: str
    reason: str
    lesson_key: tuple[str, str] | None
    policy: str


async def consume_shadow_registry(
    runtime: Any,
    *,
    now: datetime | None = None,
    max_dismissals: int = DEFAULT_MAX_DISMISSALS,
) -> dict[str, Any]:
    """Consume eligible shadow clusters into artifacts, proof, and dismissal state."""
    registry = _shadow_registry(runtime)
    cognitive_store = _cognitive_store(runtime)
    proof_chain = getattr(runtime, "proof_chain", None)
    if registry is None or cognitive_store is None or proof_chain is None:
        return {
            "available": False,
            "reason": "shadow_registry_cognitive_store_or_proof_chain_unavailable",
            "dismissed_count": 0,
        }

    now = _aware_utc(now or datetime.now(timezone.utc))
    grouped = _active_clusters(registry)
    candidates = _eligible_candidates(grouped, registry=registry, now=now)
    selected = candidates[: max(0, int(max_dismissals))]
    if not selected:
        return {
            "available": True,
            "candidate_count": 0,
            "dismissed_count": 0,
            "learned_skill_count": 0,
            "counterfactual_count": 0,
            "proof_claim_count": 0,
        }

    artifact_refs: dict[str, str] = {}
    learned_skill_count = 0
    counterfactual_count = 0
    for lesson_key, cohort in _selected_lesson_cohorts(selected).items():
        skill = await _record_learned_skill(cognitive_store, lesson_key=lesson_key, cohort=cohort)
        learned_skill_count += 1
        artifact_ref = f"learned_skill:{skill.skill_id}"
        for candidate in cohort:
            artifact_refs[candidate.fingerprint] = artifact_ref

    for candidate in selected:
        if candidate.fingerprint in artifact_refs:
            continue
        event = await _record_counterfactual_event(cognitive_store, candidate)
        counterfactual_count += 1
        artifact_refs[candidate.fingerprint] = f"cognitive_event:{event.event_id}"

    dismissed_count = 0
    proof_claim_count = 0
    for candidate in selected:
        artifact_ref = artifact_refs[candidate.fingerprint]
        claim = await _record_dismissal_proof(
            proof_chain,
            candidate=candidate,
            artifact_ref=artifact_ref,
        )
        proof_claim_count += 1
        detail = registry.triage_cluster(
            candidate.fingerprint,
            annotation=(
                f"auto-dismissed by shadow_consumer; policy={candidate.policy}; "
                f"artifact={artifact_ref}; proof_claim={claim.claim_id}; reversible=true"
            ),
            dismissed=True,
        )
        if detail.get("triage_status") == ClusterTriageStatus.DISMISSED.value:
            dismissed_count += 1
            _trace_dismissal(runtime, candidate, artifact_ref=artifact_ref, proof_claim_id=str(claim.claim_id))

    return {
        "available": True,
        "candidate_count": len(candidates),
        "dismissed_count": dismissed_count,
        "learned_skill_count": learned_skill_count,
        "counterfactual_count": counterfactual_count,
        "proof_claim_count": proof_claim_count,
        "bounded": len(candidates) > len(selected),
        "max_dismissals": max(0, int(max_dismissals)),
    }


def _shadow_registry(runtime: Any) -> Any:
    return getattr(getattr(runtime, "ctx", None), "shadow_registry", None) or getattr(
        runtime,
        "shadow_registry",
        None,
    )


def _cognitive_store(runtime: Any) -> Any:
    return getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )


def _active_clusters(registry: Any) -> dict[str, list[BlockedIntention]]:
    grouped: dict[str, list[BlockedIntention]] = defaultdict(list)
    for item in registry.store.list_all():
        grouped[item.fingerprint or item.id].append(item)
    active: dict[str, list[BlockedIntention]] = {}
    for fingerprint, entries in grouped.items():
        state = registry.store.get_cluster_state(fingerprint)
        if state is not None and state.triage_status == ClusterTriageStatus.DISMISSED:
            continue
        active[fingerprint] = entries
    return active


def _eligible_candidates(
    grouped: dict[str, list[BlockedIntention]],
    *,
    registry: Any,
    now: datetime,
) -> list[ShadowClusterCandidate]:
    records: list[tuple[str, list[BlockedIntention], BlockedIntention, timedelta, tuple[str, ...], str]] = []
    for fingerprint, entries in grouped.items():
        latest = max(entries, key=lambda item: item.captured_at)
        age = now - _aware_utc(latest.captured_at)
        tools = tuple(sorted({entry.tool_name for entry in entries if entry.tool_name}))
        records.append(
            (
                fingerprint,
                entries,
                latest,
                age,
                tools,
                latest.block_reason.value,
            )
        )

    cohort_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _fingerprint, _entries, _latest, _age, tools, reason in records:
        for tool_name in tools:
            cohort_counts[(tool_name, reason)] += 1

    candidates: list[ShadowClusterCandidate] = []
    for fingerprint, entries, latest, age, tools, reason in records:
        if age < MIN_DISMISSAL_AGE:
            continue
        lesson_key = next(
            (
                (tool_name, reason)
                for tool_name in tools
                if cohort_counts[(tool_name, reason)] >= LESSON_CLUSTER_MIN
            ),
            None,
        )
        policy = ""
        if lesson_key is not None:
            policy = "learned_skill_cohort"
        elif age >= TIME_DECAY_AGE:
            policy = "time_decay_counterfactual"
        if not policy:
            continue
        candidates.append(
            ShadowClusterCandidate(
                fingerprint=fingerprint,
                entries=entries,
                latest=latest,
                age=age,
                tools=tools,
                primary_tool=latest.tool_name,
                reason=reason,
                lesson_key=lesson_key,
                policy=policy,
            )
        )
    candidates.sort(key=lambda item: (item.age, len(item.entries)), reverse=True)
    return candidates


def _selected_lesson_cohorts(
    selected: list[ShadowClusterCandidate],
) -> dict[tuple[str, str], list[ShadowClusterCandidate]]:
    cohorts: dict[tuple[str, str], list[ShadowClusterCandidate]] = defaultdict(list)
    for candidate in selected:
        if candidate.lesson_key is not None:
            cohorts[candidate.lesson_key].append(candidate)
    return cohorts


async def _record_learned_skill(
    store: Any,
    *,
    lesson_key: tuple[str, str],
    cohort: list[ShadowClusterCandidate],
) -> LearnedSkill:
    tool_name, reason = lesson_key
    evidence_refs = [f"shadow_registry:{candidate.fingerprint}" for candidate in cohort]
    skill = LearnedSkill(
        skill_id=f"shadow:{tool_name}:{reason}",
        name=f"Shadow lesson for {tool_name} {reason}",
        description=_lesson_description(tool_name, reason, cohort),
        tool_sequence=[tool_name],
        preconditions=_lesson_preconditions(reason),
        failure_count=len(evidence_refs),
        risk_tier=_risk_tier_for_tool(tool_name),
        evidence_refs=evidence_refs,
        payload={
            "source": "shadow_consumer",
            "block_reason": reason,
            "cluster_fingerprints": [candidate.fingerprint for candidate in cohort],
        },
    )
    return await store.upsert_learned_skill(skill)


async def _record_counterfactual_event(store: Any, candidate: ShadowClusterCandidate) -> Any:
    counterfactual = recommended_counterfactual(
        objective=candidate.latest.intent_summary,
        failure_summary=candidate.latest.block_context,
        prior_tool=candidate.primary_tool,
        available_tools=candidate.tools,
        prior_attempts=len(candidate.entries),
    )
    return await store.record_event(
        CognitiveEventKind.COUNTERFACTUAL,
        f"Shadow counterfactual for {candidate.primary_tool} {candidate.reason}",
        content=candidate.latest.block_context,
        source="shadow_consumer",
        confidence=0.72,
        salience=1.2,
        evidence_refs=[f"shadow_registry:{candidate.fingerprint}"],
        payload={
            "fingerprint": candidate.fingerprint,
            "tool_name": candidate.primary_tool,
            "block_reason": candidate.reason,
            "count": len(candidate.entries),
            "latest_captured_at": candidate.latest.captured_at.isoformat(),
            "counterfactual": counterfactual,
        },
    )


async def _record_dismissal_proof(
    proof_chain: Any,
    *,
    candidate: ShadowClusterCandidate,
    artifact_ref: str,
) -> Any:
    claim = await proof_chain.record_claim(
        claim_type=ProofClaimType.MAINTENANCE,
        claim=(
            f"Shadow cluster {candidate.fingerprint} was dismissed only after downstream "
            f"learning artifact {artifact_ref} was recorded."
        ),
        subject=candidate.fingerprint,
        meta={
            "source": "shadow_consumer",
            "policy": candidate.policy,
            "artifact_ref": artifact_ref,
        },
    )
    await proof_chain.link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.RUNTIME_EVENT,
        evidence_id=f"shadow_registry:{candidate.fingerprint}",
        summary=f"Shadow cluster {candidate.fingerprint} dismissal candidate",
        supports_claim=True,
        confidence=0.85,
        meta={"policy": candidate.policy, "block_reason": candidate.reason},
    )
    await proof_chain.link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.RUNTIME_EVENT,
        evidence_id=artifact_ref,
        summary=f"Downstream artifact recorded before dismissal: {artifact_ref}",
        supports_claim=True,
        confidence=0.85,
        meta={"policy": candidate.policy},
    )
    return claim


def _trace_dismissal(
    runtime: Any,
    candidate: ShadowClusterCandidate,
    *,
    artifact_ref: str,
    proof_claim_id: str,
) -> None:
    payload = {
        "fingerprint": candidate.fingerprint,
        "policy": candidate.policy,
        "reason": candidate.reason,
        "tool_name": candidate.primary_tool,
        "artifact_ref": artifact_ref,
        "proof_claim_id": proof_claim_id,
        "reversible": True,
    }
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace("shadow_consumer_cluster_dismissed", payload)
        return
    tracer = getattr(runtime, "tracer", None)
    if tracer is not None:
        tracer.log(EventKind.TOM_EVAL, "ShadowConsumer: cluster dismissed", payload)


def _lesson_description(
    tool_name: str,
    reason: str,
    cohort: list[ShadowClusterCandidate],
) -> str:
    examples = "; ".join(candidate.latest.intent_summary for candidate in cohort[:3])
    return (
        f"{len(cohort)} shadow clusters show {tool_name} repeatedly hit {reason}. "
        f"Before retrying this shape, apply the preconditions and preserve fresh evidence. "
        f"Examples: {examples}"
    )


def _lesson_preconditions(reason: str) -> list[str]:
    if reason == "approval_denied":
        return [
            "Verify operator scope and permission boundary before retrying.",
            "Prefer a read-only or narrower alternative when possible.",
            "Carry explicit evidence for why the action is now allowed.",
        ]
    if reason == "validation_blocked":
        return [
            "Normalize paths under allowed roots before the tool call.",
            "Check required argument schema before retrying.",
        ]
    if reason == "tool_loop_guard_blocked":
        return [
            "Do not repeat identical tool arguments after a guard block.",
            "Change the evidence target or stop and summarize the blocker.",
        ]
    if reason == "retry_blocked":
        return [
            "Require fresh evidence before reopening a low-divergence retry.",
            "Use a materially different framing, not cosmetic rewording.",
        ]
    return ["Verify the blocker is stale and choose a changed strategy before retrying."]


def _risk_tier_for_tool(tool_name: str) -> str:
    if tool_name in {"web_fetch", "http_request"}:
        return "network"
    if tool_name in {"bash_run_command", "python_repl"}:
        return "shell_local"
    return "workspace_write"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
