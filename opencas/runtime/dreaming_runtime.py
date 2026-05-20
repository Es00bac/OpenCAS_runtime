"""Runtime hooks for nightly dreaming."""

from __future__ import annotations

from typing import Any, Dict

from opencas.dreaming import DreamMode
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind


async def run_runtime_nightly_dream(
    runtime: Any,
    *,
    mode: DreamMode | str = DreamMode.LIGHT,
    consolidation_result: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the nightly dream service when available."""

    service = getattr(runtime, "nightly_dreaming", None)
    if service is None:
        return {
            "available": False,
            "reason": "nightly_dreaming_unavailable",
        }
    record = await service.run(
        mode=mode,
        consolidation_result=consolidation_result or {},
        runtime=runtime,
    )
    await _record_nightly_dream_proof(runtime, record)
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(
            "nightly_dream_complete",
            {
                "dream_id": str(record.dream_id),
                "mode": record.mode.value,
                "artifact_path": record.artifact_path,
            },
        )
    return {
        "available": True,
        "dream_id": str(record.dream_id),
        "mode": record.mode.value,
        "dream_for_date": record.meta.get("dream_for_date"),
        "required_nightly_dream": bool(record.meta.get("required_nightly_dream")),
        "summary": record.summary,
        "insights": record.insights,
        "curiosity_seeds": record.curiosity_seeds,
        "action_candidates": record.action_candidates,
        "artifact_path": record.artifact_path,
    }


async def _record_nightly_dream_proof(runtime: Any, record: Any) -> None:
    proof_chain = getattr(runtime, "proof_chain", None)
    if proof_chain is None:
        return
    try:
        claim = await proof_chain.record_claim(
            claim_type=ProofClaimType.DREAM,
            claim=f"Nightly {record.mode.value} dream completed.",
            subject="nightly_dream",
            meta={
                "dream_id": str(record.dream_id),
                "artifact_path": record.artifact_path,
            },
        )
        await proof_chain.link_evidence(
            claim.claim_id,
            evidence_kind=ProofEvidenceKind.DREAM_RECORD,
            evidence_id=f"dream:{record.dream_id}",
            summary=record.summary,
            supports_claim=True,
            meta={"artifact_path": record.artifact_path},
        )
    except Exception as exc:
        trace = getattr(runtime, "_trace", None)
        if callable(trace):
            trace(
                "proof_chain_nightly_dream_failed",
                {
                    "dream_id": str(getattr(record, "dream_id", "")),
                    "error": str(exc),
                },
            )
