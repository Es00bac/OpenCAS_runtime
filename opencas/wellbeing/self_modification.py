"""Reviewable self-modification proposals from wellbeing follow-up signals."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opencas.identity.agent_name import resolve_agent_name
from opencas.thread_registry import BeadSourceKind

from .models import MaintenanceOutcome, SelfModificationProposal, WellbeingState

PROPOSAL_THREAD_ANCHOR_ID = "wellbeing-self-modification-proposals"


class SelfModificationProposalGenerator:
    """Create idempotent review artifacts for wellbeing follow-up actions."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.store = getattr(runtime, "wellbeing_store", None)

    async def ensure_for_followup(
        self,
        *,
        next_followup_action: dict[str, Any] | None,
        state: WellbeingState | None,
    ) -> SelfModificationProposal | None:
        if not _is_self_modification_action(next_followup_action):
            return None
        if self.store is None:
            return None
        recommendation_id = _clean_text(next_followup_action.get("recommendation_id"))
        if not recommendation_id:
            return None

        existing = await self._find_existing(recommendation_id)
        if existing is not None:
            return existing

        source_action_type = _clean_text(
            next_followup_action.get("source_action_type")
        )
        recent_outcomes = await self.store.list_maintenance_outcomes(
            action_type=source_action_type or None,
            limit=20,
        )
        proposal_dir = self._proposal_dir()
        proposal_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = proposal_dir / f"{timestamp}-{_slug(recommendation_id)}.md"
        content = self._render_markdown(
            next_followup_action=next_followup_action,
            state=state,
            recent_outcomes=recent_outcomes,
        )
        artifact_path.write_text(content, encoding="utf-8")
        proposal = SelfModificationProposal(
            title="Review stuck wellbeing maintenance loop",
            rationale=(
                "Wellbeing follow-up detected a repeated or unvalidated maintenance "
                "loop and requested an operator-reviewable change proposal."
            ),
            target_kind="wellbeing_self_maintenance",
            evidence_ids=[str(outcome.outcome_id) for outcome in recent_outcomes[:10]],
            review_required=True,
            artifact_path=str(artifact_path),
            risk_summary="Proposal-only; no prompt, config, or code change was applied.",
            meta={
                "recommendation_id": recommendation_id,
                "source_action_type": source_action_type,
                "source_outcome": next_followup_action.get("source_outcome"),
                "repeat_count": next_followup_action.get("repeat_count"),
                "followup_action_type": next_followup_action.get("type"),
                "artifact_kind": "wellbeing_self_modification_proposal",
            },
        )
        bead_meta = await self._record_thread_bead(
            proposal=proposal,
            recommendation_id=recommendation_id,
            content=content,
        )
        if bead_meta:
            proposal.meta = {**proposal.meta, "thread_registry": bead_meta}
        await self.store.save_self_modification_proposal(proposal)
        return proposal

    async def _find_existing(self, recommendation_id: str) -> SelfModificationProposal | None:
        proposals = await self.store.list_self_modification_proposals(limit=100)
        for proposal in proposals:
            if (proposal.meta or {}).get("recommendation_id") == recommendation_id:
                path = Path(proposal.artifact_path)
                if proposal.artifact_path and path.exists():
                    return proposal
        return None

    def _proposal_dir(self) -> Path:
        config = getattr(getattr(self.runtime, "ctx", None), "config", None)
        if config is not None and callable(getattr(config, "agent_workspace_root", None)):
            workspace_root = Path(config.agent_workspace_root())
        else:
            workspace_root = Path.cwd() / "workspace"
        return workspace_root / "self" / "proposals"

    def _render_markdown(
        self,
        *,
        next_followup_action: dict[str, Any],
        state: WellbeingState | None,
        recent_outcomes: list[MaintenanceOutcome],
    ) -> str:
        agent_name = resolve_agent_name(runtime=self.runtime, default="OpenCAS")
        source_action_type = _clean_text(next_followup_action.get("source_action_type"))
        repeat_count = int(next_followup_action.get("repeat_count") or 0)
        effect_basis = _clean_text(next_followup_action.get("effect_basis"))
        evidence_lines = [
            (
                f"- maintenance_outcome:{outcome.outcome_id} "
                f"`{outcome.action_type}` `{outcome.outcome}` "
                f"basis=`{(outcome.meta or {}).get('effect_basis', '')}` "
                f"direction=`{(outcome.meta or {}).get('effect_direction', '')}`"
            )
            for outcome in recent_outcomes[:10]
        ] or ["- No recent maintenance outcomes were available for this action type."]
        risk_line = "state unavailable"
        if state is not None:
            risk_line = (
                f"overall_risk={state.overall_risk:.3f}, "
                f"drift_load={state.drift_load:.3f}, "
                f"truth_pressure={state.truth_pressure:.3f}, "
                f"promise_load={state.promise_load:.3f}"
            )
        proposed_change = _proposed_change(source_action_type, effect_basis)
        return "\n".join(
            [
                "# Self-Modification Proposal",
                "",
                "## Detected Stuck-Loop Pattern",
                "",
                (
                    f"{agent_name} detected `{source_action_type or 'unknown'}` repeating "
                    f"{repeat_count} times with effect_basis=`{effect_basis or 'unknown'}`."
                ),
                "",
                f"Recommendation: `{next_followup_action.get('recommendation_id')}`",
                f"Current wellbeing state: {risk_line}",
                "",
                "## Evidence",
                "",
                *evidence_lines,
                "",
                "## Hypothesis",
                "",
                (
                    "The maintenance route is producing receipts without enough measured "
                    "evidence that the underlying drift source has changed. Treating the "
                    "loop as repaired would hide unresolved pressure."
                ),
                "",
                "## Proposed Change",
                "",
                proposed_change,
                "",
                "## Operator Instruction",
                "",
                (
                    "Review this proposal manually. Apply any prompt, config, skill, or "
                    "code change in a separate tooling step, or mark the recommendation "
                    "acknowledged to suppress repeated proposal generation for a bounded "
                    "number of future cycles."
                ),
                "",
                "No change has been hot-applied by this proposal.",
                "",
            ]
        )

    async def _record_thread_bead(
        self,
        *,
        proposal: SelfModificationProposal,
        recommendation_id: str,
        content: str,
    ) -> dict[str, Any]:
        service = getattr(self.runtime, "thread_registry_service", None)
        if service is None:
            return {}
        try:
            anchor = await service.ensure_thread_anchor(
                title="Wellbeing self-modification proposals",
                kind="system_insight",
                anchor_id=PROPOSAL_THREAD_ANCHOR_ID,
            )
            bead = await service.create_candidate_bead(
                thread_anchor_id=anchor.anchor_id,
                title=proposal.title,
                summary=(
                    "Reviewable wellbeing self-modification proposal generated from "
                    f"recommendation {recommendation_id}."
                ),
                source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT,
                source_ref=f"wellbeing:self_modification_proposal:{recommendation_id}",
                content=content,
                user_commissioned=False,
            )
        except Exception as exc:
            return {"error": str(exc)}
        return {"bead_id": bead.bead_id, "thread_anchor_id": bead.thread_anchor_id}


def _is_self_modification_action(action: dict[str, Any] | None) -> bool:
    return isinstance(action, dict) and action.get("type") == "self_modification_proposal"


def _proposed_change(source_action_type: str, effect_basis: str) -> str:
    if effect_basis == "record_only":
        return (
            f"Require `{source_action_type}` to produce measured effect evidence or a "
            "different action route after repeated record-only receipts. For high drift "
            "load, stop counting additional record-only receipts as repair progress."
        )
    if effect_basis == "estimated":
        return (
            f"Raise the measurement gate for `{source_action_type}` so estimated lower "
            "risk does not reduce drift pressure until the observation rate falls or an "
            "artifact-backed repair exists."
        )
    return (
        "Add a targeted maintenance action for the unresolved drift source class, then "
        "verify it with before/after observation evidence before promoting it."
    )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "recommendation"
