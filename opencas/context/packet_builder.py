"""Context packet builders for executive and reflective lanes."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .hemispheres import ContextAuthority, ContextLane
from .proposals import ContextProposal, ContextProposalStore, ProposalStatus
from .truth_arbiter import TruthArbiter, TruthSnapshot


class ContextSection(BaseModel):
    """One labeled section inside a context packet."""

    title: str
    label: str
    authority: ContextAuthority
    content: str
    refs: List[str] = Field(default_factory=list)


class ContextPacket(BaseModel):
    """A lane-specific prompt packet with explicit authority boundaries."""

    lane: ContextLane
    authority_mode: str
    truth_snapshot_id: str
    truth_epoch: int
    token_budget: int
    can_write: bool
    sections: List[ContextSection] = Field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Context lane: {self.lane.value}",
            f"Authority mode: {self.authority_mode}",
            f"Truth snapshot: {self.truth_snapshot_id} epoch={self.truth_epoch}",
            f"Write authority: {'yes' if self.can_write else 'no'}",
        ]
        for section in self.sections:
            lines.append("")
            lines.append(f"{section.label}: {section.title}")
            if section.refs:
                lines.append("refs: " + ", ".join(section.refs))
            lines.append(section.content)
        return "\n".join(lines)


class ContextPacketBuilder:
    """Build executive and reflective packets from one runtime truth source."""

    def __init__(
        self,
        *,
        runtime: Any,
        truth_arbiter: Optional[TruthArbiter] = None,
        proposal_store: Optional[ContextProposalStore] = None,
        executive_token_budget: int = 4000,
        reflective_token_budget: int = 6000,
    ) -> None:
        self.runtime = runtime
        self.truth_arbiter = truth_arbiter or getattr(runtime, "truth_arbiter", None)
        self.proposal_store = proposal_store or getattr(runtime, "context_proposals", None)
        self.executive_token_budget = executive_token_budget
        self.reflective_token_budget = reflective_token_budget

    async def build_executive_packet(
        self,
        *,
        user_input: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ContextPacket:
        snapshot = await self._snapshot(reason="executive_packet")
        sections = [
            ContextSection(
                title="Current live truth",
                label="LIVE FACT",
                authority=ContextAuthority.LIVE_OBSERVATION,
                content=self._snapshot_summary(snapshot),
                refs=[snapshot.snapshot_id],
            ),
            ContextSection(
                title="Immediate request",
                label="LIVE FACT",
                authority=ContextAuthority.LIVE_OBSERVATION,
                content=user_input,
            ),
            ContextSection(
                title="Action boundary",
                label="COMMITTED FACT",
                authority=ContextAuthority.COMMITTED_FACT,
                content=(
                    "Only executive truth, direct user request, due schedules, or accepted "
                    "arbiter proposals may create tool writes, commitments, schedules, or BAA tasks."
                ),
            ),
        ]
        sections.extend(await self._proposal_sections(project_id=project_id, task_id=task_id, accepted_only=True))
        return ContextPacket(
            lane=ContextLane.EXECUTIVE,
            authority_mode="live_truth_and_committed_actions",
            truth_snapshot_id=snapshot.snapshot_id,
            truth_epoch=snapshot.epoch,
            token_budget=self.executive_token_budget,
            can_write=True,
            sections=sections,
        )

    async def build_reflective_packet(
        self,
        *,
        user_input: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ContextPacket:
        snapshot = await self._snapshot(reason="reflective_packet")
        sections = [
            ContextSection(
                title="No write authority",
                label="NO WRITE AUTHORITY",
                authority=ContextAuthority.INTERPRETATION,
                content=(
                    "This lane may associate, compare, critique, and propose. It may not create "
                    "schedules, commitments, BAA tasks, or tool writes."
                ),
                refs=[snapshot.snapshot_id],
            ),
            ContextSection(
                title="Reflective focus",
                label="INTERPRETATION",
                authority=ContextAuthority.INTERPRETATION,
                content=user_input,
            ),
        ]
        sections.extend(await self._proposal_sections(project_id=project_id, task_id=task_id, accepted_only=False))
        return ContextPacket(
            lane=ContextLane.REFLECTIVE,
            authority_mode="proposals_only",
            truth_snapshot_id=snapshot.snapshot_id,
            truth_epoch=snapshot.epoch,
            token_budget=self.reflective_token_budget,
            can_write=False,
            sections=sections,
        )

    async def _snapshot(self, *, reason: str) -> TruthSnapshot:
        if self.truth_arbiter is None:
            self.truth_arbiter = TruthArbiter(self.runtime)
        return await self.truth_arbiter.issue_snapshot(reason=reason)

    async def _proposal_sections(
        self,
        *,
        project_id: Optional[str],
        task_id: Optional[str],
        accepted_only: bool,
    ) -> List[ContextSection]:
        if self.proposal_store is None:
            return []
        proposals: list[ContextProposal] = []
        if project_id:
            proposals.extend(await self.proposal_store.list_by_project(project_id, include_terminal=True, limit=8))
        if task_id:
            proposals.extend(await self.proposal_store.list_by_task(task_id, include_terminal=True, limit=8))
        deduped = {proposal.proposal_id: proposal for proposal in proposals}
        sections: list[ContextSection] = []
        for proposal in deduped.values():
            if accepted_only and proposal.status != ProposalStatus.ACCEPTED:
                continue
            sections.append(self._proposal_section(proposal))
        return sections

    @staticmethod
    def _proposal_section(proposal: ContextProposal) -> ContextSection:
        if proposal.status == ProposalStatus.ACCEPTED:
            label = "ACCEPTED SUPPORT"
            authority = ContextAuthority.EXECUTIVE_COMMITTED
        elif proposal.status == ProposalStatus.REJECTED:
            label = "REJECTED IDEA"
            authority = ContextAuthority.INTERPRETATION
        else:
            label = "PROPOSAL"
            authority = ContextAuthority.PROPOSAL
        return ContextSection(
            title=f"{proposal.proposal_kind} ({proposal.status.value})",
            label=label,
            authority=authority,
            content=proposal.content,
            refs=[proposal.proposal_id, proposal.source_snapshot_id, *proposal.evidence_refs],
        )

    @staticmethod
    def _snapshot_summary(snapshot: TruthSnapshot) -> str:
        payload = {
            "executive": snapshot.executive,
            "baa": snapshot.baa,
            "schedules": snapshot.schedules,
            "commitments": snapshot.commitments,
            "receipts": snapshot.receipts,
            "runtime_activity": snapshot.runtime_activity,
        }
        return json.dumps(payload, sort_keys=True, default=str)
