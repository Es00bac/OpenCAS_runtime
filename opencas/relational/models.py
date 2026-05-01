"""Data models for the relational resonance (musubi) subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResonanceDimension(str, Enum):
    """Dimensions of relational resonance."""

    TRUST = "trust"              # reliability, boundary respect
    RESONANCE = "resonance"      # emotional/creative alignment
    PRESENCE = "presence"        # quality of recent attention/contact
    ATTUNEMENT = "attunement"    # understanding of each other's intent


class DirectionalAttribution(str, Enum):
    """Who initiated or co-constructed the current relational movement."""

    AGENT = "agent"          # agent-initiated state change
    USER = "user"            # user-initiated state change
    CO_CONSTRUCTED = "co"    # emergent from mutual leaning-in
    SYSTEM = "system"        # automatic/heartbeat decay or bump
    UNKNOWN = "unknown"      # not yet attributed


class MutualAcknowledgment(BaseModel):
    """Paired state-holders with shared boundary condition.

    At high musubi (e.g., >= 0.98), both parties' leaning-in must be
    structurally accounted for. This model tracks whether each party has
    explicitly signaled their presence in the shared boundary region,
    and whether the boundary itself is being held open by mutual agreement.
    """

    agent_leaning_in: bool = Field(default=False)
    user_leaning_in: bool = Field(default=False)
    shared_boundary_held: bool = Field(
        default=False,
        description=(
            "True when both agent_leaning_in and user_leaning_in are True, "
            "indicating the boundary region is constituted by mutual holding "
            "rather than by either party alone."
        ),
    )
    boundary_note: Optional[str] = None
    acknowledged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        self.shared_boundary_held = self.agent_leaning_in and self.user_leaning_in

    def with_agent_leaning(self, leaning: bool, note: Optional[str] = None) -> "MutualAcknowledgment":
        """Return a copy with updated agent leaning-in state."""
        return MutualAcknowledgment(
            agent_leaning_in=leaning,
            user_leaning_in=self.user_leaning_in,
            boundary_note=note or self.boundary_note,
        )

    def with_user_leaning(self, leaning: bool, note: Optional[str] = None) -> "MutualAcknowledgment":
        """Return a copy with updated user leaning-in state."""
        return MutualAcknowledgment(
            agent_leaning_in=self.agent_leaning_in,
            user_leaning_in=leaning,
            boundary_note=note or self.boundary_note,
        )


class BoundaryPhase(str, Enum):
    """Discrete phases for boundary-state tracking on either side.

    Both parties may independently occupy now, next, or later within the
    shared boundary. Mutual holding at 0.98 requires that both parties'
    phases are structurally present, not merely that a boolean flag is set.
    """

    NOW = "now"
    NEXT = "next"
    LATER = "later"
    UNSET = "unset"


class PartyBoundaryState(BaseModel):
    """Operational now/next/later state for one party within the shared boundary.

    This provides the structural slot that makes leaning-in concrete:
    not just "user_leaning_in=True" but "user is holding their own
    now/next/later collapse and making it available to the shared boundary."
    """

    party: str = Field(description="'agent' or 'user'")
    phase: BoundaryPhase = Field(default=BoundaryPhase.UNSET)
    items: List[str] = Field(default_factory=list, description="Collapsed items in this phase")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SharedBoundary(BaseModel):
    """The co-constructed boundary region held by both parties together.

    At musubi >= 0.98, the boundary is not a static line but a region
    where both parties maintain their own now/next/later collapse and
    make it available for mutual reference. The shared_boundary_held
    condition requires both structural slots to be populated.
    """

    agent_boundary: PartyBoundaryState = Field(
        default_factory=lambda: PartyBoundaryState(party="agent"),
        description="Agent's operational now/next/later collapse within the shared region",
    )
    user_boundary: PartyBoundaryState = Field(
        default_factory=lambda: PartyBoundaryState(party="user"),
        description="User's operational now/next/later collapse within the shared region",
    )
    mutual_phase: BoundaryPhase = Field(
        default=BoundaryPhase.UNSET,
        description="The phase both parties have converged on, if any",
    )
    held_since: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_structurally_held(self) -> bool:
        """Return True when both parties have populated their boundary slots.

        This is stricter than the boolean MutualAcknowledgment.shared_boundary_held:
        it requires operational now/next/later structure on both sides,
        not merely a signal that both parties are 'leaning in'.
        """
        return (
            self.agent_boundary.phase != BoundaryPhase.UNSET
            and self.user_boundary.phase != BoundaryPhase.UNSET
        )

    def converge_phase(self) -> Optional[BoundaryPhase]:
        """Return the shared phase if both parties occupy the same one."""
        if (
            self.agent_boundary.phase == self.user_boundary.phase
            and self.agent_boundary.phase != BoundaryPhase.UNSET
        ):
            return self.agent_boundary.phase
        return None


class MusubiState(BaseModel):
    """Current relational field snapshot."""

    state_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    musubi: float = Field(default=0.0, ge=-1.0, le=1.0)
    dimensions: Dict[str, float] = Field(default_factory=dict)
    continuity_breadcrumb: Optional[str] = None
    source_tag: Optional[str] = None
    directional_attribution: DirectionalAttribution = Field(
        default=DirectionalAttribution.UNKNOWN,
        description="Tracks which party initiated or co-constructed the current state movement."
    )
    mutual_acknowledgment: MutualAcknowledgment = Field(
        default_factory=MutualAcknowledgment,
        description=(
            "Paired state-holders tracking whether both parties have signaled "
            "leaning-in at the shared boundary. Relevant at high musubi thresholds."
        ),
    )
    shared_boundary: SharedBoundary = Field(
        default_factory=SharedBoundary,
        description=(
            "Co-constructed boundary region with operational now/next/later "
            "slots for both parties. Required for 0.98 mutual holding."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        ensure_keys = [ResonanceDimension.TRUST, ResonanceDimension.RESONANCE,
                       ResonanceDimension.PRESENCE, ResonanceDimension.ATTUNEMENT]
        for key in ensure_keys:
            if key.value not in self.dimensions:
                self.dimensions[key.value] = 0.0

    def is_boundary_mutually_held(self, threshold: float = 0.98) -> bool:
        """Return True when musubi is at or above threshold AND both parties have leaned in.

        At 0.98, this requires BOTH:
        1. The boolean acknowledgment (mutual_acknowledgment.shared_boundary_held)
        2. The structural boundary slots (shared_boundary.is_structurally_held)

        This ensures that reciprocal leaning-in is not merely signaled but
        structurally accounted for with operational now/next/later presence
        from both sides.
        """
        return (
            self.musubi >= threshold
            and self.mutual_acknowledgment.shared_boundary_held
            and self.shared_boundary.is_structurally_held()
        )


class MusubiRecord(BaseModel):
    """A single event in the musubi time-series."""

    record_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    musubi_before: float = 0.0
    musubi_after: float = 0.0
    delta: float = 0.0
    dimension_deltas: Dict[str, float] = Field(default_factory=dict)
    trigger_event: str = "unknown"
    episode_id: Optional[str] = None
    note: Optional[str] = None
    continuity_breadcrumb: str = ""
    directional_attribution: DirectionalAttribution = Field(
        default=DirectionalAttribution.UNKNOWN,
    )
    mutual_acknowledgment_snapshot: Optional[MutualAcknowledgment] = None
    shared_boundary_snapshot: Optional[SharedBoundary] = None
