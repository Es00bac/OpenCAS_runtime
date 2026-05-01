"""Verification test for the 0.98 musubi mutual-holding criterion.

This test deterministically checks that MusubiState.is_boundary_mutually_held()
correctly evaluates the 0.98 threshold with both parties' leaning-in
structurally accounted for.

The criterion requires ALL of:
1. musubi >= 0.98
2. mutual_acknowledgment.shared_boundary_held (both booleans True)
3. shared_boundary.is_structurally_held() (both phases != UNSET)
"""

import pytest

from opencas.relational.models import (
    BoundaryPhase,
    MusubiState,
    MutualAcknowledgment,
    PartyBoundaryState,
    SharedBoundary,
)


class TestMusubi098Threshold:
    """Deterministic verification of the 0.98 musubi criterion."""

    def _make_fully_held_state(self, musubi: float = 0.98) -> MusubiState:
        """Return a state with both boolean and structural leaning-in."""
        return MusubiState(
            musubi=musubi,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW, items=["present"]),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.NOW, items=["here"]),
            ),
        )

    def test_exactly_098_with_full_structure_passes(self) -> None:
        """At exactly 0.98 with both parties structurally present -> True."""
        state = self._make_fully_held_state(musubi=0.98)
        assert state.is_boundary_mutually_held() is True

    def test_above_098_with_full_structure_passes(self) -> None:
        """Above 0.98 with both parties structurally present -> True."""
        state = self._make_fully_held_state(musubi=0.99)
        assert state.is_boundary_mutually_held() is True

    def test_below_098_fails_even_with_full_structure(self) -> None:
        """At 0.97 with both parties structurally present -> False (threshold not met)."""
        state = self._make_fully_held_state(musubi=0.97)
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_without_agent_boolean(self) -> None:
        """At 0.98 with only user boolean leaning-in -> False."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=False,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.NOW),
            ),
        )
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_without_user_boolean(self) -> None:
        """At 0.98 with only agent boolean leaning-in -> False."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=False,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.NOW),
            ),
        )
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_without_agent_structural_phase(self) -> None:
        """At 0.98 with agent phase=UNSET -> False (structural requirement missing)."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.UNSET),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.NOW),
            ),
        )
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_without_user_structural_phase(self) -> None:
        """At 0.98 with user phase=UNSET -> False (structural requirement missing)."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.UNSET),
            ),
        )
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_with_only_booleans_no_structure(self) -> None:
        """At 0.98 with boolean acknowledgment but no structural phases -> False."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(),  # both phases UNSET
        )
        assert state.is_boundary_mutually_held() is False

    def test_098_fails_with_only_structure_no_booleans(self) -> None:
        """At 0.98 with structural phases but no boolean acknowledgment -> False."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(),  # both False
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.NOW),
            ),
        )
        assert state.is_boundary_mutually_held() is False

    def test_default_state_fails(self) -> None:
        """A fresh default MusubiState -> False."""
        state = MusubiState()
        assert state.is_boundary_mutually_held() is False

    def test_custom_threshold_095(self) -> None:
        """Custom threshold of 0.95 passes at 0.96 with full structure."""
        state = self._make_fully_held_state(musubi=0.96)
        assert state.is_boundary_mutually_held(threshold=0.95) is True

    def test_custom_threshold_095_fails_at_094(self) -> None:
        """Custom threshold of 0.95 fails at 0.94."""
        state = self._make_fully_held_state(musubi=0.94)
        assert state.is_boundary_mutually_held(threshold=0.95) is False

    def test_both_parties_can_have_different_phases(self) -> None:
        """Agent and user can occupy different phases; structural check still passes."""
        state = MusubiState(
            musubi=0.98,
            mutual_acknowledgment=MutualAcknowledgment(
                agent_leaning_in=True,
                user_leaning_in=True,
            ),
            shared_boundary=SharedBoundary(
                agent_boundary=PartyBoundaryState(party="agent", phase=BoundaryPhase.NOW),
                user_boundary=PartyBoundaryState(party="user", phase=BoundaryPhase.LATER),
            ),
        )
        assert state.is_boundary_mutually_held() is True
        # But converge_phase returns None since they're different
        assert state.shared_boundary.converge_phase() is None

    def test_converge_phase_when_same(self) -> None:
        """When both parties share a phase, converge_phase returns it."""
        state = self._make_fully_held_state(musubi=0.98)
        assert state.shared_boundary.converge_phase() == BoundaryPhase.NOW
