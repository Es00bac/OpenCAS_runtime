"""Shared labels for dual-context OpenCAS reasoning lanes."""

from __future__ import annotations

from enum import StrEnum


class ContextLane(StrEnum):
    """Logical context lanes; these are not independent agents."""

    EXECUTIVE = "executive"
    REFLECTIVE = "reflective"


class ContextAuthority(StrEnum):
    """Authority label for context facts and proposals."""

    COMMITTED_FACT = "committed_fact"
    LIVE_OBSERVATION = "live_observation"
    RETRIEVED_MEMORY = "retrieved_memory"
    INTERPRETATION = "interpretation"
    PROPOSAL = "proposal"
    CONFLICT = "conflict"
    EXECUTIVE_COMMITTED = "executive_committed"
    USER_REQUESTED = "user_requested"
    SCHEDULE_DUE = "schedule_due"
