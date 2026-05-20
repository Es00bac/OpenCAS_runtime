from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.thread_registry import BeadSourceKind, ThreadStatus


@pytest.mark.asyncio
async def test_record_loaded_suppressed_reframe_thread_beads_routes_rejections_to_audit_beads() -> None:
    from opencas.runtime.suppressed_reframe_runtime import (
        record_loaded_suppressed_reframe_thread_beads,
    )

    goal = "browser_click blocked memory should not become a parked goal"

    class FakeExecutive:
        def __init__(self) -> None:
            self.rejections = [
                {
                    "goal": goal,
                    "reason": "low_divergence_reframe",
                    "source_artifact": goal,
                    "metadata": {
                        "reason": "low_divergence_reframe",
                        "source_artifact": goal,
                        "failed_framings": [goal],
                        "duplicate_of_task_id": "task-1",
                    },
                }
            ]

        def consume_suppressed_parked_goal_rejections(self):
            rejections = list(self.rejections)
            self.rejections.clear()
            return rejections

    class FakeThreadRegistryService:
        def __init__(self) -> None:
            self.anchors = []
            self.beads = []

        async def ensure_thread_anchor(self, **kwargs):
            self.anchors.append(kwargs)
            return SimpleNamespace(anchor_id=kwargs["anchor_id"])

        async def create_candidate_bead(self, **kwargs):
            self.beads.append(kwargs)
            return SimpleNamespace(bead_id="bead-1")

    executive = FakeExecutive()
    service = FakeThreadRegistryService()

    result = await record_loaded_suppressed_reframe_thread_beads(
        executive=executive,
        service=service,
    )

    assert result["candidate_count"] == 1
    assert result["recorded_count"] == 1
    assert service.anchors[0]["anchor_id"] == "suppressed-recursive-reframes"
    assert service.anchors[0]["status"] == ThreadStatus.PERIPHERAL
    bead = service.beads[0]
    assert bead["source_kind"] == BeadSourceKind.SUPPRESSED_REFRAME
    assert bead["source_ref"].startswith("suppressed_reframe:loaded:")
    assert bead["user_commissioned"] is False
    assert goal in bead["content"]
    assert "audit-only thread-registry query" in bead["content"]

    second_result = await record_loaded_suppressed_reframe_thread_beads(
        executive=executive,
        service=service,
    )

    assert second_result["candidate_count"] == 0
    assert second_result["recorded_count"] == 0
