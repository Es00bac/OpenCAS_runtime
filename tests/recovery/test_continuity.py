from __future__ import annotations

from datetime import datetime, timezone

import pytest

from opencas.recovery.continuity import ContinuityPacketBuilder
from opencas.recovery.models import RecoveryCandidate, RecoveryCandidateKind


def _candidate(payload: dict, *, kind: RecoveryCandidateKind = RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP) -> RecoveryCandidate:
    return RecoveryCandidate(
        candidate_id="loop:story",
        kind=kind,
        title="Continue creative writing project",
        status="paused",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=["loop:story"],
        payload=payload,
    )


@pytest.mark.asyncio
async def test_builder_creates_creative_writing_packet() -> None:
    builder = ContinuityPacketBuilder()
    packet = await builder.build(
        _candidate(
            {
                "project_key": "story-project",
                "project_type": "creative_writing",
                "canonical_artifact_paths": ["workspace/writing/story.md"],
                "latest_completed_unit": "chapter 2",
                "current_status": "chapter 3 needs drafting",
                "continuity_facts": ["Character A left town in chapter 2."],
                "unresolved_threads": ["Character B has not learned why A left."],
                "next_concrete_action": "Draft chapter 3 opening scene.",
                "completion_criteria": ["chapter 3 draft exists"],
            }
        )
    )

    assert packet.project_type == "creative_writing"
    assert packet.canonical_artifact_paths == ["workspace/writing/story.md"]
    assert packet.next_concrete_action == "Draft chapter 3 opening scene."
    assert "Character A left town in chapter 2." in packet.continuity_facts


@pytest.mark.asyncio
async def test_builder_creates_software_packet_with_verification() -> None:
    builder = ContinuityPacketBuilder()
    packet = await builder.build(
        _candidate(
            {
                "project_key": "app-project",
                "project_type": "software",
                "artifact_paths_touched": ["opencas/app.py", "tests/test_app.py"],
                "current_status": "implementation failed during verification",
                "known_defects": ["test_app.py::test_build fails"],
                "verification_commands": ["pytest tests/test_app.py -q"],
                "next_concrete_action": "Fix failing app build test.",
            },
            kind=RecoveryCandidateKind.FAILED_TASK,
        )
    )

    assert packet.project_type == "software"
    assert packet.canonical_artifact_paths == ["opencas/app.py", "tests/test_app.py"]
    assert "pytest tests/test_app.py -q" in packet.completion_criteria
    assert "test_app.py::test_build fails" in packet.unresolved_threads
