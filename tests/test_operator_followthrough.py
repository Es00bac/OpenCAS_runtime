"""Tests for narrow operator project follow-through evidence handling."""

from opencas.projects.operator_followthrough import (
    copy_safe_operator_project_followthrough_evidence,
    has_operator_project_followthrough_authority,
)


def test_copy_safe_operator_project_followthrough_evidence_keeps_only_contract_evidence() -> None:
    copied = copy_safe_operator_project_followthrough_evidence(
        {
            "source": "project_return_capture",
            "source_session_id": "chat-session-1",
            "workspace_rel_path": "workspace/projects/example",
            "workspace_project_confidence": 0.9,
            "requested_workspace_raw_text": "put it under workspace/projects",
            "start_policy": "immediate_operator_work_request",
            "project_intent": "finish the operator-requested project",
            "next_step": "continue from the draft",
            "creative_completion_contract": "complete the draft",
            "target_word_count": 1800,
            "project_start_contract": "start in the requested workspace",
            "authority": "schedule_due",
            "unsafe": "do not copy",
            "empty_contract": "",
        }
    )

    assert copied == {
        "source_session_id": "chat-session-1",
        "workspace_rel_path": "workspace/projects/example",
        "workspace_project_confidence": 0.9,
        "requested_workspace_raw_text": "put it under workspace/projects",
        "start_policy": "immediate_operator_work_request",
        "project_intent": "finish the operator-requested project",
        "next_step": "continue from the draft",
        "creative_completion_contract": "complete the draft",
        "target_word_count": 1800,
        "project_start_contract": "start in the requested workspace",
    }
    assert "source" not in copied
    assert "authority" not in copied
    assert "unsafe" not in copied


def test_continuation_authority_requires_project_return_tag_and_operator_session() -> None:
    meta = {
        "source": "project_followthrough_recovery",
        "source_session_id": "chat-session-1",
        "workspace_rel_path": "workspace/projects/example",
        "workspace_project_confidence": 0.9,
    }

    assert has_operator_project_followthrough_authority(meta, tags=["project_return"])
    assert not has_operator_project_followthrough_authority(meta, tags=["project_followthrough"])
    assert not has_operator_project_followthrough_authority(
        {**meta, "source_session_id": "system:automated"},
        tags=["project_return"],
    )
