from __future__ import annotations

from opencas.cognition import recommended_counterfactual


def test_recommended_counterfactual_scores_prerequisite_failures() -> None:
    review = recommended_counterfactual(
        objective="Verify that the schedule exists before reporting success.",
        failure_summary="workflow_list_schedules failed: not found / missing schedule record",
        prior_tool="workflow_list_schedules",
        available_tools=["workflow_list_schedules", "runtime_status"],
        prior_attempts=2,
    )

    recommended = review["recommended"]
    assert recommended["strategy"] == "verify_prerequisite"
    assert recommended["score"] > 0.7
    assert recommended["evidence_need"]
    assert len(review["options"]) >= 4


def test_recommended_counterfactual_escalates_permission_boundaries() -> None:
    review = recommended_counterfactual(
        objective="Fix the calendar duplicate.",
        failure_summary="permission denied; approval required for calendar deletion",
        prior_tool="google_workspace_calendar_delete",
        available_tools=["google_workspace_calendar_delete"],
        prior_attempts=1,
    )

    recommended = review["recommended"]
    assert recommended["strategy"] == "ask_or_escalate"
    assert "permission" in recommended["evidence_need"]
