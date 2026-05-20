from __future__ import annotations

from opencas.tools.loop import ToolUseLoop


def test_ongoing_assistant_request_gets_commitment_and_capability_tools() -> None:
    required = set(
        ToolUseLoop._required_tool_names_for_objective(
            "Can you be my assistant, figuring out a realistic business model "
            "and productive routine?"
        )
    )

    assert "workflow_create_commitment" in required
    assert "workflow_list_commitments" in required
    assert "workflow_create_schedule" in required
    assert "workflow_create_plan" in required
    assert "cognitive_focus_set" in required
    assert "cognitive_prospective_memory_set" in required
    assert "mcp_list_servers" in required
    assert "mcp_register_server_tools" in required


def test_conditional_future_followup_gets_commitment_tools_without_live_schedule() -> None:
    required = set(
        ToolUseLoop._required_tool_names_for_objective(
            "When I get home, create an itinerary for rental assistance and KDP publishing."
        )
    )

    assert "workflow_create_commitment" in required
    assert "workflow_list_commitments" in required
    assert "cognitive_focus_set" in required
    assert "cognitive_prospective_memory_set" in required


def test_high_priority_income_mission_gets_followthrough_tools() -> None:
    required = set(
        ToolUseLoop._required_tool_names_for_objective(
            "Treat helping me start making income as a high priority mission. "
            "Be proactive and keep track of a complex multi-step project."
        )
    )

    assert "workflow_create_commitment" in required
    assert "workflow_create_schedule" in required
    assert "workflow_create_plan" in required
    assert "workflow_list_tasks" in required
    assert "cognitive_focus_set" in required
    assert "cognitive_context_query" in required
    assert "cognitive_prospective_memory_set" in required
