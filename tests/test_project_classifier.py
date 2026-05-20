"""Tests for project-type classification."""

from opencas.projects.classifier import classify_project_type


def test_classifier_keeps_software_project_current_turn_over_stale_manuscript_context() -> None:
    result = classify_project_type(
        current_turn_text="Create a native Qt6 email client called kPony and keep working until it builds.",
        context_text="Earlier context: revise writing project 4246 and keep the manuscript moving.",
    )

    assert result.project_type == "software"
    assert "qt6" in result.evidence
    assert "email client" in result.evidence


def test_classifier_does_not_treat_attested_as_software_test_marker() -> None:
    result = classify_project_type(
        current_turn_text=(
            "Onnen is attested as a Brythonic woman's given name. "
            "Next I need to fold this naming decision back into the manuscript and keep revising writing project 4246."
        )
    )

    assert result.project_type == "writing"
    assert "test" not in result.evidence


def test_classifier_does_not_treat_write_code_as_creative_writing() -> None:
    result = classify_project_type(
        current_turn_text="Write code for the kPony Qt6 client and run the build.",
    )

    assert result.project_type == "software"


def test_classifier_can_identify_research_without_forcing_writing_or_software() -> None:
    result = classify_project_type(
        current_turn_text="Research email protocol tradeoffs and compare IMAP library options.",
    )

    assert result.project_type == "research"
