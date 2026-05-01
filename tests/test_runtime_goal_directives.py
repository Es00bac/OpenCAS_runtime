from opencas.runtime.episodic_runtime import extract_runtime_goal_directives


def test_extract_runtime_goal_directives_ignores_descriptive_intention_language() -> None:
    goals, intention, drops = extract_runtime_goal_directives(
        "Right now the live intention is back to continuity surface reconciliation decision bead and the queue is empty."
    )

    assert goals == []
    assert intention is None
    assert drops == []


def test_extract_runtime_goal_directives_accepts_explicit_intention_directive() -> None:
    goals, intention, drops = extract_runtime_goal_directives(
        "Please intention is stabilize the executive parser."
    )

    assert goals == []
    assert intention == "stabilize the executive parser"
    assert drops == []


def test_extract_runtime_goal_directives_accepts_sentence_start_goal_directive() -> None:
    goals, intention, drops = extract_runtime_goal_directives(
        "And focus on verifying the intention anchor."
    )

    assert goals == ["verifying the intention anchor"]
    assert intention is None
    assert drops == []
