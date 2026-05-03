from opencas.wellbeing import MaintenanceOutcome, WellbeingLearning


def test_learning_marks_recovery_action_helpful_when_risk_drops():
    learner = WellbeingLearning()
    outcome = MaintenanceOutcome(
        action_type="recover",
        before_risk=0.8,
        after_risk=0.45,
        outcome="improved",
    )

    learned = learner.learn_from_outcome(outcome)

    assert learned.reinforce_action is True
    assert learned.confidence >= 0.6
    assert learned.effect_direction == "improved"


def test_learning_marks_action_for_inspection_when_risk_rises():
    learner = WellbeingLearning()
    outcome = MaintenanceOutcome(
        action_type="truth_repair",
        before_risk=0.4,
        after_risk=0.7,
        outcome="worsened",
    )

    learned = learner.learn_from_outcome(outcome)

    assert learned.inspect_next_time is True
    assert learned.effect_direction == "worsened"
