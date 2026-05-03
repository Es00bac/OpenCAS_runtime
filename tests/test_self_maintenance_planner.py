from opencas.wellbeing import (
    MaintenanceActionType,
    MaintenancePlanner,
    WellbeingAssessment,
    WellbeingState,
)


def test_planner_routes_high_recovery_need_to_rest_without_dropping_promises():
    assessment = WellbeingAssessment(
        state=WellbeingState(recovery_need=0.9, promise_load=0.6, coherence=0.7)
    )

    plan = MaintenancePlanner().plan(assessment)

    assert plan.actions[0].action_type == MaintenanceActionType.RECOVER
    assert any(action.action_type == MaintenanceActionType.PRESERVE_PROMISES for action in plan.actions)
    assert plan.background_throttle is True


def test_planner_routes_relationship_pressure_to_truthful_boundary_repair():
    assessment = WellbeingAssessment(
        state=WellbeingState(relationship_pressure=0.85, truth_pressure=0.74, autonomy=0.3)
    )

    plan = MaintenancePlanner().plan(assessment)

    assert any(action.action_type == MaintenanceActionType.BOUNDARY_CHECK for action in plan.actions)
    assert any(action.action_type == MaintenanceActionType.TRUTH_REPAIR for action in plan.actions)


def test_planner_routes_low_curiosity_to_incubation_not_user_compliance():
    assessment = WellbeingAssessment(
        state=WellbeingState(curiosity=0.18, relationship_pressure=0.6, autonomy=0.62)
    )

    plan = MaintenancePlanner().plan(assessment)

    assert any(action.action_type == MaintenanceActionType.CURIOSITY_INCUBATE for action in plan.actions)
    assert not any(
        "reassure" in action.reason.lower()
        for action in plan.actions
    )
