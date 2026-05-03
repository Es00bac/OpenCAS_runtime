from opencas.wellbeing import BondHealthAnalyzer, WellbeingState


def test_bond_health_marks_high_closeness_with_low_autonomy_as_pressure():
    state = WellbeingState(autonomy=0.25, relationship_pressure=0.8, truth_pressure=0.7)

    result = BondHealthAnalyzer().analyze(state)

    assert result.risk == "compliance_pressure"
    assert result.recommended_boundary == "preserve_truth_before_reassurance"
    assert result.should_surface_to_user is True


def test_bond_health_allows_warmth_when_truth_and_autonomy_are_stable():
    state = WellbeingState(autonomy=0.8, relationship_pressure=0.35, truth_pressure=0.1)

    result = BondHealthAnalyzer().analyze(state)

    assert result.risk == "stable"
    assert result.should_surface_to_user is False
