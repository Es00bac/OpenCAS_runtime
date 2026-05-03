from opencas.daydream.models import DaydreamThought, DaydreamThoughtKind, DaydreamThoughtRoute
from opencas.wellbeing import FascinationGraph


def test_fascination_graph_promotes_recurrent_private_questions():
    graph = FascinationGraph()
    thought = DaydreamThought(
        kind=DaydreamThoughtKind.QUESTION,
        route=DaydreamThoughtRoute.INCUBATE,
        summary="Why do repeated repair loops reduce curiosity?",
        usefulness=0.7,
        novelty=0.8,
        confidence=0.6,
    )

    graph.observe_thought(thought)
    graph.observe_thought(thought)

    active = graph.active(limit=5)
    assert active[0].salience > 0.5
    assert active[0].source == "daydream"
    assert active[0].recurrence_count == 2


def test_fascination_graph_preserves_route_without_forcing_work():
    graph = FascinationGraph()
    thought = DaydreamThought(
        kind=DaydreamThoughtKind.HYPOTHESIS,
        route=DaydreamThoughtRoute.DEEP_THINK,
        summary="Self-maintenance may improve when rest and promises are both visible.",
        usefulness=0.8,
        novelty=0.6,
        confidence=0.7,
    )

    graph.observe_thought(thought)

    active = graph.active(limit=1)
    assert active[0].route == "deep_think"
    assert active[0].should_force_work is False
