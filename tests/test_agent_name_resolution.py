from types import SimpleNamespace

from opencas.identity.agent_name import resolve_agent_name


def test_resolve_agent_name_prefers_runtime_context_identity() -> None:
    runtime = SimpleNamespace(
        agent_name="FallbackName",
        ctx=SimpleNamespace(identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent"))),
    )

    assert resolve_agent_name(runtime=runtime) == "TestAgent"


def test_resolve_agent_name_uses_safe_fallback_for_missing_identity() -> None:
    assert resolve_agent_name(runtime=SimpleNamespace(), default="OpenCAS") == "OpenCAS"


def test_resolve_agent_name_normalizes_blank_and_long_names() -> None:
    runtime = SimpleNamespace(agent_name="  " + ("A" * 100) + "  ")

    assert resolve_agent_name(runtime=runtime) == "A" * 80
