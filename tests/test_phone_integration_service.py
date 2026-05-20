"""Tests for the runtime phone bridge service variant."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from opencas.phone_config import PhoneRuntimeConfig
from opencas.phone_integration_service import PhoneBridgeService, PhoneResolvedCaller


def _service(tmp_path: Path) -> PhoneBridgeService:
    workspace = tmp_path / "workspace"
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(
                state_dir=tmp_path / "state",
                agent_workspace_root=lambda: workspace,
            ),
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
        )
    )
    return PhoneBridgeService(runtime=runtime, config=PhoneRuntimeConfig(enabled=True))


def test_phone_service_defaults_use_runtime_agent_name(tmp_path: Path) -> None:
    service = _service(tmp_path)
    caller = PhoneResolvedCaller(
        phone_number="+15550001111",
        display_name="Jordan",
        trust_level="known",
        allowed_actions=("leave_message",),
    )

    assert service.screening_menu_prompt_default().startswith("Hi, this is TestAgent.")
    assert service._default_greeting(caller).endswith("message for TestAgent.")
    assert "TestAgent on a restricted workspace line" in service.menu_workspace_acceptance(caller)
    assert "WorkSafe TestAgent" in service._read_phone_prompt_profile(None)
