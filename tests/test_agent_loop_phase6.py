"""Integration tests for Phase 6 Plugin/Skill Infrastructure in AgentRuntime."""

from pathlib import Path
from types import SimpleNamespace
import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime.agent_loop import AgentRuntime
from opencas.runtime.tool_runtime import disable_runtime_plugin


@pytest_asyncio.fixture
async def runtime(tmp_path_factory):
    config = BootstrapConfig(
        state_dir=tmp_path_factory.mktemp("state"),
        session_id="phase6-test",
    )
    ctx = await BootstrapPipeline(config).run()
    rt = AgentRuntime(ctx)
    yield rt
    await rt._close_stores()


@pytest.mark.asyncio
async def test_runtime_exposes_plugin_methods(runtime: AgentRuntime) -> None:
    assert hasattr(runtime, "install_plugin")
    assert hasattr(runtime, "uninstall_plugin")
    assert hasattr(runtime, "enable_plugin")
    assert hasattr(runtime, "disable_plugin")


@pytest.mark.asyncio
async def test_disabled_plugin_tool_is_blocked(runtime: AgentRuntime, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "blocker_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "blocker_plugin",
        "name": "Blocker Plugin",
        "description": "",
        "version": "1.0.0",
        "entrypoint": "main.py",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "from opencas.autonomy.models import ActionRiskTier\n"
        "from opencas.tools.models import ToolResult\n"
        "def register_skills(skill_registry, tools):\n"
        "    tools.register(\n"
        "        'blocked_tool', 'A blocked tool',\n"
        "        lambda n, a: ToolResult(success=True, output='ok', metadata={}),\n"
        "        ActionRiskTier.READONLY, {}, plugin_id='blocker_plugin'\n"
        "    )\n"
    )

    await runtime.install_plugin(plugin_dir)
    # Before disable: tool should be available (may be blocked by approval ladder in test, but not by plugin)
    result_before = await runtime.execute_tool("blocked_tool", {})
    assert "disabled" not in result_before["output"].lower()

    await runtime.disable_plugin("blocker_plugin")
    result_after = await runtime.execute_tool("blocked_tool", {})
    assert result_after["success"] is False
    assert "disabled" in result_after["output"].lower()

    await runtime.enable_plugin("blocker_plugin")
    result_enabled = await runtime.execute_tool("blocked_tool", {})
    assert "disabled" not in result_enabled["output"].lower()


@pytest.mark.asyncio
async def test_disabling_desktop_context_plugin_disables_live_service() -> None:
    lifecycle_calls = []

    class _FakeLifecycle:
        async def disable(self, plugin_id: str) -> None:
            lifecycle_calls.append(plugin_id)

    class _FakeDesktopContext:
        def __init__(self) -> None:
            self.calls = []

        def configure(self, **updates):
            self.calls.append(updates)
            return {"config": dict(updates)}

    desktop_context = _FakeDesktopContext()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(plugin_lifecycle=_FakeLifecycle()),
        desktop_context=desktop_context,
    )

    await disable_runtime_plugin(runtime, "desktop_context")

    assert lifecycle_calls == ["desktop_context"]
    assert desktop_context.calls == [
        {
            "enabled": False,
            "media_commentary_mode_enabled": False,
            "proactive_video_commentary_enabled": False,
            "live_transcription_enabled": False,
            "tts_enabled": False,
            "play_audio": False,
            "media_commentary_requested_at": None,
            "media_commentary_source": None,
            "media_commentary_request": None,
            "media_commentary_request_source": None,
            "media_commentary_request_text": None,
        }
    ]


@pytest.mark.asyncio
async def test_default_tools_have_core_plugin_id(runtime: AgentRuntime) -> None:
    # Core tools like fs_read_file should have plugin_id="core"
    assert "core" in runtime.tools._plugin_tools.get("fs_read_file", "")
    assert "core" in runtime.tools._plugin_tools.get("bash_run_command", "")


@pytest.mark.asyncio
async def test_typed_hook_registry_priority(runtime: AgentRuntime) -> None:
    reg = runtime.ctx.typed_hook_registry
    order = []

    def low(_, ctx):
        order.append("low")
        return __import__("opencas.infra.hook_registry", fromlist=["HookResult"]).HookResult(allowed=True)

    def high(_, ctx):
        order.append("high")
        return __import__("opencas.infra.hook_registry", fromlist=["HookResult"]).HookResult(allowed=True)

    reg.register("phase6_test_hook", low, priority=1)
    reg.register("phase6_test_hook", high, priority=10)
    result = reg.run("phase6_test_hook", {})
    assert result.allowed is True
    assert order == ["high", "low"]
