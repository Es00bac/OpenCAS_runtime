from types import SimpleNamespace

from opencas.runtime.capability_context import build_runtime_capability_context


class _ToolRegistry:
    def __init__(self, names):
        self._entries = [
            entry
            if not isinstance(entry, str)
            else SimpleNamespace(name=entry, risk_tier=SimpleNamespace(value="readonly"), description="")
            for entry in names
        ]

    def list_tools(self):
        return self._entries


class _CapabilityRegistry:
    def __init__(self, descriptors):
        self._descriptors = descriptors

    def list_capabilities(self):
        return self._descriptors


def test_runtime_capability_context_marks_opencas_inspectability_surfaces():
    runtime = SimpleNamespace(
        tools=_ToolRegistry(
            [
                "runtime_status",
                "workflow_status",
                "self_inspection_query",
                "wellbeing_query",
            ]
        )
    )

    context = build_runtime_capability_context(runtime)

    assert "OpenCAS operational inspectability surfaces" in context
    assert "operator-facing diagnostic records" in context
    assert "workflow_status" in context
    assert "wellbeing_query" in context


def test_runtime_capability_context_includes_live_somatic_snapshot():
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(
                    somatic_tag="caring",
                    valence=0.08,
                    arousal=0.02,
                    energy=1.0,
                    focus=0.5,
                    fatigue=0.01,
                    tension=0.03,
                    certainty=0.6,
                )
            )
        ),
        tools=_ToolRegistry(["runtime_status", "wellbeing_query"]),
    )

    context = build_runtime_capability_context(runtime)

    assert "Current somatic snapshot:" in context
    assert "tag=caring" in context
    assert "energy=1.00" in context


def test_runtime_capability_context_includes_continuity_clock():
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            identity=SimpleNamespace(
                continuity=SimpleNamespace(
                    last_offline_duration_seconds=125.0,
                    last_offline_started_at="2026-05-07T00:28:13+00:00",
                    last_boot_time="2026-05-07T00:30:18+00:00",
                    continuous_present_score=0.99,
                    continuity_breadcrumb=(
                        "2026-05-07T00:30:00+00:00 | intent: start burst: Conversation burst "
                        "for How long have I been gone | decision: current probe"
                    ),
                    continuity_breadcrumbs=[
                        "2026-05-07T00:20:00+00:00 | intent: finish Prompt F | decision: live verify",
                        (
                            "2026-05-07T00:25:00+00:00 | intent: start burst: Cycle burst for cycle "
                            "| decision: scheduler housekeeping"
                        ),
                        (
                            "2026-05-07T00:26:00+00:00 | intent: resume context after 1 minutes "
                            "| decision: boot bookkeeping"
                        ),
                        (
                            "2026-05-07T00:30:00+00:00 | intent: start burst: Conversation burst "
                            "for How long have I been gone | decision: current probe"
                        ),
                    ],
                )
            )
        ),
        tools=_ToolRegistry(["runtime_status"]),
    )

    context = build_runtime_capability_context(runtime)

    assert "Continuity clock:" in context
    assert "elapsed_since_last_shutdown_or_persistence=2 minutes" in context
    assert "last_offline_started_at=2026-05-07T00:28:13+00:00" in context
    assert "continuous_present_score=0.99" in context
    assert "latest_breadcrumb=2026-05-07T00:20:00+00:00 | intent: finish Prompt F" in context
    assert "current probe" not in context
    assert "scheduler housekeeping" not in context
    assert "boot bookkeeping" not in context


def test_runtime_capability_context_summarizes_high_level_capability_concepts():
    runtime = SimpleNamespace(
        capability_registry=_CapabilityRegistry(
            [
                SimpleNamespace(
                    capability_id="plugin:desktop_context.observe",
                    source=SimpleNamespace(value="plugin"),
                    owner_id="desktop_context",
                    status=SimpleNamespace(value="enabled"),
                    metadata={},
                    tool_names=["desktop_context_speak"],
                    description="Desktop context body-double plugin.",
                ),
                SimpleNamespace(
                    capability_id="mcp:github.search",
                    source=SimpleNamespace(value="mcp"),
                    owner_id="mcp:github",
                    status=SimpleNamespace(value="enabled"),
                    metadata={},
                    tool_names=["mcp__github__search"],
                    description="MCP tool.",
                ),
            ]
        ),
        tools=_ToolRegistry(
            [
                "browser_start",
                "browser_navigate",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "mcp_list_servers",
                "mcp_register_server_tools",
                "desktop_context_speak",
                "phone_get_status",
                "workflow_create_schedule",
                "workflow_list_schedules",
                "workflow_cancel_schedule",
                "bash_run_command",
                "pty_start",
                "tui_playwright_open",
                "tui_playwright_keys",
                "process_start",
                "google_workspace_calendar_schedule",
                "himalaya_email_headlines",
                "self_inspection_query",
                "wellbeing_query",
                "workflow_cancel_project",
            ]
        ),
        skill_registry=SimpleNamespace(
            list_skills=lambda: [
                SimpleNamespace(skill_id="google_workspace"),
                SimpleNamespace(skill_id="desktop_context"),
            ]
        ),
    )

    context = build_runtime_capability_context(runtime)

    assert "feature plugin/extension system: available" in context
    assert "feature skills system: available" in context
    assert "feature browser automation: available" in context
    assert "feature MCP integration: available" in context
    assert "feature voice/contact channels: available" in context
    assert "feature scheduled follow-up: available" in context
    assert "feature Linux terminal/process use: available" in context
    assert "feature browser-driven terminal UI use: available" in context
    assert "tui_playwright_keys" in context
    assert "feature Google Workspace access: available" in context
    assert "feature local email access: available" in context
    assert "feature self-inspection/wellbeing diagnostics: available" in context


def test_runtime_capability_context_renders_tool_descriptions():
    runtime = SimpleNamespace(
        tools=_ToolRegistry(
            [
                SimpleNamespace(
                    name="workflow_update_schedule",
                    risk_tier=SimpleNamespace(value="workspace_write"),
                    description=(
                        "Update OpenCAS schedule records; set status=cancelled to unschedule "
                        "a future OpenCAS task or reminder."
                    ),
                ),
                SimpleNamespace(
                    name="workflow_cancel_schedule",
                    risk_tier=SimpleNamespace(value="workspace_write"),
                    description="Cancel or unschedule an OpenCAS scheduled task, event, or reminder.",
                ),
                SimpleNamespace(
                    name="workflow_cancel_task",
                    risk_tier=SimpleNamespace(value="workspace_write"),
                    description="Cancel or delete a BAA task, not an OpenCAS schedule.",
                ),
            ]
        )
    )

    context = build_runtime_capability_context(runtime)

    assert (
        "- tool workflow_update_schedule; risk=workspace_write; description=Update OpenCAS "
        "schedule records; set status=cancelled to unschedule a future OpenCAS task or reminder."
    ) in context
    assert (
        "- tool workflow_cancel_schedule; risk=workspace_write; description=Cancel or unschedule "
        "an OpenCAS scheduled task, event, or reminder."
    ) in context
    assert (
        "- tool workflow_cancel_task; risk=workspace_write; description=Cancel or delete a BAA task, "
        "not an OpenCAS schedule."
    ) in context


def test_runtime_capability_context_excludes_disabled_plugin_tools():
    runtime = SimpleNamespace(
        tools=_ToolRegistry(
            [
                SimpleNamespace(
                    name="desktop_context_configure",
                    risk_tier=SimpleNamespace(value="workspace_write"),
                    description="Configure desktop context.",
                ),
                SimpleNamespace(
                    name="system_status",
                    risk_tier=SimpleNamespace(value="readonly"),
                    description="Read host status.",
                ),
            ]
        ),
        ctx=SimpleNamespace(
            plugin_lifecycle=SimpleNamespace(
                is_tool_disabled=lambda name: name == "desktop_context_configure"
            )
        ),
    )

    context = build_runtime_capability_context(runtime)

    assert "desktop_context_configure" not in context
    assert "system_status" in context
