from __future__ import annotations

from pathlib import Path

from opencas.desktop_context.region_prompt_portal import build_region_prompt_command


def test_region_prompt_portal_command_defaults_to_system_python_and_repo_script() -> None:
    command = build_region_prompt_command(python_executable="/usr/bin/python3")

    assert command[0] == "/usr/bin/python3"
    assert command[1].endswith("scripts/opencas_region_prompt.py")


def test_region_prompt_portal_command_accepts_explicit_script() -> None:
    command = build_region_prompt_command(
        python_executable="/custom/python",
        script_path=Path("/tmp/custom-region.py"),
    )

    assert command == ["/custom/python", "/tmp/custom-region.py"]
