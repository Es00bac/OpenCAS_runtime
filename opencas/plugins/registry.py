"""Registries for plugins and skills in OpenCAS."""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import PluginEntry, SkillEntry


class SkillRegistry:
    """Registry for loaded skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillEntry] = {}

    def register(self, entry: SkillEntry) -> None:
        """Register a skill entry."""
        self._skills[entry.skill_id] = entry

    def get(self, skill_id: str) -> Optional[SkillEntry]:
        return self._skills.get(skill_id)

    def list_skills(self) -> List[SkillEntry]:
        return list(self._skills.values())

    def resolve_for_tool(self, tool_name: str) -> Optional[SkillEntry]:
        """Find a skill that claims to provide *tool_name*."""
        for skill in self._skills.values():
            if tool_name in skill.capabilities:
                return skill
        return None

    def unregister(self, skill_id: str) -> None:
        """Remove a skill from the registry."""
        self._skills.pop(skill_id, None)


class PluginRegistry:
    """Registry for loaded plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, PluginEntry] = {}

    def register(self, entry: PluginEntry) -> None:
        """Register a plugin entry."""
        self._plugins[entry.plugin_id] = entry

    def get(self, plugin_id: str) -> Optional[PluginEntry]:
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> List[PluginEntry]:
        return list(self._plugins.values())

    def unregister(self, plugin_id: str) -> None:
        """Remove a plugin from the registry."""
        self._plugins.pop(plugin_id, None)
