"""Plugin and skill registry for OpenCAS."""

from .lifecycle import PluginLifecycleManager
from .loader import (
    load_builtin_plugins,
    load_builtin_skills,
    load_plugin_from_manifest,
    load_skill_from_path,
)
from .models import PluginEntry, SkillEntry
from .registry import PluginRegistry, SkillRegistry
from .store import PluginStore

__all__ = [
    "PluginEntry",
    "PluginLifecycleManager",
    "PluginRegistry",
    "PluginStore",
    "SkillEntry",
    "SkillRegistry",
    "load_builtin_plugins",
    "load_builtin_skills",
    "load_plugin_from_manifest",
    "load_skill_from_path",
]
