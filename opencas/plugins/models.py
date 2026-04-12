"""Data models for the plugins / skills subsystem."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry


@dataclass
class SkillEntry:
    """Registered skill metadata."""

    skill_id: str
    name: str
    description: str
    entrypoint: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    register_fn: Optional[Callable[[ToolRegistry], None]] = None
    plugin_id: Optional[str] = None


@dataclass
class PluginEntry:
    """Registered plugin metadata."""

    plugin_id: str
    name: str
    description: str
    version: str = "0.0.1"
    source: str = "installed"  # or "builtin"
    path: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    skills: List[SkillEntry] = field(default_factory=list)
    on_load_fn: Optional[Callable[[], None]] = None
    on_unload_fn: Optional[Callable[[], None]] = None
