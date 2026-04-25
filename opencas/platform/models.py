"""Canonical platform capability and extension descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class CapabilitySource(str, Enum):
    """Origin of a capability."""

    CORE = "core"
    PLUGIN = "plugin"
    MCP = "mcp"


class CapabilityStatus(str, Enum):
    """Lifecycle status for a capability or extension."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    FAILED_VALIDATION = "failed_validation"
    MISSING_DEPENDENCY = "missing_dependency"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Canonical description of one capability exposed by the platform."""

    capability_id: str
    display_name: str
    kind: str
    source: CapabilitySource
    owner_id: str
    status: CapabilityStatus
    description: str = ""
    tool_names: list[str] = field(default_factory=list)
    declared_dependencies: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    entrypoint: str | None = None
    manifest_path: str | None = None
    source_path: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_status(
        self,
        status: CapabilityStatus,
        *,
        errors: list[str] | None = None,
    ) -> "CapabilityDescriptor":
        """Return a copy of the descriptor with updated status metadata."""

        return replace(self, status=status, validation_errors=list(errors or []))


@dataclass(frozen=True)
class ExtensionDescriptor:
    """Canonical description of a platform extension."""

    extension_id: str
    extension_kind: str
    display_name: str
    status: CapabilityStatus
    capability_ids: list[str] = field(default_factory=list)
    manifest_version: int | None = None
    version: str | None = None
    manifest_path: str | None = None
    dependencies: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    release_notes: str = ""
    config_schema_summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
