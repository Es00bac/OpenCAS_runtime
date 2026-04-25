"""Tools module for OpenCAS: registry, filesystem, and shell tools."""

from .models import ToolEntry, ToolResult
from .registry import ToolRegistry
from .adapters.fs import FileSystemToolAdapter
from .adapters.shell import ShellToolAdapter
from .validation import (
    CommandSafetyValidator,
    ContentSizeValidator,
    FilesystemPathValidator,
    FilesystemWatchlistValidator,
    ToolValidationContext,
    ToolValidationPipeline,
    ToolValidationResult,
    create_default_tool_validation_pipeline,
)
from .schema import build_tool_schemas
from .context import ToolUseContext, ToolUseResult, UserInputRequired
from .loop import ToolUseLoop

__all__ = [
    "build_tool_schemas",
    "CommandSafetyValidator",
    "ContentSizeValidator",
    "create_default_tool_validation_pipeline",
    "FileSystemToolAdapter",
    "FilesystemPathValidator",
    "FilesystemWatchlistValidator",
    "ShellToolAdapter",
    "ToolEntry",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationContext",
    "ToolValidationPipeline",
    "ToolValidationResult",
    "ToolUseContext",
    "ToolUseLoop",
    "ToolUseResult",
    "UserInputRequired",
]
