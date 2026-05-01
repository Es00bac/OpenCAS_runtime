"""Tools module for OpenCAS: registry, filesystem, and shell tools."""

from .adapters.fs import FileSystemToolAdapter
from .adapters.shell import ShellToolAdapter
from .context import ToolUseContext, ToolUseResult, UserInputRequired
from .loop import ToolUseLoop
from .models import ToolEntry, ToolResult
from .registry import ToolRegistry
from .schema import build_tool_schemas
from .tool_use_memory import ToolUseLesson, ToolUseMemoryStore
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
    "ToolUseLesson",
    "ToolValidationContext",
    "ToolValidationPipeline",
    "ToolValidationResult",
    "ToolUseContext",
    "ToolUseLoop",
    "ToolUseMemoryStore",
    "ToolUseResult",
    "UserInputRequired",
]
