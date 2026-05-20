"""Project helpers."""

from .classifier import ProjectClassification, classify_project_type
from .workspace_registry import (
    WorkspaceProjectCandidate,
    WorkspacePathRequest,
    resolve_requested_workspace_path,
    resolve_recent_workspace_project,
    resolve_workspace_project,
    resolve_workspace_project_from_text,
)

__all__ = [
    "ProjectClassification",
    "ProjectCancelResult",
    "TaskCancelResult",
    "WorkspaceProjectCandidate",
    "WorkspacePathRequest",
    "cancel_project",
    "cancel_task",
    "classify_project_type",
    "resolve_requested_workspace_path",
    "resolve_recent_workspace_project",
    "resolve_workspace_project",
    "resolve_workspace_project_from_text",
]


def __getattr__(name: str):
    if name in {"ProjectCancelResult", "TaskCancelResult", "cancel_project", "cancel_task"}:
        from . import lifecycle

        return getattr(lifecycle, name)
    raise AttributeError(name)
