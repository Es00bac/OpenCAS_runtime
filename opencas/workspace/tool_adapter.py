import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencas.tools.models import ToolResult
from opencas.workspace.scanner import sha256_file_sync
from opencas.workspace.service import WorkspaceIndexService


class RefreshWorkspaceIndexSchema(BaseModel):
    roots: Optional[List[str]] = Field(None, description="Optional specific roots to refresh.")
    force: bool = Field(False, description="Force re-hash even if unmodified.")


class GetFileGistSchema(BaseModel):
    abs_path: str = Field(description="Absolute path to the file.")
    refresh_if_stale: bool = Field(False, description="Refresh the gist if stale.")


class SearchFileGistsSchema(BaseModel):
    query: str = Field(description="Semantic search query.")
    limit: int = Field(8, description="Max results.")


class ListDirectoryGistsSchema(BaseModel):
    dir_path: str = Field(description="Absolute path to the directory.")
    recursive: bool = Field(False, description="Include subdirectories.")


class WorkspaceIndexerToolAdapter:
    """Tool adapter for semantic workspace file gisting and discovery."""

    def __init__(self, service: WorkspaceIndexService) -> None:
        self.service = service

    def schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "workspace_get_file_gist",
                    "description": (
                        "Returns the workspace gist for a file when one exists, "
                        "or 'gist_pending' status with checksum and live metadata "
                        "when the file exists on disk but no gist has been generated yet. "
                        "Never silently reports a missing file as 'no gist'."
                    ),
                    "parameters": GetFileGistSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_search_file_gists",
                    "description": (
                        "Semantically search the workspace for files related to a query "
                        "using gist embeddings."
                    ),
                    "parameters": SearchFileGistsSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_list_directory_gists",
                    "description": (
                        "Snapshot of the workspace semantic index for a directory, "
                        "plus a live filesystem fallback when the index is empty for "
                        "an existing directory. Always returns index_status with "
                        "last_scan_age_seconds. Prefer artifact_lookup for "
                        "authorship/origin questions about specific files."
                    ),
                    "parameters": ListDirectoryGistsSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_refresh_index",
                    "description": "Force a refresh of the workspace semantic index.",
                    "parameters": RefreshWorkspaceIndexSchema.model_json_schema(),
                },
            },
        ]

    async def __call__(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name == "workspace_get_file_gist":
            args = GetFileGistSchema(**arguments)
            path = Path(args.abs_path).expanduser().resolve()
            result = await self.service.get_gist_for_path(path, refresh_if_stale=args.refresh_if_stale)
            if not result:
                if path.is_file():
                    try:
                        stat = path.stat()
                    except OSError:
                        stat = None
                    output = {
                        "status": "gist_pending",
                        "abs_path": str(path),
                        "exists_on_disk": True,
                        "size_bytes": stat.st_size if stat else None,
                        "mtime": stat.st_mtime if stat else None,
                        "mtime_ns": stat.st_mtime_ns if stat else None,
                        "checksum": sha256_file_sync(path),
                    }
                    return ToolResult(
                        success=True,
                        output=json.dumps(output, indent=2),
                        metadata={"path": str(path), "gist_pending": True},
                    )
                return ToolResult(
                    success=False,
                    output=(
                        f"No workspace_paths row and no file on disk at {path}."
                    ),
                    metadata={"path": str(path)},
                )
            
            output = {
                "status": "gist" if result.gist_text else "gist_pending",
                "abs_path": str(result.abs_path),
                "checksum": result.checksum,
                "file_kind": result.file_kind,
                "size_bytes": result.size_bytes,
                "gist": result.gist_text,
                "cosine_similarity": result.cosine_similarity,
                "needs_further_reading": result.needs_further_reading,
            }
            return ToolResult(
                success=True,
                output=json.dumps(output, indent=2),
                metadata={
                    "path": str(result.abs_path),
                    "gist_pending": result.gist_text is None,
                },
            )

        elif name == "workspace_search_file_gists":
            args = SearchFileGistsSchema(**arguments)
            results = await self.service.search(args.query, limit=args.limit)

            formatted = []
            for res in results:
                formatted.append(
                    {
                        "path": res.get("path"),
                        "score": res.get("score"),
                        "checksum": res.get("checksum"),
                        "fallback": res.get("fallback", False),
                    }
                )
            
            return ToolResult(success=True, output=json.dumps({"results": formatted}, indent=2), metadata={})

        elif name == "workspace_list_directory_gists":
            args = ListDirectoryGistsSchema(**arguments)
            path = Path(args.dir_path).expanduser().resolve()
            if hasattr(self.service, "list_directory_with_status"):
                payload = await self.service.list_directory_with_status(path)
            else:
                results = await self.service.list_directory(path)
                payload = {
                    "directory": str(path),
                    "indexed_files": [
                        {
                            "name": str(r.abs_path.name),
                            "kind": r.file_kind,
                            "gist": r.gist_text,
                            "needs_further_reading": r.needs_further_reading,
                        }
                        for r in results
                    ],
                    "disk_listing": [],
                    "index_status": {
                        "last_scan_age_seconds": None,
                        "indexed_count": len(results),
                        "disk_count": None,
                        "fallback_used": False,
                        "scan_root": None,
                    },
                }

            return ToolResult(
                success=True,
                output=json.dumps(payload, indent=2),
                metadata={},
            )

        elif name == "workspace_refresh_index":
            args = RefreshWorkspaceIndexSchema(**arguments)
            await self.service.full_scan(force=args.force)
            return ToolResult(
                success=True,
                output="Workspace index refresh triggered.",
                metadata={},
            )

        raise ValueError(f"Unknown tool: {name}")
