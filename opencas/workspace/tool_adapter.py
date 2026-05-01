import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencas.tools.models import ToolResult
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
                    "description": "Get a highly compressed semantic gist of a file without reading its full content.",
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
                        "List all files in a directory along with their 1-line gists "
                        "to understand a subsystem."
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
                return ToolResult(
                    success=False,
                    output=(
                        f"No gist found for {path}. The file may not exist, "
                        "be ignored, or indexing hasn't finished."
                    ),
                    metadata={"path": str(path)},
                )
            
            output = {
                "abs_path": str(result.abs_path),
                "checksum": result.checksum,
                "file_kind": result.file_kind,
                "size_bytes": result.size_bytes,
                "gist": result.gist_text,
                "cosine_similarity": result.cosine_similarity,
                "needs_further_reading": result.needs_further_reading,
            }
            return ToolResult(success=True, output=json.dumps(output, indent=2), metadata={})

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
            results = await self.service.list_directory(path)
            
            formatted = []
            for r in results:
                formatted.append({
                    "name": str(r.abs_path.name),
                    "kind": r.file_kind,
                    "gist": r.gist_text,
                    "needs_further_reading": r.needs_further_reading,
                })

            return ToolResult(
                success=True,
                output=json.dumps({"directory": str(path), "files": formatted}, indent=2),
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
