from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

FileKind = Literal["text", "binary", "image", "audio", "archive", "db", "unknown"]
ContentTextStatus = Literal["full", "chunked", "metadata_only", "unreadable"]

class WorkspacePathRecord(BaseModel):
    abs_path: Path
    rel_path: Optional[Path] = None
    parent_dir: Path
    file_name: str
    extension: Optional[str] = None
    exists_flag: bool = True
    file_kind: FileKind
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
    current_checksum: Optional[str] = None
    first_seen_at: datetime
    last_seen_at: datetime
    last_scan_id: Optional[int] = None
    last_error: Optional[str] = None

class WorkspaceChecksumRecord(BaseModel):
    checksum: str = Field(min_length=64, max_length=64)
    size_bytes: int
    file_kind: FileKind
    mime_type: Optional[str] = None
    content_text_status: ContentTextStatus
    content_preview: Optional[str] = None
    content_embedding_ref: Optional[str] = None
    content_embedding_model: Optional[str] = None
    content_embedding_dim: Optional[int] = None
    first_seen_at: datetime
    last_seen_at: datetime

class WorkspaceGistRecord(BaseModel):
    checksum: str
    gist_text: str
    gist_json: Optional[str] = None
    llm_model: str
    prompt_version: str
    gist_embedding_ref: Optional[str] = None
    gist_embedding_model: Optional[str] = None
    gist_embedding_dim: Optional[int] = None
    cosine_similarity: Optional[float] = None
    drift_score: Optional[float] = None
    accepted_flag: bool = False
    needs_further_reading: bool = False
    attempt_count: int = 0
    updated_at: datetime

class WorkspaceGistAttempt(BaseModel):
    checksum: str
    attempt_no: int
    gist_text: str
    gist_json: Optional[str] = None
    cosine_similarity: Optional[float] = None
    drift_score: Optional[float] = None
    accepted_flag: bool = False
    failure_reason: Optional[str] = None
    created_at: datetime

class WorkspaceGistLookupResult(BaseModel):
    abs_path: Path
    checksum: Optional[str] = None
    gist_text: Optional[str] = None
    cosine_similarity: Optional[float] = None
    needs_further_reading: bool = False
    file_kind: FileKind
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
