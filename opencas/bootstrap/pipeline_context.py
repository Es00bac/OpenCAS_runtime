"""Workspace-index and context-assembly helpers for ``BootstrapPipeline``."""

from __future__ import annotations

from typing import Any

from opencas.api import LLMClient
from opencas.embeddings import EmbeddingService
from opencas.workspace.service import WorkspaceIndexService
from opencas.workspace.store import WorkspaceStore

from .context import BootstrapContext
from .config import BootstrapConfig


async def initialize_workspace_index(
    config: BootstrapConfig,
    embeddings: EmbeddingService,
    llm: LLMClient,
) -> WorkspaceIndexService:
    """Start the managed workspace index using the runtime defaults."""
    workspace_store = WorkspaceStore(config.state_dir / "workspace.db")
    workspace_index = WorkspaceIndexService(
        store=workspace_store,
        embeddings_client=embeddings,
        llm_client=llm,
        workspace_roots=config.workspace_roots,
        llm_model=config.default_llm_model or "kimi-coding/k2p5",
        embedding_model=config.embedding_model_id or "google/gemini-embedding-2-preview",
    )
    await workspace_index.start()
    return workspace_index


def build_bootstrap_context(**components: Any) -> BootstrapContext:
    """Construct the final bootstrap context from staged runtime components."""
    return BootstrapContext(**components)
