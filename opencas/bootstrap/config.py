"""Bootstrap configuration for OpenCAS."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencas.model_routing import ModelRoutingConfig
from opencas.sandbox import SandboxConfig


class BootstrapConfig(BaseModel):
    """Configuration required to boot OpenCAS."""

    # State directories
    state_dir: Path = Field(default_factory=lambda: Path("~/.opencas/state").expanduser())
    memory_db: Optional[Path] = None
    tasks_db: Optional[Path] = None
    telemetry_dir: Optional[Path] = None
    embedding_db: Optional[Path] = None
    context_db: Optional[Path] = None
    work_db: Optional[Path] = None
    relational_db: Optional[Path] = None
    daydream_db: Optional[Path] = None
    conflict_db: Optional[Path] = None
    harness_db: Optional[Path] = None
    tom_db: Optional[Path] = None
    plugins_db: Optional[Path] = None

    # Session
    session_id: Optional[str] = None
    agent_profile_id: str = "general_technical_operator"

    # Explicit workspaces the agent is allowed to operate in
    workspace_root: Optional[Path] = None
    workspace_roots: List[Path] = Field(default_factory=list)
    managed_workspace_root: Optional[Path] = None

    # Embedding model override
    embedding_model_id: Optional[str] = "google/gemini-embedding-2-preview"

    # Optional Qdrant vector backend
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: Optional[str] = "opencas_embeddings"

    # Local HNSW vector backend tuning
    hnsw_enabled: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200

    # Persistent plans
    plans_db: Optional[Path] = None
    schedules_db: Optional[Path] = None

    # On-demand MCP servers
    mcp_servers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    mcp_auto_register: bool = False

    # LLM gateway default model
    default_llm_model: Optional[str] = "anthropic/claude-sonnet-4-6"
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)

    # Per-project OpenLLMAuth configuration
    provider_config_path: Optional[Path] = None
    provider_env_path: Optional[Path] = None
    credential_source_config_path: Optional[Path] = None
    credential_source_env_path: Optional[Path] = None
    credential_profile_ids: List[str] = Field(default_factory=list)
    credential_env_keys: List[str] = Field(default_factory=list)

    # Telegram channel integration
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_dm_policy: str = "pairing"
    telegram_allow_from: List[str] = Field(default_factory=list)
    telegram_poll_interval_seconds: float = 1.0
    telegram_pairing_ttl_seconds: int = 3600
    telegram_api_base_url: str = "https://api.telegram.org"

    # Sandbox configuration
    sandbox: Optional[SandboxConfig] = None

    # First-boot identity seeding
    clean_boot: bool = False
    persona_name: Optional[str] = None
    user_bio: Optional[str] = None
    user_name: Optional[str] = None

    def resolve_paths(self) -> "BootstrapConfig":
        """Ensure all derived paths are resolved relative to state_dir."""
        self.state_dir = self.state_dir.expanduser().resolve()
        if self.memory_db is None:
            self.memory_db = self.state_dir / "memory.db"
        if self.tasks_db is None:
            self.tasks_db = self.state_dir / "tasks.db"
        if self.telemetry_dir is None:
            self.telemetry_dir = self.state_dir / "telemetry"
        if self.embedding_db is None:
            self.embedding_db = self.state_dir / "embeddings.db"
        if self.context_db is None:
            self.context_db = self.state_dir / "context.db"
        if self.work_db is None:
            self.work_db = self.state_dir / "work.db"
        if self.relational_db is None:
            self.relational_db = self.state_dir / "relational.db"
        if self.daydream_db is None:
            self.daydream_db = self.state_dir / "daydream.db"
        if self.conflict_db is None:
            self.conflict_db = self.state_dir / "conflict.db"
        if self.harness_db is None:
            self.harness_db = self.state_dir / "harness.db"
        if self.tom_db is None:
            self.tom_db = self.state_dir / "tom.db"
        if self.plugins_db is None:
            self.plugins_db = self.state_dir / "plugins.db"
        if self.plans_db is None:
            self.plans_db = self.state_dir / "plans.db"
        if self.schedules_db is None:
            self.schedules_db = self.state_dir / "schedules.db"
        if self.workspace_root is not None:
            self.workspace_root = self.workspace_root.expanduser().resolve()
        self.workspace_roots = [
            Path(root).expanduser().resolve() for root in self.workspace_roots
        ]
        if self.managed_workspace_root is not None:
            self.managed_workspace_root = (
                self.managed_workspace_root.expanduser().resolve()
            )
        if self.provider_config_path is not None:
            self.provider_config_path = self.provider_config_path.expanduser().resolve()
        if self.provider_env_path is not None:
            self.provider_env_path = self.provider_env_path.expanduser().resolve()
        if self.credential_source_config_path is not None:
            self.credential_source_config_path = (
                self.credential_source_config_path.expanduser().resolve()
            )
        if self.credential_source_env_path is not None:
            self.credential_source_env_path = (
                self.credential_source_env_path.expanduser().resolve()
            )
        if self.sandbox is None:
            self.sandbox = SandboxConfig()
        return self

    def all_workspace_roots(self) -> List[Path]:
        """Return deduplicated configured workspace roots with a stable primary root."""
        roots: List[Path] = []
        if self.workspace_root is not None:
            roots.append(self.workspace_root)
        roots.extend(self.workspace_roots)
        if self.managed_workspace_root is not None and not any(
            self.managed_workspace_root.is_relative_to(root) for root in roots
        ):
            roots.append(self.managed_workspace_root)
        if not roots:
            roots.append(self.state_dir.parent)
        deduped = []
        seen = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(root)
        return deduped

    def primary_workspace_root(self) -> Path:
        """Return the primary workspace root for execution defaults."""
        return self.all_workspace_roots()[0]

    def agent_workspace_root(self) -> Path:
        """Return the dedicated workspace root for agent-created artifacts and projects."""
        if self.managed_workspace_root is not None:
            return self.managed_workspace_root
        return (self.primary_workspace_root() / "workspace").resolve()
