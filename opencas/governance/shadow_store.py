"""File-backed store for ShadowRegistry blocked intentions."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .shadow_models import BlockedIntention, ShadowClusterTriageState


class ShadowRegistryStore:
    """Persist blocked intentions as individual JSON documents."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cluster_path = self.path / "_clusters"
        self.cluster_path.mkdir(parents=True, exist_ok=True)

    def save(self, intention: BlockedIntention) -> BlockedIntention:
        target = self.path / f"{intention.id}.json"
        target.write_text(intention.model_dump_json(indent=2), encoding="utf-8")
        return intention

    def get(self, intention_id: str) -> Optional[BlockedIntention]:
        target = self.path / f"{intention_id}.json"
        if not target.exists():
            return None
        try:
            return BlockedIntention.model_validate_json(target.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_all(self) -> List[BlockedIntention]:
        items: list[tuple[int, BlockedIntention]] = []
        for target in self.path.glob("*.json"):
            try:
                items.append(
                    (
                        target.stat().st_mtime_ns,
                        BlockedIntention.model_validate_json(target.read_text(encoding="utf-8")),
                    )
                )
            except Exception:
                continue
        items.sort(key=lambda item: (item[1].captured_at, item[0]), reverse=True)
        return [item for _, item in items]

    def list_recent(self, limit: int = 10, offset: int = 0) -> List[BlockedIntention]:
        items = self.list_all()
        return items[offset : offset + limit]

    def save_cluster_state(self, state: ShadowClusterTriageState) -> ShadowClusterTriageState:
        target = self.cluster_path / f"{state.fingerprint}.json"
        target.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return state

    def get_cluster_state(self, fingerprint: str) -> Optional[ShadowClusterTriageState]:
        target = self.cluster_path / f"{fingerprint}.json"
        if not target.exists():
            return None
        try:
            return ShadowClusterTriageState.model_validate_json(target.read_text(encoding="utf-8"))
        except Exception:
            return None
