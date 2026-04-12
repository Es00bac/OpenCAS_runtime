"""Persistence for identity, user-model, and continuity state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .models import ContinuityState, SelfModel, UserModel


class IdentityStore:
    """File-based store for identity documents.

    Uses atomic writes (write-to-temp then rename) for durability.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.base_path / f"{name}.json"

    def _write(self, name: str, model: BaseModel) -> None:
        target = self._path(name)
        temp = self._path(f"{name}.tmp")
        temp.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(target)

    def _read(self, name: str, model_class: type) -> Optional[BaseModel]:
        target = self._path(name)
        if not target.exists():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None

    def save_self(self, model: SelfModel) -> None:
        self._write("self", model)

    def load_self(self) -> SelfModel:
        loaded = self._read("self", SelfModel)
        return loaded if isinstance(loaded, SelfModel) else SelfModel()

    def save_user(self, model: UserModel) -> None:
        self._write("user", model)

    def load_user(self) -> UserModel:
        loaded = self._read("user", UserModel)
        return loaded if isinstance(loaded, UserModel) else UserModel()

    def save_continuity(self, state: ContinuityState) -> None:
        self._write("continuity", state)

    def load_continuity(self) -> ContinuityState:
        loaded = self._read("continuity", ContinuityState)
        return loaded if isinstance(loaded, ContinuityState) else ContinuityState()
