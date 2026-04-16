"""Shared model-routing config for OpenCAS chat/reasoning workloads."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ComplexityTier(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    HIGH = "high"
    EXTRA_HIGH = "extra_high"


class ModelRoutingMode(str, Enum):
    SINGLE = "single"
    TIERED = "tiered"


class ModelRoutingConfig(BaseModel):
    """Persisted chat/reasoning model selection policy.

    The routing policy is intentionally small: it determines which model OpenCAS
    should use for each reasoning tier and whether automatic escalation is
    allowed while a workflow is still in progress.
    """

    mode: ModelRoutingMode = ModelRoutingMode.SINGLE
    single_model: Optional[str] = None
    light_model: Optional[str] = None
    standard_model: Optional[str] = None
    high_model: Optional[str] = None
    extra_high_model: Optional[str] = None
    auto_escalation: bool = True

    def normalized(self, default_model: Optional[str]) -> "ModelRoutingConfig":
        """Return a copy with all fallbacks resolved against *default_model*."""
        fallback = (self.single_model or default_model or "").strip() or None
        if self.mode == ModelRoutingMode.SINGLE:
            return self.model_copy(
                update={
                    "single_model": fallback,
                    "light_model": fallback,
                    "standard_model": fallback,
                    "high_model": fallback,
                    "extra_high_model": fallback,
                }
            )

        standard = (self.standard_model or fallback or "").strip() or None
        light = (self.light_model or standard or fallback or "").strip() or None
        high = (self.high_model or standard or fallback or "").strip() or None
        extra_high = (
            self.extra_high_model or high or standard or fallback or ""
        ).strip() or None
        return self.model_copy(
            update={
                "single_model": fallback or standard,
                "light_model": light,
                "standard_model": standard,
                "high_model": high,
                "extra_high_model": extra_high,
            }
        )

    def resolve_model(
        self,
        *,
        default_model: Optional[str],
        complexity: ComplexityTier = ComplexityTier.STANDARD,
    ) -> Optional[str]:
        """Resolve the configured model for *complexity* with stable fallback."""
        normalized = self.normalized(default_model)
        if complexity == ComplexityTier.LIGHT:
            return normalized.light_model or normalized.standard_model
        if complexity == ComplexityTier.HIGH:
            return normalized.high_model or normalized.standard_model
        if complexity == ComplexityTier.EXTRA_HIGH:
            return (
                normalized.extra_high_model
                or normalized.high_model
                or normalized.standard_model
            )
        return normalized.standard_model or normalized.single_model

    def effective_map(self, default_model: Optional[str]) -> Dict[str, Optional[str]]:
        """Return the resolved model mapping for dashboard/API display."""
        normalized = self.normalized(default_model)
        return {
            ComplexityTier.LIGHT.value: normalized.light_model,
            ComplexityTier.STANDARD.value: normalized.standard_model,
            ComplexityTier.HIGH.value: normalized.high_model,
            ComplexityTier.EXTRA_HIGH.value: normalized.extra_high_model,
        }


class PersistedModelRoutingState(BaseModel):
    """Minimal runtime settings persisted under the OpenCAS state directory."""

    default_llm_model: Optional[str] = None
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)


def model_routing_settings_path(state_dir: Path) -> Path:
    """Return the canonical state file used for model-routing persistence."""
    return Path(state_dir).expanduser() / "runtime_model_routing.json"


def load_persisted_model_routing_state(
    state_dir: Path,
) -> Optional[PersistedModelRoutingState]:
    """Load persisted model-routing state when present and parseable."""
    path = model_routing_settings_path(state_dir)
    if not path.exists():
        return None
    try:
        return PersistedModelRoutingState.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def save_persisted_model_routing_state(
    state_dir: Path,
    state: PersistedModelRoutingState,
) -> Path:
    """Persist model-routing state beneath *state_dir*."""
    path = model_routing_settings_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def normalize_complexity_tier(
    value: ComplexityTier | str | None,
) -> ComplexityTier:
    """Accept dashboard/user strings while keeping a typed internal tier."""
    if isinstance(value, ComplexityTier):
        return value
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "default"}:
        return ComplexityTier.STANDARD
    return ComplexityTier(text)


def next_complexity_tier(
    value: ComplexityTier | str | None,
) -> ComplexityTier:
    """Return the next higher complexity tier without exceeding extra-high."""
    current = normalize_complexity_tier(value)
    if current == ComplexityTier.LIGHT:
        return ComplexityTier.STANDARD
    if current == ComplexityTier.STANDARD:
        return ComplexityTier.HIGH
    if current == ComplexityTier.HIGH:
        return ComplexityTier.EXTRA_HIGH
    return ComplexityTier.EXTRA_HIGH
