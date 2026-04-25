"""Shared model-routing config for OpenCAS chat/reasoning workloads."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional

from pydantic import BaseModel, Field


class ComplexityTier(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    HIGH = "high"
    EXTRA_HIGH = "extra_high"


class ModelRoutingMode(str, Enum):
    SINGLE = "single"
    TIERED = "tiered"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTRA_HIGH = "xhigh"


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
    single_reasoning_effort: Optional[ReasoningEffort] = None
    light_reasoning_effort: Optional[ReasoningEffort] = None
    standard_reasoning_effort: Optional[ReasoningEffort] = None
    high_reasoning_effort: Optional[ReasoningEffort] = None
    extra_high_reasoning_effort: Optional[ReasoningEffort] = None
    auto_escalation: bool = True

    def normalized(self, default_model: Optional[str]) -> "ModelRoutingConfig":
        """Return a copy with all fallbacks resolved against *default_model*."""
        fallback = (self.single_model or default_model or "").strip() or None
        fallback_effort = self.single_reasoning_effort
        if self.mode == ModelRoutingMode.SINGLE:
            return self.model_copy(
                update={
                    "single_model": fallback,
                    "light_model": fallback,
                    "standard_model": fallback,
                    "high_model": fallback,
                    "extra_high_model": fallback,
                    "light_reasoning_effort": fallback_effort,
                    "standard_reasoning_effort": fallback_effort,
                    "high_reasoning_effort": fallback_effort,
                    "extra_high_reasoning_effort": fallback_effort,
                }
            )

        standard = (self.standard_model or fallback or "").strip() or None
        light = (self.light_model or standard or fallback or "").strip() or None
        high = (self.high_model or standard or fallback or "").strip() or None
        extra_high = (
            self.extra_high_model or high or standard or fallback or ""
        ).strip() or None
        standard_effort = self.standard_reasoning_effort or fallback_effort
        light_effort = self.light_reasoning_effort or standard_effort
        high_effort = self.high_reasoning_effort or standard_effort
        extra_high_effort = (
            self.extra_high_reasoning_effort or high_effort or standard_effort
        )
        return self.model_copy(
            update={
                "single_model": fallback or standard,
                "light_model": light,
                "standard_model": standard,
                "high_model": high,
                "extra_high_model": extra_high,
                "light_reasoning_effort": light_effort,
                "standard_reasoning_effort": standard_effort,
                "high_reasoning_effort": high_effort,
                "extra_high_reasoning_effort": extra_high_effort,
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

    def resolve_reasoning_effort(
        self,
        *,
        complexity: ComplexityTier = ComplexityTier.STANDARD,
    ) -> Optional[ReasoningEffort]:
        """Resolve the configured reasoning-effort override for *complexity*."""
        normalized = self.normalized(default_model=self.single_model)
        if complexity == ComplexityTier.LIGHT:
            return (
                normalized.light_reasoning_effort
                or normalized.standard_reasoning_effort
            )
        if complexity == ComplexityTier.HIGH:
            return (
                normalized.high_reasoning_effort
                or normalized.standard_reasoning_effort
            )
        if complexity == ComplexityTier.EXTRA_HIGH:
            return (
                normalized.extra_high_reasoning_effort
                or normalized.high_reasoning_effort
                or normalized.standard_reasoning_effort
            )
        return normalized.standard_reasoning_effort or normalized.single_reasoning_effort

    def effective_reasoning_map(
        self,
        default_model: Optional[str],
    ) -> Dict[str, Optional[str]]:
        """Return the resolved reasoning-effort mapping for dashboard/API display."""
        normalized = self.normalized(default_model)
        return {
            ComplexityTier.LIGHT.value: (
                normalized.light_reasoning_effort.value
                if normalized.light_reasoning_effort
                else None
            ),
            ComplexityTier.STANDARD.value: (
                normalized.standard_reasoning_effort.value
                if normalized.standard_reasoning_effort
                else None
            ),
            ComplexityTier.HIGH.value: (
                normalized.high_reasoning_effort.value
                if normalized.high_reasoning_effort
                else None
            ),
            ComplexityTier.EXTRA_HIGH.value: (
                normalized.extra_high_reasoning_effort.value
                if normalized.extra_high_reasoning_effort
                else None
            ),
        }


class PersistedModelRoutingState(BaseModel):
    """Minimal runtime settings persisted under the OpenCAS state directory."""

    default_llm_model: Optional[str] = None
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)


MODEL_ROUTING_MODEL_FIELDS = (
    "single_model",
    "light_model",
    "standard_model",
    "high_model",
    "extra_high_model",
)


def _clean_model_id(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _resolve_available_model(
    candidate: Optional[str],
    ordered_available: list[str],
    available_models: set[str],
) -> Optional[str]:
    clean = _clean_model_id(candidate)
    if not clean:
        return None
    if clean in available_models:
        return clean
    if "/" not in clean:
        suffix = f"/{clean}"
        matches = [item for item in ordered_available if item.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    return None


def _pick_available_model(
    default_model: Optional[str],
    model_routing: ModelRoutingConfig,
    ordered_available: list[str],
    available_models: set[str],
) -> Optional[str]:
    ordered_candidates = [
        default_model,
        model_routing.standard_model,
        model_routing.single_model,
        model_routing.high_model,
        model_routing.extra_high_model,
        model_routing.light_model,
    ]
    for candidate in ordered_candidates:
        resolved = _resolve_available_model(candidate, ordered_available, available_models)
        if resolved:
            return resolved
    return ordered_available[0] if ordered_available else _clean_model_id(default_model)


def sanitize_model_routing_state(
    default_model: Optional[str],
    model_routing: Optional[ModelRoutingConfig],
    available_models: Iterable[str],
) -> PersistedModelRoutingState:
    """Clamp persisted/runtime routing to models that actually exist.

    This prevents stale routed model IDs from surviving provider changes and
    poisoning future startups with models that no longer exist in the active
    OpenLLMAuth material.
    """

    routing = model_routing or ModelRoutingConfig()
    ordered_available: list[str] = []
    available: set[str] = set()
    for clean in (_clean_model_id(item) for item in available_models):
        if not clean or clean in available:
            continue
        ordered_available.append(clean)
        available.add(clean)
    if not available:
        normalized = routing.normalized(_clean_model_id(default_model))
        return PersistedModelRoutingState(
            default_llm_model=_clean_model_id(default_model),
            model_routing=normalized,
        )

    fallback = _pick_available_model(default_model, routing, ordered_available, available)
    if fallback is None and ordered_available:
        fallback = ordered_available[0]
    updates: Dict[str, Optional[str]] = {}
    for field in MODEL_ROUTING_MODEL_FIELDS:
        updates[field] = _resolve_available_model(
            getattr(routing, field, None),
            ordered_available,
            available,
        )
    normalized = routing.model_copy(update=updates).normalized(fallback)
    return PersistedModelRoutingState(
        default_llm_model=fallback,
        model_routing=normalized,
    )


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
