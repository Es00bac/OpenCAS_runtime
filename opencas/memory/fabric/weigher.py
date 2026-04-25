"""Context-aware edge-weight fusion for memory fabric."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, TypedDict

from opencas.memory import EdgeKind


class ContextProfile(str, Enum):
    RETRIEVAL = "retrieval"
    CONSOLIDATION = "consolidation"
    CAUSAL_INFERENCE = "causal_inference"
    BRIDGE = "bridge"


class WeightRecipe(TypedDict):
    conceptual: float
    emotional: float
    relational: float
    temporal: float
    causal: float


class FusionResult(TypedDict):
    confidence: float
    kind: EdgeKind
    weights: Dict[str, float]


class EdgeWeigher:
    """Fuse multiple scorer outputs into a single confidence and elected EdgeKind."""

    RECIPES: Dict[ContextProfile, WeightRecipe] = {
        ContextProfile.RETRIEVAL: {
            "conceptual": 0.40,
            "emotional": 0.20,
            "relational": 0.20,
            "temporal": 0.10,
            "causal": 0.10,
        },
        ContextProfile.CONSOLIDATION: {
            "conceptual": 0.30,
            "emotional": 0.25,
            "relational": 0.20,
            "temporal": 0.15,
            "causal": 0.10,
        },
        ContextProfile.CAUSAL_INFERENCE: {
            "conceptual": 0.10,
            "emotional": 0.10,
            "relational": 0.15,
            "temporal": 0.20,
            "causal": 0.45,
        },
        ContextProfile.BRIDGE: {
            "conceptual": 0.2,
            "emotional": 0.1,
            "relational": 0.3,
            "temporal": 0.3,
            "causal": 0.1,
        },
    }

    def __init__(
        self,
        profile: ContextProfile = ContextProfile.CONSOLIDATION,
    ) -> None:
        self.profile = profile

    def fuse(
        self,
        scores: Dict[str, float],
        context: Optional[Any] = None,
    ) -> FusionResult:
        """Fuse scorer outputs into a single confidence and dominant EdgeKind."""
        recipe = self.RECIPES.get(
            self.profile, self.RECIPES[ContextProfile.CONSOLIDATION]
        )
        weights: Dict[str, float] = {}
        confidence = 0.0
        for key, weight in recipe.items():
            s = scores.get(key, 0.0)
            weighted = s * weight
            weights[key] = round(weighted, 4)
            confidence += weighted

        confidence = round(min(1.0, confidence), 4)
        kind = self._elect_kind(weights)
        return FusionResult(confidence=confidence, kind=kind, weights=weights)

    @staticmethod
    def _elect_kind(weights: Dict[str, float]) -> EdgeKind:
        max_key = max(weights, key=weights.get)
        mapping = {
            "conceptual": EdgeKind.CONCEPTUAL,
            "emotional": EdgeKind.EMOTIONAL,
            "relational": EdgeKind.RELATIONAL,
            "temporal": EdgeKind.TEMPORAL,
            "causal": EdgeKind.CAUSAL,
        }
        return mapping.get(max_key, EdgeKind.SEMANTIC)
