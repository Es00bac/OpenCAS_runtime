"""Operational trust engine for approval and tool-outcome feedback."""

from __future__ import annotations

from typing import Any, Optional

from opencas.autonomy.models import ActionRiskTier


class TrustEngine:
    """Compute trust-based approval thresholds from operational history."""

    def __init__(
        self,
        identity: Any,
        base_trust: float = 0.5,
        success_boost: float = 0.03,
        failure_penalty: float = 0.08,
        max_trust: float = 0.98,
        min_trust: float = 0.05,
    ) -> None:
        self.identity = identity
        self.base_trust = base_trust
        self.success_boost = success_boost
        self.failure_penalty = failure_penalty
        self.max_trust = max_trust
        self.min_trust = min_trust

    def current_trust(self) -> float:
        raw = getattr(self.identity.user_model, "trust_level", self.base_trust)
        return max(self.min_trust, min(self.max_trust, float(raw)))

    async def record_success(
        self,
        *,
        tool_name: Optional[str] = None,
        tier: Optional[ActionRiskTier] = None,
    ) -> float:
        trust = min(self.max_trust, self.current_trust() + self.success_boost)
        self._record_tool_success(tool_name, tier)
        await self._update_identity_trust(trust)
        return trust

    async def record_failure(
        self,
        *,
        tool_name: Optional[str] = None,
        tier: Optional[ActionRiskTier] = None,
        is_recoverable: bool = True,
    ) -> float:
        penalty = self.failure_penalty * (1.0 if is_recoverable else 1.5)
        trust = max(self.min_trust, self.current_trust() - penalty)
        self._record_tool_failure(tool_name, tier)
        await self._update_identity_trust(trust)
        return trust

    def threshold_for(self, tier: ActionRiskTier) -> float:
        """Return the score threshold under which a tier can proceed."""
        trust = self.current_trust()
        base = {
            ActionRiskTier.READONLY: 0.50,
            ActionRiskTier.WORKSPACE_WRITE: 0.55,
            ActionRiskTier.SHELL_LOCAL: 0.65,
            ActionRiskTier.NETWORK: 0.60,
            ActionRiskTier.EXTERNAL_WRITE: 0.80,
            ActionRiskTier.DESTRUCTIVE: 0.99,
        }.get(tier, 0.70)
        adjustment = (trust - 0.5) * 0.30
        return max(0.30, min(0.99, base + adjustment))

    async def _update_identity_trust(self, trust: float) -> None:
        self.identity.user_model.trust_level = round(trust, 4)
        save = getattr(self.identity, "save", None)
        if not callable(save):
            return
        result = save()
        if hasattr(result, "__await__"):
            await result

    def _record_tool_success(
        self,
        tool_name: Optional[str],
        tier: Optional[ActionRiskTier],
    ) -> None:
        beliefs = self.identity.self_model.self_beliefs
        if tier is not None:
            key = f"success_rate_tier_{tier.value}"
            beliefs[key] = round(float(beliefs.get(key, 0.5)) * 0.9 + 0.1, 4)
        if tool_name:
            key = f"success_rate_tool_{tool_name}"
            beliefs[key] = round(float(beliefs.get(key, 0.5)) * 0.9 + 0.1, 4)

    def _record_tool_failure(
        self,
        tool_name: Optional[str],
        tier: Optional[ActionRiskTier],
    ) -> None:
        beliefs = self.identity.self_model.self_beliefs
        if tier is not None:
            key = f"success_rate_tier_{tier.value}"
            beliefs[key] = round(float(beliefs.get(key, 0.5)) * 0.9, 4)
        if tool_name:
            key = f"success_rate_tool_{tool_name}"
            beliefs[key] = round(float(beliefs.get(key, 0.5)) * 0.9, 4)
