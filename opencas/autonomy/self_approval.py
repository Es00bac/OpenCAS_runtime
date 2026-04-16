"""Self-approval ladder for OpenCAS autonomy.

Determines whether the agent can self-approve an action based on:
- action risk tier
- user trust level
- historical success evidence
- somatic state
- explicit user boundaries
"""

from __future__ import annotations

from typing import List, Optional

from opencas.governance import ApprovalLedger
from opencas.identity import IdentityManager
from opencas.relational import RelationalEngine
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, Tracer

from .models import ActionRequest, ActionRiskTier, ApprovalDecision, ApprovalLevel


# Base risk score per tier (0.0 = safe, 1.0 = dangerous)
_BASE_RISK: dict[ActionRiskTier, float] = {
    ActionRiskTier.READONLY: 0.05,
    ActionRiskTier.WORKSPACE_WRITE: 0.20,
    ActionRiskTier.SHELL_LOCAL: 0.40,
    ActionRiskTier.NETWORK: 0.30,
    ActionRiskTier.EXTERNAL_WRITE: 0.65,
    ActionRiskTier.DESTRUCTIVE: 0.95,
}

# Circuit breakers — prevent degraded inner state from distorting safety decisions.
# These bounds are well outside the normal operating range; they only activate when
# somatic/relational state is severely degraded.
_SOMATIC_APPROVAL_DELTA_CAP: float = 0.20   # max somatic can add to a risk score
_MUSUBI_APPROVAL_ABS_CAP: float = 0.12      # musubi modifier clamped to ±this


class SelfApprovalLadder:
    """Evaluates action requests and returns an approval decision."""

    def __init__(
        self,
        identity: IdentityManager,
        somatic: Optional[SomaticManager] = None,
        tracer: Optional[Tracer] = None,
        relational: Optional[RelationalEngine] = None,
        ledger: Optional[ApprovalLedger] = None,
    ) -> None:
        self.identity = identity
        self.somatic = somatic
        self.tracer = tracer
        self.relational = relational
        self.ledger = ledger

    def evaluate(self, request: ActionRequest) -> ApprovalDecision:
        """Compute approval level for *request*."""
        score = self._base_score(request.tier)
        reasons: List[str] = [f"base_risk={request.tier.value}"]

        # 1. Trust modulation
        trust = self.identity.user_model.trust_level
        trust_delta = (0.5 - trust) * 0.15
        score += trust_delta
        reasons.append(f"trust_mod={trust_delta:+.3f}")

        # 2. Historical evidence modulation
        history_delta = self._history_modulation(request)
        score += history_delta
        reasons.append(f"history_mod={history_delta:+.3f}")

        # 3. Somatic modulation
        somatic_delta = self._somatic_modulation()
        score += somatic_delta
        reasons.append(f"somatic_mod={somatic_delta:+.3f}")

        # 4. Musubi / relational risk appetite (circuit-breaker: clamp to ±_MUSUBI_APPROVAL_ABS_CAP)
        musubi_delta = 0.0
        if self.relational:
            raw_musubi = self.relational.to_approval_risk_modifier()
            musubi_delta = max(-_MUSUBI_APPROVAL_ABS_CAP, min(_MUSUBI_APPROVAL_ABS_CAP, raw_musubi))
            score += musubi_delta
            reasons.append(f"musubi_mod={musubi_delta:+.3f}")

        # 5. Structured payload modulation
        payload_delta = self._payload_modulation(request)
        score += payload_delta
        reasons.append(f"payload_mod={payload_delta:+.3f}")

        # 6. Explicit boundary check
        boundary_hits = self._matching_boundaries(request)
        if boundary_hits:
            score = 1.0
            reasons.append("explicit_boundary_hit=" + ",".join(boundary_hits))

        score = max(0.0, min(1.0, score))
        
        threshold_adjustment = 0.0
        if self.relational:
            raw_musubi = self.relational.state.musubi
            if raw_musubi > 0.5:
                threshold_adjustment = -0.05
                reasons.append("relational_autonomy: high musubi")
            elif raw_musubi < -0.3:
                threshold_adjustment = 0.05
                reasons.append("relational_caution: low musubi")
        
        level, confidence, reasoning = self._score_to_level(score, reasons, threshold_adjustment)

        decision = ApprovalDecision(
            level=level,
            action_id=request.action_id,
            confidence=confidence,
            reasoning=reasoning,
            score=score,
        )

        if self.tracer:
            somatic_state = None
            if self.somatic:
                s = self.somatic.state
                somatic_state = {
                    "arousal": round(s.arousal, 3),
                    "fatigue": round(s.fatigue, 3),
                    "tension": round(s.tension, 3),
                    "valence": round(s.valence, 3),
                }
            self.tracer.log(
                EventKind.SELF_APPROVAL,
                f"Self-approval: {level.value}",
                {
                    "action_id": str(request.action_id),
                    "tier": request.tier.value,
                    "tool_name": request.tool_name,
                    "score": round(score, 3),
                    "level": level.value,
                    "confidence": round(decision.confidence, 3),
                    "trust_level": round(trust, 3),
                    "somatic_state": somatic_state,
                    "boundary_hit": bool(boundary_hits),
                    "boundary_matches": boundary_hits,
                    "history_mod": round(history_delta, 3),
                    "somatic_mod": round(somatic_delta, 3),
                    "payload_mod": round(payload_delta, 3),
                    "reasoning": reasoning,
                },
            )

        return decision

    def evaluate_conversational(self, text: str) -> ApprovalDecision:
        """Evaluate a user conversational input as a synthetic READONLY action."""
        request = ActionRequest(
            tier=ActionRiskTier.READONLY,
            description=text,
            tool_name="conversation",
        )
        return self.evaluate(request)

    async def maybe_record(
        self,
        decision: ApprovalDecision,
        request: ActionRequest,
        score: float,
    ) -> None:
        """Record the decision to the ledger if one is configured."""
        if self.ledger is None:
            return
        somatic_state = None
        if self.somatic:
            s = self.somatic.state
            somatic_state = (
                f"arousal={round(s.arousal, 3)},"
                f"fatigue={round(s.fatigue, 3)},"
                f"tension={round(s.tension, 3)},"
                f"valence={round(s.valence, 3)}"
            )
        await self.ledger.record(decision, request, score, somatic_state)

    @staticmethod
    def _base_score(tier: ActionRiskTier) -> float:
        return _BASE_RISK.get(tier, 0.5)

    def _history_modulation(self, request: ActionRequest) -> float:
        """Adjust score based on prior success rates stored in self-beliefs."""
        delta = 0.0
        keys_checked = 0

        # Check tier-level history
        tier_key = f"success_rate_tier_{request.tier.value}"
        tier_rate = self.identity.self_model.self_beliefs.get(tier_key)
        if isinstance(tier_rate, (int, float)):
            delta += (tier_rate - 0.5) * -0.20
            keys_checked += 1

        # Check tool-level history
        if request.tool_name:
            tool_key = f"success_rate_tool_{request.tool_name}"
            tool_rate = self.identity.self_model.self_beliefs.get(tool_key)
            if isinstance(tool_rate, (int, float)):
                delta += (tool_rate - 0.5) * -0.15
                keys_checked += 1

        # If no history, slight uncertainty penalty for risky tiers
        if keys_checked == 0 and request.tier in (
            ActionRiskTier.SHELL_LOCAL,
            ActionRiskTier.EXTERNAL_WRITE,
            ActionRiskTier.DESTRUCTIVE,
        ):
            delta += 0.05

        return delta

    def _somatic_modulation(self) -> float:
        """Increase caution when the body is in a stressed or exhausted state.

        Circuit-breaker: total somatic contribution is capped at
        _SOMATIC_APPROVAL_DELTA_CAP so a maximally stressed state cannot
        override tier-based safety decisions (e.g. DESTRUCTIVE still escalates).
        """
        if self.somatic is None:
            return 0.0
        delta = 0.0
        state = self.somatic.state
        if state.tension > 0.7:
            delta += 0.08
        if state.fatigue > 0.8:
            delta += 0.08
        if state.arousal > 0.9:
            delta += 0.05
        return min(delta, _SOMATIC_APPROVAL_DELTA_CAP)

    def _payload_modulation(self, request: ActionRequest) -> float:
        """Adjust risk using structured command/tool payload details."""
        payload = request.payload or {}
        permission_class = str(payload.get("command_permission_class", "")).lower()
        command_family = str(payload.get("command_family", "")).lower()
        command_scope = str(payload.get("command_scope", "")).lower()
        effective_permission_class = str(
            payload.get("command_effective_permission_class", "")
        ).lower()
        effective_family = str(payload.get("command_effective_family", "")).lower()
        write_scope = str(payload.get("write_scope", "")).lower()

        if (
            request.tier == ActionRiskTier.SHELL_LOCAL
            and command_scope == "managed_workspace"
            and effective_family == "safe"
            and effective_permission_class in {"read_only", "bounded_write"}
        ):
            return -0.30
        if permission_class == "read_only" and request.tier == ActionRiskTier.SHELL_LOCAL:
            return -0.18
        if permission_class == "bounded_write" and request.tier == ActionRiskTier.SHELL_LOCAL:
            return -0.12
        if (
            request.tier == ActionRiskTier.WORKSPACE_WRITE
            and write_scope in {"managed_workspace", "plans"}
        ):
            return -0.12
        if permission_class == "network":
            return 0.08
        if permission_class == "dangerous":
            return 0.40
        if command_family == "unsafe_shell":
            return 0.25
        if command_family in {"filesystem_destructive", "privilege_escalation"}:
            return 0.45
        return 0.0

    def _matching_boundaries(self, request: ActionRequest) -> List[str]:
        """Return boundary strings that semantically match the request."""
        boundaries = self.identity.user_model.known_boundaries
        if not boundaries:
            return []

        tier_str = request.tier.value
        tool_str = (request.tool_name or "").lower()
        payload = request.payload or {}
        command_family = str(payload.get("command_family", "")).lower()
        permission_class = str(payload.get("command_permission_class", "")).lower()
        description = request.description.lower()
        matches: List[str] = []

        for boundary in boundaries:
            normalized = boundary.strip().lower()
            if not normalized:
                continue
            if normalized == tier_str or normalized == tool_str:
                matches.append(boundary)
                continue
            if normalized == command_family or normalized == permission_class:
                matches.append(boundary)
                continue
            if self._boundary_matches_semantics(
                normalized,
                tier_str=tier_str,
                tool_str=tool_str,
                description=description,
                command_family=command_family,
                permission_class=permission_class,
            ):
                matches.append(boundary)
        return matches

    @staticmethod
    def _boundary_matches_semantics(
        boundary: str,
        *,
        tier_str: str,
        tool_str: str,
        description: str,
        command_family: str,
        permission_class: str,
    ) -> bool:
        if "destructive" in boundary and (
            tier_str == ActionRiskTier.DESTRUCTIVE.value
            or command_family == "filesystem_destructive"
            or permission_class == "dangerous"
        ):
            return True
        if "external write" in boundary and tier_str == ActionRiskTier.EXTERNAL_WRITE.value:
            return True
        if "network" in boundary and (
            tier_str == ActionRiskTier.NETWORK.value or permission_class == "network"
        ):
            return True
        if "browser" in boundary and (
            "browser" in tool_str
            or "browser" in description
            or tier_str == ActionRiskTier.NETWORK.value
        ):
            return True
        if ("shell" in boundary or "command" in boundary) and tier_str == ActionRiskTier.SHELL_LOCAL.value:
            return True
        if "workspace write" in boundary and tier_str == ActionRiskTier.WORKSPACE_WRITE.value:
            return True
        return False

    def _score_to_level(self, score: float, reasons: List[str], threshold_adjustment: float = 0.0) -> tuple[ApprovalLevel, float, str]:
        t1 = 0.20 + threshold_adjustment
        t2 = 0.45 + threshold_adjustment
        t3 = 0.70 + threshold_adjustment
        if score < t1:
            return ApprovalLevel.CAN_DO_NOW, 1.0 - score, "; ".join(reasons)
        if score < t2:
            return ApprovalLevel.CAN_DO_WITH_CAUTION, 1.0 - score, "; ".join(reasons)
        if score < t3:
            evidence = ["provide more context", "confirm target scope"]
            if threshold_adjustment > 0:
                evidence.extend(["verify relational trust", "check boundary alignment"])
            return (
                ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE,
                1.0 - score,
                "; ".join(reasons) + " | suggested_evidence: " + ", ".join(evidence),
            )
        return ApprovalLevel.MUST_ESCALATE, score, "; ".join(reasons)
