"""Anomaly detection and intervention surfacing for the provenance mirror.

Deploys anomaly detection on provenance mirrors to flag when emotional state
dependencies predict quality degradation, triggering team health alerts before
code failures propagate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .affect_models import (
    AffectDimension,
    AffectSnapshot,
    AffectTrajectory,
    QualitySignal,
    TeamHealthAlert,
)
from .affect_store import AffectStore


class AnomalyDetector:
    """Detects anomalies in affect trajectories that predict quality degradation."""

    def __init__(self, store: AffectStore) -> None:
        self.store = store

    def scan_session(self, session_id: str) -> List[TeamHealthAlert]:
        """Scan a session for affect-quality anomalies."""
        alerts: List[TeamHealthAlert] = []
        trajectories = list(self.store.iter_trajectories(session_id=session_id))

        for traj in trajectories:
            alert = self._check_trajectory(traj)
            if alert:
                self.store.save_alert(alert)
                alerts.append(alert)

        return alerts

    def scan_all_sessions(self) -> List[TeamHealthAlert]:
        """Scan all sessions for anomalies."""
        alerts: List[TeamHealthAlert] = []
        seen_sessions: set = set()
        for traj in self.store.iter_trajectories():
            if traj.session_id not in seen_sessions:
                seen_sessions.add(traj.session_id)
                alerts.extend(self.scan_session(traj.session_id))
        return alerts

    def _check_trajectory(self, traj: AffectTrajectory) -> Optional[TeamHealthAlert]:
        """Check a single trajectory for anomaly patterns."""
        if len(traj.snapshots) < 2:
            return None

        # Pattern 1: Stress spike
        latest_stress = traj.snapshots[-1].composite_stress
        if latest_stress > 0.7:
            return self._create_alert(
                traj,
                "stress_spike",
                "critical" if latest_stress > 0.85 else "warning",
                f"Stress level spiked to {latest_stress:.2f} for artifact {traj.artifact_id}",
                {
                    "latest_stress": latest_stress,
                    "stress_trend": traj.stress_trend,
                    "snapshot_count": len(traj.snapshots),
                },
            )

        # Pattern 2: Flow depletion
        latest_flow = traj.snapshots[-1].composite_flow
        if latest_flow < -0.5:
            return self._create_alert(
                traj,
                "flow_depletion",
                "warning",
                f"Flow state depleted to {latest_flow:.2f} for artifact {traj.artifact_id}",
                {
                    "latest_flow": latest_flow,
                    "valence_trend": traj.valence_trend,
                },
            )

        # Pattern 3: Volatility cascade
        if traj.volatility > 0.4:
            return self._create_alert(
                traj,
                "volatility_cascade",
                "warning",
                f"High emotional volatility ({traj.volatility:.2f}) detected for artifact {traj.artifact_id}",
                {
                    "volatility": traj.volatility,
                    "snapshot_count": len(traj.snapshots),
                },
            )

        # Pattern 4: Negative valence cliff
        latest_valence = traj.snapshots[-1].dimensions.get(AffectDimension.VALENCE.value, 0.0)
        if latest_valence < -0.6 and traj.valence_trend < -0.1:
            return self._create_alert(
                traj,
                "valence_cliff",
                "critical" if latest_valence < -0.8 else "warning",
                f"Valence dropped sharply to {latest_valence:.2f} for artifact {traj.artifact_id}",
                {
                    "latest_valence": latest_valence,
                    "valence_trend": traj.valence_trend,
                },
            )

        # Pattern 5: Urgency + low certainty (dangerous combination)
        latest = traj.snapshots[-1]
        urgency = latest.dimensions.get(AffectDimension.URGENCY.value, 0.0)
        certainty = latest.dimensions.get(AffectDimension.CERTAINTY.value, 0.0)
        if urgency > 0.7 and certainty < -0.3:
            return self._create_alert(
                traj,
                "uncertain_urgency",
                "critical",
                f"High urgency ({urgency:.2f}) combined with low certainty ({certainty:.2f})",
                {
                    "urgency": urgency,
                    "certainty": certainty,
                },
            )

        return None

    def _create_alert(
        self,
        traj: AffectTrajectory,
        alert_type: str,
        severity: str,
        message: str,
        observed_affect: Dict[str, Any],
    ) -> TeamHealthAlert:
        """Create a team health alert from trajectory anomaly."""
        # Predict quality impact
        predicted_impact = self._predict_quality_impact(traj)

        # Determine recommended action
        recommended = self._recommend_intervention(alert_type, traj)

        return TeamHealthAlert(
            severity=severity,
            alert_type=alert_type,
            session_id=traj.session_id,
            artifact_id=traj.artifact_id,
            actor=traj.actor,
            observed_affect=observed_affect,
            predicted_quality_impact=predicted_impact,
            recommended_action=recommended,
        )

    def _predict_quality_impact(self, traj: AffectTrajectory) -> Dict[str, Any]:
        """Predict likely quality degradation based on affect pattern."""
        if not traj.snapshots:
            return {}

        latest = traj.snapshots[-1]
        stress = latest.composite_stress
        flow = latest.composite_flow
        valence = latest.dimensions.get(AffectDimension.VALENCE.value, 0.0)
        urgency = latest.dimensions.get(AffectDimension.URGENCY.value, 0.0)
        certainty = latest.dimensions.get(AffectDimension.CERTAINTY.value, 0.0)

        # Heuristic predictions
        bug_risk = min(1.0, stress * 0.6 + (1.0 - certainty) * 0.3 + urgency * 0.1)
        rollback_risk = min(1.0, stress * 0.5 + urgency * 0.4 + (1.0 - flow) * 0.1)
        review_friction = min(1.0, (1.0 - coherence) * 0.5 + stress * 0.3) if (coherence := latest.dimensions.get(AffectDimension.COHERENCE.value, 0.0)) else 0.0

        return {
            "predicted_bug_density_increase": round(bug_risk * 0.05, 4),  # bugs per LOC
            "predicted_rollback_probability": round(rollback_risk, 4),
            "predicted_review_rounds": round(2 + review_friction * 3),
            "confidence": round(0.5 + stress * 0.3, 4),
        }

    def _recommend_intervention(self, alert_type: str, traj: AffectTrajectory) -> str:
        """Generate a recommended intervention based on alert type."""
        recommendations = {
            "stress_spike": (
                "Consider pausing the session. Suggest a break, pair programming, "
                "or re-scoping the task. High stress correlates with defect introduction."
            ),
            "flow_depletion": (
                "Flow state is depleted. Recommend switching to a lower-cognitive-load task, "
                "or scheduling the complex work for when the developer is more energized."
            ),
            "volatility_cascade": (
                "Emotional volatility detected. Recommend stabilizing the context: "
                "reduce interruptions, clarify requirements, or bring in a second reviewer."
            ),
            "valence_cliff": (
                "Significant negative emotional shift. Check for blockers, unclear requirements, "
                "or interpersonal friction. Consider team check-in."
            ),
            "uncertain_urgency": (
                "CRITICAL: High urgency with low confidence is the highest-risk combination. "
                "Mandate peer review, add extra tests, and consider delaying deployment."
            ),
        }
        return recommendations.get(
            alert_type,
            "Review the affect trajectory and consider team health intervention.",
        )


class InterventionEngine:
    """Engine for triggering and managing interventions."""

    def __init__(self, store: AffectStore, detector: AnomalyDetector) -> None:
        self.store = store
        self.detector = detector

    def run_health_check(self, session_id: Optional[str] = None) -> List[TeamHealthAlert]:
        """Run a full health check and return any alerts."""
        if session_id:
            alerts = self.detector.scan_session(session_id)
        else:
            alerts = self.detector.scan_all_sessions()

        # Auto-trigger interventions for critical alerts
        for alert in alerts:
            if alert.severity == "critical":
                alert.auto_intervention_triggered = True
                self._execute_intervention(alert)

        return alerts

    def _execute_intervention(self, alert: TeamHealthAlert) -> None:
        """Execute an automatic intervention for critical alerts."""
        # In a real system, this would:
        # - Post to Slack/Teams
        # - Create a Jira ticket
        # - Trigger a CI hold
        # - Notify team lead
        # For now, we just mark it
        alert.resolution_notes = (
            f"Auto-intervention triggered at {datetime.now(timezone.utc).isoformat()}. "
            f"Recommended: {alert.recommended_action}"
        )
        self.store.save_alert(alert)

    def get_team_health_dashboard(self) -> Dict[str, Any]:
        """Generate a team health dashboard summary."""
        all_alerts = list(self.store.iter_alerts())
        unresolved = list(self.store.iter_alerts(resolved=False))
        critical = [a for a in unresolved if a.severity == "critical"]
        warnings = [a for a in unresolved if a.severity == "warning"]

        # Aggregate by actor
        actor_stress: Dict[str, List[float]] = {}
        for traj in self.store.iter_trajectories():
            if traj.actor:
                if traj.actor not in actor_stress:
                    actor_stress[traj.actor] = []
                for snap in traj.snapshots:
                    actor_stress[traj.actor].append(snap.composite_stress)

        actor_health = {}
        for actor, stresses in actor_stress.items():
            avg_stress = sum(stresses) / len(stresses) if stresses else 0.0
            actor_health[actor] = {
                "avg_stress": round(avg_stress, 4),
                "alert_count": sum(1 for a in unresolved if a.actor == actor),
                "status": "at_risk" if avg_stress > 0.6 else "healthy" if avg_stress < 0.3 else "watch",
            }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_alerts": len(all_alerts),
            "unresolved_alerts": len(unresolved),
            "critical_alerts": len(critical),
            "warning_alerts": len(warnings),
            "actor_health": actor_health,
            "top_risk_factors": self._aggregate_risk_factors(unresolved),
        }

    def _aggregate_risk_factors(self, alerts: List[TeamHealthAlert]) -> List[Dict[str, Any]]:
        """Aggregate the most common risk factors across alerts."""
        from collections import Counter
        types = Counter(a.alert_type for a in alerts)
        total = len(alerts) if alerts else 1
        return [
            {"type": t, "count": c, "percentage": round(c / total * 100, 2)}
            for t, c in types.most_common(5)
        ]
