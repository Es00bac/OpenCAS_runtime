"""Dependency graph engine linking affect trajectories to quality signals.

Constructs and analyzes graphs that surface hidden dependencies between
emotional states and code quality outcomes through the provenance chain.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .affect_models import (
    AffectDimension,
    AffectQualityEdge,
    AffectSnapshot,
    AffectTrajectory,
    QualitySignal,
    TeamHealthAlert,
)
from .affect_store import AffectStore


class AffectQualityGraph:
    """Graph connecting affect trajectories to quality outcomes.

    Nodes: AffectTrajectories and QualitySignals
    Edges: Temporal correlations and predictive links
    """

    def __init__(self, store: AffectStore) -> None:
        self.store = store
        self._trajectory_cache: Dict[str, AffectTrajectory] = {}
        self._quality_cache: Dict[str, QualitySignal] = {}
        self._edge_cache: Dict[str, AffectQualityEdge] = {}

    def refresh(self) -> None:
        """Reload all data from store into memory."""
        self._trajectory_cache = {
            str(t.trajectory_id): t
            for t in self.store.iter_trajectories()
        }
        self._quality_cache = {
            str(q.signal_id): q
            for q in self.store.iter_quality_signals()
        }
        self._edge_cache = {
            str(e.edge_id): e
            for e in self.store.iter_edges()
        }

    def build_edges_for_artifact(self, artifact_id: str) -> List[AffectQualityEdge]:
        """Build correlation edges between affect and quality for an artifact."""
        trajectories = list(self.store.iter_trajectories(artifact_id=artifact_id))
        quality_signals = list(self.store.iter_quality_signals(artifact_id=artifact_id))

        edges: List[AffectQualityEdge] = []
        for traj in trajectories:
            for qsig in quality_signals:
                if not traj.snapshots:
                    continue
                # Affect must precede quality signal
                last_snapshot = traj.snapshots[-1]
                if last_snapshot.timestamp >= qsig.timestamp:
                    continue

                lead_time = (qsig.timestamp - last_snapshot.timestamp).total_seconds()

                # Compute simple correlation between stress trend and quality
                corr = self._compute_correlation(traj, qsig)

                edge = AffectQualityEdge(
                    trajectory_id=traj.trajectory_id,
                    quality_signal_id=qsig.signal_id,
                    artifact_id=artifact_id,
                    session_id=traj.session_id,
                    affect_lead_time_seconds=lead_time,
                    correlation_coefficient=corr,
                    is_predictive=abs(corr) > 0.5,
                )
                self.store.save_edge(edge)
                edges.append(edge)

        return edges

    def _compute_correlation(self, traj: AffectTrajectory, qsig: QualitySignal) -> float:
        """Compute a heuristic correlation between affect trajectory and quality."""
        if not traj.snapshots or qsig.composite_quality is None:
            return 0.0

        # Get average stress and flow across trajectory
        avg_stress = sum(s.composite_stress for s in traj.snapshots) / len(traj.snapshots)
        avg_flow = sum(s.composite_flow for s in traj.snapshots) / len(traj.snapshots)
        valence_trend = traj.valence_trend

        # Heuristic: high stress + negative valence trend -> lower quality
        # High flow -> higher quality
        predicted_quality = (
            0.5
            - (avg_stress * 0.4)          # stress hurts quality
            + (avg_flow * 0.3)            # flow helps quality
            + (valence_trend * 0.2)       # improving mood helps
        )
        predicted_quality = max(0.0, min(1.0, predicted_quality))

        # Correlation is inverse of difference between predicted and actual
        diff = abs(predicted_quality - qsig.composite_quality)
        corr = 1.0 - diff
        return round(corr, 4)

    def find_predictive_patterns(self, min_correlation: float = 0.5) -> List[Dict[str, Any]]:
        """Find trajectories that strongly predict quality outcomes."""
        patterns: List[Dict[str, Any]] = []
        for edge in self.store.iter_edges():
            if edge.correlation_coefficient is None:
                continue
            if abs(edge.correlation_coefficient) < min_correlation:
                continue

            traj = self.store.get_trajectory(str(edge.trajectory_id))
            if traj is None:
                continue

            patterns.append({
                "trajectory_id": str(edge.trajectory_id),
                "artifact_id": edge.artifact_id,
                "session_id": edge.session_id,
                "correlation": edge.correlation_coefficient,
                "lead_time_seconds": edge.affect_lead_time_seconds,
                "snapshot_count": len(traj.snapshots),
                "avg_stress": (
                    sum(s.composite_stress for s in traj.snapshots) / len(traj.snapshots)
                    if traj.snapshots else 0.0
                ),
                "valence_trend": traj.valence_trend,
                "stress_trend": traj.stress_trend,
                "volatility": traj.volatility,
            })

        # Sort by correlation strength
        patterns.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        return patterns

    def get_artifact_risk_profile(self, artifact_id: str) -> Dict[str, Any]:
        """Get a comprehensive risk profile for an artifact."""
        trajectories = list(self.store.iter_trajectories(artifact_id=artifact_id))
        quality_signals = list(self.store.iter_quality_signals(artifact_id=artifact_id))
        edges = list(self.store.iter_edges(artifact_id=artifact_id))

        if not trajectories:
            return {"artifact_id": artifact_id, "risk_level": "unknown", "reason": "no_affect_data"}

        all_stress = []
        all_flow = []
        all_valence = []
        for traj in trajectories:
            for snap in traj.snapshots:
                all_stress.append(snap.composite_stress)
                all_flow.append(snap.composite_flow)
                all_valence.append(snap.dimensions.get(AffectDimension.VALENCE.value, 0.0))

        avg_stress = sum(all_stress) / len(all_stress) if all_stress else 0.0
        avg_flow = sum(all_flow) / len(all_flow) if all_flow else 0.0
        avg_valence = sum(all_valence) / len(all_valence) if all_valence else 0.0

        # Risk assessment
        risk_factors = []
        if avg_stress > 0.6:
            risk_factors.append("high_stress")
        if avg_flow < -0.3:
            risk_factors.append("flow_depletion")
        if avg_valence < -0.4:
            risk_factors.append("negative_valence")

        latest_quality = quality_signals[-1] if quality_signals else None
        quality_risk = False
        if latest_quality and latest_quality.composite_quality is not None:
            if latest_quality.composite_quality < 0.4:
                quality_risk = True
                risk_factors.append("low_quality")

        predictive_edges = [e for e in edges if e.is_predictive]

        risk_level = "low"
        if len(risk_factors) >= 3:
            risk_level = "critical"
        elif len(risk_factors) >= 2:
            risk_level = "high"
        elif len(risk_factors) >= 1:
            risk_level = "medium"

        return {
            "artifact_id": artifact_id,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "avg_stress": round(avg_stress, 4),
            "avg_flow": round(avg_flow, 4),
            "avg_valence": round(avg_valence, 4),
            "trajectory_count": len(trajectories),
            "snapshot_count": sum(len(t.snapshots) for t in trajectories),
            "quality_signal_count": len(quality_signals),
            "predictive_edge_count": len(predictive_edges),
            "latest_quality": latest_quality.composite_quality if latest_quality else None,
        }

    def get_session_dependency_graph(self, session_id: str) -> Dict[str, Any]:
        """Build a dependency graph for all artifacts in a session."""
        trajectories = list(self.store.iter_trajectories(session_id=session_id))
        quality_signals = list(self.store.iter_quality_signals(session_id=session_id))
        edges = [
            e for e in self.store.iter_edges()
            if e.session_id == session_id
        ]

        artifacts = set()
        for t in trajectories:
            if t.artifact_id:
                artifacts.add(t.artifact_id)
        for q in quality_signals:
            artifacts.add(q.artifact_id)

        nodes = []
        for artifact_id in artifacts:
            profile = self.get_artifact_risk_profile(artifact_id)
            nodes.append({
                "artifact_id": artifact_id,
                "risk_level": profile["risk_level"],
                "risk_factors": profile["risk_factors"],
                "avg_stress": profile["avg_stress"],
            })

        edge_data = []
        for e in edges:
            edge_data.append({
                "from_trajectory": str(e.trajectory_id),
                "to_quality_signal": str(e.quality_signal_id),
                "artifact_id": e.artifact_id,
                "correlation": e.correlation_coefficient,
                "is_predictive": e.is_predictive,
            })

        return {
            "session_id": session_id,
            "nodes": nodes,
            "edges": edge_data,
            "artifact_count": len(artifacts),
            "trajectory_count": len(trajectories),
            "quality_signal_count": len(quality_signals),
        }
