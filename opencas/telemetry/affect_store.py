"""Storage layer for affect telemetry and provenance mirror data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .affect_models import (
    AffectQualityEdge,
    AffectSnapshot,
    AffectTrajectory,
    QualitySignal,
    TeamHealthAlert,
)


class AffectStore:
    """JSONL-backed store for affect telemetry, quality signals, and edges.

    Mirrors the provenance registry by linking emotional state telemetry
    to build artifacts and quality outcomes.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._affect_path = self.base_path / "affect_snapshots.jsonl"
        self._trajectory_path = self.base_path / "affect_trajectories.jsonl"
        self._quality_path = self.base_path / "quality_signals.jsonl"
        self._edge_path = self.base_path / "affect_quality_edges.jsonl"
        self._alert_path = self.base_path / "team_health_alerts.jsonl"
        self._subscribers: List[Callable[[str, dict], None]] = []

    def subscribe(self, subscriber: Callable[[str, dict], None]) -> None:
        """Register a callback receiving (topic, payload) for every write."""
        self._subscribers.append(subscriber)

    def _notify(self, topic: str, payload: dict) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber(topic, payload)
            except Exception:
                pass

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _read_jsonl(self, path: Path) -> Iterator[dict]:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # -- Affect Snapshots --

    def save_snapshot(self, snapshot: AffectSnapshot) -> AffectSnapshot:
        record = snapshot.model_dump(mode="json")
        self._append_jsonl(self._affect_path, record)
        self._notify("affect_snapshot", record)
        return snapshot

    def iter_snapshots(
        self,
        session_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> Iterator[AffectSnapshot]:
        for raw in self._read_jsonl(self._affect_path):
            try:
                snap = AffectSnapshot.model_validate(raw)
                if session_id and snap.session_id != session_id:
                    continue
                if artifact_id and snap.artifact_id != artifact_id:
                    continue
                if actor and snap.actor != actor:
                    continue
                yield snap
            except Exception:
                continue

    def get_snapshots_for_session(self, session_id: str) -> List[AffectSnapshot]:
        return list(self.iter_snapshots(session_id=session_id))

    # -- Trajectories --

    def save_trajectory(self, trajectory: AffectTrajectory) -> AffectTrajectory:
        record = trajectory.model_dump(mode="json")
        self._append_jsonl(self._trajectory_path, record)
        self._notify("affect_trajectory", record)
        return trajectory

    def get_trajectory(self, trajectory_id: str) -> Optional[AffectTrajectory]:
        for raw in self._read_jsonl(self._trajectory_path):
            try:
                traj = AffectTrajectory.model_validate(raw)
                if str(traj.trajectory_id) == trajectory_id:
                    return traj
            except Exception:
                continue
        return None

    def iter_trajectories(
        self,
        session_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Iterator[AffectTrajectory]:
        for raw in self._read_jsonl(self._trajectory_path):
            try:
                traj = AffectTrajectory.model_validate(raw)
                if session_id and traj.session_id != session_id:
                    continue
                if artifact_id and traj.artifact_id != artifact_id:
                    continue
                yield traj
            except Exception:
                continue

    # -- Quality Signals --

    def save_quality_signal(self, signal: QualitySignal) -> QualitySignal:
        record = signal.model_dump(mode="json")
        self._append_jsonl(self._quality_path, record)
        self._notify("quality_signal", record)
        return signal

    def iter_quality_signals(
        self,
        artifact_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[QualitySignal]:
        for raw in self._read_jsonl(self._quality_path):
            try:
                sig = QualitySignal.model_validate(raw)
                if artifact_id and sig.artifact_id != artifact_id:
                    continue
                if session_id and sig.session_id != session_id:
                    continue
                yield sig
            except Exception:
                continue

    # -- Affect-Quality Edges --

    def save_edge(self, edge: AffectQualityEdge) -> AffectQualityEdge:
        record = edge.model_dump(mode="json")
        self._append_jsonl(self._edge_path, record)
        self._notify("affect_quality_edge", record)
        return edge

    def iter_edges(
        self,
        artifact_id: Optional[str] = None,
        trajectory_id: Optional[str] = None,
    ) -> Iterator[AffectQualityEdge]:
        for raw in self._read_jsonl(self._edge_path):
            try:
                edge = AffectQualityEdge.model_validate(raw)
                if artifact_id and edge.artifact_id != artifact_id:
                    continue
                if trajectory_id and str(edge.trajectory_id) != trajectory_id:
                    continue
                yield edge
            except Exception:
                continue

    # -- Team Health Alerts --

    def save_alert(self, alert: TeamHealthAlert) -> TeamHealthAlert:
        record = alert.model_dump(mode="json")
        self._append_jsonl(self._alert_path, record)
        self._notify("team_health_alert", record)
        return alert

    def iter_alerts(
        self,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> Iterator[TeamHealthAlert]:
        for raw in self._read_jsonl(self._alert_path):
            try:
                alert = TeamHealthAlert.model_validate(raw)
                if severity and alert.severity != severity:
                    continue
                if resolved is not None:
                    is_resolved = alert.resolved_at is not None
                    if is_resolved != resolved:
                        continue
                yield alert
            except Exception:
                continue

    def acknowledge_alert(self, alert_id: str) -> Optional[TeamHealthAlert]:
        """Acknowledge an alert by ID (requires rewriting the file)."""
        alerts: List[TeamHealthAlert] = []
        target: Optional[TeamHealthAlert] = None
        for raw in self._read_jsonl(self._alert_path):
            try:
                alert = TeamHealthAlert.model_validate(raw)
                if str(alert.alert_id) == alert_id:
                    from datetime import datetime, timezone
                    alert.acknowledged_at = datetime.now(timezone.utc)
                    target = alert
                alerts.append(alert)
            except Exception:
                continue
        if target is None:
            return None
        # Rewrite
        with open(self._alert_path, "w", encoding="utf-8") as f:
            for alert in alerts:
                f.write(json.dumps(alert.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":")) + "\n")
        return target
