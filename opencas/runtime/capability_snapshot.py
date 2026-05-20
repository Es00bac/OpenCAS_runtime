"""Detect and surface tool-capability drift across runtime restarts.

Past episodes and self-authored capability docs (e.g. ``capability_inventory_audit.md``)
record successful uses of tools that may no longer be registered — for example, the
``google_workspace_*`` family disappears whenever the runtime boots without
``gws`` on PATH. Without an explicit drift signal, retrieval surfaces those past
successes, the model concludes "I have this tool", and emits text claiming a call
it cannot actually make. The user sees apparent gaslighting; the underlying issue
is registry/memory mismatch.

This helper:
  1. Loads the previous boot's tool snapshot.
  2. Diffs it against the currently registered tool set.
  3. Persists the new snapshot.
  4. Logs a telemetry event when drift is non-empty.
  5. Records a high-salience observation episode listing the lost tools so future
     retrieval is annotated with the correction, not just the stale claim.
  6. Exposes a ``CapabilityDriftReport`` so the prompt-time capability context can
     render a "tools previously available but unavailable now" warning, which is
     what actually changes the model's tool routing on the next turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

_LOG = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "capability_snapshot.json"


@dataclass
class CapabilityDriftReport:
    """Per-boot tool-registry diff against the previous boot's snapshot."""

    lost: tuple[str, ...] = ()
    gained: tuple[str, ...] = ()
    previous_recorded_at: Optional[str] = None
    current_recorded_at: str = ""
    snapshot_path: Optional[str] = None

    @property
    def has_drift(self) -> bool:
        return bool(self.lost) or bool(self.gained)


def _snapshot_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / SNAPSHOT_FILENAME


def _load_previous(snapshot_path: Path) -> tuple[set[str], Optional[str]]:
    if not snapshot_path.exists():
        return set(), None
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("capability snapshot unreadable at %s: %s", snapshot_path, exc)
        return set(), None
    tools = data.get("tools") or []
    return {str(name) for name in tools}, data.get("recorded_at")


def _write_current(snapshot_path: Path, tools: Iterable[str], recorded_at: str) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": recorded_at,
        "tools": sorted({str(name) for name in tools}),
    }
    snapshot_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def detect_capability_drift(
    state_dir: Path | str,
    current_tools: Iterable[str],
    *,
    tracer: Optional[Any] = None,
) -> CapabilityDriftReport:
    """Compare current tool set against the previous boot's snapshot.

    Always persists the new snapshot. Emits a telemetry event when drift is
    detected. Returns the report so callers can surface drift in the prompt
    and (asynchronously) record an observation episode.
    """
    current = {str(name) for name in current_tools}
    snapshot_path = _snapshot_path(state_dir)
    previous, previous_at = _load_previous(snapshot_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    # First boot has no comparison baseline — skip diff so we don't surface every
    # registered tool as "gained" or fire spurious warnings during initialization.
    if previous_at is None:
        lost: tuple[str, ...] = ()
        gained: tuple[str, ...] = ()
    else:
        lost = tuple(sorted(previous - current))
        gained = tuple(sorted(current - previous))

    try:
        _write_current(snapshot_path, current, now_iso)
    except OSError as exc:
        _LOG.warning("capability snapshot write failed: %s", exc)

    report = CapabilityDriftReport(
        lost=lost,
        gained=gained,
        previous_recorded_at=previous_at,
        current_recorded_at=now_iso,
        snapshot_path=str(snapshot_path),
    )

    if tracer is not None and report.has_drift:
        try:
            from opencas.telemetry import EventKind

            event_kind = EventKind.WARNING if lost else EventKind.BOOTSTRAP_STAGE
            tracer.log(
                event_kind,
                "capability_drift_detected",
                {
                    "lost": list(lost),
                    "gained": list(gained),
                    "previous_recorded_at": previous_at,
                    "current_recorded_at": now_iso,
                },
            )
        except Exception as exc:  # telemetry must never block boot
            _LOG.warning("capability drift telemetry failed: %s", exc)

    return report


def render_capability_drift_warning(report: Optional[CapabilityDriftReport]) -> Optional[str]:
    """Return a single-line prompt warning for the LLM, or None when no drift.

    Surfaces only the ``lost`` set — newly gained tools don't confuse routing.
    The directive at the end is what actually breaks the gaslighting loop:
    it tells the model what to do *instead* of attempting a missing tool.
    """
    if report is None or not report.lost:
        return None
    listed = ", ".join(report.lost)
    return (
        "- WARNING capability_drift: tools registered in the previous boot are "
        f"NOT registered now: {listed}. Memory or self-authored capability docs "
        "may still reference these. Do not claim to call them in this session. "
        "If a turn requires one, tell the operator the capability is missing and "
        "propose how to restore it (e.g., reinstall the underlying CLI or fix the "
        "runtime PATH/credentials), instead of falling back to bash, web_fetch, "
        "or browser tooling."
    )


async def record_drift_episode(
    memory: Any,
    embeddings: Any,
    report: CapabilityDriftReport,
) -> None:
    """Persist a high-salience observation episode listing the lost tools.

    Without this, future retrieval surfaces the stale "I have tool X" claims
    from older episodes with no opposing evidence. With it, semantic recall on
    "do I have google_workspace_gmail_headlines?" returns the correction too.
    """
    if not report.lost or memory is None:
        return

    from opencas.memory.models import Episode, EpisodeKind

    lost_str = ", ".join(report.lost)
    content = (
        f"Capability drift observed at runtime boot ({report.current_recorded_at}). "
        f"Tools registered in the previous boot ({report.previous_recorded_at or 'unknown'}) "
        f"are not registered in this runtime: {lost_str}. "
        "Past memory of using these tools is no longer current. Do not claim to "
        "call them. If a turn requires one, tell the operator the capability is "
        "missing and propose restoration instead of attempting bash, web_fetch, "
        "or browser fallbacks."
    )

    embedding_id = None
    if embeddings is not None:
        try:
            record = await embeddings.embed(content, task_type="capability_drift")
            embedding_id = record.source_hash
        except Exception as exc:
            _LOG.warning("capability drift embedding failed: %s", exc)

    episode = Episode(
        episode_id=str(uuid4()),
        kind=EpisodeKind.OBSERVATION,
        session_id=None,
        content=content,
        embedding_id=embedding_id,
        salience=2.0,
        payload={
            "capability_drift": {
                "lost": list(report.lost),
                "gained": list(report.gained),
                "previous_recorded_at": report.previous_recorded_at,
                "current_recorded_at": report.current_recorded_at,
            },
            "source_lane": "system_observation",
            "context_authority": "ground_truth",
        },
    )
    try:
        await memory.save_episodes_batch([episode])
    except Exception as exc:
        _LOG.warning("capability drift episode save failed: %s", exc)
