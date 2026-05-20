from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from opencas.daydream.signal_store import DaydreamSignalStore
from opencas.daydream.signals import PossibilitySignal, PossibilitySignalRoute
from opencas.maintenance.daydream_signal_status import (
    repair_legacy_daydream_signal_statuses,
)


async def _seed_signal(
    db_path: Path,
    *,
    route: PossibilitySignalRoute,
    status: str,
    reason: str,
) -> str:
    store = await DaydreamSignalStore(db_path).connect()
    signal = PossibilitySignal(
        source_reflection_id=f"reflection-{route.value}",
        summary=f"{route.value} legacy signal",
        suggested_route=route,
        route_status=status,
        route_reason=reason,
    )
    await store.save_signal(signal)
    await store.record_route(signal.signal_id, route=route, status=status, reason=reason)
    await store.close()
    return signal.signal_id


def _row(db_path: Path, table: str, key_column: str, key_value: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return dict(
            conn.execute(
                f"SELECT * FROM {table} WHERE {key_column} = ?",
                (key_value,),
            ).fetchone()
        )


async def test_repair_reclassifies_legacy_held_daydream_statuses(tmp_path: Path) -> None:
    db_path = tmp_path / "daydream_signals.db"
    ask_id = await _seed_signal(
        db_path,
        route=PossibilitySignalRoute.ASK_USER,
        status="held",
        reason="fallback_hold_low_intensity",
    )
    incubate_id = await _seed_signal(
        db_path,
        route=PossibilitySignalRoute.INCUBATE,
        status="held",
        reason="handler suggested incubate",
    )
    research_id = await _seed_signal(
        db_path,
        route=PossibilitySignalRoute.RESEARCH,
        status="held",
        reason="research handler not wired",
    )
    thread_id = await _seed_signal(
        db_path,
        route=PossibilitySignalRoute.THREAD,
        status="held",
        reason="unrelated held route",
    )

    preview = repair_legacy_daydream_signal_statuses(db_path)
    assert preview.applied is False
    assert preview.signal_updates == {
        "ask_user:held->deferred": 1,
        "incubate:held->incubated": 1,
        "research:held->deferred": 1,
    }
    assert _row(db_path, "daydream_signals", "signal_id", ask_id)["route_status"] == "held"

    applied = repair_legacy_daydream_signal_statuses(db_path, apply=True)

    assert applied.total_updates == 6
    assert _row(db_path, "daydream_signals", "signal_id", ask_id)["route_status"] == "deferred"
    assert (
        _row(db_path, "daydream_signals", "signal_id", incubate_id)["route_status"]
        == "incubated"
    )
    assert _row(db_path, "daydream_signals", "signal_id", research_id)["route_status"] == "deferred"
    assert _row(db_path, "daydream_signals", "signal_id", thread_id)["route_status"] == "held"

    raw = json.loads(_row(db_path, "daydream_signals", "signal_id", ask_id)["raw"])
    assert raw["meta"]["route_schema_version"] == 2
    assert raw["meta"]["legacy_status_repair"]["to_status"] == "deferred"

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT route, status, COUNT(*) AS count
            FROM daydream_signal_routes
            GROUP BY route, status
            """
        ).fetchall()
    route_statuses = {(route, status): count for route, status, count in rows}
    assert route_statuses[("ask_user", "deferred")] == 1
    assert route_statuses[("incubate", "incubated")] == 1
    assert route_statuses[("research", "deferred")] == 1
    assert route_statuses[("thread", "held")] == 1
