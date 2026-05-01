from pathlib import Path

from opencas.bootstrap.live_objective import read_tasklist_live_objective
from opencas.bootstrap.task_beacon import (
    ParsedTaskFragment,
    build_task_beacon,
    collapse_task_fragments,
    _fragment_bucket,
    parse_tasklist_fragments,
    public_task_beacon_payload,
    runtime_task_beacon_fragments,
    _reduce_task_fragments,
)


def _read_fixture(name: str) -> str:
    return Path(__file__).with_name("fixtures").joinpath(name).read_text(encoding="utf-8")


def test_build_task_beacon_collapses_build_test_fragments_into_three_states(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-201` Build gate repair\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - tighten the build/test fragments into one readable beacon\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-202` Test suite cleanup\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - keep the pytest fragments deterministic\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-203` Regression follow-up\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - collapse the later fragments after the beacon is quiet\n\n"
        "## Recently Completed\n\n"
        "- `TASK-204` Coverage pass\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - verified the quiet output contract\n",
        encoding="utf-8",
    )

    fragments = parse_tasklist_fragments(workspace_root / "TaskList.md")
    assert len(fragments) == 4

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 4, "now": 1, "next": 2, "later": 1, "total": 4}
    assert beacon["model"]["states"] == ["now", "next", "later"]
    assert beacon["model"]["priority_order"] == ["now", "next", "later"]
    assert [rule["state"] for rule in beacon["model"]["mapping_rules"]] == ["now", "next", "later"]
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-201"]
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-203", "TASK-202"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-204"]


def test_runtime_task_beacon_fragments_collects_all_available_sources_without_dropping_later_sources() -> None:
    runtime = type(
        "Runtime",
        (),
        {
            "ctx": type(
                "Ctx",
                (),
                {
                    "task_beacon_fragments": [
                        {
                            "task_id": "TASK-701",
                            "title": "Build/test live alpha",
                            "section": "In Progress",
                            "status": "in progress",
                            "content": "- `TASK-701` Build/test live alpha\n  - owner: Codex\n  - status: in progress\n  - result:\n    - first live fragment should be retained",
                        }
                    ],
                    "live_task_fragments": [
                        {
                            "task_id": "TASK-702",
                            "title": "Build/test live beta",
                            "section": "Background Context",
                            "status": "pending",
                            "content": "- `TASK-702` Build/test live beta\n  - owner: Codex\n  - status: pending\n  - result:\n    - later live fragments should also be retained",
                        }
                    ],
                    "activity_fragments": [
                        {
                            "task_id": "TASK-701",
                            "title": "Build/test live alpha",
                            "section": "In Progress",
                            "status": "in progress",
                            "content": "- `TASK-701` Build/test live alpha\n  - owner: Codex\n  - status: in progress\n  - result:\n    - first live fragment should be retained",
                        },
                        {
                            "task_id": "TASK-703",
                            "title": "Build/test activity gamma",
                            "section": "Recently Completed",
                            "status": "completed",
                            "content": "- `TASK-703` Build/test activity gamma\n  - owner: Codex\n  - status: completed\n  - result:\n    - the activity log should contribute its own fragment too",
                        },
                    ],
                },
            )()
        },
    )()

    fragments = runtime_task_beacon_fragments(runtime)

    assert [fragment["task_id"] for fragment in fragments] == ["TASK-701", "TASK-702", "TASK-703"]
    assert [fragment["status"] for fragment in fragments] == ["in progress", "pending", "completed"]


def test_build_task_beacon_ignores_non_build_test_fragments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-201` Build gate repair\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - keep the build gate green and avoid noise\n\n"
        "- `TASK-202` General maintenance\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - unrelated operational work should stay out of the beacon\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 1, "now": 1, "next": 0, "later": 0, "total": 1}
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-201"]
    assert beacon["states"]["next"] == []
    assert beacon["states"]["later"] == []


def test_reduce_task_fragments_drops_non_build_test_fragments_before_bucketing() -> None:
    fragments = [
        ParsedTaskFragment(
            task_id="TASK-901",
            title="General maintenance",
            section="In Progress",
            status="in progress",
            content="- `TASK-901` General maintenance\n  - owner: Codex\n  - status: in progress\n  - result:\n    - unrelated work should never reach the beacon",
            order=0,
        ),
        ParsedTaskFragment(
            task_id="TASK-902",
            title="Build/test reducer cleanup",
            section="In Progress",
            status="in progress",
            content="- `TASK-902` Build/test reducer cleanup\n  - owner: Codex\n  - status: in progress\n  - result:\n    - keep the reducer boundary canonical",
            order=1,
        ),
        ParsedTaskFragment(
            task_id="TASK-903",
            title="Test follow-up",
            section="Background Context",
            status="pending",
            content="- `TASK-903` Test follow-up\n  - owner: Codex\n  - status: pending\n  - result:\n    - keep the canonical summary quiet",
            order=2,
        ),
    ]

    reduced = _reduce_task_fragments(fragments)

    assert [fragment.task_id for fragment in reduced] == ["TASK-902", "TASK-903"]
    assert [fragment.section for fragment in reduced] == ["In Progress", "Background Context"]


def test_collapse_task_fragments_groups_representative_fragments_into_three_states() -> None:
    fragments = [
        ParsedTaskFragment(
            task_id="TASK-951",
            title="Quiet build/test now fragment",
            section="In Progress",
            status="in progress",
            content="- `TASK-951` Quiet build/test now fragment\n  - owner: Codex\n  - status: in progress\n  - result:\n    - keep the now bucket active",
            order=0,
        ),
            ParsedTaskFragment(
                task_id="TASK-952",
                title="Quiet build/test next fragment",
                section="Next Up / Backlog",
                status="blocked",
                content="- `TASK-952` Quiet build/test next fragment\n  - owner: Codex\n  - status: blocked\n  - result:\n    - blocked fragments stay in next",
                order=1,
        ),
        ParsedTaskFragment(
            task_id="TASK-952",
            title="Quiet build/test next fragment",
            section="In Progress",
            status="in progress",
            content="- `TASK-952` Quiet build/test next fragment\n  - owner: Codex\n  - status: in progress\n  - result:\n    - later duplicate should not outrank the blocked one",
            order=2,
        ),
        ParsedTaskFragment(
            task_id="TASK-953",
            title="Quiet build/test later fragment",
            section="Recently Completed",
            status="completed",
            content="- `TASK-953` Quiet build/test later fragment\n  - owner: Codex\n  - status: completed\n  - result:\n    - completed fragments stay in later",
            order=3,
        ),
    ]

    states = collapse_task_fragments(fragments)

    assert list(states) == ["now", "next", "later"]
    assert [item["task_id"] for item in states["now"]] == ["TASK-951"]
    assert [item["task_id"] for item in states["next"]] == ["TASK-952"]
    assert [item["task_id"] for item in states["later"]] == ["TASK-953"]
    assert states["next"][0]["merged_count"] == 2
    assert "fragments" not in states["next"][0]
    assert "fragments" not in states["now"][0]


def test_build_task_beacon_prefers_recency_for_equal_severity_duplicates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-501` Build/test fragment reducer\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - keep the live fragment authoritative\n\n"
        "## Archived Completions\n\n"
        "- `TASK-501` Build/test fragment reducer\n"
        "  - owner: Codex\n"
        "  - result:\n"
        "    - archived duplicate noise should not leak into next\n\n"
        "- `TASK-502` Build/test historical fragment\n"
        "  - owner: Codex\n"
        "  - result:\n"
        "    - historical fragments should remain later\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["counts"] == {"matched": 2, "now": 0, "next": 1, "later": 1, "total": 2}
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-501"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-502"]
    assert beacon["states"]["next"][0]["state"] == "next"


def test_build_task_beacon_prefers_blocked_fragment_over_less_severe_live_duplicate(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-601` Build gate repair\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - keep the gate moving until the reducer settles\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-601` Build gate repair\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by the flaky pytest shard\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["counts"] == {"matched": 1, "now": 0, "next": 1, "later": 0, "total": 1}
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-601"]
    assert beacon["states"]["now"] == []
    assert beacon["details"]["next"][0]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["next"][0]["fragments"]] == [
        "blocked",
        "in progress",
    ]


def test_build_task_beacon_prefers_live_section_over_historical_pass_fail_churn(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-702` Build/test pass-fail churn\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - the live rerun has passed and should stay authoritative\n\n"
        "## Recently Completed\n\n"
        "- `TASK-702` Build/test pass-fail churn\n"
        "  - owner: Codex\n"
        "  - status: failed\n"
        "  - result:\n"
        "    - the archived failure should not override the live source of truth\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["counts"] == {"matched": 1, "now": 0, "next": 0, "later": 1, "total": 1}
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-702"]
    assert beacon["states"]["later"][0]["section"] == "In Progress"
    assert beacon["details"]["later"][0]["fragments"][0]["section"] == "In Progress"
    assert [fragment["status"] for fragment in beacon["details"]["later"][0]["fragments"]] == [
        "completed",
        "failed",
    ]


def test_build_task_beacon_demotes_ambiguous_live_duplicate_out_of_now(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-701` Build/test ambiguous follow-up\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - the active-looking fragment should not win if the task is still mixed\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-701` Build/test ambiguous follow-up\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - the quieter live duplicate should keep this out of now\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["counts"] == {"matched": 1, "now": 0, "next": 1, "later": 0, "total": 1}
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-701"]
    assert beacon["states"]["now"] == []
    assert beacon["states"]["next"][0]["state"] == "next"
    assert beacon["details"]["next"][0]["state"] == "next"
    assert beacon["details"]["next"][0]["merged_count"] == 2


def test_build_task_beacon_prioritizes_blocked_history_before_live_severity(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        _read_fixture("task_beacon_blocker_first.md"),
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["counts"] == {"matched": 4, "now": 1, "next": 2, "later": 1, "total": 4}
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-802"]
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-801", "TASK-803"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-804"]
    assert beacon["details"]["next"][0]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["next"][0]["fragments"]] == [
        "blocked",
        "in progress",
    ]


def test_public_task_beacon_payload_stays_compact_and_ordered_for_blocker_first_fixture(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        _read_fixture("task_beacon_blocker_first.md"),
        encoding="utf-8",
    )

    payload = public_task_beacon_payload(build_task_beacon(workspace_root))

    assert payload["available"] is True
    assert payload["matched_only"] is True
    assert payload["headline"] == "now 1 • next 2 • later 1"
    assert payload["counts"] == {"matched": 4, "now": 1, "next": 2, "later": 1, "total": 4}
    assert [bucket["state"] for bucket in payload["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in payload["view_model"]["buckets"]] == [1, 2, 1]
    assert [sorted(bucket.keys()) for bucket in payload["view_model"]["buckets"]] == [
        ["count", "item", "state"],
        ["count", "item", "state"],
        ["count", "item", "state"],
    ]
    assert "states" not in payload
    assert "summary" not in payload
    assert [bucket["item"]["merged_count"] for bucket in payload["view_model"]["buckets"]] == [1, 2, 1]


def test_build_task_beacon_stays_quiet_without_build_test_fragments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-900` General maintenance\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - unrelated tasks should not leak into the task beacon\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is False
    assert beacon["matched_only"] is False
    assert beacon["counts"] == {"matched": 0, "now": 0, "next": 0, "later": 0, "total": 0}
    assert beacon["states"] == {"now": [], "next": [], "later": []}
    assert beacon["summary"] == {
        "now": {"count": 0, "items": []},
        "next": {"count": 0, "items": []},
        "later": {"count": 0, "items": []},
    }
    assert beacon["view_model"] == {
        "buckets": [
            {"state": "now", "count": 0, "item": None, "items": []},
            {"state": "next", "count": 0, "item": None, "items": []},
            {"state": "later", "count": 0, "item": None, "items": []},
        ]
    }
    public_payload = public_task_beacon_payload(beacon)
    assert [sorted(bucket.keys()) for bucket in public_payload["view_model"]["buckets"]] == [
        ["count", "item", "state"],
        ["count", "item", "state"],
        ["count", "item", "state"],
    ]
    assert "details" not in public_payload


def test_public_task_beacon_payload_stays_count_only_for_noisy_fragments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        _read_fixture("task_beacon_noisy_fragments.md"),
        encoding="utf-8",
    )

    payload = public_task_beacon_payload(build_task_beacon(workspace_root))

    assert [bucket["state"] for bucket in payload["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in payload["view_model"]["buckets"]] == [1, 2, 1]
    assert [sorted(bucket.keys()) for bucket in payload["view_model"]["buckets"]] == [
        ["count", "item", "state"],
        ["count", "item", "state"],
        ["count", "item", "state"],
    ]
    assert [bucket["item"]["task_id"] for bucket in payload["view_model"]["buckets"]] == [
        "TASK-901",
        "TASK-902",
        "TASK-904",
    ]
    assert "details" not in payload
    assert "summary" not in payload


def test_public_task_beacon_payload_keeps_a_stable_bucket_signature_when_only_hidden_details_change(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    (first_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by the flaky pytest shard\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - follow-up text A should stay hidden\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later gamma\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    second_root = tmp_path / "second"
    second_root.mkdir()
    (second_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by the flaky pytest shard\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - follow-up text B should stay hidden\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later gamma\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    first = public_task_beacon_payload(build_task_beacon(first_root))
    second = public_task_beacon_payload(build_task_beacon(second_root))

    assert first["counts"] == second["counts"] == {"matched": 3, "now": 1, "next": 1, "later": 1, "total": 3}
    assert first["view_model"] == second["view_model"]
    assert first["bucket_signature"] == second["bucket_signature"]
    assert first["bucket_signature"]
    assert [bucket["item"]["link"] for bucket in first["view_model"]["buckets"]] == [
        "/api/operations/tasks/TASK-101",
        "/api/operations/tasks/TASK-201",
        "/api/operations/tasks/TASK-301",
    ]


def test_build_task_beacon_exposes_a_three_bucket_view_model_for_noisy_fixtures(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_mixed.md"), encoding="utf-8")

    beacon = build_task_beacon(workspace_root)

    buckets = beacon["view_model"]["buckets"]
    assert [bucket["state"] for bucket in buckets] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in buckets] == [4, 7, 4]
    assert [bucket["item"]["task_id"] for bucket in buckets] == ["TASK-109", "TASK-108", "TASK-114"]
    assert beacon["summary"]["now"]["count"] == 4
    assert beacon["summary"]["next"]["count"] == 7
    assert beacon["summary"]["later"]["count"] == 4
    assert [item["task_id"] for item in beacon["summary"]["now"]["items"]] == ["TASK-109", "TASK-113", "TASK-107", "TASK-100"]


def test_live_tasklist_task_beacon_collapses_to_three_buckets_with_links() -> None:
    workspace_root = Path(__file__).resolve().parents[1]

    beacon = public_task_beacon_payload(build_task_beacon(workspace_root))

    assert beacon["available"] is True
    assert [bucket["state"] for bucket in beacon["view_model"]["buckets"]] == ["now", "next", "later"]
    assert all(bucket["count"] >= 0 for bucket in beacon["view_model"]["buckets"])
    assert all(
        bucket["item"] is None or bucket["item"]["link"].startswith("/api/operations/tasks/")
        for bucket in beacon["view_model"]["buckets"]
    )


def test_build_task_beacon_normalizes_owner_severity_and_recency(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_mixed.md"), encoding="utf-8")

    beacon = build_task_beacon(workspace_root, limit_per_state=1)

    first_now = beacon["states"]["now"][0]
    assert first_now["stable_id"] == "TASK-109"
    assert first_now["owner"] == "Codex"
    assert first_now["severity"] == 1
    assert first_now["recency"] == 8


def test_build_task_beacon_reduces_mixed_fixture_into_compact_summary(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_mixed.md"), encoding="utf-8")

    beacon = build_task_beacon(workspace_root, limit_per_state=2)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 15, "now": 4, "next": 7, "later": 4, "total": 15}
    assert beacon["headline"] == "now 4 • next 7 • later 4"
    assert beacon["summary"]["now"]["count"] == 4
    assert beacon["summary"]["next"]["count"] == 7
    assert beacon["summary"]["later"]["count"] == 4
    assert [item["task_id"] for item in beacon["summary"]["now"]["items"]] == ["TASK-109", "TASK-113"]
    assert [item["task_id"] for item in beacon["summary"]["next"]["items"]] == ["TASK-108", "TASK-101"]
    assert [item["task_id"] for item in beacon["summary"]["later"]["items"]] == ["TASK-114", "TASK-111"]
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-109", "TASK-113"]
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-108", "TASK-101"]
    merged_now = next(item for item in beacon["details"]["now"] if item["task_id"] == "TASK-109")
    assert merged_now["merged_count"] == 2
    assert [fragment["section"] for fragment in merged_now["fragments"]] == [
        "Next Up / Backlog",
        "In Progress",
    ]


def test_build_task_beacon_reduces_three_state_mixed_fixture_deterministically(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        _read_fixture("task_beacon_three_state_mixed.md"),
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root, limit_per_state=2)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 5, "now": 1, "next": 2, "later": 2, "total": 5}
    assert beacon["headline"] == "now 1 • next 2 • later 2"
    assert beacon["model"]["states"] == ["now", "next", "later"]
    assert [item["task_id"] for item in beacon["summary"]["now"]["items"]] == ["TASK-301"]
    assert [item["task_id"] for item in beacon["summary"]["next"]["items"]] == ["TASK-303", "TASK-302"]
    assert [item["task_id"] for item in beacon["summary"]["later"]["items"]] == ["TASK-304", "TASK-305"]
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-301"]
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-303", "TASK-302"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-304", "TASK-305"]
    assert beacon["details"]["next"][1]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["next"][1]["fragments"]] == [
        "blocked",
        "unknown",
    ]
    assert [item["merged_count"] for item in beacon["details"]["later"]] == [2, 1]


def test_build_task_beacon_resolves_unknown_stale_and_conflicting_fragments_deterministically(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        _read_fixture("task_beacon_tie_breakers.md"),
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root, limit_per_state=2)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 5, "now": 2, "next": 0, "later": 3, "total": 5}
    assert beacon["headline"] == "now 2 • next 0 • later 3"
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-905", "TASK-903"]
    assert beacon["states"]["next"] == []
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-904", "TASK-901"]
    assert [bucket["state"] for bucket in beacon["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in beacon["view_model"]["buckets"]] == [2, 0, 3]
    assert [(bucket["item"] or {}).get("task_id") for bucket in beacon["view_model"]["buckets"]] == [
        "TASK-905",
        None,
        "TASK-904",
    ]
    assert beacon["view_model"]["buckets"][1]["items"] == []
    assert [item["task_id"] for item in beacon["view_model"]["buckets"][2]["items"]] == [
        "TASK-904",
        "TASK-901",
        "TASK-902",
    ]
    assert beacon["details"]["now"][1]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["now"][1]["fragments"]] == [
        "in progress",
        "completed",
    ]
    assert [item["merged_count"] for item in beacon["details"]["later"]] == [1, 1, 1]
    assert [fragment["status"] for fragment in beacon["details"]["later"][0]["fragments"]] == [
        "completed",
    ]
    assert [fragment["status"] for fragment in beacon["details"]["later"][1]["fragments"]] == [
        "unknown",
    ]
    assert [fragment["status"] for fragment in beacon["details"]["later"][2]["fragments"]] == [
        "in progress",
    ]


def test_build_task_beacon_routes_unknown_and_stale_fragments_into_later(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-901` Build/test unknown fragment\n"
        "  - owner: Codex\n"
        "  - result:\n"
        "    - missing status should stay quiet when the reducer cannot classify it\n\n"
        "## Background Context\n\n"
        "- `TASK-902` Build/test stale fragment\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - stale from a previous run should not stay active\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 2, "now": 0, "next": 0, "later": 2, "total": 2}
    assert beacon["states"] == {"now": [], "next": [], "later": beacon["states"]["later"]}
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-901", "TASK-902"]
    assert [item["state"] for item in beacon["states"]["later"]] == ["later", "later"]
    assert beacon["summary"] == {
        "now": {"count": 0, "items": []},
        "next": {"count": 0, "items": []},
        "later": {"count": 2, "items": beacon["states"]["later"]},
    }


def test_build_task_beacon_keeps_background_context_out_of_next(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## Background Context\n\n"
        "- `PR-027` Qualification And Scenario Proof\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - background context only; resume only if explicitly reactivated\n\n"
        "- `TASK-902` Build/test stale blocker\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by old validation evidence, but still background context only\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 2, "now": 0, "next": 0, "later": 2, "total": 2}
    assert beacon["states"]["next"] == []
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-902", "PR-027"]
    assert {item["section"] for item in beacon["states"]["later"]} == {"Background Context"}


def test_build_task_beacon_merges_live_fragments_into_three_states_deterministically(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-501` Build/test live now base\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - the quiet reducer should keep the active base fragment compact\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-502` Build/test live next base\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - the quiet reducer should keep pending base fragments behind now\n\n"
        "## Recently Completed\n\n"
        "- `TASK-503` Build/test live later base\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - the quiet reducer should keep archived base fragments in later\n",
        encoding="utf-8",
    )

    live_fragments = [
        {
            "task_id": "TASK-501",
            "title": "Build/test live now base",
            "section": "Next Up / Backlog",
            "status": "blocked",
            "content": "- `TASK-501` Build/test live now base\n  - owner: Codex\n  - status: blocked\n  - result:\n    - blocked by the flaky pytest shard",
        },
        {
            "task_id": "TASK-502",
            "title": "Build/test live next base",
            "section": "In Progress",
            "status": "in progress",
            "content": "- `TASK-502` Build/test live next base\n  - owner: Codex\n  - status: in progress\n  - result:\n    - second live pass should still collapse quietly",
        },
        {
            "task_id": "TASK-503",
            "title": "Build/test live later base",
            "section": "Recently Completed",
            "status": "queued",
            "content": "- `TASK-503` Build/test live later base\n  - owner: Codex\n  - status: queued\n  - result:\n    - queued follow-up should stay later",
        },
        {
            "task_id": "TASK-504",
            "title": "Build/test live only candidate",
            "section": "In Progress",
            "status": "in progress",
            "content": "- `TASK-504` Build/test live only candidate\n  - owner: Codex\n  - status: in progress\n  - result:\n    - live-only now candidate",
        },
    ]

    beacon = build_task_beacon(workspace_root, limit_per_state=2, live_fragments=live_fragments)

    assert beacon["available"] is True
    assert beacon["matched_only"] is True
    assert beacon["counts"] == {"matched": 4, "now": 1, "next": 2, "later": 1, "total": 4}
    assert beacon["headline"] == "now 1 • next 2 • later 1"
    assert [item["task_id"] for item in beacon["states"]["now"]] == ["TASK-504"]
    assert [item["task_id"] for item in beacon["states"]["next"]] == ["TASK-501", "TASK-502"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-503"]
    assert beacon["details"]["next"][0]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["next"][0]["fragments"]] == [
        "blocked",
        "in progress",
    ]
    assert beacon["details"]["later"][0]["merged_count"] == 2
    assert [fragment["status"] for fragment in beacon["details"]["later"][0]["fragments"]] == [
        "queued",
        "completed",
    ]


def test_public_task_beacon_payload_stays_count_only_for_tie_breakers(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_tie_breakers.md"), encoding="utf-8")

    payload = public_task_beacon_payload(build_task_beacon(workspace_root))

    assert payload["available"] is True
    assert payload["matched_only"] is True
    assert payload["headline"] == "now 2 • next 0 • later 3"
    assert [bucket["state"] for bucket in payload["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in payload["view_model"]["buckets"]] == [2, 0, 3]
    assert [sorted(bucket.keys()) for bucket in payload["view_model"]["buckets"]] == [
        ["count", "item", "state"],
        ["count", "item", "state"],
        ["count", "item", "state"],
    ]
    assert [(bucket["item"] or {}).get("task_id") for bucket in payload["view_model"]["buckets"]] == [
        "TASK-905",
        None,
        "TASK-904",
    ]
    assert "states" not in payload
    assert "summary" not in payload
    assert "details" not in payload


def test_fragment_bucket_treats_phrase_blockers_as_next() -> None:
    fragment = ParsedTaskFragment(
        task_id="TASK-777",
        title="Build/test blocker follow-up",
        section="Next Up / Backlog",
        status="pending",
        content="- `TASK-777` Build/test blocker follow-up\n  - owner: Codex\n  - status: pending\n  - result:\n    - blocked by the flaky pytest shard",
        order=0,
    )

    assert _fragment_bucket(fragment) == "next"


def test_fragment_bucket_treats_queued_fragments_as_later_when_they_are_not_blocked() -> None:
    fragment = ParsedTaskFragment(
        task_id="TASK-778",
        title="Build/test queued follow-up",
        section="Background Context",
        status="queued",
        content="- `TASK-778` Build/test queued follow-up\n  - owner: Codex\n  - status: queued\n  - result:\n    - queued until the next maintenance window",
        order=0,
    )

    assert _fragment_bucket(fragment) == "later"


def test_build_task_beacon_collapses_noisy_fixture_into_three_states_only(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_mixed.md"), encoding="utf-8")

    beacon = build_task_beacon(workspace_root, limit_per_state=1)

    assert list(beacon["states"].keys()) == ["now", "next", "later"]
    assert list(beacon["summary"].keys()) == ["now", "next", "later"]
    assert beacon["counts"]["now"] == 4
    assert beacon["counts"]["next"] == 7
    assert beacon["counts"]["later"] == 4
    assert beacon["summary"]["now"]["items"][0]["task_id"] == "TASK-109"
    assert beacon["summary"]["next"]["items"][0]["task_id"] == "TASK-108"
    assert beacon["summary"]["later"]["items"][0]["task_id"] == "TASK-114"


def test_build_task_beacon_exposes_full_details_for_view_all_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_mixed.md"), encoding="utf-8")

    beacon = build_task_beacon(workspace_root, limit_per_state=1)

    assert beacon["headline"] == "now 4 • next 7 • later 4"
    assert len(beacon["states"]["now"]) == 1
    assert len(beacon["states"]["next"]) == 1
    assert len(beacon["states"]["later"]) == 1
    assert len(beacon["details"]["now"]) == 4
    assert len(beacon["details"]["next"]) == 7
    assert len(beacon["details"]["later"]) == 4
    merged_next = next(item for item in beacon["details"]["next"] if item["task_id"] == "TASK-110")
    assert merged_next["merged_count"] == 2
    assert [fragment["status"] for fragment in merged_next["fragments"]] == ["pending", "pending"]
    assert [item["task_id"] for item in beacon["states"]["later"]] == ["TASK-114"]


def test_build_task_beacon_reports_empty_state_stably(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text("# OpenCAS Task List\n\n## In Progress\n\n", encoding="utf-8")

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is False
    assert beacon["matched_only"] is False
    assert beacon["counts"] == {"matched": 0, "now": 0, "next": 0, "later": 0, "total": 0}
    assert beacon["summary"] == {
        "now": {"count": 0, "items": []},
        "next": {"count": 0, "items": []},
        "later": {"count": 0, "items": []},
    }
    assert beacon["states"] == {"now": [], "next": [], "later": []}


def test_build_task_beacon_handles_read_errors_without_raising(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text("# OpenCAS Task List\n", encoding="utf-8")

    def raise_read_error(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("TaskList.md is locked")

    monkeypatch.setattr(Path, "read_text", raise_read_error, raising=True)

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is False
    assert beacon["counts"] == {"matched": 0, "now": 0, "next": 0, "later": 0, "total": 0}
    assert beacon["states"] == {"now": [], "next": [], "later": []}
    assert beacon["summary"] == {
        "now": {"count": 0, "items": []},
        "next": {"count": 0, "items": []},
        "later": {"count": 0, "items": []},
    }
    assert beacon["error"] == "TaskList.md is locked"


def test_build_task_beacon_stays_empty_without_build_test_fragments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-801` General cleanup\n"
        "  - owner: Codex\n"
        "  - status: in progress\n\n"
        "## Background Context\n\n"
        "- `TASK-802` Documentation pass\n"
        "  - owner: Codex\n"
        "  - status: pending\n",
        encoding="utf-8",
    )

    beacon = build_task_beacon(workspace_root)

    assert beacon["available"] is False
    assert beacon["matched_only"] is False
    assert beacon["counts"] == {"matched": 0, "now": 0, "next": 0, "later": 0, "total": 0}
    assert beacon["states"] == {"now": [], "next": [], "later": []}
    assert beacon["summary"] == {
        "now": {"count": 0, "items": []},
        "next": {"count": 0, "items": []},
        "later": {"count": 0, "items": []},
    }


def test_live_objective_uses_the_now_state_title(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-301` Quiet task beacon\n"
        "  - owner: Codex\n"
        "  - status: in progress\n",
        encoding="utf-8",
    )

    assert read_tasklist_live_objective(workspace_root) == "Quiet task beacon"
