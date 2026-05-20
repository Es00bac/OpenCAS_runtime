from pathlib import Path

from scripts.cognitive_health_query import build_report
from opencas.maintenance.state_hygiene import (
    remove_empty_whitespace_sqlite_files,
    sqlite_path_hygiene_violations,
)


def test_state_path_hygiene_finds_whitespace_sqlite_names(tmp_path: Path) -> None:
    bad = tmp_path / "work.db work_objects"
    bad.touch()
    (tmp_path / "work.db").touch()

    violations = sqlite_path_hygiene_violations(tmp_path)

    assert [item.path.name for item in violations] == ["work.db work_objects"]


def test_state_path_hygiene_removes_only_empty_whitespace_sqlite_files(tmp_path: Path) -> None:
    empty_bad = tmp_path / "work.db work_objects"
    nonempty_bad = tmp_path / "wellbeing.db wellbeing_states"
    empty_bad.touch()
    nonempty_bad.write_text("not empty", encoding="utf-8")

    removed = remove_empty_whitespace_sqlite_files(tmp_path)

    assert removed == [empty_bad]
    assert not empty_bad.exists()
    assert nonempty_bad.exists()


def test_deprecated_probe_dbs_are_not_runtime_store_targets(tmp_path: Path) -> None:
    (tmp_path / "initiative_contact.db").touch()
    (tmp_path / "baa_tasks.db").touch()

    report = build_report(tmp_path)

    assert report["checks"]["deprecated_probe_dbs"]["status"] == "info"
    assert sorted(report["checks"]["deprecated_probe_dbs"]["empty_files"]) == [
        "baa_tasks.db",
        "initiative_contact.db",
    ]
    assert "deprecated_probe_dbs" not in report["summary"]["unhealthy_checks"]
