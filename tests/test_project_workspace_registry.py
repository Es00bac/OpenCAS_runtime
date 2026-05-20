"""Tests for workspace project-root resolution."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.projects.workspace_registry import (
    resolve_recent_workspace_project,
    resolve_workspace_project,
    resolve_workspace_project_from_text,
)


class _Config:
    def __init__(self, root: Path):
        self._root = root

    def primary_workspace_root(self) -> Path:
        return self._root

    def agent_workspace_root(self) -> Path:
        return self._root / "workspace"


def _runtime(root: Path) -> SimpleNamespace:
    return SimpleNamespace(ctx=SimpleNamespace(config=_Config(root)))


def test_resolves_branded_workspace_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "kPony"
    (project_root / "src").mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (project_root / "README.md").write_text("# kPony\n", encoding="utf-8")

    candidate = resolve_workspace_project(
        _runtime(tmp_path),
        project_key="kpony",
        project_title="kPony",
    )

    assert candidate is not None
    assert candidate.project_title == "kPony"
    assert candidate.path == project_root.resolve()
    assert candidate.workspace_rel_path == Path("workspace/kPony")
    assert "CMakeLists.txt" in candidate.evidence


def test_resolves_nested_creative_workspace_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "novels" / "active-workspace-project"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "bible").mkdir()
    (project_root / "notes").mkdir()
    (project_root / "research").mkdir()
    (project_root / "review").mkdir()
    (project_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")
    (project_root / "bible" / "world_bible.md").write_text("world\n", encoding="utf-8")
    (project_root / "bible" / "character_bible.md").write_text("characters\n", encoding="utf-8")

    candidate = resolve_workspace_project(
        _runtime(tmp_path),
        project_key="active-workspace-project",
        project_title="Active Workspace Project",
    )

    assert candidate is not None
    assert candidate.project_title == "Active Workspace Project"
    assert candidate.project_key == "active-workspace-project"
    assert candidate.path == project_root.resolve()
    assert candidate.workspace_rel_path == Path("workspace/novels/active-workspace-project")
    assert "PROJECT.md" in candidate.evidence
    assert "drafts" in candidate.evidence
    assert "bible/world_bible.md" in candidate.evidence


def test_does_not_resolve_generic_project_to_unrelated_workspace_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "kPony"
    (project_root / "src").mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")

    candidate = resolve_workspace_project(
        _runtime(tmp_path),
        project_key="project",
        project_title="project",
    )

    assert candidate is None


def test_ignores_composted_projects_when_live_project_exists(tmp_path: Path) -> None:
    live_root = tmp_path / "workspace" / "kPony"
    (live_root / "src").mkdir(parents=True)
    (live_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    compost_root = tmp_path / "workspace" / "_compost" / "kpony-20260503T225951Z"
    compost_root.mkdir(parents=True)
    (compost_root / "README.md").write_text("# old kPony\n", encoding="utf-8")

    candidate = resolve_workspace_project(
        _runtime(tmp_path),
        project_key="kpony",
        project_title="kPony",
    )

    assert candidate is not None
    assert candidate.path == live_root.resolve()


def test_rejects_symlinked_project_root_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_root = tmp_path / "outside" / "escape-project"
    outside_root.mkdir(parents=True)
    (outside_root / "PROJECT.md").write_text("# Escape Project\n", encoding="utf-8")
    symlink_root = workspace_root / "escape-project"
    try:
        symlink_root.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    candidate = resolve_workspace_project(
        _runtime(tmp_path),
        project_key="escape-project",
        project_title="escape-project",
    )

    assert candidate is None


def test_resolves_workspace_project_from_named_path_text(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "novels" / "active-workspace-project"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "bible").mkdir()
    (project_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")

    candidate = resolve_workspace_project_from_text(
        _runtime(tmp_path),
        "Return to workspace/novels/active-workspace-project and continue the new novel.",
    )

    assert candidate is not None
    assert candidate.project_title == "Active Workspace Project"
    assert candidate.project_key == "active-workspace-project"
    assert candidate.workspace_rel_path == Path("workspace/novels/active-workspace-project")


def test_does_not_resolve_build_directory_from_imperative_verb(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "kPony"
    build_root = project_root / "build"
    build_root.mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (build_root / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")

    candidate = resolve_workspace_project_from_text(
        _runtime(tmp_path),
        "Build a name/place audit before revising the manuscript.",
    )

    assert candidate is None


def test_resolves_named_project_without_matching_build_child(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace" / "kPony"
    build_root = project_root / "build"
    build_root.mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (build_root / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")

    candidate = resolve_workspace_project_from_text(
        _runtime(tmp_path),
        "Continue kPony in the workspace and finish the build proof.",
    )

    assert candidate is not None
    assert candidate.project_title == "kPony"
    assert candidate.path == project_root.resolve()


def test_resolves_recent_writing_project_and_excludes_archived_keys(tmp_path: Path) -> None:
    old_root = tmp_path / "workspace" / "novels" / "archived-workspace-project"
    old_root.mkdir(parents=True)
    (old_root / "PROJECT.md").write_text("# Archived Workspace Project\n", encoding="utf-8")
    (old_root / "drafts").mkdir()
    new_root = tmp_path / "workspace" / "novels" / "active-workspace-project"
    new_root.mkdir(parents=True)
    (new_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")
    (new_root / "drafts").mkdir()
    (new_root / "bible").mkdir()
    os.utime(old_root / "PROJECT.md", (1_700_000_000, 1_700_000_000))
    os.utime(new_root / "PROJECT.md", (1_800_000_000, 1_800_000_000))

    candidate = resolve_recent_workspace_project(
        _runtime(tmp_path),
        project_type="writing",
        excluded_project_keys=("archived-workspace-project",),
    )

    assert candidate is not None
    assert candidate.project_title == "Active Workspace Project"
    assert candidate.path == new_root.resolve()
