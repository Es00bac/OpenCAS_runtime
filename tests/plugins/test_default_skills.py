"""Tests for default built-in skills."""

from pathlib import Path

import pytest

from opencas.plugins.skills.git_skill import _git_commit, _git_review_pr
from opencas.plugins.skills.test_skill import _run_test
from opencas.plugins.skills.doc_skill import _generate_docstring, _run_docs
from opencas.tools import ToolRegistry
from opencas.plugins import SkillRegistry, load_builtin_skills


def test_git_commit(tmp_path: Path) -> None:
    # Initialize a temp git repo
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True)

    file_path = repo / "hello.txt"
    file_path.write_text("world", encoding="utf-8")

    result = _git_commit({"message": "Initial commit", "cwd": str(repo)})
    assert result.success is True
    assert "Initial commit" in result.output or "1 file changed" in result.output


def test_git_review_pr_no_gh() -> None:
    result = _git_review_pr({"pr_number": "123"})
    assert result.success is False
    assert "gh CLI" in result.output or "not available" in result.output


def test_test_skill_runs_pytest(tmp_path: Path) -> None:
    test_file = tmp_path / "test_example.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")
    result = _run_test({"target": str(test_file), "cwd": str(tmp_path)})
    assert result.success is True
    assert "passed" in result.output.lower()


def test_doc_skill_preview(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def foo():\n    pass\n", encoding="utf-8")
    result = _run_docs({"file_path": str(source), "write": False})
    assert result.success is True
    assert '"""foo' in result.output or "preview" in str(result.metadata)


def test_doc_skill_write(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def bar():\n    pass\n", encoding="utf-8")
    result = _run_docs({"file_path": str(source), "write": True})
    assert result.success is True
    assert "Updated docstrings" in result.output
    updated = source.read_text(encoding="utf-8")
    assert '"""bar"""' in updated


def test_generate_docstring_with_existing_docstring(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text('def baz():\n    """Already documented."""\n    pass\n', encoding="utf-8")
    result = _generate_docstring(str(source))
    # Should not insert a new docstring
    assert result.count('"""') == 2


def test_builtin_skills_load_from_package() -> None:
    import opencas.plugins.skills as skills_pkg

    registry = SkillRegistry()
    builtin_dir = Path(skills_pkg.__file__).parent
    loaded = load_builtin_skills(builtin_dir, registry)
    skill_ids = {s.skill_id for s in loaded}
    assert "git_skill" in skill_ids
    assert "test_skill" in skill_ids
    assert "doc_skill" in skill_ids


def test_builtin_skills_register_tools() -> None:
    import opencas.plugins.skills as skills_pkg

    skill_registry = SkillRegistry()
    tool_registry = ToolRegistry()
    builtin_dir = Path(skills_pkg.__file__).parent
    load_builtin_skills(builtin_dir, skill_registry)

    for skill in skill_registry.list_skills():
        if skill.register_fn is not None:
            skill.register_fn(tool_registry)

    assert tool_registry.get("git_commit") is not None
    assert tool_registry.get("git_review_pr") is not None
    assert tool_registry.get("run_tests") is not None
    assert tool_registry.get("generate_docs") is not None
