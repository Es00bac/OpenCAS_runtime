"""Tests for the skill loader."""

import pytest
from pathlib import Path
from opencas.plugins import SkillEntry, SkillRegistry, load_builtin_skills, load_skill_from_path


def test_load_skill_from_path_with_skill_entry(tmp_path: Path) -> None:
    skill_file = tmp_path / "my_skill.py"
    skill_file.write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='my_skill', name='My Skill', description='Does a thing.')\n"
    )
    registry = SkillRegistry()
    entry = load_skill_from_path(skill_file, registry)

    assert entry is not None
    assert entry.skill_id == "my_skill"
    assert entry.name == "My Skill"
    assert registry.get("my_skill") == entry


def test_load_skill_from_path_with_register_fn(tmp_path: Path) -> None:
    skill_file = tmp_path / "reg_skill.py"
    skill_file.write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='reg_skill', name='Reg Skill', description='Registers tools.')\n"
        "def register_skills(registry):\n"
        "    registry.register('dummy_tool', 'A dummy tool', None, None, {})\n"
    )
    registry = SkillRegistry()
    entry = load_skill_from_path(skill_file, registry)

    assert entry is not None
    assert entry.skill_id == "reg_skill"
    assert entry.register_fn is not None


def test_load_skill_from_path_nonexistent() -> None:
    registry = SkillRegistry()
    assert load_skill_from_path(Path("/does/not/exist.py"), registry) is None


def test_load_skill_from_path_ignores_private_files(tmp_path: Path) -> None:
    skill_file = tmp_path / "_private.py"
    skill_file.write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='private', name='Private', description='')\n"
    )
    registry = SkillRegistry()
    assert load_skill_from_path(skill_file, registry) is None


def test_load_builtin_skills(tmp_path: Path) -> None:
    (tmp_path / "skill_a.py").write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='a', name='A', description='')\n"
    )
    (tmp_path / "skill_b.py").write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='b', name='B', description='')\n"
    )
    (tmp_path / "_ignored.py").write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='ignored', name='Ignored', description='')\n"
    )
    (tmp_path / "not_a_skill.txt").write_text("hello")

    registry = SkillRegistry()
    loaded = load_builtin_skills(tmp_path, registry)

    assert len(loaded) == 2
    skill_ids = {s.skill_id for s in loaded}
    assert skill_ids == {"a", "b"}
    assert registry.get("a") is not None
    assert registry.get("b") is not None
    assert registry.get("ignored") is None


def test_load_builtin_skills_nonexistent_dir() -> None:
    registry = SkillRegistry()
    loaded = load_builtin_skills(Path("/does/not/exist"), registry)
    assert loaded == []
