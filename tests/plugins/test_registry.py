"""Tests for the SkillRegistry."""

import pytest
from opencas.plugins import SkillEntry, SkillRegistry


def test_register_and_get() -> None:
    registry = SkillRegistry()
    entry = SkillEntry(
        skill_id="test_skill",
        name="Test Skill",
        description="A test skill.",
        capabilities=["test_tool"],
    )
    registry.register(entry)

    assert registry.get("test_skill") == entry
    assert registry.get("missing") is None


def test_list_skills() -> None:
    registry = SkillRegistry()
    entry1 = SkillEntry(skill_id="s1", name="Skill 1", description="")
    entry2 = SkillEntry(skill_id="s2", name="Skill 2", description="")
    registry.register(entry1)
    registry.register(entry2)

    skills = registry.list_skills()
    assert len(skills) == 2
    assert entry1 in skills
    assert entry2 in skills


def test_resolve_for_tool() -> None:
    registry = SkillRegistry()
    entry = SkillEntry(
        skill_id="fs_skill",
        name="Filesystem Skill",
        description="",
        capabilities=["fs_read_file", "fs_write_file"],
    )
    registry.register(entry)

    assert registry.resolve_for_tool("fs_read_file") == entry
    assert registry.resolve_for_tool("fs_write_file") == entry
    assert registry.resolve_for_tool("missing_tool") is None


def test_register_overwrites_existing() -> None:
    registry = SkillRegistry()
    entry1 = SkillEntry(skill_id="s1", name="Original", description="")
    entry2 = SkillEntry(skill_id="s1", name="Updated", description="")
    registry.register(entry1)
    registry.register(entry2)

    assert registry.get("s1") == entry2
