"""Tests for persisted phone bridge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencas.phone_config import (
    PhoneContactPolicy,
    load_phone_menu_config,
    PhoneRuntimeConfig,
    load_phone_runtime_config,
    normalize_phone_number,
    summarize_phone_session_profiles,
    save_phone_menu_config,
    save_phone_runtime_config,
    phone_dashboard_menu_path,
)


def test_normalize_phone_number_handles_common_inputs() -> None:
    assert normalize_phone_number("+1 (555) 123-4567") == "+15551234567"
    assert normalize_phone_number("5551234567") == "+15551234567"
    assert normalize_phone_number("1-555-123-4567") == "+15551234567"
    assert normalize_phone_number("442071838750") == "+442071838750"
    assert normalize_phone_number("") is None
    assert normalize_phone_number(None) is None


def test_phone_contact_policy_rejects_owner_trust_and_workspace_escape() -> None:
    with pytest.raises(ValueError, match="contacts must use low trust"):
        PhoneContactPolicy(phone_number="+15551234567", trust_level="owner")

    with pytest.raises(ValueError, match="managed workspace"):
        PhoneContactPolicy(phone_number="+15551234567", workspace_subdir="../escape")


def test_phone_contact_policy_preserves_explicit_empty_actions() -> None:
    contact = PhoneContactPolicy(phone_number="+15551234567", allowed_actions=[])

    assert contact.allowed_actions == []


def test_phone_runtime_config_round_trips_and_dedupes_contacts(tmp_path: Path) -> None:
    config = PhoneRuntimeConfig(
        enabled=True,
        public_base_url="https://opencas.example.com/",
        menu_config_path="operator_seed/phone/menu.json",
        twilio_from_number="5551112222",
        owner_phone_number="+1 (555) 123-4567",
        owner_workspace_subdir="phone/owner",
        owner_pin="123456",
        contacts=[
            {
                "phone_number": "5550001111",
                "display_name": "Alex",
                "allowed_actions": ["knowledge_qa"],
                "workspace_subdir": "phone/contacts/alex",
            },
            {
                "phone_number": "+1 555 000 1111",
                "display_name": "Duplicate Alex",
                "allowed_actions": ["leave_message"],
            },
        ],
    )

    saved_path = save_phone_runtime_config(tmp_path, config)
    loaded = load_phone_runtime_config(tmp_path)

    assert saved_path == Path(tmp_path) / "phone" / "config.json"
    assert loaded.enabled is True
    assert loaded.public_base_url == "https://opencas.example.com"
    assert loaded.menu_config_path == "operator_seed/phone/menu.json"
    assert loaded.twilio_from_number == "+15551112222"
    assert loaded.owner_phone_number == "+15551234567"
    assert loaded.owner_pin == "123456"
    assert len(loaded.contacts) == 1
    assert loaded.contacts[0].phone_number == "+15550001111"
    assert loaded.contacts[0].display_name == "Alex"
    assert loaded.contacts[0].allowed_actions == ["knowledge_qa"]
    assert loaded.contacts[0].workspace_subdir == "phone/contacts/alex"


def test_phone_runtime_config_rejects_owner_in_contacts() -> None:
    with pytest.raises(ValueError, match="must not also appear in contacts"):
        PhoneRuntimeConfig(
            owner_phone_number="+15551234567",
            contacts=[{"phone_number": "+15551234567"}],
        )


def test_phone_runtime_config_rejects_non_six_digit_owner_pin() -> None:
    with pytest.raises(ValueError, match="exactly 6 digits"):
        PhoneRuntimeConfig(owner_pin="12345")


def test_phone_menu_config_loads_default_menu_json() -> None:
    menu = load_phone_menu_config(Path("operator_seed/phone/menu.json"))

    assert menu.default_menu_key == "public_main"
    assert menu.owner_menu_key == "owner_entry"
    assert menu.menus
    owner_menu = next(item for item in menu.menus if item.key == "owner_entry")
    public_menu = next(item for item in menu.menus if item.key == "public_main")
    assert owner_menu.options[0].action == "owner_conversation"
    assert owner_menu.options[1].action == "submenu"
    assert owner_menu.options[1].target_menu == "public_main"
    employer = next(option for option in public_menu.options if option.key == "employer")
    assert employer.action == "workspace_assistant"
    assert employer.prompt_profile == "worksafe_owner"
    assert [mount.access for mount in employer.workspace_mounts] == ["read_only", "append_only"]


def test_phone_menu_config_summary_and_save_round_trip(tmp_path: Path) -> None:
    menu = load_phone_menu_config(Path("operator_seed/phone/menu.json"))
    summary = summarize_phone_session_profiles(menu)

    assert summary["owner_entry"]["continue_digit"] == "1"
    assert summary["employer"]["enabled"] is True
    assert summary["employer"]["shared_workspace_subdir"] == "phone/employer_shared"
    assert summary["reject"]["enabled"] is True

    saved = save_phone_menu_config(phone_dashboard_menu_path(tmp_path), menu)
    loaded = load_phone_menu_config(saved)
    assert loaded.default_menu_key == "public_main"
