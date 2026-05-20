"""Tests for the self-approval ladder."""

from pathlib import Path
import pytest

from opencas.autonomy import (
    ActionRequest,
    ActionRiskTier,
    AuthorizationStore,
    ApprovalMode,
    ApprovalLevel,
    SelfApprovalLadder,
)
from opencas.governance import (
    WebActionClass,
    WebDomainObservation,
    WebTrustLevel,
    WebTrustPolicy,
    WebTrustService,
    WebTrustStore,
)
from opencas.identity import IdentityManager, IdentityStore
from opencas.relational import MusubiState, RelationalEngine, MusubiStore
from opencas.somatic import SomaticManager


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def somatic(tmp_path: Path):
    return SomaticManager(tmp_path / "somatic.json")


@pytest.fixture
def ladder(identity, somatic):
    return SelfApprovalLadder(identity=identity, somatic=somatic)


def test_readonly_safe(ladder: SelfApprovalLadder) -> None:
    req = ActionRequest(tier=ActionRiskTier.READONLY, description="list files")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_NOW


def test_workspace_write_with_trust(ladder: SelfApprovalLadder, identity: IdentityManager) -> None:
    identity.adjust_trust(0.4)  # high trust
    req = ActionRequest(tier=ActionRiskTier.WORKSPACE_WRITE, description="edit file")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_NOW


def test_destructive_escalates(ladder: SelfApprovalLadder) -> None:
    req = ActionRequest(tier=ActionRiskTier.DESTRUCTIVE, description="rm -rf /")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_boundary_blocks(ladder: SelfApprovalLadder, identity: IdentityManager) -> None:
    identity.user_model.known_boundaries.append(ActionRiskTier.SHELL_LOCAL.value)
    identity.save()
    req = ActionRequest(tier=ActionRiskTier.SHELL_LOCAL, description="run shell")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_tool_boundary_blocks(ladder: SelfApprovalLadder, identity: IdentityManager) -> None:
    identity.user_model.known_boundaries.append("browser_navigate")
    identity.save()
    req = ActionRequest(
        tier=ActionRiskTier.NETWORK,
        description="browse",
        tool_name="browser_navigate",
    )
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_natural_language_destructive_boundary_blocks(
    ladder: SelfApprovalLadder, identity: IdentityManager
) -> None:
    identity.user_model.known_boundaries = [
        "no destructive actions without explicit confirmation"
    ]
    identity.save()
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="dangerous shell action",
        tool_name="bash_run_command",
        payload={
            "command_family": "filesystem_destructive",
            "command_permission_class": "dangerous",
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_readonly_shell_gets_higher_self_trust(
    ladder: SelfApprovalLadder, identity: IdentityManager
) -> None:
    identity.adjust_trust(0.2)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="inspect repository",
        tool_name="bash_run_command",
        payload={
            "command_family": "safe",
            "command_permission_class": "read_only",
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )


def test_bounded_shell_command_gets_payload_credit(
    ladder: SelfApprovalLadder, identity: IdentityManager, somatic: SomaticManager
) -> None:
    identity.user_model.trust_level = 0.95
    identity.save()
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="launch codex in a PTY",
        tool_name="pty_interact",
        payload={
            "command_family": "safe",
            "command_permission_class": "bounded_write",
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )


def test_history_modulation_improves(ladder: SelfApprovalLadder, identity: IdentityManager) -> None:
    identity.update_self_belief("success_rate_tier_shell_local", 0.95)
    req = ActionRequest(tier=ActionRiskTier.SHELL_LOCAL, description="safe shell")
    dec = ladder.evaluate(req)
    # High historical success should push it below escalation
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
        ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE,
    )


def test_history_modulation_worsens(ladder: SelfApprovalLadder, identity: IdentityManager) -> None:
    identity.update_self_belief("success_rate_tier_shell_local", 0.10)
    req = ActionRequest(tier=ActionRiskTier.SHELL_LOCAL, description="risky shell")
    dec = ladder.evaluate(req)
    # Low historical success should escalate or demand evidence
    assert dec.level in (
        ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE,
        ApprovalLevel.MUST_ESCALATE,
    )


def test_ordinary_safe_shell_does_not_request_user_evidence(
    ladder: SelfApprovalLadder,
    identity: IdentityManager,
) -> None:
    identity.update_self_belief("success_rate_tier_shell_local", 0.1)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="inspect the repository with a read-only command",
        tool_name="bash_run_command",
        payload={
            "command_family": "safe",
            "command_permission_class": "read_only",
            "ordinary_action": True,
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_WITH_CAUTION
    assert "ordinary_action_self_approved" in dec.reasoning


def test_somatic_tension_increases_caution(
    ladder: SelfApprovalLadder, somatic: SomaticManager
) -> None:
    somatic.set_tension(0.8)
    req = ActionRequest(tier=ActionRiskTier.WORKSPACE_WRITE, description="edit while tense")
    dec = ladder.evaluate(req)
    # With default trust, workspace_write + tension should tip into caution
    assert dec.level in (
        ApprovalLevel.CAN_DO_WITH_CAUTION,
        ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE,
    )


def test_managed_workspace_write_gets_payload_credit_under_stress(
    ladder: SelfApprovalLadder,
    identity: IdentityManager,
    somatic: SomaticManager,
) -> None:
    identity.user_model.trust_level = 0.95
    identity.save()
    somatic.set_tension(1.0)
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.WORKSPACE_WRITE,
        description="write experiment scaffold file",
        tool_name="fs_write_file",
        payload={
            "file_path": "/tmp/workspace/experiment/README.md",
            "write_scope": "managed_workspace",
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )


def test_managed_workspace_shell_verification_gets_payload_credit_under_stress(
    ladder: SelfApprovalLadder,
    identity: IdentityManager,
    somatic: SomaticManager,
) -> None:
    identity.user_model.trust_level = 0.95
    identity.save()
    somatic.set_tension(1.0)
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="verify generated parser inside managed workspace",
        tool_name="bash_run_command",
        payload={
            "command_family": "safe",
            "command_permission_class": "bounded_write",
            "command_scope": "managed_workspace",
            "command_effective_family": "safe",
            "command_effective_permission_class": "bounded_write",
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )


def test_web_research_is_ordinary_under_stress(
    ladder: SelfApprovalLadder,
    somatic: SomaticManager,
) -> None:
    somatic.set_tension(1.0)
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.NETWORK,
        description="research a topic for a grounded daydream",
        tool_name="web_fetch",
        payload={"web_action_class": "fetch", "web_domain": "example.com"},
    )

    dec = ladder.evaluate(req)

    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )
    assert "ordinary_action_self_approved" in dec.reasoning


def test_browser_navigation_research_is_ordinary_under_stress(
    ladder: SelfApprovalLadder,
    somatic: SomaticManager,
) -> None:
    somatic.set_tension(1.0)
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.NETWORK,
        description="open a source page while researching a user-requested topic",
        tool_name="browser_navigate",
        payload={
            "web_action_class": WebActionClass.NAVIGATE.value,
            "web_domain": "example.com",
            "web_url": "https://example.com/source",
        },
    )

    dec = ladder.evaluate(req)

    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )
    assert "ordinary_action_self_approved" in dec.reasoning


def test_project_workspace_verification_is_ordinary_under_stress(
    ladder: SelfApprovalLadder,
    somatic: SomaticManager,
) -> None:
    somatic.set_tension(1.0)
    somatic.set_fatigue(1.0)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run project verification inside an allowed workspace",
        tool_name="bash_run_command",
        payload={
            "command_family": "safe",
            "command_permission_class": "bounded_write",
            "command_scope": "project_workspace",
            "command_effective_family": "safe",
            "command_effective_permission_class": "bounded_write",
        },
    )

    dec = ladder.evaluate(req)

    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )
    assert "ordinary_action_self_approved" in dec.reasoning


def test_no_somatic_does_not_crash(identity: IdentityManager) -> None:
    ladder_no_somatic = SelfApprovalLadder(identity=identity)
    req = ActionRequest(tier=ActionRiskTier.READONLY, description="read")
    dec = ladder_no_somatic.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_NOW


def test_standing_authorization_fast_paths_gmail_read(
    identity: IdentityManager,
    tmp_path: Path,
) -> None:
    store = AuthorizationStore(tmp_path / "authorizations.db")
    store.grant(
        action_class="gmail_read",
        scope="google_workspace:gmail",
        evidence_episode_id="episode-1",
    )
    ladder = SelfApprovalLadder(identity=identity, authorization_store=store)

    req = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="tool google_workspace_gmail_headlines",
        tool_name="google_workspace_gmail_headlines",
    )

    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_NOW
    assert dec.reasoning.startswith("standing_authorization:gmail_read")


def test_standing_authorization_does_not_override_boundary(
    identity: IdentityManager,
    tmp_path: Path,
) -> None:
    identity.user_model.known_boundaries = ["google_workspace_gmail_headlines"]
    identity.save()
    store = AuthorizationStore(tmp_path / "authorizations.db")
    store.grant(action_class="gmail_read", scope="google_workspace:gmail")
    ladder = SelfApprovalLadder(identity=identity, authorization_store=store)

    req = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="tool google_workspace_gmail_headlines",
        tool_name="google_workspace_gmail_headlines",
    )

    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE
    assert "explicit_boundary_hit" in dec.reasoning


def test_musubi_approval_modulator_is_env_gated_and_reduces_high_trust_risk(
    identity: IdentityManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relational = RelationalEngine(MusubiStore(Path(":memory:")))
    relational._state = MusubiState(musubi=0.8)
    req = ActionRequest(tier=ActionRiskTier.SHELL_LOCAL, description="run safe test")

    monkeypatch.delenv("AUTH_USE_MUSUBI_MODULATOR", raising=False)
    disabled = SelfApprovalLadder(identity=identity, relational=relational).evaluate(req)

    monkeypatch.setenv("AUTH_USE_MUSUBI_MODULATOR", "1")
    enabled = SelfApprovalLadder(identity=identity, relational=relational).evaluate(req)

    assert "musubi_mod" not in disabled.reasoning
    assert enabled.score < disabled.score
    assert "musubi_mod=-" in enabled.reasoning


def test_yolo_readonly_is_approved(identity: IdentityManager) -> None:
    ladder = SelfApprovalLadder(identity=identity, mode=ApprovalMode.FULLY_AUTONOMOUS)
    req = ActionRequest(tier=ActionRiskTier.READONLY, description="list files")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_NOW


def test_yolo_shell_is_caution(identity: IdentityManager) -> None:
    ladder = SelfApprovalLadder(identity=identity, mode=ApprovalMode.FULLY_AUTONOMOUS)
    req = ActionRequest(tier=ActionRiskTier.SHELL_LOCAL, description="run test")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.CAN_DO_WITH_CAUTION


def test_yolo_destructive_escalates(identity: IdentityManager) -> None:
    ladder = SelfApprovalLadder(identity=identity, mode=ApprovalMode.FULLY_AUTONOMOUS)
    req = ActionRequest(tier=ActionRiskTier.DESTRUCTIVE, description="rm -rf /")
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_yolo_boundary_hit_escalates(identity: IdentityManager) -> None:
    identity.user_model.known_boundaries = ["shell"]
    identity.save()
    ladder = SelfApprovalLadder(identity=identity, mode=ApprovalMode.FULLY_AUTONOMOUS)
    req = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run command",
        payload={"command_family": "shell"},
    )
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_set_mode_switches_evaluation(identity: IdentityManager) -> None:
    ladder = SelfApprovalLadder(identity=identity)
    ladder.set_mode("yolo")
    assert ladder.mode == ApprovalMode.FULLY_AUTONOMOUS


def test_blocked_web_domain_escalates(identity: IdentityManager, somatic: SomaticManager, tmp_path: Path) -> None:
    web_trust = WebTrustService(WebTrustStore(tmp_path / "web_trust.db"))
    web_trust._policies["example.com"] = WebTrustPolicy(
        domain="example.com",
        level=WebTrustLevel.BLOCKED,
        source="user",
        note="blocked",
    )
    ladder = SelfApprovalLadder(identity=identity, somatic=somatic, web_trust=web_trust)
    req = ActionRequest(
        tier=ActionRiskTier.NETWORK,
        description="navigate to blocked site",
        tool_name="browser_navigate",
        payload={
            "web_domain": "example.com",
            "web_url": "https://example.com",
            "web_action_class": WebActionClass.NAVIGATE.value,
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level == ApprovalLevel.MUST_ESCALATE


def test_trusted_web_interaction_can_self_approve_with_evidence(
    identity: IdentityManager,
    somatic: SomaticManager,
    tmp_path: Path,
) -> None:
    identity.user_model.trust_level = 1.0
    identity.update_self_belief("success_rate_tier_external_write", 0.95)
    identity.update_self_belief("success_rate_tool_browser_click", 0.95)
    web_trust = WebTrustService(WebTrustStore(tmp_path / "web_trust.db"))
    web_trust._observations[("docs.python.org", WebActionClass.NAVIGATE)] = WebDomainObservation(
        domain="docs.python.org",
        action_class=WebActionClass.NAVIGATE,
        success_count=12,
        failure_count=0,
    )
    ladder = SelfApprovalLadder(identity=identity, somatic=somatic, web_trust=web_trust)
    req = ActionRequest(
        tier=ActionRiskTier.EXTERNAL_WRITE,
        description="click trusted navigation element",
        tool_name="browser_click",
        payload={
            "web_domain": "docs.python.org",
            "web_url": "https://docs.python.org/3/",
            "web_action_class": WebActionClass.INTERACT.value,
        },
    )
    dec = ladder.evaluate(req)
    assert dec.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )
