"""Counterfactual option scoring for recovery and retry decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CounterfactualOption:
    """A concrete alternate path the agent can choose after friction."""

    strategy: str
    action: str
    score: float
    reason: str
    evidence_need: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_counterfactual_options(
    *,
    objective: str = "",
    failure_summary: str = "",
    prior_tool: str = "",
    available_tools: Iterable[str] = (),
    prior_attempts: int = 1,
) -> list[CounterfactualOption]:
    """Score alternative recovery strategies from concrete failure evidence.

    This is deliberately deterministic and inspectable. It is not a canned
    response generator: it returns candidate actions and their evidence needs so
    downstream runtime code can choose a changed hypothesis before retrying.
    """

    objective_l = _compact(objective).lower()
    failure_l = _compact(failure_summary).lower()
    tools = {str(tool).strip() for tool in available_tools if str(tool).strip()}
    attempts = max(1, int(prior_attempts or 1))

    options = [
        _option(
            "verify_prerequisite",
            "Check the missing prerequisite or runtime state before repeating the failed action.",
            0.55,
            "Failure text suggests the precondition may be false or unverified.",
            "runtime proof, file existence, auth status, or command help output",
        ),
        _option(
            "change_tool_or_arguments",
            "Retry only with a different tool, path, arguments, or permission evidence.",
            0.50,
            "The previous attempt used a concrete tool/action shape that may be wrong.",
            "diff between prior and next tool invocation",
        ),
        _option(
            "narrow_scope",
            "Reduce the objective to the smallest verifiable slice, complete it, then expand.",
            0.46,
            "Broad retries are more likely to repeat the same mistake without isolating cause.",
            "one narrow artifact or one targeted command result",
        ),
        _option(
            "ask_or_escalate",
            "Ask a focused question or escalate only if the blocker cannot be resolved from local evidence.",
            0.34,
            "Some blockers require owner preference, credentials, or external authorization.",
            "specific missing decision or permission boundary",
        ),
    ]

    if any(token in failure_l for token in ("missing", "not found", "no such", "unavailable", "not installed")):
        options = _boost(options, "verify_prerequisite", 0.24)
    if any(token in failure_l for token in ("permission", "denied", "blocked", "approval", "forbidden")):
        options = _boost(options, "ask_or_escalate", 0.38)
        options = _boost(options, "verify_prerequisite", 0.12)
    if any(token in failure_l for token in ("timeout", "rate limit", "429", "server error", "connection")):
        options = _boost(options, "change_tool_or_arguments", 0.18)
        options = _boost(options, "narrow_scope", 0.12)
    if attempts >= 2:
        options = _boost(options, "narrow_scope", min(0.25, attempts * 0.06))
    if prior_tool and prior_tool in tools:
        options = _boost(options, "change_tool_or_arguments", 0.08)
    if any(token in objective_l for token in ("verify", "proof", "evidence", "audit", "check")):
        options = _boost(options, "verify_prerequisite", 0.10)
    if any(token in objective_l for token in ("build", "implement", "fix", "repair")):
        options = _boost(options, "narrow_scope", 0.10)

    ranked = sorted(
        (
            CounterfactualOption(
                strategy=option.strategy,
                action=option.action,
                score=round(max(0.0, min(1.0, option.score)), 3),
                reason=option.reason,
                evidence_need=option.evidence_need,
            )
            for option in options
        ),
        key=lambda item: (item.score, item.strategy),
        reverse=True,
    )
    return ranked


def recommended_counterfactual(
    *,
    objective: str = "",
    failure_summary: str = "",
    prior_tool: str = "",
    available_tools: Iterable[str] = (),
    prior_attempts: int = 1,
) -> dict[str, Any]:
    """Return the top scored option plus the full option set."""

    options = build_counterfactual_options(
        objective=objective,
        failure_summary=failure_summary,
        prior_tool=prior_tool,
        available_tools=available_tools,
        prior_attempts=prior_attempts,
    )
    return {
        "recommended": options[0].to_dict() if options else {},
        "options": [option.to_dict() for option in options],
    }


def _option(strategy: str, action: str, score: float, reason: str, evidence_need: str) -> CounterfactualOption:
    return CounterfactualOption(
        strategy=strategy,
        action=action,
        score=score,
        reason=reason,
        evidence_need=evidence_need,
    )


def _boost(options: list[CounterfactualOption], strategy: str, amount: float) -> list[CounterfactualOption]:
    boosted: list[CounterfactualOption] = []
    for option in options:
        if option.strategy != strategy:
            boosted.append(option)
            continue
        boosted.append(
            CounterfactualOption(
                strategy=option.strategy,
                action=option.action,
                score=option.score + amount,
                reason=option.reason,
                evidence_need=option.evidence_need,
            )
        )
    return boosted


def _compact(text: str) -> str:
    return " ".join(str(text or "").split())
