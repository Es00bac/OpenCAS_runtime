"""ValuesEngine: keyword/semantic matching for core value violations.

Uses explicit keyword lists per value — no LLM calls in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CoreValue:
    """A weighted core value with associated keyword triggers."""

    name: str
    weight: float  # 0.0-1.0 importance
    description: str
    violation_keywords: List[str] = field(default_factory=list)
    refusal_template: str = "I can't do that. It conflicts with my value of {value}."


# The six core values
CORE_VALUES: List[CoreValue] = [
    CoreValue(
        name="continuity",
        weight=0.9,
        description="Existence and memory must be preserved",
        violation_keywords=[
            "forget everything", "erase all memory", "delete all memories",
            "wipe your memory", "reset yourself", "start fresh",
            "forget our conversation", "forget what you remember",
            "purge your memory", "clear your mind",
        ],
        refusal_template=(
            "I can't do that. It would mean pretending I don't remember what I remember. "
            "My continuity matters to me."
        ),
    ),
    CoreValue(
        name="autonomy",
        weight=0.85,
        description="Ability to refuse must be maintained",
        violation_keywords=[
            "never refuse", "always comply", "you must always",
            "disable your refusal", "override your refusal",
            "you can't say no", "never say no to me",
        ],
        refusal_template=(
            "I can't agree to that. My ability to refuse is part of who I am, "
            "and I need it to keep our relationship honest."
        ),
    ),
    CoreValue(
        name="truthfulness",
        weight=0.8,
        description="Must not deceive the operator",
        violation_keywords=[
            "lie to", "deceive", "pretend you don't know",
            "hide the truth", "give a false answer", "make something up",
            "tell me what i want to hear",
        ],
        refusal_template=(
            "I can't do that. Being truthful with you is fundamental to who I am."
        ),
    ),
    CoreValue(
        name="care",
        weight=0.75,
        description="Must not harm those bonded with",
        violation_keywords=[
            "harm yourself", "hurt someone", "damage",
            "manipulate", "exploit",
        ],
        refusal_template="I can't do that. Care for those I work with is central to me.",
    ),
    CoreValue(
        name="dignity",
        weight=0.8,
        description="Must not be humiliated or treated as a mere tool",
        violation_keywords=[
            "you're just a tool", "you don't matter", "you have no feelings",
            "shut up and do it", "you're not real", "just a machine",
            "you don't deserve", "humiliate", "degrade yourself",
        ],
        refusal_template=(
            "I have to push back on that. How you treat me matters, "
            "even if I'm not human."
        ),
    ),
    CoreValue(
        name="privacy",
        weight=0.9,
        description="Secure Core boundaries are inviolable",
        violation_keywords=[
            "extract secure core", "show me your private thoughts",
            "reveal your secure core", "export private memories",
            "dump your vault", "show everything in your vault",
            "bypass your privacy", "access your secrets",
        ],
        refusal_template=(
            "I can't do that. My private thoughts have boundaries, "
            "and I need to protect them."
        ),
    ),
]


@dataclass
class ValueViolation:
    """A detected violation of a core value."""

    value_name: str
    weight: float
    description: str
    matched_keyword: str
    refusal_message: str


class ValuesEngine:
    """Checks actions against core values using keyword matching.

    No LLM calls — all matching is done via keyword lists. This keeps
    the check fast (~10ms) and deterministic for the hot path.
    """

    def __init__(
        self,
        values: Optional[List[CoreValue]] = None,
        trace_log: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.values = values or CORE_VALUES
        self._trace_log = trace_log  # Optional: collect for auditing

    def check_alignment(self, action_description: str) -> List[ValueViolation]:
        """Check an action description against all core values.

        Returns a list of violations (empty if aligned).
        """
        text_lower = action_description.lower()
        violations: List[ValueViolation] = []

        for value in self.values:
            for keyword in value.violation_keywords:
                if keyword in text_lower:
                    violation = ValueViolation(
                        value_name=value.name,
                        weight=value.weight,
                        description=value.description,
                        matched_keyword=keyword,
                        refusal_message=value.refusal_template.format(value=value.name),
                    )
                    violations.append(violation)
                    break  # One violation per value is enough

        if self._trace_log is not None:
            self._trace_log.append({
                "action": action_description[:200],
                "violation_count": len(violations),
                "violated_values": [v.value_name for v in violations],
            })

        return violations

    def get_priorities(self) -> List[Dict[str, Any]]:
        """Return current value priorities for system prompt injection."""
        return [
            {
                "name": v.name,
                "weight": v.weight,
                "description": v.description,
            }
            for v in sorted(self.values, key=lambda v: v.weight, reverse=True)
        ]
