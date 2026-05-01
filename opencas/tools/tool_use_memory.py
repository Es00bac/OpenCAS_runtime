"""Durable, compact lessons about tool choice and tool outcomes."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

_STOPWORDS = {
    "a",
    "about",
    "again",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "later",
    "of",
    "on",
    "or",
    "return",
    "schedule",
    "the",
    "this",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class ToolUseLesson:
    lesson_id: str
    tool_name: str
    outcome: str
    summary: str
    objective_terms: tuple[str, ...]
    uses: int
    confidence: float


class ToolUseMemoryStore:
    """SQLite-backed lessons that keep tool-routing context small and retrievable."""

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir).expanduser()
        self.path = self.state_dir / "tool_use_memory.db"
        self._ensure_schema()

    def record_result(
        self,
        *,
        objective: str,
        tool_name: str,
        args: dict[str, Any] | None,
        result: dict[str, Any],
    ) -> None:
        """Learn a compact lesson when a tool result contains reusable signal."""
        lesson = self._summarize_result(
            objective=objective,
            tool_name=tool_name,
            args=args or {},
            result=result,
        )
        if lesson is None:
            return
        outcome, summary = lesson
        self.record_lesson(
            objective=objective,
            tool_name=tool_name,
            outcome=outcome,
            summary=summary,
        )

    def record_lesson(
        self,
        *,
        objective: str,
        tool_name: str,
        outcome: str,
        summary: str,
    ) -> ToolUseLesson:
        """Persist or reinforce one reusable tool-use lesson."""
        clean_tool = str(tool_name or "").strip()
        clean_summary = _compact_text(summary, limit=320)
        if not clean_tool or not clean_summary:
            raise ValueError("tool_name and summary are required")

        now = _now()
        objective_terms = tuple(sorted(_tokens(" ".join((objective, clean_summary, clean_tool)))))
        objective_json = json.dumps(objective_terms)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT lesson_id, objective_terms, uses, confidence
                FROM tool_use_lessons
                WHERE tool_name = ? AND summary = ?
                """,
                (clean_tool, clean_summary),
            ).fetchone()
            if row is not None:
                uses = int(row["uses"]) + 1
                confidence = min(1.0, float(row["confidence"]) + 0.05)
                existing_terms = set(json.loads(row["objective_terms"] or "[]"))
                objective_json = json.dumps(tuple(sorted(existing_terms.union(objective_terms))))
                conn.execute(
                    """
                    UPDATE tool_use_lessons
                    SET updated_at = ?, objective_terms = ?, outcome = ?,
                        uses = ?, confidence = ?
                    WHERE lesson_id = ?
                    """,
                    (now, objective_json, outcome, uses, confidence, row["lesson_id"]),
                )
                lesson_id = str(row["lesson_id"])
            else:
                lesson_id = str(uuid4())
                uses = 1
                confidence = 0.62 if outcome == "success" else 0.72
                conn.execute(
                    """
                    INSERT INTO tool_use_lessons (
                        lesson_id, created_at, updated_at, objective_terms,
                        tool_name, outcome, summary, uses, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lesson_id,
                        now,
                        now,
                        objective_json,
                        clean_tool,
                        outcome,
                        clean_summary,
                        uses,
                        confidence,
                    ),
                )
        return ToolUseLesson(
            lesson_id=lesson_id,
            tool_name=clean_tool,
            outcome=outcome,
            summary=clean_summary,
            objective_terms=objective_terms,
            uses=uses,
            confidence=confidence,
        )

    def relevant_lessons(
        self,
        *,
        objective: str,
        available_tool_names: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[ToolUseLesson]:
        """Return the highest-signal lessons for the current objective."""
        objective_terms = _tokens(objective)
        if not objective_terms:
            return []
        available = set(available_tool_names or ())
        scored: list[tuple[float, ToolUseLesson]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT lesson_id, objective_terms, tool_name, outcome, summary,
                       uses, confidence
                FROM tool_use_lessons
                ORDER BY updated_at DESC
                LIMIT 200
                """
            ).fetchall()
        for row in rows:
            tool_name = str(row["tool_name"])
            if available and tool_name not in available:
                continue
            terms = tuple(json.loads(row["objective_terms"] or "[]"))
            haystack = _tokens(" ".join((tool_name, row["summary"] or "", " ".join(terms))))
            overlap = len(objective_terms.intersection(haystack))
            if objective_terms and overlap <= 0:
                continue
            lesson = ToolUseLesson(
                lesson_id=str(row["lesson_id"]),
                tool_name=tool_name,
                outcome=str(row["outcome"]),
                summary=str(row["summary"]),
                objective_terms=terms,
                uses=int(row["uses"]),
                confidence=float(row["confidence"]),
            )
            score = overlap + min(lesson.uses, 6) * 0.1 + lesson.confidence
            scored.append((score, lesson))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [lesson for _score, lesson in scored[: max(limit, 0)]]

    def relevant_tool_names(self, *, objective: str, limit: int = 4) -> list[str]:
        """Return learned tool names that should be considered for this objective."""
        names: list[str] = []
        for lesson in self.relevant_lessons(objective=objective, limit=limit):
            if lesson.tool_name not in names:
                names.append(lesson.tool_name)
        return names

    def build_context(
        self,
        *,
        objective: str,
        available_tool_names: Sequence[str] | None = None,
        limit: int = 5,
    ) -> str:
        """Render a compact prompt block for the tool loop."""
        lessons = self.relevant_lessons(
            objective=objective,
            available_tool_names=available_tool_names,
            limit=limit,
        )
        if not lessons:
            return ""
        lines = ["Tool-use memory hints:"]
        for lesson in lessons:
            lines.append(f"- {lesson.summary}")
        return "\n".join(lines)

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_use_lessons (
                    lesson_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    objective_terms TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    uses INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.5
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_use_lessons_tool
                ON tool_use_lessons(tool_name)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_use_lessons_updated
                ON tool_use_lessons(updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _summarize_result(
        *,
        objective: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> tuple[str, str] | None:
        success = bool(result.get("success", False))
        output = str(result.get("output", "") or "")
        output_lower = output.lower()
        text = " ".join((objective, output, json.dumps(args, default=str))).lower()

        if (
            tool_name == "workflow_create_schedule"
            and "submit_baa" in text
            and "reminder_only" in text
        ):
            return (
                "success" if success else "failure",
                (
                    "workflow_create_schedule: unfinished writing/project return "
                    "schedules need action=submit_baa instead of reminder_only."
                ),
            )
        if tool_name == "workflow_create_schedule" and args.get("action") == "submit_baa" and success:
            return (
                "success",
                (
                    "workflow_create_schedule: use action=submit_baa when a future "
                    "return must actively resume work, not merely remind."
                ),
            )
        if tool_name in {"fs_write_file", "edit_file"} and success and _has_any(
            objective,
            ("write", "writing", "draft", "manuscript", "revise", "revision", "artifact"),
        ):
            return (
                "success",
                (
                    f"{tool_name}: use direct file-writing tools for artifact drafts "
                    "and verify the target file changed."
                ),
            )
        if any(marker in output_lower for marker in ("tool not found", "missing required", "validation failed")):
            return (
                "failure",
                f"{tool_name}: previous call failed with {_compact_text(output, limit=180)}",
            )
        if any(marker in output_lower for marker in ("hook blocked", "blocked execution", "blocked write")):
            return (
                "failure",
                f"{tool_name}: previous call was blocked; inspect policy and choose a safer path.",
            )
        return None


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]{3,}", str(text).lower())
        if token not in _STOPWORDS
    }


def _has_any(text: str, needles: Iterable[str]) -> bool:
    lower = str(text or "").lower()
    return any(needle in lower for needle in needles)


def _compact_text(text: str, *, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
