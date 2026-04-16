"""Summarize live validation reports into a compact qualification view."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / ".opencas_live_test_state"
SUMMARY_SCOPE = "retained_runs_dir_snapshot"


@dataclass
class ReportSummary:
    run_id: str
    started_at: str
    finished_at: str
    model: str
    embedding_model: str
    direct_successes: int
    direct_total: int
    agent_successes: int
    agent_total: int
    agent_failures: int
    duration_seconds: float


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration_seconds(report: Dict[str, Any]) -> float:
    started = _parse_iso8601(report.get("started_at"))
    finished = _parse_iso8601(report.get("finished_at"))
    if not started or not finished:
        return 0.0
    return max(0.0, (finished - started).total_seconds())


def _direct_check_success(payload: Dict[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload.get("success"))
    return bool(payload.get("available"))


def summarize_report(report: Dict[str, Any]) -> ReportSummary:
    direct_checks = report.get("direct_checks", {}) or {}
    agent_checks = report.get("agent_checks", []) or []
    direct_successes = sum(1 for payload in direct_checks.values() if _direct_check_success(payload))
    direct_total = len(direct_checks)
    agent_successes = sum(1 for item in agent_checks if _agent_check_success(item))
    agent_total = len(agent_checks)
    return ReportSummary(
        run_id=str(report.get("run_id", "")),
        started_at=str(report.get("started_at", "")),
        finished_at=str(report.get("finished_at", "")),
        model=str(report.get("model", "")),
        embedding_model=str(report.get("embedding_model", "")),
        direct_successes=direct_successes,
        direct_total=direct_total,
        agent_successes=agent_successes,
        agent_total=agent_total,
        agent_failures=max(0, agent_total - agent_successes),
        duration_seconds=_duration_seconds(report),
    )


def load_reports(runs_dir: Path, limit: int | None = None) -> List[Tuple[Path, Dict[str, Any]]]:
    reports: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted(runs_dir.glob("*/live_debug_validation_report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reports.append((path, payload))
    reports.sort(key=lambda item: item[1].get("started_at", ""), reverse=True)
    if limit is not None:
        reports = reports[:limit]
    return reports


def aggregate_agent_checks(reports: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "runs": 0,
        "successes": 0,
        "failures": 0,
        "timeouts": 0,
        "tool_messages_total": 0,
        "outcomes": Counter(),
        "recent_failures": [],
    })
    for report in reports:
        for item in report.get("agent_checks", []) or []:
            label = str(item.get("label", "unknown"))
            slot = by_label[label]
            slot["runs"] += 1
            if _agent_check_success(item):
                slot["successes"] += 1
            else:
                slot["failures"] += 1
                if len(slot["recent_failures"]) < 3:
                    slot["recent_failures"].append({
                        "run_id": report.get("run_id", ""),
                        "outcome": _agent_check_outcome(item),
                        "response": item.get("response", "")[:240],
                    })
            if item.get("timed_out", False):
                slot["timeouts"] += 1
            slot["tool_messages_total"] += int(item.get("tool_message_delta", 0) or 0)
            slot["outcomes"][_agent_check_outcome(item)] += 1
    result: Dict[str, Dict[str, Any]] = {}
    for label, slot in by_label.items():
        runs = slot["runs"] or 1
        result[label] = {
            "runs": slot["runs"],
            "successes": slot["successes"],
            "failures": slot["failures"],
            "success_rate": round(slot["successes"] / runs, 3),
            "timeouts": slot["timeouts"],
            "average_tool_messages": round(slot["tool_messages_total"] / runs, 2),
            "outcomes": dict(slot["outcomes"]),
            "recent_failures": slot["recent_failures"],
        }
    return dict(sorted(result.items()))


def _agent_check_success(item: Dict[str, Any]) -> bool:
    if "material_success" in item:
        return bool(item.get("material_success"))
    if item.get("timed_out", False):
        return False
    if "expected_file" in item:
        return bool(item.get("expected_file_exists", False))
    return True


def _agent_check_outcome(item: Dict[str, Any]) -> str:
    explicit = item.get("outcome")
    if explicit:
        return str(explicit)
    if item.get("timed_out", False):
        return "timed_out"
    if "expected_file" in item:
        return "artifact_verified" if item.get("expected_file_exists", False) else "artifact_missing"
    return "completed"


def aggregate_reports(reports: List[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
    payloads = [payload for _, payload in reports]
    summaries = [summarize_report(payload) for payload in payloads]
    total_runs = len(summaries)
    total_direct = sum(item.direct_total for item in summaries)
    total_direct_successes = sum(item.direct_successes for item in summaries)
    total_agent = sum(item.agent_total for item in summaries)
    total_agent_successes = sum(item.agent_successes for item in summaries)
    durations = [item.duration_seconds for item in summaries if item.duration_seconds > 0]
    return {
        "summary_scope": SUMMARY_SCOPE,
        "total_runs": total_runs,
        "total_direct_checks": total_direct,
        "total_direct_successes": total_direct_successes,
        "total_agent_checks": total_agent,
        "total_agent_successes": total_agent_successes,
        "direct_success_rate": round(total_direct_successes / total_direct, 3) if total_direct else None,
        "agent_success_rate": round(total_agent_successes / total_agent, 3) if total_agent else None,
        "average_run_duration_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "models": sorted({item.model for item in summaries if item.model}),
        "embedding_models": sorted({item.embedding_model for item in summaries if item.embedding_model}),
        "recent_runs": [
            {
                "run_id": item.run_id,
                "started_at": item.started_at,
                "finished_at": item.finished_at,
                "model": item.model,
                "direct_successes": item.direct_successes,
                "direct_total": item.direct_total,
                "agent_successes": item.agent_successes,
                "agent_total": item.agent_total,
                "duration_seconds": item.duration_seconds,
            }
            for item in summaries
        ],
        "agent_checks": aggregate_agent_checks(payloads),
    }


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# OpenCAS Live Validation Qualification Summary",
        "",
        f"- Scope: `current retained run folders`",
        f"- Runs analyzed: `{summary['total_runs']}`",
        f"- Summary scope id: `{summary.get('summary_scope', SUMMARY_SCOPE)}`",
        f"- Direct success rate: `{_format_scalar(summary['direct_success_rate'])}`",
        f"- Agent success rate: `{_format_scalar(summary['agent_success_rate'])}`",
        f"- Average run duration (s): `{_format_scalar(summary['average_run_duration_seconds'])}`",
        f"- Models: `{', '.join(summary['models']) or '-'}`",
        f"- Embedding models: `{', '.join(summary['embedding_models']) or '-'}`",
        "- Historical note: this file reflects only the run folders currently retained under `.opencas_live_test_state`; use `qualification_remediation_rollup.md` and readiness/task docs for rerun-history decisions.",
        "",
        "## Recent Runs",
        "",
    ]
    if not summary["recent_runs"]:
        lines.append("- None")
    else:
        for item in summary["recent_runs"]:
            lines.append(
                f"- `{item['run_id']}` direct `{item['direct_successes']}/{item['direct_total']}` "
                f"agent `{item['agent_successes']}/{item['agent_total']}` "
                f"duration `{item['duration_seconds']}`s model `{item['model']}`"
            )

    lines.extend(["", "## Agent Checks", ""])
    if not summary["agent_checks"]:
        lines.append("- None")
    else:
        for label, item in summary["agent_checks"].items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"- Runs: `{item['runs']}`")
            lines.append(f"- Successes: `{item['successes']}`")
            lines.append(f"- Failures: `{item['failures']}`")
            lines.append(f"- Success rate: `{item['success_rate']}`")
            lines.append(f"- Timeouts: `{item['timeouts']}`")
            lines.append(f"- Average tool messages: `{item['average_tool_messages']}`")
            lines.append(f"- Outcomes: `{json.dumps(item['outcomes'], sort_keys=True)}`")
            if item["recent_failures"]:
                lines.append("- Recent failures:")
                for failure in item["recent_failures"]:
                    lines.append(
                        f"  - `{failure['run_id']}` outcome `{failure['outcome']}`: "
                        f"{failure['response']}"
                    )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OpenCAS live validation runs")
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory containing live validation run folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of most-recent runs to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write summary JSON/Markdown files.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    reports = load_reports(runs_dir, limit=args.limit)
    summary = aggregate_reports(reports)

    print(render_markdown(summary))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "live_validation_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (output_dir / "live_validation_summary.md").write_text(
            render_markdown(summary),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
