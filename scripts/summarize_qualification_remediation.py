#!/usr/bin/env python3
"""Summarize recent qualification reruns into actionable remediation guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / ".opencas_live_test_state"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "qualification"
DEFAULT_HISTORY_PATH = DEFAULT_RUNS_DIR / "qualification_rerun_history.jsonl"
RATE_SCOPE = "retained_label_runs"


def _agent_check_success(item: Dict[str, Any]) -> bool:
    if "material_success" in item:
        return bool(item.get("material_success"))
    if item.get("timed_out", False):
        return False
    if "expected_file" in item:
        return bool(item.get("expected_file_exists", False))
    return True


def _load_rerun_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def _load_label_runs(runs_dir: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*/live_debug_validation_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(report.get("run_id", path.parent.name))
        matched = next((item for item in report.get("agent_checks", []) or [] if str(item.get("label", "")) == label), None)
        if matched is None:
            continue
        rows.append({
            "run_id": run_id,
            "success": _agent_check_success(matched),
            "outcome": matched.get("outcome"),
            "response": matched.get("response"),
        })
    return rows


def _before_after_stats(label_runs: List[Dict[str, Any]], latest_run_id: Optional[str]) -> Dict[str, Any]:
    if not label_runs:
        return {}
    if latest_run_id is None:
        before_rate = round(sum(1 for item in label_runs if item["success"]) / len(label_runs), 3) if label_runs else None
        return {
            "before_rate": before_rate,
            "after_rate": before_rate,
            "previous_run": label_runs[-1] if label_runs else None,
            "latest_run": None,
        }
    latest_index = next((idx for idx, item in enumerate(label_runs) if item["run_id"] == latest_run_id), None)
    if latest_index is None:
        return {
            "before_rate": None,
            "after_rate": None,
            "previous_run": None,
            "latest_run": None,
        }
    before = label_runs[:latest_index]
    after = label_runs[: latest_index + 1]

    def _rate(items: List[Dict[str, Any]]) -> Optional[float]:
        if not items:
            return None
        return round(sum(1 for item in items if item["success"]) / len(items), 3)

    previous = label_runs[latest_index - 1] if latest_index > 0 else None
    latest = label_runs[latest_index] if latest_index < len(label_runs) else None
    return {
        "before_rate": _rate(before),
        "after_rate": _rate(after),
        "previous_run": previous,
        "latest_run": latest,
    }


def _classify_action(completion: Dict[str, Any], stats: Dict[str, Any]) -> str:
    if int(completion.get("returncode", 0) or 0) != 0 and not completion.get("latest_run_id"):
        return "investigate_runner"
    previous = stats.get("previous_run")
    latest = stats.get("latest_run")
    if latest is None:
        return "investigate_runner"
    if latest.get("success") and (previous is None or not previous.get("success")):
        return "continue_testing"
    if latest.get("success"):
        return "watch_only"
    return "code_change_justified"


def build_rollup(runs_dir: Path, history_path: Path, limit: int = 10) -> Dict[str, Any]:
    history = _load_rerun_history(history_path)
    completions = [item for item in history if str(item.get("event", "")) == "completed"]
    rows: List[Dict[str, Any]] = []
    for completion in reversed(completions[-limit:]):
        labels = [str(item) for item in completion.get("labels", []) if str(item)]
        label = labels[0] if len(labels) == 1 else None
        label_runs = _load_label_runs(runs_dir, label) if label else []
        stats = _before_after_stats(label_runs, completion.get("latest_run_id"))
        rows.append({
            "request_id": completion.get("request_id"),
            "label": label,
            "returncode": completion.get("returncode"),
            "latest_run_id": completion.get("latest_run_id"),
            "generated_run_ids": completion.get("generated_run_ids", []),
            "before_rate": stats.get("before_rate"),
            "after_rate": stats.get("after_rate"),
            "previous_run": stats.get("previous_run"),
            "latest_run": stats.get("latest_run"),
            "recommended_action": _classify_action(completion, stats),
        })
    return {
        "history_path": str(history_path),
        "runs_dir": str(runs_dir),
        "rate_scope": RATE_SCOPE,
        "count": len(rows),
        "items": list(reversed(rows)),
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Qualification Remediation Rollup",
        "",
        f"- Reruns summarized: `{payload.get('count', 0)}`",
        f"- History path: `{payload.get('history_path', '')}`",
        f"- Rate scope id: `{payload.get('rate_scope', RATE_SCOPE)}`",
        "- Historical note: `before_rate` and `after_rate` are computed from the label runs currently retained under `.opencas_live_test_state`; use the request IDs plus task/readiness docs for longer-history decisions.",
        "",
    ]
    for item in payload.get("items", []):
        lines.append(f"## {item.get('label') or item.get('request_id')}")
        lines.append("")
        lines.append(f"- Request ID: `{item.get('request_id', '-')}`")
        lines.append(f"- Return code: `{item.get('returncode', '-')}`")
        lines.append(f"- Latest run: `{item.get('latest_run_id', '-')}`")
        lines.append(f"- Before rate: `{item.get('before_rate', '-')}`")
        lines.append(f"- After rate: `{item.get('after_rate', '-')}`")
        latest = item.get("latest_run") or {}
        if latest:
            lines.append(f"- Latest outcome: `{latest.get('outcome', '-')}` success `{latest.get('success', False)}`")
        prev = item.get("previous_run") or {}
        if prev:
            lines.append(f"- Previous outcome: `{prev.get('outcome', '-')}` success `{prev.get('success', False)}`")
        lines.append(f"- Recommended action: `{item.get('recommended_action', '-')}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize qualification reruns into remediation guidance.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    history_path = Path(args.history_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = build_rollup(runs_dir, history_path, limit=max(1, args.limit))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qualification_remediation_rollup.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    md = render_markdown(payload)
    (output_dir / "qualification_remediation_rollup.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
