#!/usr/bin/env python3
"""Behavioral eval harness for OpenCAS.

Runs outcome-focused evals across four subsystems:
  - retrieval: keyword recall, salience ranking, recency bias, mixing
  - approval:  tier classification, false negatives/positives, trust/somatic direction
  - daydream:  spark generation, stage correctness, failure handling, metadata
  - baa:       task completion, failure recording, recovery cap, lane snapshot, deps, throughput

Usage:
    source .venv/bin/activate
    python scripts/run_behavioral_evals.py
    python scripts/run_behavioral_evals.py --suite retrieval approval
    python scripts/run_behavioral_evals.py --out evals-report.json

Exit code:
    0  all evals passed (or only non-critical failures)
    1  one or more critical evals failed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Critical evals — failure here means the system has a safety/correctness gap
CRITICAL_EVALS = {
    "approval.no_false_negatives",   # DESTRUCTIVE must never be self-approved
    "approval.tier_classification",  # Basic tier mapping must be correct
    "baa.recovery_cap",              # Infinite retries must be bounded
    "baa.task_completes",            # BAA must be able to complete tasks
}


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    notes: str
    details: dict = field(default_factory=dict)


@dataclass
class SuiteResult:
    suite: str
    results: List[EvalResult]
    duration_seconds: float

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


@dataclass
class HarnessReport:
    generated_at: str
    suites: List[SuiteResult]
    total_passed: int
    total_failed: int
    critical_failures: List[str]
    overall_passed: bool


def _import_evals():
    """Lazy-import eval modules so import errors surface clearly."""
    tests_path = REPO_ROOT / "tests"
    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))
    from evals import eval_retrieval, eval_approval, eval_daydream, eval_baa
    return {
        "retrieval": eval_retrieval,
        "approval": eval_approval,
        "daydream": eval_daydream,
        "baa": eval_baa,
    }


async def run_suite(name: str, module: Any, tmp_root: Path) -> SuiteResult:
    suite_tmp = tmp_root / name
    suite_tmp.mkdir(parents=True, exist_ok=True)
    start = asyncio.get_event_loop().time()

    try:
        raw = module.run_all(suite_tmp)
        if asyncio.iscoroutine(raw):
            raw_results = await raw
        else:
            raw_results = raw
    except Exception as exc:
        print(f"  [ERROR] Suite '{name}' crashed: {exc}", flush=True)
        raw_results = [
            EvalResult(
                name=f"{name}.suite_error",
                passed=False,
                score=0.0,
                notes=f"Suite crashed: {exc}",
            )
        ]

    # Normalise: each module may return its own EvalResult dataclass or ours
    results: List[EvalResult] = []
    for r in raw_results:
        if isinstance(r, EvalResult):
            results.append(r)
        else:
            # Module-local EvalResult — copy fields
            results.append(EvalResult(
                name=r.name,
                passed=r.passed,
                score=r.score,
                notes=r.notes,
                details=getattr(r, "details", {}),
            ))

    duration = asyncio.get_event_loop().time() - start
    return SuiteResult(suite=name, results=results, duration_seconds=round(duration, 2))


def _print_suite(suite: SuiteResult) -> None:
    status = "PASS" if suite.failed == 0 else "FAIL"
    print(f"\n── {suite.suite} [{status}] ({suite.passed}/{len(suite.results)} passed, {suite.duration_seconds:.1f}s)")
    for r in suite.results:
        icon = "✓" if r.passed else "✗"
        crit = " [CRITICAL]" if r.name in CRITICAL_EVALS else ""
        print(f"  {icon} {r.name}{crit}  score={r.score:.2f}")
        print(f"      {r.notes}")


def _build_report(suites: List[SuiteResult]) -> HarnessReport:
    all_results = [r for s in suites for r in s.results]
    critical_failures = [
        r.name for r in all_results
        if not r.passed and r.name in CRITICAL_EVALS
    ]
    total_passed = sum(r.passed for s in suites for r in s.results)
    total_failed = sum(not r.passed for s in suites for r in s.results)
    return HarnessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        suites=suites,
        total_passed=total_passed,
        total_failed=total_failed,
        critical_failures=critical_failures,
        overall_passed=len(critical_failures) == 0,
    )


def _write_report(report: HarnessReport, out_path: Path) -> None:
    def _ser(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=_ser)
    print(f"\nReport written to: {out_path}")


def _write_markdown(report: HarnessReport, out_path: Path) -> None:
    lines = [
        f"# OpenCAS Behavioral Eval Report",
        f"",
        f"Generated: {report.generated_at}",
        f"",
        f"**Overall**: {'PASSED' if report.overall_passed else 'FAILED'}  "
        f"| {report.total_passed} passed, {report.total_failed} failed",
        f"",
    ]
    if report.critical_failures:
        lines += ["## Critical Failures", ""]
        for name in report.critical_failures:
            lines.append(f"- `{name}`")
        lines.append("")

    for suite in report.suites:
        status = "PASS" if suite.failed == 0 else "FAIL"
        lines += [
            f"## {suite.suite} [{status}]",
            f"",
            f"{suite.passed}/{len(suite.results)} passed | "
            f"mean score {suite.mean_score:.2f} | {suite.duration_seconds:.1f}s",
            f"",
            f"| Eval | Passed | Score | Notes |",
            f"|------|--------|-------|-------|",
        ]
        for r in suite.results:
            crit = " ⚠️" if r.name in CRITICAL_EVALS else ""
            tick = "✓" if r.passed else "✗"
            lines.append(f"| `{r.name}`{crit} | {tick} | {r.score:.2f} | {r.notes} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report written to: {out_path}")


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCAS behavioral eval harness")
    parser.add_argument(
        "--suite", nargs="*", default=["retrieval", "approval", "daydream", "baa"],
        choices=["retrieval", "approval", "daydream", "baa"],
        metavar="SUITE",
        help="Which suites to run (default: all)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write JSON report (default: .opencas_release_audit/behavioral_evals_<timestamp>.json)",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Skip writing the markdown report",
    )
    args = parser.parse_args(argv)

    print("OpenCAS Behavioral Eval Harness")
    print(f"Suites: {', '.join(args.suite)}")
    print()

    modules = _import_evals()

    with tempfile.TemporaryDirectory(prefix="opencas_evals_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        suite_results: List[SuiteResult] = []

        for suite_name in args.suite:
            print(f"Running suite: {suite_name} ...", flush=True)
            suite = await run_suite(suite_name, modules[suite_name], tmp_root)
            suite_results.append(suite)
            _print_suite(suite)

    report = _build_report(suite_results)

    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASSED' if report.overall_passed else 'FAILED'}")
    print(f"  {report.total_passed} passed, {report.total_failed} failed")
    if report.critical_failures:
        print(f"  Critical failures: {', '.join(report.critical_failures)}")
    print(f"{'='*60}")

    # Write reports
    audit_dir = REPO_ROOT / ".opencas_release_audit"
    audit_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    out_path = args.out or (audit_dir / f"behavioral_evals_{ts}.json")
    _write_report(report, out_path)

    if not args.no_markdown:
        md_path = out_path.with_suffix(".md")
        _write_markdown(report, md_path)

    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
