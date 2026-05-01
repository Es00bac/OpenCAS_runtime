"""CLI entry-point for manual affective registry captures.

Usage::

    python -m opencas.affective_registry \
        --registry .opencas/affective-registry.jsonl \
        --phase boot \
        --valence 0.2 \
        --arousal 0.6 \
        --fatigue 0.1 \
        --tension 0.0 \
        --focus 0.8 \
        --energy 0.7 \
        --certainty 0.9

This is useful for smoke-testing the registry path and validating that
historical entries are preserved across runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .models import AffectiveRegistryEntry, AffectiveState, ExecutionContext, ExecutionPhase
from .writer import AffectiveRegistryWriter

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affective_registry",
        description="Append-only affective state registry writer",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(".opencas/affective-registry.jsonl"),
        help="Path to the registry file (default: %(default)s)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="manual",
        choices=[p.value for p in ExecutionPhase],
        help="Execution phase for this entry",
    )
    parser.add_argument("--primary-emotion", type=str, default="neutral")
    parser.add_argument("--valence", type=float, default=0.0)
    parser.add_argument("--arousal", type=float, default=0.5)
    parser.add_argument("--fatigue", type=float, default=0.0)
    parser.add_argument("--tension", type=float, default=0.0)
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--energy", type=float, default=0.5)
    parser.add_argument("--certainty", type=float, default=0.5)
    parser.add_argument("--musubi", type=float, default=None)
    parser.add_argument("--somatic-tag", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--span-id", type=str, default=None)
    parser.add_argument("--trace-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip write validation",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Disable POSIX advisory locking",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print total entry count and exit",
    )
    parser.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help="Print the N most recent entries and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    writer = AffectiveRegistryWriter(
        registry_path=args.registry,
        enable_locking=not args.no_lock,
        validate_writes=not args.no_validate,
    )

    if args.count:
        count = writer.count_entries()
        print(count)
        return 0

    if args.latest:
        entries = writer.get_latest(args.latest)
        for entry in entries:
            print(entry.model_dump_json(indent=2))
        return 0

    affective = AffectiveState(
        primary_emotion=args.primary_emotion,
        valence=args.valence,
        arousal=args.arousal,
        fatigue=args.fatigue,
        tension=args.tension,
        focus=args.focus,
        energy=args.energy,
        certainty=args.certainty,
        musubi=args.musubi,
        somatic_tag=args.somatic_tag,
    )

    ctx = ExecutionContext(
        session_id=args.session_id,
        span_id=args.span_id,
        trace_id=args.trace_id,
        user_id=args.user_id,
    )

    entry = AffectiveRegistryEntry(
        phase=ExecutionPhase(args.phase),
        affective_state=affective,
        execution_context=ctx,
    )

    try:
        writer.append(entry)
    except Exception as exc:
        logger.error("Failed to append entry: %s", exc)
        return 1

    logger.info(
        "Successfully appended entry %s to %s (total entries: %d)",
        entry.entry_id,
        writer.registry_path,
        writer.count_entries(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
