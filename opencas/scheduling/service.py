"""Scheduling service that advances due items into OpenCAS execution."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opencas.execution.models import RepairTask
from opencas.projects.operator_followthrough import has_operator_project_followthrough_authority
from opencas.telemetry import EventKind, Tracer

from .dedupe import (
    attach_dedupe_metadata,
    choose_survivor,
    merge_schedule_details,
    schedules_are_duplicates,
)
from .models import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleStatus,
)
from .store import ScheduleStore


class ScheduleService:
    """Coordinates due schedule detection, execution, and recurrence advancement."""

    def __init__(
        self,
        store: ScheduleStore,
        runtime: Optional[Any] = None,
        tracer: Optional[Tracer] = None,
        default_timezone: str = "America/Denver",
        claimed_run_recovery_seconds: float = 60.0,
        claimed_run_max_recovery_attempts: int = 3,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.tracer = tracer
        self.default_timezone = default_timezone
        self.claimed_run_recovery_seconds = claimed_run_recovery_seconds
        self.claimed_run_max_recovery_attempts = claimed_run_max_recovery_attempts

    async def create_schedule(self, **kwargs: Any) -> ScheduleItem:
        item = ScheduleItem(**kwargs)
        item.next_run_at = item.next_run_at or item.start_at
        await self._copy_claim_id_from_linked_commitment(item)
        duplicate = await self._dedupe_active_schedule(item)
        if duplicate is not None:
            return duplicate
        attach_dedupe_metadata(
            item,
            dedupe_action="created",
            survivor_schedule_id=str(item.schedule_id),
        )
        await self.store.save(item)
        self._trace("schedule_created", {"schedule_id": str(item.schedule_id), "kind": item.kind.value})
        return item

    async def _copy_claim_id_from_linked_commitment(self, item: ScheduleItem) -> None:
        claim_id = str((item.meta or {}).get("claim_id") or "").strip()
        if claim_id or not str(item.commitment_id or "").strip():
            return
        store = None
        if self.runtime is not None:
            store = getattr(self.runtime, "commitment_store", None)
            if store is None:
                store = getattr(getattr(self.runtime, "ctx", None), "commitment_store", None)
        get_commitment = getattr(store, "get", None)
        if not callable(get_commitment):
            return
        try:
            commitment = await get_commitment(str(item.commitment_id))
        except Exception:
            return
        if commitment is None:
            return
        claim_id = str((getattr(commitment, "meta", {}) or {}).get("claim_id") or "").strip()
        if claim_id:
            item.meta = {**dict(item.meta or {}), "claim_id": claim_id}

    async def _dedupe_active_schedule(self, candidate: ScheduleItem) -> ScheduleItem | None:
        active_items = await self.store.list_items(status=ScheduleStatus.ACTIVE, limit=1000)
        matches = [item for item in active_items if schedules_are_duplicates(item, candidate)]
        if not matches:
            return None

        survivor = choose_survivor(matches)
        changed = merge_schedule_details(survivor, candidate)
        duplicate_ids: list[str] = []
        merged_ids: list[str] = []
        for duplicate in matches:
            if duplicate.schedule_id == survivor.schedule_id:
                continue
            if merge_schedule_details(survivor, duplicate):
                changed = True
            duplicate.status = ScheduleStatus.CANCELLED
            duplicate.next_run_at = None
            duplicate_id = str(duplicate.schedule_id)
            duplicate_ids.append(duplicate_id)
            merged_ids.append(duplicate_id)
            attach_dedupe_metadata(
                duplicate,
                dedupe_action="cancelled_duplicate",
                survivor_schedule_id=str(survivor.schedule_id),
                merged_schedule_ids=[],
                duplicate_schedule_ids=[],
            )
            await self.store.save(duplicate)

        attach_dedupe_metadata(
            survivor,
            dedupe_action="merged" if changed or merged_ids else "reused",
            survivor_schedule_id=str(survivor.schedule_id),
            merged_schedule_ids=merged_ids,
            duplicate_schedule_ids=duplicate_ids,
        )
        await self.store.save(survivor)
        self._trace(
            "schedule_deduped",
            {
                "schedule_id": str(survivor.schedule_id),
                "dedupe_action": survivor.meta.get("dedupe_action"),
                "duplicate_schedule_ids": duplicate_ids,
            },
        )
        return survivor

    async def process_due(self, now: Optional[datetime] = None, limit: int = 50) -> Dict[str, Any]:
        now = self._as_utc(now or datetime.now(timezone.utc))
        recovered = await self._recover_claimed_runs(now=now, limit=limit)
        due = await self.store.list_due(now, limit=limit)
        processed = 0
        submitted = recovered["submitted"]
        recorded = 0
        skipped = 0
        failed = recovered["failed"]
        for item in due:
            try:
                run = await self.trigger(item, now=now, manual=False)
                processed += 1
                if run.status == ScheduleRunStatus.SUBMITTED:
                    submitted += 1
                elif run.status == ScheduleRunStatus.RECORDED:
                    recorded += 1
                elif run.status == ScheduleRunStatus.SKIPPED:
                    skipped += 1
                elif run.status == ScheduleRunStatus.FAILED:
                    failed += 1
            except Exception as exc:
                failed += 1
                self._trace("schedule_process_error", {"schedule_id": str(item.schedule_id), "error": str(exc)})
        return {
            "processed": processed,
            "submitted": submitted,
            "recorded": recorded,
            "skipped": skipped,
            "failed": failed,
        }

    async def temporal_agenda(
        self,
        now: Optional[datetime] = None,
        *,
        horizon_hours: float = 24.0,
        upcoming_limit: int = 8,
        recent_limit: int = 8,
    ) -> Dict[str, Any]:
        """Return the agent-facing temporal agenda from durable schedule state."""
        now = self._as_utc(now or datetime.now(timezone.utc))
        horizon_hours = max(0.25, float(horizon_hours))
        upcoming_limit = max(1, int(upcoming_limit))
        recent_limit = max(1, int(recent_limit))
        horizon_end = now + timedelta(hours=horizon_hours)

        active_items = await self.store.list_items(status=ScheduleStatus.ACTIVE, limit=1000)
        due_items = [
            item
            for item in active_items
            if item.next_run_at is not None and self._as_utc(item.next_run_at) <= now
        ]
        future_items = [
            item
            for item in active_items
            if item.next_run_at is not None and now < self._as_utc(item.next_run_at) <= horizon_end
        ]
        due_items.sort(key=lambda item: (item.next_run_at or item.start_at, -item.priority))
        future_items.sort(key=lambda item: (item.next_run_at or item.start_at, -item.priority))
        recent_runs = await self.store.list_runs(limit=recent_limit)

        due_payload = [self._agenda_item_payload(item, now) for item in due_items[:upcoming_limit]]
        upcoming_payload = [
            self._agenda_item_payload(item, now) for item in future_items[:upcoming_limit]
        ]
        runs_payload = [await self._agenda_run_payload(run) for run in recent_runs]
        next_item = due_payload[0] if due_payload else upcoming_payload[0] if upcoming_payload else None
        counts = {
            "active": len(active_items),
            "due_now": len(due_items),
            "upcoming": len(future_items),
            "recent_runs": len(runs_payload),
            "tasks": sum(1 for item in active_items if item.kind == ScheduleKind.TASK),
            "events": sum(1 for item in active_items if item.kind == ScheduleKind.EVENT),
        }
        return {
            "now": now.isoformat(),
            "horizon_hours": horizon_hours,
            "horizon_end": horizon_end.isoformat(),
            "timezone": self.default_timezone,
            "counts": counts,
            "due_now": due_payload,
            "upcoming": upcoming_payload,
            "recent_runs": runs_payload,
            "next": next_item,
            "summary": self._agenda_summary(counts, next_item),
        }

    async def trigger(
        self,
        item_or_id: ScheduleItem | str,
        now: Optional[datetime] = None,
        manual: bool = True,
    ) -> ScheduleRun:
        now = self._as_utc(now or datetime.now(timezone.utc))
        item = item_or_id if isinstance(item_or_id, ScheduleItem) else await self.store.get(str(item_or_id))
        if item is None:
            raise ValueError("schedule not found")
        scheduled_for = item.next_run_at or item.start_at
        if manual:
            scheduled_for = now
        elif scheduled_for <= now:
            scheduled_for = self.latest_due_occurrence(item, now) or scheduled_for
        run = ScheduleRun(
            schedule_id=item.schedule_id,
            scheduled_for=scheduled_for,
            started_at=now,
            status=ScheduleRunStatus.RECORDED,
            meta={
                "manual": manual,
                "action": item.action.value,
                **(
                    {"claim_id": str(item.meta.get("claim_id"))}
                    if str((item.meta or {}).get("claim_id") or "").strip()
                    else {}
                ),
            },
        )
        if await self._linked_commitment_finished(item):
            run.status = ScheduleRunStatus.SKIPPED
            run.finished_at = datetime.now(timezone.utc)
            run.meta["skip_reason"] = "linked_commitment_finished"
            item.status = ScheduleStatus.COMPLETED
            item.next_run_at = None
            await self.store.record_run(run)
            await self.store.save(item)
            self._trace(
                "schedule_triggered",
                {
                    "schedule_id": str(item.schedule_id),
                    "run_id": str(run.run_id),
                    "status": run.status.value,
                    "task_id": run.task_id,
                    "skip_reason": run.meta["skip_reason"],
                },
            )
            return run
        if (
            item.action == ScheduleAction.SUBMIT_BAA
            and self.runtime is not None
            and getattr(self.runtime, "baa", None)
        ):
            return await self._trigger_baa(
                item=item,
                run=run,
                scheduled_for=scheduled_for,
                now=now,
                manual=manual,
            )
        try:
            if item.action == ScheduleAction.SUBMIT_BAA:
                if self.runtime is None or not getattr(self.runtime, "baa", None):
                    raise RuntimeError("runtime BAA is not available")
            elif item.action == ScheduleAction.GMAIL_ALERT:
                alert_result = await self._run_gmail_alert(item, now=now)
                run.meta["gmail_alert"] = alert_result
                if alert_result.get("skipped"):
                    run.status = ScheduleRunStatus.SKIPPED
                    skip_reason = str(alert_result.get("skip_reason") or "").strip()
                    if skip_reason:
                        run.meta["skip_reason"] = skip_reason
                elif not alert_result.get("success"):
                    raise RuntimeError(str(alert_result.get("error") or "gmail alert check failed"))
                else:
                    run.status = ScheduleRunStatus.RECORDED
            else:
                run.status = ScheduleRunStatus.RECORDED
            run.finished_at = datetime.now(timezone.utc)
        except Exception as exc:
            run.status = ScheduleRunStatus.FAILED
            run.error = str(exc)
            run.finished_at = datetime.now(timezone.utc)
        await self.store.record_run(run)
        if not manual:
            if run.status == ScheduleRunStatus.FAILED:
                await self._defer_failed_item(item, run=run, now=now)
                await self._notify_owner_schedule_failure(item, run=run)
            elif run.status == ScheduleRunStatus.SKIPPED and self._run_auth_failure(run) is not None:
                # Auth preflight skips update the item's next probe time before returning.
                pass
            else:
                await self._advance_item(item, scheduled_for=scheduled_for, now=now)
        self._trace(
            "schedule_triggered",
            {
                "schedule_id": str(item.schedule_id),
                "run_id": str(run.run_id),
                "status": run.status.value,
                "task_id": run.task_id,
            },
        )
        return run

    def _executive_pause_payload(self) -> Optional[Dict[str, str]]:
        executive = getattr(self.runtime, "executive", None) if self.runtime is not None else None
        if executive is None:
            return None
        try:
            paused = bool(executive.recommend_pause())
        except Exception:
            return None
        if not paused:
            return None
        reason = "executive_pause"
        pause_reason = getattr(executive, "pause_reason", None)
        if callable(pause_reason):
            try:
                reason = str(pause_reason() or reason)
            except Exception:
                reason = "executive_pause"
        return {"skip_reason": "executive_recommended_pause", "pause_reason": reason}

    @staticmethod
    def _can_bypass_executive_pause(item: ScheduleItem, pause_payload: Dict[str, str]) -> bool:
        """Let operator-grounded project returns reach BAA's overload recovery gate."""

        pause_reason = str(pause_payload.get("pause_reason") or "").strip().lower()
        if pause_reason != "overload":
            return False
        return ScheduleService._has_operator_project_return_evidence(item)

    @staticmethod
    def _has_operator_project_return_evidence(item: ScheduleItem) -> bool:
        return has_operator_project_followthrough_authority(
            item.meta if isinstance(item.meta, dict) else {},
            tags=item.tags or (),
        )

    async def _trigger_baa(
        self,
        *,
        item: ScheduleItem,
        run: ScheduleRun,
        scheduled_for: datetime,
        now: datetime,
        manual: bool,
    ) -> ScheduleRun:
        task = await self._build_baa_task(item=item, scheduled_for=scheduled_for, manual=manual)
        run.status = ScheduleRunStatus.CLAIMED
        run.meta = {
            **dict(run.meta or {}),
            "task_objective": task.objective,
            "task_meta": dict(task.meta or {}),
        }
        await self.store.record_run(run)
        if not manual:
            await self._advance_item(item, scheduled_for=scheduled_for, now=now)
        await self.runtime.baa.submit(task)
        run.task_id = str(task.task_id)
        run.status = ScheduleRunStatus.SUBMITTED
        run.finished_at = datetime.now(timezone.utc)
        await self.store.record_run(run)
        self._trace(
            "schedule_triggered",
            {
                "schedule_id": str(item.schedule_id),
                "run_id": str(run.run_id),
                "status": run.status.value,
                "task_id": run.task_id,
            },
        )
        return run

    async def _build_baa_task(
        self,
        *,
        item: ScheduleItem,
        scheduled_for: datetime,
        manual: bool,
    ) -> RepairTask:
        truth_meta = await self._truth_context_metadata(
            item=item,
            scheduled_for=scheduled_for,
            manual=manual,
        )
        return RepairTask(
            objective=item.objective or item.title,
            commitment_id=item.commitment_id,
            project_id=item.plan_id,
            meta={
                **(item.meta or {}),
                "source": "schedule",
                "schedule_id": str(item.schedule_id),
                "schedule_title": item.title,
                "scheduled_for": scheduled_for.isoformat(),
                "manual": manual,
                "recurrence": item.recurrence.value,
                **truth_meta,
            },
        )

    async def _recover_claimed_runs(self, *, now: datetime, limit: int) -> Dict[str, int]:
        cutoff = now - timedelta(seconds=self.claimed_run_recovery_seconds)
        runs = await self.store.list_runs(limit=1000)
        claimed = [
            run
            for run in runs
            if run.status == ScheduleRunStatus.CLAIMED and self._as_utc(run.started_at) <= cutoff
        ][:limit]
        submitted = 0
        failed = 0
        for run in claimed:
            item = await self.store.get(str(run.schedule_id))
            if item is None:
                run.status = ScheduleRunStatus.FAILED
                run.error = "schedule missing for claimed run recovery"
                run.finished_at = now
                await self.store.record_run(run)
                failed += 1
                continue
            attempts = int((run.meta or {}).get("recovery_attempts") or 0) + 1
            run.meta = {**dict(run.meta or {}), "recovery_attempts": attempts}
            try:
                if self.runtime is None or not getattr(self.runtime, "baa", None):
                    raise RuntimeError("runtime BAA is not available")
                task = await self._build_baa_task(
                    item=item,
                    scheduled_for=run.scheduled_for,
                    manual=bool((run.meta or {}).get("manual", False)),
                )
                await self.runtime.baa.submit(task)
                run.task_id = str(task.task_id)
                run.status = ScheduleRunStatus.SUBMITTED
                run.error = None
                run.finished_at = now
                await self.store.record_run(run)
                submitted += 1
            except Exception as exc:
                run.meta["last_recovery_error"] = str(exc)
                if attempts >= self.claimed_run_max_recovery_attempts:
                    run.status = ScheduleRunStatus.FAILED
                    run.error = str(exc)
                    run.finished_at = now
                await self.store.record_run(run)
                failed += 1
        return {"submitted": submitted, "failed": failed}

    async def _truth_context_metadata(
        self,
        *,
        item: ScheduleItem,
        scheduled_for: datetime,
        manual: bool,
    ) -> Dict[str, Any]:
        """Build compact truth metadata for schedule-created BAA tasks."""

        metadata: Dict[str, Any] = {
            "origin_context_lane": "executive",
            "authority": "schedule_due",
        }
        accepted_proposals = (item.meta or {}).get("accepted_proposal_ids")
        if isinstance(accepted_proposals, list):
            metadata["accepted_proposal_ids"] = [str(value) for value in accepted_proposals]

        runtime = self.runtime
        arbiter = getattr(runtime, "truth_arbiter", None) if runtime is not None else None
        if arbiter is None and runtime is not None:
            arbiter = getattr(getattr(runtime, "ctx", None), "truth_arbiter", None)
        issue_snapshot = getattr(arbiter, "issue_snapshot", None)
        if not callable(issue_snapshot):
            return metadata

        reason = (
            f"schedule_due:{item.schedule_id}:"
            f"{scheduled_for.isoformat()}:manual={manual}"
        )
        try:
            snapshot = issue_snapshot(reason=reason)
            if inspect.isawaitable(snapshot):
                snapshot = await snapshot
        except Exception as exc:
            metadata["context_truth_error"] = str(exc)
            return metadata

        snapshot_id = str(getattr(snapshot, "snapshot_id", "") or "")
        if snapshot_id:
            metadata["context_truth_snapshot_id"] = snapshot_id
            metadata["source_snapshot_id"] = snapshot_id
        epoch = getattr(snapshot, "epoch", None)
        if epoch is not None:
            metadata["context_truth_epoch"] = int(epoch)
        return metadata

    async def _linked_commitment_finished(self, item: ScheduleItem) -> bool:
        commitment_id = str(item.commitment_id or "").strip()
        if not commitment_id or self.runtime is None:
            return False
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            store = getattr(getattr(self.runtime, "ctx", None), "commitment_store", None)
        if store is None:
            executive = getattr(getattr(self.runtime, "ctx", None), "executive", None)
            store = getattr(executive, "commitment_store", None)
        if store is None:
            return False
        try:
            commitment = await store.get(commitment_id)
        except Exception:
            return False
        if commitment is None:
            return False
        status = str(getattr(getattr(commitment, "status", None), "value", getattr(commitment, "status", "")))
        return status in {"completed", "abandoned"}

    async def _run_gmail_alert(self, item: ScheduleItem, *, now: datetime) -> Dict[str, Any]:
        """Check Gmail through the runtime tool surface and notify once for new matches."""
        if self.runtime is None:
            return {"success": False, "error": "runtime is not available"}
        execute_tool = getattr(self.runtime, "execute_tool", None)
        if not callable(execute_tool):
            return {"success": False, "error": "runtime tool execution is not available"}

        meta = dict(item.meta or {})
        query = str(meta.get("gmail_query") or meta.get("query") or "in:inbox").strip() or "in:inbox"
        max_results = max(1, min(int(meta.get("max_results", 10)), 25))
        include_snippet = bool(meta.get("include_snippet", False))
        preflight = await self._preflight_google_workspace_auth_recovery(
            item,
            meta=meta,
            query=query,
            now=now,
        )
        if preflight is not None:
            return preflight
        meta = dict(item.meta or {})
        args = {
            "query": query,
            "max_results": max_results,
            "include_snippet": include_snippet,
        }
        result = execute_tool(
            "google_workspace_gmail_headlines",
            args,
            session_id=f"schedule:{item.schedule_id}",
        )
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            return {"success": False, "error": "gmail headline tool returned a non-object result"}
        if not result.get("success"):
            error = str(result.get("output") or result.get("error") or "gmail headline tool failed")
            auth_failure = await self._diagnose_google_workspace_auth_failure(
                item,
                error=error,
                tool_result=result,
            )
            if auth_failure is not None:
                return {
                    "success": False,
                    "error": error,
                    "query": query,
                    "auth_failure": auth_failure,
                }
            return {
                "success": False,
                "error": error,
                "query": query,
            }

        try:
            payload = json.loads(str(result.get("output") or "{}"))
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"gmail headline output was not JSON: {exc}", "query": query}

        messages = [msg for msg in list(payload.get("messages") or []) if isinstance(msg, dict)]
        seen_records = self._gmail_seen_records(meta.get("seen_message_ids"), now=now)
        returned_ids: List[str] = []
        messages_to_deliver: List[Dict[str, Any]] = []
        for message in messages:
            message_id = str(message.get("id") or "").strip()
            if not message_id:
                continue
            returned_ids.append(message_id)
            if seen_records.get(message_id, {}).get("persisted_at"):
                continue
            messages_to_deliver.append(message)

        contact_result: Any = None
        if messages_to_deliver:
            contact_result = await self._send_gmail_alert_contact(item, messages_to_deliver)
            if not self._contact_result_sent(contact_result):
                return {
                    "success": False,
                    "error": self._contact_failure_reason(contact_result),
                    "query": query,
                    "result_count": len(messages),
                    "new_count": len(messages_to_deliver),
                    "returned_message_ids": returned_ids,
                    "contact_result": contact_result,
                }

        delivered_ids: List[str] = []
        delivered_at = now.isoformat()
        for message in messages_to_deliver:
            message_id = str(message.get("id") or "").strip()
            if not message_id:
                continue
            entry = dict(seen_records.get(message_id) or {})
            entry["delivered_at"] = delivered_at
            entry.setdefault("persisted_at", None)
            seen_records[message_id] = entry
            delivered_ids.append(message_id)
        meta.update(
            {
                "gmail_query": query,
                "max_results": max_results,
                "last_checked_at": now.isoformat(),
                "last_result_count": len(messages),
                "last_new_count": len(messages_to_deliver),
                "seen_message_ids": self._trim_gmail_seen_records(seen_records),
            }
        )
        if contact_result is not None:
            meta["last_contact_result"] = contact_result
        item.meta = meta
        await self.store.save(item)
        if delivered_ids:
            persisted_at = datetime.now(timezone.utc).isoformat()
            seen_records = self._gmail_seen_records(item.meta.get("seen_message_ids"), now=now)
            for message_id in delivered_ids:
                entry = dict(seen_records.get(message_id) or {})
                entry.setdefault("delivered_at", delivered_at)
                entry["persisted_at"] = persisted_at
                seen_records[message_id] = entry
            item.meta = {
                **dict(item.meta or {}),
                "seen_message_ids": self._trim_gmail_seen_records(seen_records),
            }
            await self.store.save(item)
        return {
            "success": True,
            "query": query,
            "result_count": len(messages),
            "new_count": len(messages_to_deliver),
            "alert_sent": bool(messages_to_deliver),
            "returned_message_ids": returned_ids,
            "new_message_ids": [str(message.get("id")) for message in messages_to_deliver],
        }

    async def _send_gmail_alert_contact(
        self,
        item: ScheduleItem,
        messages: List[Dict[str, Any]],
    ) -> Any:
        contact_owner = getattr(self.runtime, "initiative_contact_owner", None)
        if not callable(contact_owner):
            service = getattr(self.runtime, "initiative_contact", None)
            contact_owner = getattr(service, "request_contact", None)
        if not callable(contact_owner):
            return {"status": "failed", "error": "owner contact is not available", "channel": "telegram"}

        meta = dict(item.meta or {})
        alert_name = str(meta.get("alert_name") or item.title or "Gmail alert").strip()
        channel = str(meta.get("alert_channel") or "telegram").strip() or "telegram"
        urgency = str(meta.get("alert_urgency") or "high").strip() or "high"
        lines = [
            f"{alert_name}: {len(messages)} new matching email"
            f"{'' if len(messages) == 1 else 's'}."
        ]
        for message in messages[:10]:
            subject = str(message.get("subject") or "(no subject)").strip()
            sender = str(message.get("from") or "(unknown sender)").strip()
            date = str(message.get("date") or "(unknown date)").strip()
            lines.append(f"- {date} | {sender} | {subject}")
        if len(messages) > 10:
            lines.append(f"- Plus {len(messages) - 10} more matching emails.")

        result = contact_owner(
            message="\n".join(lines),
            reason=str(meta.get("alert_reason") or "gmail_alert_match"),
            urgency=urgency,
            channel=channel,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _preflight_google_workspace_auth_recovery(
        self,
        item: ScheduleItem,
        *,
        meta: Dict[str, Any],
        query: str,
        now: datetime,
    ) -> Dict[str, Any] | None:
        existing_auth = dict(meta.get("auth_failure") or {})
        if existing_auth.get("kind") != "google_workspace_auth":
            return None

        status_result = await self._google_workspace_auth_status(item)
        auth_status = self._compact_auth_status(status_result)
        if self._google_workspace_auth_status_recovered(auth_status):
            item.meta = self._clear_google_workspace_auth_failure(meta, now=now, auth_status=auth_status)
            await self.store.save(item)
            return None

        next_probe_at = now + self._google_workspace_auth_probe_delay(existing_auth)
        failure_count = int(existing_auth.get("failure_count") or 0) + 1
        notification_count = int(existing_auth.get("notification_count") or 0)
        preflight_count = int(existing_auth.get("preflight_count") or 0) + 1
        first_seen_at = str(existing_auth.get("first_seen_at") or now.isoformat())
        auth_failure = {
            **existing_auth,
            "kind": "google_workspace_auth",
            "reason": str(existing_auth.get("reason") or "auth_error"),
            "first_seen_at": first_seen_at,
            "last_seen_at": now.isoformat(),
            "failure_count": failure_count,
            "preflight_count": preflight_count,
            "notification_count": notification_count,
            "reauth_required": True,
            "recovery_strategy": "auth_status_preflight",
            "last_probe_at": now.isoformat(),
            "next_retry_at": next_probe_at.isoformat(),
            "notification_suppressed": notification_count > 0,
            "auth_status": auth_status,
        }
        item.status = ScheduleStatus.ACTIVE
        item.last_run_at = now
        item.next_run_at = next_probe_at
        item.meta = {
            **meta,
            "last_failure_at": now.isoformat(),
            "last_failure_error": "Google Workspace authentication still requires reauthorization",
            "auth_failure": auth_failure,
        }
        await self.store.save(item)
        return {
            "success": False,
            "skipped": True,
            "skip_reason": "google_workspace_auth_unresolved",
            "error": "Google Workspace authentication still requires reauthorization; skipped Gmail call.",
            "query": query,
            "auth_failure": auth_failure,
        }

    async def _google_workspace_auth_status(self, item: ScheduleItem) -> Any:
        execute_tool = getattr(self.runtime, "execute_tool", None) if self.runtime is not None else None
        if not callable(execute_tool):
            return {"success": False, "error": "runtime tool execution is not available"}
        try:
            status_result = execute_tool(
                "google_workspace_auth_status",
                {"timeout_seconds": 30},
                session_id=f"schedule:{item.schedule_id}:auth",
            )
            if inspect.isawaitable(status_result):
                status_result = await status_result
            return status_result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _clear_google_workspace_auth_failure(
        meta: Dict[str, Any],
        *,
        now: datetime,
        auth_status: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        cleared = dict(meta)
        for key in ("auth_failure", "last_failure_at", "last_failure_error", "last_failed_run_id"):
            cleared.pop(key, None)
        cleared["auth_recovered_at"] = now.isoformat()
        if auth_status is not None:
            cleared["auth_recovery_status"] = auth_status
        return cleared

    @staticmethod
    def _google_workspace_auth_probe_delay(auth_failure: Dict[str, Any]) -> timedelta:
        return timedelta(hours=float(auth_failure.get("probe_interval_hours") or 6))

    @classmethod
    def _google_workspace_auth_status_recovered(cls, auth_status: Dict[str, Any] | None) -> bool:
        if not isinstance(auth_status, dict) or not bool(auth_status.get("success")):
            return False
        text = json.dumps(auth_status, default=str).lower()
        if cls._google_workspace_auth_failure_payload(text) is not None:
            return False
        broken_tokens = (
            "not authenticated",
            "unauthenticated",
            "login required",
            "no credential",
            "no credentials",
            "missing credential",
            "missing credentials",
            "not configured",
            "failed to get token",
        )
        return not any(token in text for token in broken_tokens)

    @staticmethod
    def _contact_result_sent(contact_result: Any) -> bool:
        if not isinstance(contact_result, dict):
            return False
        if contact_result.get("status") == "sent":
            return True
        if contact_result.get("sent") is True:
            return True
        dispatch = contact_result.get("dispatch")
        return isinstance(dispatch, dict) and dispatch.get("sent") is True

    @staticmethod
    def _contact_failure_reason(contact_result: Any) -> str:
        if not isinstance(contact_result, dict):
            return "owner contact failed"
        return str(
            contact_result.get("error")
            or contact_result.get("reason")
            or contact_result.get("status")
            or "owner contact failed"
        )

    async def _diagnose_google_workspace_auth_failure(
        self,
        item: ScheduleItem,
        *,
        error: str,
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        parsed = self._google_workspace_auth_failure_payload(error, tool_result.get("metadata"))
        if parsed is None:
            return None
        status_result: Any = None
        execute_tool = getattr(self.runtime, "execute_tool", None) if self.runtime is not None else None
        if callable(execute_tool):
            try:
                status_result = execute_tool(
                    "google_workspace_auth_status",
                    {"timeout_seconds": 30},
                    session_id=f"schedule:{item.schedule_id}:auth",
                )
                if inspect.isawaitable(status_result):
                    status_result = await status_result
            except Exception as exc:
                status_result = {"success": False, "error": str(exc)}
        return {
            **parsed,
            "tool": "google_workspace_gmail_headlines",
            "repair_hint": (
                "Google Workspace OAuth needs re-authentication. Run `gws auth login` "
                "or the project-approved Google Workspace auth setup, then resume the schedule."
            ),
            "auth_status": self._compact_auth_status(status_result),
        }

    @staticmethod
    def _google_workspace_auth_failure_payload(
        error: str,
        metadata: Any = None,
    ) -> Dict[str, Any] | None:
        text = str(error or "")
        lowered = text.lower()
        meta = metadata if isinstance(metadata, dict) else {}
        if not (
            meta.get("auth_error")
            or "invalid_grant" in lowered
            or "expired or revoked" in lowered
            or "authentication failed" in lowered
            or '"reason": "autherror"' in lowered
            or '"reason":"autherror"' in lowered
        ):
            return None
        reason = "auth_error"
        if "invalid_grant" in lowered:
            reason = "invalid_grant"
        elif "expired or revoked" in lowered:
            reason = "expired_or_revoked"
        code: Any = 401 if "401" in lowered else None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    code = err.get("code", code)
                    message = str(err.get("message") or text)
                    reason = "invalid_grant" if "invalid_grant" in message.lower() else str(err.get("reason") or reason)
                    return {
                        "kind": "google_workspace_auth",
                        "reason": reason,
                        "code": code,
                        "message": message,
                    }
        except json.JSONDecodeError:
            pass
        return {
            "kind": "google_workspace_auth",
            "reason": reason,
            "code": code,
            "message": text[:1000],
        }

    @staticmethod
    def _compact_auth_status(status_result: Any) -> Dict[str, Any] | None:
        if status_result is None:
            return None
        if not isinstance(status_result, dict):
            return {"success": False, "output": str(status_result)[:1000]}
        return {
            "success": bool(status_result.get("success")),
            "output": str(status_result.get("output") or status_result.get("error") or "")[:1000],
            "metadata": {
                key: value
                for key, value in dict(status_result.get("metadata") or {}).items()
                if key in {"auth_error", "missing_command", "exit_code", "command", "resolved_command"}
            },
        }

    @staticmethod
    def _gmail_seen_records(raw: Any, *, now: datetime) -> Dict[str, Dict[str, Any]]:
        migrated_at = now.isoformat()
        records: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                message_id = str(key).strip()
                if not message_id:
                    continue
                entry = dict(value) if isinstance(value, dict) else {}
                records[message_id] = {
                    **entry,
                    "delivered_at": entry.get("delivered_at") or None,
                    "persisted_at": entry.get("persisted_at") or None,
                }
            return records

        if raw is None:
            values: List[Any] = []
        elif isinstance(raw, list):
            values = raw
        elif isinstance(raw, tuple):
            values = list(raw)
        else:
            values = [raw]

        for value in values:
            message_id = str(value).strip()
            if message_id:
                records[message_id] = {"delivered_at": None, "persisted_at": migrated_at}
        return records

    @staticmethod
    def _trim_gmail_seen_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return dict(list(records.items())[-500:])

    async def _advance_item(self, item: ScheduleItem, scheduled_for: datetime, now: datetime) -> None:
        item.last_run_at = now
        item.occurrence_count += 1
        next_run = self.compute_next_run(item, after=scheduled_for, now=now)
        if next_run is None:
            item.next_run_at = None
            item.status = ScheduleStatus.COMPLETED
        else:
            item.next_run_at = next_run
        await self.store.save(item)

    async def _defer_failed_item(self, item: ScheduleItem, *, run: ScheduleRun, now: datetime) -> None:
        """Keep failed schedule submissions alive so a transient failure cannot erase work."""
        auth_failure = self._run_auth_failure(run)
        if auth_failure is not None:
            item_meta = dict(item.meta or {})
            existing_auth = dict(item_meta.get("auth_failure") or {})
            failure_count = int(existing_auth.get("failure_count") or 0) + 1
            notification_count = int(existing_auth.get("notification_count") or 0)
            previous_auth_failure = self._google_workspace_auth_failure_payload(
                str(item_meta.get("last_failure_error") or "")
            )
            if notification_count == 0 and previous_auth_failure is not None:
                notification_count = 1
            first_seen_at = str(existing_auth.get("first_seen_at") or now.isoformat())
            item.status = ScheduleStatus.ACTIVE
            item.last_run_at = now
            item.next_run_at = now + timedelta(hours=6)
            item.meta = {
                **item_meta,
                "last_failure_at": now.isoformat(),
                "last_failure_error": run.error or "schedule trigger failed",
                "last_failed_run_id": str(run.run_id),
                "auth_failure": {
                    **auth_failure,
                    "first_seen_at": first_seen_at,
                    "last_seen_at": now.isoformat(),
                    "failure_count": failure_count,
                    "preflight_count": int(existing_auth.get("preflight_count") or 0),
                    "notification_count": notification_count,
                    "reauth_required": True,
                    "recovery_strategy": "auth_status_preflight",
                    "next_retry_at": item.next_run_at.isoformat(),
                    "notification_suppressed": notification_count > 0,
                },
            }
            await self.store.save(item)
            return
        item.status = ScheduleStatus.ACTIVE
        item.last_run_at = now
        item.next_run_at = now + timedelta(minutes=5)
        item.meta = {
            **(item.meta or {}),
            "last_failure_at": now.isoformat(),
            "last_failure_error": run.error or "schedule trigger failed",
            "last_failed_run_id": str(run.run_id),
        }
        await self.store.save(item)

    async def _notify_owner_schedule_failure(self, item: ScheduleItem, *, run: ScheduleRun) -> None:
        if self.runtime is None:
            return
        contact_owner = getattr(self.runtime, "initiative_contact_owner", None)
        if not callable(contact_owner):
            service = getattr(self.runtime, "initiative_contact", None)
            contact_owner = getattr(service, "request_contact", None)
        if not callable(contact_owner):
            return
        stored_auth_failure = dict((item.meta or {}).get("auth_failure") or {})
        auth_failure = {**stored_auth_failure, **(self._run_auth_failure(run) or {})}
        if auth_failure.get("kind") == "google_workspace_auth":
            if int(auth_failure.get("notification_count") or 0) > 0:
                return
            message = self._auth_failure_owner_message(item, run=run, auth_failure=auth_failure)
            try:
                result = contact_owner(
                    message=message,
                    reason="schedule_auth_failed",
                    urgency="medium",
                    channel="telegram",
                )
                if inspect.isawaitable(result):
                    await result
                stored_auth = dict((item.meta or {}).get("auth_failure") or {})
                stored_auth["notification_count"] = int(stored_auth.get("notification_count") or 0) + 1
                stored_auth["last_notified_at"] = datetime.now(timezone.utc).isoformat()
                item.meta = {**dict(item.meta or {}), "auth_failure": stored_auth}
                await self.store.save(item)
            except Exception as exc:
                self._trace(
                    "schedule_failure_contact_error",
                    {"schedule_id": str(item.schedule_id), "error": str(exc), "failure_class": "auth"},
                )
            return
        project_title = str((item.meta or {}).get("project_title") or "").strip()
        lines = ["OpenCAS could not start scheduled work."]
        if project_title:
            lines.append(f"Project: {project_title}")
        lines.append(f"Schedule: {item.title}")
        lines.append(f"Error: {run.error or 'schedule trigger failed'}")
        if item.next_run_at:
            lines.append(f"Retry scheduled: {item.next_run_at.isoformat()}")
        try:
            result = contact_owner(
                message="\n".join(lines),
                reason="schedule_submission_failed",
                urgency="high",
                channel="telegram",
            )
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._trace(
                "schedule_failure_contact_error",
                {"schedule_id": str(item.schedule_id), "error": str(exc)},
            )

    @staticmethod
    def _run_auth_failure(run: ScheduleRun) -> Dict[str, Any] | None:
        gmail_alert = (run.meta or {}).get("gmail_alert")
        if isinstance(gmail_alert, dict):
            auth_failure = gmail_alert.get("auth_failure")
            if isinstance(auth_failure, dict):
                return dict(auth_failure)
        return ScheduleService._google_workspace_auth_failure_payload(run.error or "")

    @staticmethod
    def _auth_failure_owner_message(
        item: ScheduleItem,
        *,
        run: ScheduleRun,
        auth_failure: Dict[str, Any],
    ) -> str:
        lines = [
            "OpenCAS put a scheduled Gmail check into degraded-auth recovery because Google Workspace authentication failed.",
            f"Schedule: {item.title}",
        ]
        reason = str(auth_failure.get("reason") or "auth_error")
        if reason:
            lines.append(f"Reason: {reason}")
        message = str(auth_failure.get("message") or run.error or "").strip()
        if message:
            lines.append(f"Auth error: {message[:500]}")
        repair_hint = str(auth_failure.get("repair_hint") or "").strip()
        if repair_hint:
            lines.append(f"Repair: {repair_hint}")
        if item.next_run_at:
            lines.append(f"Next auth-status probe: {item.next_run_at.isoformat()}")
        lines.append("Repeated notifications for this auth incident will be suppressed.")
        lines.append("After re-authentication, OpenCAS will resume the Gmail check from the next probe.")
        return "\n".join(lines)

    def compute_next_run(
        self,
        item: ScheduleItem,
        after: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> Optional[datetime]:
        now = self._as_utc(now or datetime.now(timezone.utc))
        after = self._as_utc(after or item.next_run_at or item.start_at)
        if item.max_occurrences is not None and item.occurrence_count >= item.max_occurrences:
            return None
        if item.recurrence == ScheduleRecurrence.NONE:
            return None

        candidate = self._next_candidate(item, after)
        if candidate is None:
            return None
        if item.end_at is not None and candidate > item.end_at:
            return None

        # Catch-up policy: run once if late, then continue with the next future occurrence.
        guard = 0
        while candidate <= now and guard < 1000:
            candidate = self._next_candidate(item, candidate)
            if candidate is None:
                return None
            if item.end_at is not None and candidate > item.end_at:
                return None
            guard += 1
        return candidate

    def occurrences_between(
        self,
        item: ScheduleItem,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> List[datetime]:
        start = self._as_utc(start)
        end = self._as_utc(end)
        if end < start:
            return []
        occurrences: List[datetime] = []
        candidate = item.start_at
        guard = 0
        while candidate and candidate <= end and len(occurrences) < limit and guard < limit * 4:
            if candidate >= start and (item.end_at is None or candidate <= item.end_at):
                occurrences.append(candidate)
            candidate = self._next_candidate(item, candidate)
            if candidate is None:
                break
            guard += 1
        return occurrences

    def latest_due_occurrence(self, item: ScheduleItem, now: datetime) -> Optional[datetime]:
        now = self._as_utc(now)
        candidate = item.next_run_at or item.start_at
        if candidate > now:
            return None
        latest = candidate
        guard = 0
        while guard < 1000:
            next_candidate = self._next_candidate(item, latest)
            if next_candidate is None or next_candidate > now:
                return latest
            if item.end_at is not None and next_candidate > item.end_at:
                return latest
            latest = next_candidate
            guard += 1
        return latest

    def _next_candidate(self, item: ScheduleItem, after: datetime) -> Optional[datetime]:
        after = self._as_utc(after)
        tz = self._zone(item.timezone)
        local_after = after.astimezone(tz)
        if item.recurrence == ScheduleRecurrence.INTERVAL_HOURS:
            return after + timedelta(hours=float(item.interval_hours or 1))
        if item.recurrence == ScheduleRecurrence.DAILY:
            return (local_after + timedelta(days=1)).astimezone(timezone.utc)
        if item.recurrence == ScheduleRecurrence.WEEKDAYS:
            weekdays = [0, 1, 2, 3, 4]
        elif item.recurrence == ScheduleRecurrence.WEEKLY:
            weekdays = item.weekdays or [local_after.weekday()]
        else:
            return None
        for offset in range(1, 8):
            candidate = local_after + timedelta(days=offset)
            if candidate.weekday() in weekdays:
                return candidate.astimezone(timezone.utc)
        return None

    async def calendar_range(
        self,
        start: datetime,
        end: datetime,
        status: ScheduleStatus = ScheduleStatus.ACTIVE,
    ) -> List[Dict[str, Any]]:
        items = await self.store.list_items(status=status, limit=1000)
        entries: List[Dict[str, Any]] = []
        for item in items:
            for occurrence in self.occurrences_between(item, start=start, end=end):
                entries.append(
                    {
                        "schedule_id": str(item.schedule_id),
                        "title": item.title,
                        "kind": item.kind.value,
                        "action": item.action.value,
                        "status": item.status.value,
                        "scheduled_for": occurrence.isoformat(),
                        "recurrence": item.recurrence.value,
                        "tags": item.tags,
                    }
                )
        entries.sort(key=lambda entry: entry["scheduled_for"])
        return entries

    def _agenda_item_payload(self, item: ScheduleItem, now: datetime) -> Dict[str, Any]:
        next_run_at = item.next_run_at or item.start_at
        next_run_utc = self._as_utc(next_run_at)
        return {
            "schedule_id": str(item.schedule_id),
            "title": item.title,
            "kind": item.kind.value,
            "action": item.action.value,
            "status": item.status.value,
            "next_run_at": next_run_utc.isoformat(),
            "seconds_until": round((next_run_utc - now).total_seconds(), 3),
            "is_due": next_run_utc <= now,
            "recurrence": item.recurrence.value,
            "priority": item.priority,
            "tags": item.tags,
            "objective": item.objective,
            "description": item.description,
            "commitment_id": item.commitment_id,
            "plan_id": item.plan_id,
            "meta": self._compact_payload(item.meta),
        }

    async def _agenda_run_payload(self, run: ScheduleRun) -> Dict[str, Any]:
        receipts = await self._receipt_refs_for_run(run)
        return {
            "run_id": str(run.run_id),
            "schedule_id": str(run.schedule_id),
            "scheduled_for": run.scheduled_for.isoformat(),
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "status": run.status.value,
            "task_id": run.task_id,
            "error": self._truncate_text(run.error, 1000) if run.error else None,
            "meta": self._compact_payload(run.meta),
            "receipt_count": len(receipts),
            "receipts": receipts,
        }

    async def _receipt_refs_for_run(self, run: ScheduleRun) -> List[Dict[str, Any]]:
        task_id = str(run.task_id or "").strip()
        if not task_id:
            return []
        runtime_ctx = getattr(self.runtime, "ctx", None)
        receipt_store = getattr(runtime_ctx, "receipt_store", None) or getattr(
            self.runtime,
            "receipt_store",
            None,
        )
        list_by_task = getattr(receipt_store, "list_by_task", None)
        if not callable(list_by_task):
            return []
        try:
            receipts = await list_by_task(task_id, limit=5)
        except Exception:
            return []
        return [
            {
                "receipt_id": str(getattr(receipt, "receipt_id", "")),
                "task_id": str(getattr(receipt, "task_id", "")),
                "success": bool(getattr(receipt, "success", False)),
                "completed_at": (
                    receipt.completed_at.isoformat()
                    if getattr(receipt, "completed_at", None)
                    else None
                ),
                "objective": self._truncate_text(getattr(receipt, "objective", "") or "", 300),
                "output": str(getattr(receipt, "output", "") or "")[:500],
            }
            for receipt in receipts
        ]

    @classmethod
    def _compact_payload(
        cls,
        value: Any,
        *,
        depth: int = 0,
        max_depth: int = 4,
        max_items: int = 30,
        max_text: int = 1000,
    ) -> Any:
        if depth >= max_depth:
            return cls._truncate_text(value, 240)
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            items = list(value.items())
            for key, child in items[:max_items]:
                compact[str(key)] = cls._compact_payload(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_text=max_text,
                )
            if len(items) > max_items:
                compact["_truncated_key_count"] = len(items) - max_items
            return compact
        if isinstance(value, list):
            compact_list = [
                cls._compact_payload(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_text=max_text,
                )
                for child in value[:max_items]
            ]
            if len(value) > max_items:
                compact_list.append({"_truncated_item_count": len(value) - max_items})
            return compact_list
        if isinstance(value, tuple):
            return cls._compact_payload(
                list(value),
                depth=depth,
                max_depth=max_depth,
                max_items=max_items,
                max_text=max_text,
            )
        if isinstance(value, str):
            return cls._truncate_text(value, max_text)
        return value

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return f"{text[:limit]}... [truncated {omitted} chars]"

    @staticmethod
    def _agenda_summary(counts: Dict[str, int], next_item: Optional[Dict[str, Any]]) -> str:
        due = int(counts.get("due_now", 0))
        upcoming = int(counts.get("upcoming", 0))
        if due:
            return f"{due} schedule item(s) due now; next: {next_item.get('title') if next_item else 'unknown'}."
        if next_item:
            return f"No schedule items due now; next: {next_item.get('title')} at {next_item.get('next_run_at')}."
        if upcoming:
            return f"No schedule items due now; {upcoming} upcoming in the horizon."
        return "No active schedule items due or upcoming in the horizon."

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _zone(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name or self.default_timezone)
        except ZoneInfoNotFoundError:
            return ZoneInfo(self.default_timezone)

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(EventKind.TOOL_CALL, f"ScheduleService: {event}", payload)
