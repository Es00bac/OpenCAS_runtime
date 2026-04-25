"""Plan mode tool adapter for constrained planning phases."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from opencas.planning import PlanStore

from ..models import ToolResult


class PlanToolAdapter:
    """Adapter for entering and exiting plan mode with persistent plan storage."""

    def __init__(self, store: Optional[PlanStore] = None) -> None:
        self.store = store

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "enter_plan_mode":
                return self._enter_plan_mode(args)
            if name == "exit_plan_mode":
                return self._exit_plan_mode(args)
            return ToolResult(success=False, output=f"Unknown plan tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _enter_plan_mode(self, args: Dict[str, Any]) -> ToolResult:
        plan_id = str(args.get("plan_id", "")) or f"plan-{uuid.uuid4().hex[:8]}"
        content = str(args.get("content", ""))
        project_id = args.get("project_id") or None
        task_id = args.get("task_id") or None

        if self.store is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Try to get existing plan
                plan = None
                try:
                    plan = asyncio.run_coroutine_threadsafe(
                        self.store.get_plan(plan_id), loop
                    ).result()
                except Exception:
                    pass
                if plan is None:
                    asyncio.run_coroutine_threadsafe(
                        self.store.create_plan(plan_id, content=content, project_id=project_id, task_id=task_id),
                        loop,
                    ).result()
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.store.set_status(plan_id, "active"), loop
                    ).result()
                    if content:
                        asyncio.run_coroutine_threadsafe(
                            self.store.update_content(plan_id, content), loop
                        ).result()
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output=f"Failed to persist plan: {exc}",
                    metadata={},
                )

        return ToolResult(
            success=True,
            output="Entered plan mode. You may now use read-only tools and write files to the plans directory.",
            metadata={"plan_mode": True, "plan_id": plan_id},
        )

    def _exit_plan_mode(self, args: Dict[str, Any]) -> ToolResult:
        plan_id = str(args.get("plan_id", ""))
        content = str(args.get("content", ""))
        action_count = 0

        if self.store is not None and plan_id:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                if content:
                    asyncio.run_coroutine_threadsafe(
                        self.store.update_content(plan_id, content), loop
                    ).result()
                asyncio.run_coroutine_threadsafe(
                    self.store.set_status(plan_id, "completed"), loop
                ).result()
                actions = asyncio.run_coroutine_threadsafe(
                    self.store.get_actions(plan_id), loop
                ).result()
                action_count = len(actions)
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output=f"Failed to update plan on exit: {exc}",
                    metadata={},
                )

        return ToolResult(
            success=True,
            output=f"Exited plan mode. Plan ID: {plan_id or 'none'}. Content length: {len(content)} chars. Actions recorded: {action_count}.",
            metadata={"plan_mode": False, "plan_id": plan_id or None, "content_length": len(content), "action_count": action_count},
        )
