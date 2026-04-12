"""Agent tool adapter for spawning lightweight subagents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import ToolResult


class AgentToolAdapter:
    """Adapter that spawns a subagent via a temporary ToolUseLoop."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "agent":
                return await self._agent(args)
            return ToolResult(success=False, output=f"Unknown agent tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    async def _agent(self, args: Dict[str, Any]) -> ToolResult:
        description = str(args.get("description", ""))
        agent_type = str(args.get("agent_type", "general-purpose"))
        prompt = str(args.get("prompt", ""))
        if not prompt:
            return ToolResult(success=False, output="prompt is required", metadata={})

        if not self.runtime or not hasattr(self.runtime, "llm"):
            return ToolResult(
                success=False,
                output="Agent runtime is not available for subagent execution.",
                metadata={},
            )

        from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
        from opencas.autonomy.models import ActionRiskTier

        # Build a filtered registry that excludes the agent tool to avoid recursion
        sub_registry = ToolRegistry(
            tracer=self.runtime.tools.tracer,
            validation_pipeline=self.runtime.tools.validation_pipeline,
            hook_bus=self.runtime.tools.hook_bus,
        )
        for entry in self.runtime.tools.list_tools():
            if entry.name == "agent":
                continue
            sub_registry.register(
                entry.name,
                entry.description,
                entry.adapter,
                entry.risk_tier,
                entry.parameters,
            )

        loop = ToolUseLoop(
            llm=self.runtime.llm,
            tools=sub_registry,
            approval=self.runtime.approval,
            tracer=self.runtime.tracer,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a specialized subagent ({agent_type}). "
                    "Use available tools to complete the task. Be concise."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        ctx = ToolUseContext(
            runtime=self.runtime,
            session_id=f"subagent-{description}"[:32],
        )
        result = await loop.run(
            objective=description or f"Subagent ({agent_type})",
            messages=messages,
            ctx=ctx,
        )
        return ToolResult(
            success=True,
            output=result.final_output,
            metadata={
                "agent_type": agent_type,
                "description": description,
                "iterations": result.iterations,
            },
        )
