import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencas.bootstrap.config import BootstrapConfig
from opencas.bootstrap.pipeline import BootstrapPipeline
from opencas.runtime.agent_loop import AgentRuntime
from opencas.execution.models import RepairTask
from opencas.autonomy.models import ActionRiskTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("live_test")

async def main():
    test_dir = Path("/tmp/opencas-public-fixture/.opencas_live_test_state")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 1. Bootstrapping OpenCAS Substrate ===")
    config = BootstrapConfig(
        state_dir=test_dir,
        session_id="live-test-01",
        clean_boot=True,
        # Setting a small memory budget to trigger compaction faster
    )

    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)

    logger.info("=== 2. Starting BAA (Bounded Assistant Agent) ===")
    await runtime.baa.start()

    try:
        logger.info("=== 3. Testing Conversational Engine, Memory, and ToM ===")
        msg = "Hello! I am a test operator. My favorite color is green. Your goal is to write a poem."
        logger.info(f"Sending message: '{msg}'")

        try:
            response = await runtime.converse(msg)
            logger.info(f"Agent Response: {response}")
        except Exception as e:
            logger.error(f"Error during LLM conversation: {e}")
            logger.warning("If this is an auth error, ensure ANTHROPIC_API_KEY is set. Skipping to next test.")

        # Verify Memory
        eps = await runtime.memory.list_recent_episodes("live-test-01", limit=5)
        logger.info(f"Saved {len(eps)} episodes to memory fabric.")
        assert len(eps) >= 2, "Failed to save conversation to memory."

        # Verify ToM
        metacog = runtime.check_metacognition()
        logger.info(f"ToM extracted {metacog['belief_count']} beliefs.")

        # Verify Executive
        goals = list(runtime.executive.active_goals)
        logger.info(f"Executive active goals: {goals}")

        # Verify Relational Engine
        musubi = runtime.ctx.relational.state.musubi
        logger.info(f"Relational Musubi score: {musubi}")

        logger.info("=== 4. Testing SmartCommandValidator ===")
        logger.info("Attempting to execute an unknown command: 'some_made_up_command_xyz --flag'")
        tool_res = await runtime.execute_tool("bash_run_command", {"command": "some_made_up_command_xyz --flag"})
        logger.info(f"Tool Result: {tool_res['output']}")
        assert tool_res['success'] == False, "Unknown command should have been blocked by SmartCommandValidator."
        assert "validation" in tool_res['output'].lower() or "blocked" in tool_res['output'].lower() or "caution" in tool_res['output'].lower(), "Should be caught by validation"

        logger.info("Attempting to execute a known safe command: 'echo hello'")
        tool_res_safe = await runtime.execute_tool("bash_run_command", {"command": "echo hello"})
        logger.info(f"Tool Result (Safe): {tool_res_safe['output']}")
        # Note: execute_tool goes through SelfApprovalLadder. It might escalate if trust is low,
        # so success isn't strictly guaranteed without operator approval in the loop. We just log it.

        logger.info("=== 5. Testing BAA Execution & File System Tools ===")
        test_file = test_dir / "output.txt"
        task = RepairTask(
            objective=f"Write the word SUCCESS to the file {test_file.absolute()}",
            verification_command=f"cat {test_file.absolute()}"
        )
        logger.info(f"Submitting repair task to BAA: {task.objective}")
        future = await runtime.submit_repair(task)

        # Wait for the BAA to finish execution
        logger.info("Waiting for BAA to complete the task...")
        result = await future
        logger.info(f"BAA Task Completed. Success: {result.success}")
        logger.info(f"BAA Output: {result.output}")
        if result.success:
            assert test_file.exists(), "BAA reported success but file was not created."

        logger.info("=== 6. Testing Proactive Loops (Daydream & Consolidation) ===")
        logger.info("Running Daydream Generator...")
        try:
            dd_res = await runtime.run_daydream()
            logger.info(f"Daydreams generated: {dd_res['daydreams']}, Reflections: {dd_res['reflections']}")
        except Exception as e:
            logger.warning(f"Daydreaming failed (expected if LLM auth is missing): {e}")

        logger.info("Running Nightly Consolidation...")
        try:
            cons_res = await runtime.run_consolidation()
            logger.info(f"Consolidation Clusters Merged: {cons_res.get('clustersMerged', 0)}")
        except Exception as e:
            logger.warning(f"Consolidation failed (expected if LLM auth is missing): {e}")

    except AssertionError as ae:
        logger.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logger.error(f"Live test failed with exception: {e}", exc_info=True)
        raise
    finally:
        logger.info("=== 7. Initiating Graceful Shutdown ===")
        await runtime.baa.stop()
        await runtime._close_stores()

    logger.info("=== LIVE TEST SUITE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(main())