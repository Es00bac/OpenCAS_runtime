"""Live integration test for embedding-based tool routing.

Bootstraps the real OpenCAS runtime, builds the tool embedding index,
and tests that paraphrased objectives surface the right tools.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.tools.context import ToolUseContext
from opencas.tools.loop import ToolUseLoop


passed = 0
failed = 0


def _ok(label, detail=""):
    global passed
    passed += 1
    print(f"  PASS  {label}" + (f" — {detail}" if detail else ""))


def _fail(label, detail=""):
    global failed
    failed += 1
    print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))


async def main():
    global passed, failed

    # --- Bootstrap the real runtime ---
    print("[bootstrap]")
    config = BootstrapConfig(
        state_dir=Path(".opencas").resolve(),
        workspace_root=Path(".").resolve(),
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    await runtime.tom.load()
    print(f"  Tools registered: {len(runtime.tools.list_tools())}")

    # --- Build the tool embedding index ---
    print("\n[build tool embedding index]")
    embeddings_svc = ctx.embeddings
    if embeddings_svc is None:
        print("  SKIP  EmbeddingService not available, cannot test semantic routing")
        print(f"\n=== {passed} passed, {failed} failed ===")
        return 0 if failed == 0 else 1

    from opencas.tools.tool_embedding_index import ToolEmbeddingIndex

    tools = runtime.tools.list_tools()
    idx = await ToolEmbeddingIndex.build(embeddings_svc, tools)

    if not idx.is_ready:
        print("  SKIP  Tool embedding index failed to build")
        print(f"\n=== {passed} passed, {failed} failed ===")
        return 0 if failed == 0 else 1

    _ok("index built", f"{len(idx._tool_names)} tools, dim={idx._dimension}")

    # --- Wire into ToolUseLoop ---
    loop = runtime.tool_loop
    loop.tool_embedding_index = idx
    loop._index_built = True

    # --- Test semantic routing with paraphrased objectives ---
    print("\n[semantic routing tests]")
    test_ctx = ToolUseContext(runtime=runtime, session_id="live-test", plan_mode=False)

    cases = [
        # (objective, expected_tool_substring, description)
        ("how hot is the server", "system_status", "paraphrased system check"),
        ("what time is it right now", "time_now", "time query"),
        ("are these two files the same", "diff", "diff via paraphrase"),
        ("turn this into a web-friendly url", "slugify", "slugify via paraphrase"),
        ("make this safe for urls", "url_encode", "url encoding via paraphrase"),
        ("how much memory is the machine using", "system_status", "memory usage paraphrase"),
        ("what day is tomorrow", "time_now", "date query paraphrase"),
        ("is this valid json", "json_validate", "json validation paraphrase"),
        ("convert 100 fahrenheit to celsius", "unit_convert", "unit conversion"),
        ("compute the square root of 144", "calculate", "math computation"),
    ]

    for objective, expected_substring, desc in cases:
        # Embed the objective
        obj_vector = await loop._embed_objective(objective, test_ctx)

        # Get tool selection
        selected = loop._filter_tools(test_ctx, objective=objective, objective_vector=obj_vector)
        tool_names = [t.name for t in selected]

        matched = any(expected_substring in n for n in tool_names)
        if matched:
            _ok(desc, f"objective={objective!r}, matched {expected_substring}")
        else:
            _fail(desc, f"objective={objective!r}, expected {expected_substring}, got {tool_names[:8]}")

    # --- Test keyword routing still works ---
    print("\n[keyword routing (existing behavior preserved)]")
    keyword_cases = [
        ("check the cpu load", "system_status"),
        ("hash this string with sha256", "hash_text"),
        ("show me the diff between these two files", "diff_files"),
        ("send a POST request to the api endpoint", "http_request"),
        ("save this as a note for later", "note_save"),
    ]

    for objective, expected_name in keyword_cases:
        obj_vector = await loop._embed_objective(objective, test_ctx)
        selected = loop._filter_tools(test_ctx, objective=objective, objective_vector=obj_vector)
        tool_names = [t.name for t in selected]

        if expected_name in tool_names:
            _ok(f"keyword: {objective[:40]}", f"found {expected_name}")
        else:
            _fail(f"keyword: {objective[:40]}", f"expected {expected_name}, got {tool_names[:8]}")

    # --- Test conversational guard ---
    print("\n[conversational guard]")
    obj_vector = await loop._embed_objective("Tell me how you understand your role in this session.", test_ctx)
    selected = loop._filter_tools(test_ctx, objective="Tell me how you understand your role in this session.", objective_vector=obj_vector)
    if len(selected) == 0:
        _ok("conversational turn returns no tools")
    else:
        _fail("conversational turn returns no tools", f"got {len(selected)} tools: {[t.name for t in selected]}")

    print(f"\n=== {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
