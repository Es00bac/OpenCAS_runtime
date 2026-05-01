"""Live install/enable/test harness for the plugin suite + core tool enhancements.

Coverage:
  * Install + enable each plugin via the real PluginLifecycleManager.
  * Exercise every plugin tool (success and failure paths).
  * Verify the enhanced core web_fetch (parse_json / raw / content-type) and
    grep_search (glob filter) using the real adapters.
  * Verify keyword routing in ToolUseLoop._select_tools_for_objective so the
    resident agent surfaces the new plugin tools.
  * Sanity-check lifecycle disable/enable gating.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from opencas.platform import CapabilityRegistry, CapabilityStatus
from opencas.infra.hook_registry import TypedHookRegistry
from opencas.plugins import (
    PluginLifecycleManager,
    PluginRegistry,
    PluginStore,
    SkillRegistry,
)
from opencas.tools import ToolRegistry
from opencas.tools.adapters.web import WebToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter

PLUGINS_DIR = REPO_ROOT / "plugins"
PLUGIN_IDS = [
    "json_tools",
    "system_stats",
    "notes",
    "time_tools",
    "http_request",
    "calc",
    "diff",
    "codec",
]


def _ok(label: str, msg: str = "") -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"  PASS  {label}{suffix}")


def _fail(label: str, msg: str) -> None:
    print(f"  FAIL  {label} — {msg}")


async def _setup() -> tuple[PluginLifecycleManager, ToolRegistry, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="opencas-plugin-test-")
    tmp_root = Path(tmp.name)
    store = PluginStore(tmp_root / "plugins.db")
    await store.connect()

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    capability_registry = CapabilityRegistry()
    hook_registry = TypedHookRegistry()
    tools = ToolRegistry()

    install_root = tmp_root / "installed_plugins"
    install_root.mkdir()

    mgr = PluginLifecycleManager(
        store=store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=tools,
        hook_registry=hook_registry,
        builtin_dir=None,
        install_root=install_root,
        capability_registry=capability_registry,
    )
    return mgr, tools, tmp


async def _install_all(mgr: PluginLifecycleManager) -> bool:
    print("\n[install + enable]")
    success = True
    for pid in PLUGIN_IDS:
        plugin = await mgr.install(PLUGINS_DIR / pid)
        if plugin is None or not await mgr.store.is_enabled(pid):
            _fail(pid, "install/enable failed")
            success = False
        else:
            _ok(pid, "enabled")
    return success


def _exec(tools: ToolRegistry, name: str, args: dict) -> tuple[bool, str, dict]:
    entry = tools.get(name)
    if entry is None:
        return False, f"tool {name!r} not registered", {}
    res = entry.adapter(name, args)
    return res.success, res.output, res.metadata


def _test_json_tools(tools: ToolRegistry) -> bool:
    print("\n[test json_tools]")
    overall = True
    data = {"users": [{"name": "ada"}, {"name": "grace"}], "count": 2}

    ok, out, _ = _exec(tools, "json_query", {"data": data, "path": "users[1].name"})
    if not ok or out.strip() != "grace":
        _fail("json_query nested", out); overall = False
    else:
        _ok("json_query nested")

    schema = {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer"}}}
    ok, _, _ = _exec(tools, "json_validate", {"data": data, "schema": schema})
    if not ok:
        _fail("json_validate ok", out); overall = False
    else:
        _ok("json_validate ok")

    ok, _, meta = _exec(tools, "json_validate", {"data": {}, "schema": schema})
    if ok or meta.get("error_count", 0) < 1:
        _fail("json_validate fail-detect", str(meta)); overall = False
    else:
        _ok("json_validate fail-detect")
    return overall


def _test_system_stats(tools: ToolRegistry) -> bool:
    print("\n[test system_stats]")
    ok, out, _ = _exec(tools, "system_status", {"disk_paths": ["/"]})
    if not ok:
        _fail("system_status", out); return False
    parsed = json.loads(out)
    if "/" not in parsed.get("disks", {}):
        _fail("system_status disks", "missing /"); return False
    _ok("system_status", f"disks={list(parsed['disks'])}")
    return True


def _test_notes(tools: ToolRegistry, sandbox: Path) -> bool:
    print("\n[test notes]")
    notes_dir = sandbox / "notes"
    overall = True
    ok, _, meta = _exec(
        tools,
        "note_save",
        {"title": "T1", "body": "x", "tags": ["alpha"], "notes_dir": str(notes_dir)},
    )
    if not ok:
        _fail("note_save", str(meta)); return False
    saved = Path(meta["path"])
    _ok("note_save", saved.name)

    ok, _, meta = _exec(tools, "note_list", {"notes_dir": str(notes_dir), "tag": "alpha"})
    if not ok or meta.get("count") != 1:
        _fail("note_list tag", str(meta)); overall = False
    else:
        _ok("note_list tag")

    ok, out, _ = _exec(tools, "note_read", {"name": saved.name, "notes_dir": str(notes_dir)})
    if not ok or "T1" not in out:
        _fail("note_read", out); overall = False
    else:
        _ok("note_read")
    return overall


def _test_time_tools(tools: ToolRegistry) -> bool:
    print("\n[test time_tools]")
    overall = True
    ok, out, _ = _exec(tools, "time_now", {"tz": "America/Denver"})
    if not ok:
        _fail("time_now America/Denver", out); overall = False
    else:
        parsed = json.loads(out)
        if "iso" not in parsed or "weekday" not in parsed:
            _fail("time_now payload", str(parsed)); overall = False
        else:
            _ok("time_now", parsed["iso"][:19])

    ok, out, _ = _exec(tools, "time_parse", {"value": "2026-04-26T18:00:00Z", "tz": "America/Denver"})
    if not ok:
        _fail("time_parse", out); overall = False
    else:
        parsed = json.loads(out)
        if "America/Denver" not in parsed["tz"]:
            _fail("time_parse tz", parsed["tz"]); overall = False
        else:
            _ok("time_parse tz convert")

    ok, out, _ = _exec(tools, "time_diff", {"from": "2026-01-01T00:00:00Z", "to": "2026-01-02T03:04:05Z"})
    if not ok:
        _fail("time_diff", out); overall = False
    else:
        parsed = json.loads(out)
        if abs(parsed["total_hours"] - 27.067916666) > 0.01:
            _fail("time_diff hours", parsed["total_hours"]); overall = False
        else:
            _ok("time_diff", parsed["human"])

    ok, out, _ = _exec(tools, "time_age", {"path": str(Path(__file__))})
    if not ok or "modified_age" not in json.loads(out):
        _fail("time_age", out); overall = False
    else:
        _ok("time_age")

    ok, out, _ = _exec(tools, "time_now", {"tz": "Mars/Olympus"})
    if ok or "unknown timezone" not in out:
        _fail("time_now bad tz", out); overall = False
    else:
        _ok("time_now rejects bad tz")
    return overall


def _test_http_request(tools: ToolRegistry) -> bool:
    print("\n[test http_request]")
    overall = True
    ok, out, meta = _exec(tools, "http_request", {"url": "ftp://nope"})
    if ok:
        _fail("http_request reject scheme", out); overall = False
    else:
        _ok("http_request rejects non-http")

    ok, out, meta = _exec(
        tools,
        "http_request",
        {"url": "https://httpbin.org/get", "method": "GET", "parse_json": True, "timeout_seconds": 10},
    )
    if not ok:
        print(f"  SKIP  http_request GET httpbin (network) — {out[:120]}")
    else:
        _ok("http_request GET httpbin", f"status={meta.get('status')}")

    ok, out, meta = _exec(
        tools,
        "http_request",
        {
            "url": "https://httpbin.org/post",
            "method": "POST",
            "json": {"hello": "opencas"},
            "headers": {"X-Test": "1"},
            "parse_json": True,
            "timeout_seconds": 10,
        },
    )
    if not ok:
        print(f"  SKIP  http_request POST httpbin (network) — {out[:120]}")
    else:
        try:
            payload = json.loads(out)
            sent_json = payload.get("json")
            if sent_json != {"hello": "opencas"}:
                _fail("http_request POST body echo", str(sent_json)); overall = False
            elif payload.get("headers", {}).get("X-Test") != "1":
                _fail("http_request POST header echo", "missing X-Test"); overall = False
            else:
                _ok("http_request POST round-trip")
        except json.JSONDecodeError as exc:
            _fail("http_request POST parse", str(exc)); overall = False
    return overall


def _test_calc(tools: ToolRegistry) -> bool:
    print("\n[test calc]")
    overall = True
    ok, out, _ = _exec(tools, "calculate", {"expression": "2 + 3 * 4 - sqrt(16)"})
    if not ok or out.strip() != "10.0":
        _fail("calculate basic", out); overall = False
    else:
        _ok("calculate basic")

    ok, out, _ = _exec(tools, "calculate", {"expression": "factorial(7)"})
    if not ok or out.strip() != "5040":
        _fail("calculate factorial", out); overall = False
    else:
        _ok("calculate factorial")

    ok, out, _ = _exec(tools, "calculate", {"expression": "__import__('os').system('echo bad')"})
    if ok:
        _fail("calculate sandbox", out); overall = False
    else:
        _ok("calculate rejects __import__")

    ok, out, _ = _exec(tools, "unit_convert", {"value": 5, "from": "km", "to": "mi"})
    if not ok:
        _fail("unit_convert km->mi", out); overall = False
    else:
        parsed = json.loads(out)
        if abs(parsed["value"] - 3.10685596) > 0.001:
            _fail("unit_convert km->mi value", parsed["value"]); overall = False
        else:
            _ok("unit_convert km->mi")

    ok, out, _ = _exec(tools, "unit_convert", {"value": 100, "from": "F", "to": "C"})
    if not ok:
        _fail("unit_convert F->C", out); overall = False
    else:
        parsed = json.loads(out)
        if abs(parsed["value"] - 37.7777777) > 0.01:
            _fail("unit_convert F->C value", parsed["value"]); overall = False
        else:
            _ok("unit_convert F->C")

    ok, out, _ = _exec(tools, "unit_convert", {"value": 1, "from": "m", "to": "kg"})
    if ok:
        _fail("unit_convert family mismatch", out); overall = False
    else:
        _ok("unit_convert rejects mismatch")
    return overall


def _test_diff(tools: ToolRegistry, sandbox: Path) -> bool:
    print("\n[test diff]")
    overall = True
    ok, out, meta = _exec(tools, "diff_text", {"a": "alpha\nbeta\n", "b": "alpha\nGAMMA\n"})
    if not ok or "alpha" not in out or meta.get("insertions") != 1 or meta.get("deletions") != 1:
        _fail("diff_text", f"out={out!r} meta={meta}"); overall = False
    else:
        _ok("diff_text", f"+{meta['insertions']}/-{meta['deletions']}")

    ok, out, meta = _exec(tools, "diff_text", {"a": "same", "b": "same"})
    if not ok or not meta.get("identical"):
        _fail("diff_text identical", str(meta)); overall = False
    else:
        _ok("diff_text identical")

    a = sandbox / "a.txt"
    b = sandbox / "b.txt"
    a.write_text("line1\nline2\nline3\n")
    b.write_text("line1\nline2-new\nline3\n")
    ok, out, meta = _exec(tools, "diff_files", {"a": str(a), "b": str(b)})
    if not ok or "line2-new" not in out or meta.get("insertions") != 1:
        _fail("diff_files", f"out={out[:200]!r} meta={meta}"); overall = False
    else:
        _ok("diff_files")
    return overall


def _test_codec(tools: ToolRegistry, sandbox: Path) -> bool:
    print("\n[test codec]")
    overall = True
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    ok, out, _ = _exec(tools, "hash_text", {"text": "hello", "algorithm": "sha256"})
    if not ok or out != expected:
        _fail("hash_text sha256", out); overall = False
    else:
        _ok("hash_text sha256")

    sample = sandbox / "sample.bin"
    sample.write_bytes(b"hello")
    ok, out, _ = _exec(tools, "hash_text", {"path": str(sample), "algorithm": "sha256"})
    if not ok or out != expected:
        _fail("hash_text path", out); overall = False
    else:
        _ok("hash_text path")

    ok, out, _ = _exec(tools, "base64_encode", {"text": "hello world"})
    if not ok or out != "aGVsbG8gd29ybGQ=":
        _fail("base64_encode", out); overall = False
    else:
        _ok("base64_encode")

    ok, out, _ = _exec(tools, "base64_decode", {"text": "aGVsbG8gd29ybGQ="})
    if not ok or out != "hello world":
        _fail("base64_decode", out); overall = False
    else:
        _ok("base64_decode")

    ok, out, _ = _exec(tools, "url_encode", {"text": "a b/c?d=e&f=g"})
    if not ok or out != "a%20b/c%3Fd%3De%26f%3Dg":
        _fail("url_encode", out); overall = False
    else:
        _ok("url_encode")

    ok, out, _ = _exec(tools, "url_decode", {"text": "a%20b%2Fc"})
    if not ok or out != "a b/c":
        _fail("url_decode", out); overall = False
    else:
        _ok("url_decode")

    ok, out, _ = _exec(tools, "slugify", {"text": "Hello, World! Plug-in #5"})
    if not ok or out != "hello-world-plug-in-5":
        _fail("slugify", out); overall = False
    else:
        _ok("slugify")
    return overall


async def _test_core_web_fetch() -> bool:
    print("\n[test core web_fetch enhancements]")
    overall = True
    adapter = WebToolAdapter()

    res = await adapter("web_fetch", {"url": "ftp://nope"})
    if res.success or "http://" not in res.output:
        _fail("web_fetch reject scheme", res.output); overall = False
    else:
        _ok("web_fetch rejects non-http (was missing before)")

    try:
        res = await adapter(
            "web_fetch",
            {"url": "https://example.com", "max_length": 4000, "raw": True},
        )
    except Exception as exc:
        print(f"  SKIP  web_fetch raw example.com (network) — {exc}")
        return overall
    if not res.success:
        print(f"  SKIP  web_fetch raw example.com (network) — {res.output[:120]}")
        return overall
    if res.metadata.get("mode") != "raw" or "<html" not in res.output.lower():
        _fail("web_fetch raw mode", f"meta={res.metadata}"); overall = False
    else:
        _ok("web_fetch raw mode", f"status={res.metadata.get('status')} ct={res.metadata.get('content_type')}")

    res = await adapter(
        "web_fetch",
        {"url": "https://httpbin.org/json", "parse_json": True, "max_length": 16384},
    )
    if not res.success:
        print(f"  SKIP  web_fetch parse_json httpbin (network) — {res.output[:120]}")
        return overall
    try:
        json.loads(res.output)
    except json.JSONDecodeError as exc:
        _fail("web_fetch parse_json", str(exc)); overall = False
    else:
        if res.metadata.get("mode") == "json":
            _ok("web_fetch parse_json")
        else:
            _fail("web_fetch parse_json mode", f"meta={res.metadata}"); overall = False
    return overall


def _test_core_grep_glob() -> bool:
    print("\n[test core grep_search enhancements]")
    overall = True
    adapter = SearchToolAdapter(allowed_roots=[str(REPO_ROOT)])

    res = adapter(
        "grep_search",
        {
            "pattern": "register_skills",
            "path": str(REPO_ROOT / "plugins"),
            "glob": "*.py",
            "max_count": 3,
            "output_mode": "content",
        },
    )
    if not res.success:
        _fail("grep_search glob", res.output); overall = False
    else:
        try:
            payload = json.loads(res.output)
        except json.JSONDecodeError as exc:
            _fail("grep_search glob parse", str(exc)); return False
        matches = payload.get("matches", [])
        if not matches or not all(m["path"].endswith(".py") for m in matches):
            _fail("grep_search glob filter", f"matches sample={matches[:2]}"); overall = False
        else:
            _ok("grep_search glob filter", f"matches={len(matches)}")

    res = adapter(
        "grep_search",
        {
            "pattern": "register_skills",
            "path": str(REPO_ROOT / "plugins"),
            "glob": "*.json",
            "output_mode": "files_with_matches",
        },
    )
    if not res.success:
        _fail("grep_search glob json (none)", res.output); overall = False
    else:
        payload = json.loads(res.output)
        if payload.get("files"):
            _fail("grep_search glob exclude", str(payload)); overall = False
        else:
            _ok("grep_search glob excludes non-matching")
    return overall


def _test_keyword_routing() -> bool:
    print("\n[test agent keyword routing]")
    from opencas.tools.loop import ToolUseLoop, ToolUseContext
    from opencas.tools.models import ToolEntry
    from opencas.autonomy.models import ActionRiskTier

    overall = True
    tools_registry = ToolRegistry()
    # Register a fake stand-in so the tool entry exists for selection
    fake_tools = [
        ("time_now", ActionRiskTier.READONLY),
        ("time_parse", ActionRiskTier.READONLY),
        ("time_diff", ActionRiskTier.READONLY),
        ("time_age", ActionRiskTier.READONLY),
        ("http_request", ActionRiskTier.NETWORK),
        ("calculate", ActionRiskTier.READONLY),
        ("unit_convert", ActionRiskTier.READONLY),
        ("diff_text", ActionRiskTier.READONLY),
        ("diff_files", ActionRiskTier.READONLY),
        ("hash_text", ActionRiskTier.READONLY),
        ("base64_encode", ActionRiskTier.READONLY),
        ("base64_decode", ActionRiskTier.READONLY),
        ("url_encode", ActionRiskTier.READONLY),
        ("url_decode", ActionRiskTier.READONLY),
        ("slugify", ActionRiskTier.READONLY),
        ("json_query", ActionRiskTier.READONLY),
        ("json_validate", ActionRiskTier.READONLY),
        ("system_status", ActionRiskTier.READONLY),
        ("note_save", ActionRiskTier.WORKSPACE_WRITE),
        ("note_list", ActionRiskTier.READONLY),
        ("note_read", ActionRiskTier.READONLY),
    ]
    for name, tier in fake_tools:
        tools_registry.register(name, "stub", lambda *a, **k: None, tier, {"type": "object"})

    class _StubLLM:
        model_routing = type("_R", (), {"auto_escalation": True})()

    _stub_approval = type("_A", (), {})()
    loop = ToolUseLoop(llm=_StubLLM(), tools=tools_registry, approval=_stub_approval)

    cases = [
        ("what time is it tomorrow", {"time_now", "time_parse"}),
        ("send a POST request to the api endpoint", {"http_request"}),
        ("compute 5! + sqrt(16)", {"calculate"}),
        ("convert 5 km to miles", {"calculate", "unit_convert"}),
        ("show me the diff between these two files", {"diff_files", "diff_text"}),
        ("hash this string with sha256", {"hash_text"}),
        ("base64 decode this token", {"base64_encode", "base64_decode"}),
        ("validate this json against my schema", {"json_query", "json_validate"}),
        ("what is the cpu and memory usage", {"system_status"}),
        ("save this as a note for later", {"note_save", "note_list", "note_read"}),
    ]

    for objective, expected in cases:
        ctx = ToolUseContext(runtime=type("_RT", (), {})(), session_id="t", plan_mode=False)
        selected = loop._filter_tools(ctx, objective)
        names = {entry.name for entry in selected}
        missing = expected - names
        if missing:
            _fail(f"routing[{objective!r}]", f"missing {missing}"); overall = False
        else:
            _ok(f"routing[{objective[:48]!r}]", f"+{len(expected & names)}")
    return overall


async def _disable_check(mgr: PluginLifecycleManager) -> bool:
    print("\n[lifecycle disable/enable]")
    overall = True
    pid = "calc"
    await mgr.disable(pid)
    if not mgr.is_tool_disabled("calculate"):
        _fail("disable gates calculate", "not flagged"); overall = False
    else:
        _ok("disable gates calculate")
    if mgr.capability_registry is not None:
        descs = mgr.capability_registry.list_capabilities(owner_id=pid)
        if any(d.status != CapabilityStatus.DISABLED for d in descs):
            _fail("disable status projection", str([d.status for d in descs])); overall = False
        else:
            _ok("disable projects DISABLED")
    await mgr.enable(pid)
    if mgr.is_tool_disabled("calculate"):
        _fail("re-enable", "still gated"); overall = False
    else:
        _ok("re-enable clears gate")
    return overall


async def main() -> int:
    print(f"plugins root: {PLUGINS_DIR}")
    mgr, tools, tmp = await _setup()
    try:
        if not await _install_all(mgr):
            return 1

        results: dict[str, bool] = {}
        with tempfile.TemporaryDirectory(prefix="opencas-pluginsbox-") as sandbox:
            sandbox_path = Path(sandbox)
            results["json_tools"] = _test_json_tools(tools)
            results["system_stats"] = _test_system_stats(tools)
            results["notes"] = _test_notes(tools, sandbox_path)
            results["time_tools"] = _test_time_tools(tools)
            results["http_request"] = _test_http_request(tools)
            results["calc"] = _test_calc(tools)
            results["diff"] = _test_diff(tools, sandbox_path)
            results["codec"] = _test_codec(tools, sandbox_path)

        results["core_web_fetch"] = await _test_core_web_fetch()
        results["core_grep_glob"] = _test_core_grep_glob()
        results["agent_routing"] = _test_keyword_routing()
        results["lifecycle_disable"] = await _disable_check(mgr)

        print("\n=== summary ===")
        for label, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        passed = sum(1 for v in results.values() if v)
        print(f"\n{passed}/{len(results)} groups passed")
        return 0 if passed == len(results) else 2
    finally:
        await mgr.store.close()
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
