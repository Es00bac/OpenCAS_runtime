"""Embedding-based semantic tool router for ToolUseLoop.

Replaces keyword-based tool selection with cosine similarity between
objective embeddings and pre-computed tool description embeddings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from opencas.tools.registry import ToolEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic hints — expanded use-case text per tool name
# ---------------------------------------------------------------------------

TOOL_SEMANTIC_HINTS: Dict[str, str] = {
    # --- Core retrieval (always available but hinted for richness) ---
    "search_memories": "Use when: searching past conversations, recalling what was discussed, finding previous interactions, querying episode memory, looking up history.",
    "recall_concepts": "Use when: recalling concepts, remembering ideas, retrieving knowledge from memory, what did we talk about, semantic memory lookup.",
    "web_search": "Use when: searching the internet, looking something up online, finding information on the web, googling, web query, search engine.",
    "web_fetch": "Use when: fetching a URL, retrieving a web page, downloading content from a link, reading a website, get page contents, HTTP GET a resource.",
    # --- Runtime / workflow ---
    "runtime_status": "Use when: checking runtime status, agent state, current configuration, operating roots, constraints, what is the agent doing.",
    "workflow_status": "Use when: checking workflow status, current plans, active tasks, what work is in progress.",
    "workflow_create_commitment": "Use when: creating a commitment, tracking a goal, making a promise, setting a deadline, pledging to do something.",
    "workflow_update_commitment": "Use when: updating a commitment, changing a goal, modifying a deadline, progress update on a promise.",
    "workflow_list_commitments": "Use when: listing commitments, showing goals, what am I tracking, what have I promised, show deadlines.",
    "workflow_create_writing_task": "Use when: writing a document, drafting an article, creating an essay, composing text, writing task, content creation.",
    "workflow_create_plan": "Use when: creating a plan, making a checklist, building a roadmap, planning steps, organizing work, structured approach.",
    "workflow_update_plan": "Use when: updating a plan, modifying checklist, changing roadmap, plan progress, revise plan.",
    "workflow_repo_triage": "Use when: repo triage, project overview, codebase summary, audit the repository, what files are in the project, repository health.",
    "workflow_supervise_session": "Use when: supervising a session, delegating to another agent, launching Claude or Codex, operator control, managing sub-agents.",
    # --- Browser tools ---
    "browser_start": "Use when: starting a browser session, opening a browser, preparing to inspect or interact with a web page.",
    "browser_navigate": "Use when: navigating to a website, opening a URL, going to a web page, visiting a link, browsing.",
    "browser_snapshot": "Use when: taking a snapshot of a web page, reading page content, what is on the screen, accessibility snapshot.",
    "browser_click": "Use when: clicking a button on a web page, interacting with a page element, pressing a link.",
    "browser_type": "Use when: typing text into a web page field, filling in a form, entering input on a page.",
    "browser_take_screenshot": "Use when: screenshot a web page, capture the screen, visual snapshot of the browser.",
    # --- Google Workspace ---
    "google_workspace_gmail": "Use when: reading email, checking Gmail, sending email, inbox, messages.",
    "google_workspace_calendar": "Use when: calendar events, schedule, appointments, meetings, what is on my calendar.",
    "google_workspace_drive": "Use when: Google Drive files, documents, spreadsheets, shared files, cloud storage.",
    # --- PTY / terminal ---
    "pty_interact": "Use when: interacting with a terminal, PTY session, shell, command-line interface, TUI application, running an interactive program.",
    "pty_remove": "Use when: removing a terminal session, cleaning up PTY, closing a shell session.",
    "pty_clear": "Use when: clearing a terminal, resetting PTY output, clean screen.",
    "pty_kill": "Use when: killing a process, terminating a terminal session, force stop.",
    # --- Process management ---
    "process_start": "Use when: starting a process, launching a background server, running a daemon, spawning a background task.",
    "process_list": "Use when: listing processes, showing running services, what is running, background tasks.",
    "process_stop": "Use when: stopping a process, killing a background task, terminating a server, shutdown daemon.",
    # --- Filesystem / search ---
    "fs_read_file": "Use when: reading a file, viewing file contents, showing a file, cat a file, display file.",
    "fs_list_dir": "Use when: listing a directory, showing folder contents, what files are here, ls, dir listing.",
    "fs_write_file": "Use when: writing a file, creating a file, saving content to disk, output to file, store data.",
    "grep_search": "Use when: searching code, grep, finding text in files, searching for a pattern, ripgrep, code search.",
    "glob_search": "Use when: finding files by name, glob pattern, file search, where is a file, locate files.",
    "bash_run_command": "Use when: running a shell command, executing bash, system command, terminal command, run script.",
    "lsp_diagnostics": "Use when: language server diagnostics, code errors, type checking, linting, code issues, compiler errors.",
    "agent": "Use when: delegating to a sub-agent, spawning a helper task, parallel research, agent tool.",
    # --- Plan mode ---
    "enter_plan_mode": "Use when: entering plan mode, planning mode, design before coding, think first.",
    "exit_plan_mode": "Use when: exiting plan mode, done planning, ready to implement, start coding.",
    # --- Time tools (plugin) ---
    "time_now": "Use when: current time, what time is it, date today, timestamp, what day is it, clock, timezone, what is the date.",
    "time_parse": "Use when: parsing a date, converting timezone, interpreting a timestamp, ISO date, when is that.",
    "time_diff": "Use when: time difference, duration between dates, how long between, elapsed time, time interval.",
    "time_age": "Use when: how old is this file, when was this modified, file age, time since creation, last modified.",
    # --- HTTP request (plugin) ---
    "http_request": "Use when: making an HTTP request, POST to an API, REST endpoint, PUT data, DELETE resource, PATCH update, webhook call, bearer token, API call, send a request, hit an endpoint.",
    # --- Calculator (plugin) ---
    "calculate": "Use when: calculate, compute, arithmetic, math, evaluate expression, factorial, square root, logarithm, sum, multiply, divide.",
    "unit_convert": "Use when: unit conversion, convert kilometers to miles, Celsius to Fahrenheit, bytes to gigabytes, measurement conversion, metric to imperial.",
    # --- Diff (plugin) ---
    "diff_text": "Use when: compare two texts, diff strings, text difference, string comparison, are these the same text, unified diff of strings, check if text matches, see differences.",
    "diff_files": "Use when: compare two files, diff files, file difference, what changed in this file, are these files identical, unified diff of files, are these two files the same, check if files match.",
    # --- Codec (plugin) ---
    "hash_text": "Use when: hash a string, checksum, SHA-256, MD5, SHA-1, BLAKE2, compute hash, verify integrity, file hash.",
    "base64_encode": "Use when: base64 encode, encode to base64, binary to text encoding, encode string.",
    "base64_decode": "Use when: base64 decode, decode base64, decode encoded text, decode token.",
    "url_encode": "Use when: URL encode, percent encode, encode for URL, encode special characters, make safe for URLs, encode string for web.",
    "url_decode": "Use when: URL decode, decode percent encoding, decode URL string.",
    "slugify": "Use when: slugify, create slug, URL-safe name, convert to dashes, clean filename from text, turn into web-friendly URL, make URL safe, generate clean slug.",
    # --- JSON tools (plugin) ---
    "json_query": "Use when: query JSON, JSON path lookup, extract value from JSON, dotted path access, nested field lookup.",
    "json_validate": "Use when: validate JSON, check JSON schema, is this valid JSON, validate structure.",
    # --- System stats (plugin) ---
    "system_status": "Use when: system status, CPU usage, memory usage, disk space, RAM, load average, host stats, host health, server performance, how hot is the server, machine resources, system health check, how much memory is free, server health, is the machine overloaded, resource usage, performance check, how much memory is the machine using, what resources are available, system load.",
    # --- Notes (plugin) ---
    "note_save": "Use when: save a note, write a note, remember this, jot down, scratchpad, save for later, make a note, quick note.",
    "note_list": "Use when: list notes, show my notes, what notes do I have, find notes, notes with tag.",
    "note_read": "Use when: read a note, show note contents, display saved note, retrieve note.",
}


@dataclass
class ToolEmbeddingIndex:
    """Pre-computed vector index for semantic tool routing.

    Built lazily on first use via the ``build`` classmethod.  Query-time
    selection (``select_tools``) is synchronous and sub-millisecond for
    typical tool counts (~60 tools).
    """

    SIMILARITY_THRESHOLD: float = 0.35
    MIN_TOOLS: int = 8
    MAX_TOOLS: int = 20

    ALWAYS_AVAILABLE: frozenset = field(
        default_factory=lambda: frozenset(
            {"search_memories", "recall_concepts", "web_search", "web_fetch"}
        )
    )

    # Populated by build()
    _tool_names: List[str] = field(default_factory=list)
    _matrix: Optional[np.ndarray] = None  # shape (N, D), L2-normalized rows
    _dimension: int = 0
    _built_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls,
        embeddings_svc: Any,
        tools: List[ToolEntry],
    ) -> "ToolEmbeddingIndex":
        """Pre-compute embeddings for all registered tools.

        Returns a ready-to-query index.  If embedding fails the index is
        returned but ``is_ready`` will be False.
        """
        if os.environ.get("OPENCAS_DISABLE_TOOL_EMBEDDINGS", "").strip() == "1":
            log.info("tool embedding index disabled by env var")
            return cls()

        if not tools:
            return cls()

        texts = [_build_tool_semantic_text(t) for t in tools]
        try:
            records = await embeddings_svc.embed_batch(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except Exception:
            log.warning("tool embedding index build failed", exc_info=True)
            return cls()

        vectors = [r.vector for r in records]
        matrix = np.array(vectors, dtype=np.float32)
        # L2-normalize rows for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        idx = cls()
        idx._tool_names = [t.name for t in tools]
        idx._matrix = matrix
        idx._dimension = matrix.shape[1]
        idx._built_at = datetime.now(timezone.utc)
        log.info(
            "tool embedding index built: %d tools, dim=%d",
            len(idx._tool_names),
            idx._dimension,
        )
        return idx

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._matrix is not None and len(self._tool_names) > 0

    def select_tools(
        self,
        objective_vector: np.ndarray,
        all_tools: List[ToolEntry],
    ) -> List[ToolEntry]:
        """Rank tools by cosine similarity and return the best subset.

        Always prepends tools in ``ALWAYS_AVAILABLE``.  Then picks the
        top-K remaining tools by similarity, respecting MIN/M MAX bounds.
        """
        tool_map = {t.name: t for t in all_tools}

        # Always-available tools go first
        always: List[ToolEntry] = []
        remaining_names: set[str] = set(self._tool_names)
        for name in self.ALWAYS_AVAILABLE:
            entry = tool_map.get(name)
            if entry:
                always.append(entry)
                remaining_names.discard(name)

        if not self.is_ready or objective_vector.ndim != 1:
            return always + [t for t in all_tools if t.name not in self.ALWAYS_AVAILABLE][: self.MAX_TOOLS - len(always)]

        # Normalize objective vector
        vec = objective_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Compute similarities for non-always tools
        name_set = remaining_names
        indices = [i for i, n in enumerate(self._tool_names) if n in name_set]
        if not indices:
            return always

        sub_matrix = self._matrix[indices]
        similarities = sub_matrix @ vec  # dot product on L2-normalized = cosine

        # Rank by similarity
        ranked_idx = np.argsort(similarities)[::-1]

        above_threshold: List[str] = []
        for i in ranked_idx:
            if similarities[i] >= self.SIMILARITY_THRESHOLD:
                above_threshold.append(self._tool_names[indices[i]])
            else:
                break

        # Floor: ensure at least MIN_TOOLS total
        min_extra = max(0, self.MIN_TOOLS - len(always))
        if len(above_threshold) < min_extra:
            seen = set(above_threshold)
            for i in ranked_idx:
                name = self._tool_names[indices[i]]
                if name not in seen:
                    above_threshold.append(name)
                    seen.add(name)
                if len(above_threshold) >= min_extra:
                    break

        # Ceiling: cap at MAX_TOOLS total
        budget = self.MAX_TOOLS - len(always)
        above_threshold = above_threshold[:budget]

        selected = always + [tool_map[n] for n in above_threshold if n in tool_map]
        selected = self._include_browser_session_starter(selected, tool_map)
        return selected

    def _include_browser_session_starter(
        self,
        selected: List[ToolEntry],
        tool_map: Dict[str, ToolEntry],
    ) -> List[ToolEntry]:
        selected_names = {entry.name for entry in selected}
        needs_browser_session = any(
            name.startswith("browser_") and name != "browser_start"
            for name in selected_names
        )
        if not needs_browser_session or "browser_start" in selected_names:
            return selected
        starter = tool_map.get("browser_start")
        if starter is None:
            return selected
        always_count = sum(1 for entry in selected if entry.name in self.ALWAYS_AVAILABLE)
        selected.insert(always_count, starter)
        if len(selected) > self.MAX_TOOLS:
            selected.pop()
        return selected

    def rebuild_for_tools(
        self,
        embeddings_svc: Any,
        tools: List[ToolEntry],
    ) -> Any:
        """Return an awaitable that rebuilds the index (e.g. after plugin load)."""
        return self.build(embeddings_svc, tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tool_semantic_text(entry: ToolEntry) -> str:
    """Compose the rich text to embed for a single tool."""
    hint = TOOL_SEMANTIC_HINTS.get(entry.name, "")
    parts = [f"{entry.name}: {entry.description}"]
    if hint:
        parts.append(hint)
    return " ".join(parts)
