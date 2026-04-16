"""OpenCAS Import Monitoring Dashboard.

Serves a live HTML dashboard on http://localhost:8765 showing real-time
import progress, DB sizes, phase timeline, and log tail.

Usage:
    source .venv/bin/activate
    python scripts/import_dashboard.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / ".opencas"
CHECKPOINT_PATH = REPO_ROOT / ".opencas_import_checkpoint.json"
LOG_FILE = Path("/tmp/opencas_import.log")
TOTAL_EPISODES = 3108
ALL_PHASES = [
    "episodes", "edges", "embeddings", "identity", "daydreams",
    "workspaces", "somatic", "executive", "continuity", "tasks",
    "skills", "governance", "execution_receipts", "harness",
    "relational", "executive_events", "task_plans", "sessions",
    "memory_aux", "cutover", "finalized",
]
IMPORT_START_TIME = "2026-04-11 21:27:32"  # from log


# ── Data collection ───────────────────────────────────────────────────────────

def get_checkpoint() -> dict:
    try:
        return json.loads(CHECKPOINT_PATH.read_text())
    except Exception:
        return {"completed_phases": [], "counts": {}}


def get_db_sizes() -> dict[str, int]:
    sizes = {}
    if STATE_DIR.exists():
        for p in sorted(STATE_DIR.glob("*.db")):
            sizes[p.stem] = p.stat().st_size
    return sizes


def get_row_counts() -> dict[str, int]:
    counts = {}
    db_queries = {
        "memory": [
            ("episodes", "SELECT COUNT(*) FROM episodes"),
            ("episode_edges", "SELECT COUNT(*) FROM episode_edges"),
        ],
        "embeddings": [
            ("embeddings", "SELECT COUNT(*) FROM embedding_records"),
        ],
        "daydream": [
            ("daydream_sparks", "SELECT COUNT(*) FROM daydream_sparks"),
        ],
        "tom": [
            ("beliefs", "SELECT COUNT(*) FROM beliefs"),
            ("intentions", "SELECT COUNT(*) FROM intentions"),
        ],
    }
    for db_name, queries in db_queries.items():
        db_path = STATE_DIR / f"{db_name}.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
            for label, sql in queries:
                try:
                    row = conn.execute(sql).fetchone()
                    counts[label] = row[0] if row else 0
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass
    return counts


def get_log_stats() -> dict:
    if not LOG_FILE.exists():
        return {"embed_calls": 0, "recent_lines": [], "last_ts": ""}

    # Count embedding calls
    try:
        result = subprocess.run(
            ["grep", "-c", "HTTP Request: POST.*embeddings", str(LOG_FILE)],
            capture_output=True, text=True, timeout=5,
        )
        embed_calls = int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        embed_calls = 0

    # Recent non-HTTP lines (last 20)
    try:
        result = subprocess.run(
            ["grep", "-v", "HTTP Request", str(LOG_FILE)],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        recent_lines = lines[-20:] if lines else []
        last_ts = recent_lines[-1][:23] if recent_lines else ""
    except Exception:
        recent_lines = []
        last_ts = ""

    return {
        "embed_calls": embed_calls,
        "recent_lines": recent_lines,
        "last_ts": last_ts,
    }


def get_elapsed() -> str:
    try:
        start = time.strptime(IMPORT_START_TIME, "%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - time.mktime(start)
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    except Exception:
        return "?"


def build_status() -> dict:
    checkpoint = get_checkpoint()
    completed = [p.lower() for p in checkpoint.get("completed_phases", [])]
    counts = checkpoint.get("counts", {})

    # Determine current active phase
    active_phase = None
    for phase in ALL_PHASES:
        if phase not in completed:
            active_phase = phase
            break

    db_sizes = get_db_sizes()
    row_counts = get_row_counts()
    log_stats = get_log_stats()

    embed_calls = log_stats["embed_calls"]
    embed_pct = round(embed_calls / TOTAL_EPISODES * 100, 1) if TOTAL_EPISODES else 0

    phases_info = []
    for phase in ALL_PHASES:
        if phase in completed:
            status = "complete"
        elif phase == active_phase:
            status = "active"
        else:
            status = "pending"
        phases_info.append({"name": phase, "status": status})

    return {
        "elapsed": get_elapsed(),
        "phases": phases_info,
        "completed_count": len(completed),
        "total_phases": len(ALL_PHASES),
        "pct_phases": round(len(completed) / len(ALL_PHASES) * 100, 1),
        "active_phase": active_phase or "done",
        "import_counts": {
            "episodes": counts.get("episodes", 0),
            "edges": counts.get("edges", 0),
            "embed_calls": embed_calls,
            "embed_total": TOTAL_EPISODES,
            "embed_pct": embed_pct,
        },
        "row_counts": row_counts,
        "db_sizes": db_sizes,
        "recent_log": log_stats["recent_lines"],
        "last_ts": log_stats["last_ts"],
    }


# ── HTTP server ───────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenCAS Import Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --purple: #bc8cff; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; }
  .badge { padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-running { background: #1c3a1c; color: var(--green); border: 1px solid var(--green); }
  .badge-done    { background: #1a2f52; color: var(--blue);  border: 1px solid var(--blue); }
  .elapsed { margin-left: auto; color: var(--muted); font-size: 13px; }
  .last-ts { color: var(--muted); font-size: 12px; }
  main { padding: 20px 24px; display: grid; gap: 20px; }

  /* stats strip */
  .stats-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .stat-card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
  .stat-card .value { font-size: 28px; font-weight: 700; }
  .stat-card .sub   { color: var(--muted); font-size: 12px; margin-top: 2px; }

  /* phase timeline */
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .panel h2 { font-size: 14px; font-weight: 600; margin-bottom: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .phases { display: flex; flex-wrap: wrap; gap: 6px; }
  .phase-chip { padding: 5px 11px; border-radius: 6px; font-size: 12px; font-weight: 500; }
  .phase-complete { background: #1c3a1c; color: var(--green); border: 1px solid #2ea043; }
  .phase-active   { background: #2d2016; color: var(--yellow); border: 1px solid #d29922; animation: pulse 1.5s infinite; }
  .phase-pending  { background: #1a1f26; color: var(--muted); border: 1px solid var(--border); }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.6 } }

  /* progress bar */
  .progress-wrap { margin-top: 14px; }
  .progress-label { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; color: var(--muted); }
  .progress-bar-bg { background: #1a1f26; border-radius: 6px; height: 12px; overflow: hidden; }
  .progress-bar-fill { height: 100%; border-radius: 6px; transition: width .5s ease; }
  .fill-green  { background: var(--green); }
  .fill-yellow { background: var(--yellow); }

  /* charts row */
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .charts-row { grid-template-columns: 1fr; } }
  .chart-wrap { position: relative; height: 280px; }

  /* log */
  .log-box { font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; background: #090d12;
             border: 1px solid var(--border); border-radius: 6px; padding: 12px;
             height: 260px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
  .log-line { line-height: 1.6; }
  .log-info    { color: #58a6ff; }
  .log-warning { color: #d29922; }
  .log-error   { color: #f85149; }
</style>
</head>
<body>
<header>
  <h1>OpenCAS Import Monitor</h1>
  <span id="status-badge" class="badge badge-running">RUNNING</span>
  <span class="elapsed">Elapsed: <strong id="elapsed">—</strong></span>
  <span class="last-ts">Last event: <span id="last-ts">—</span></span>
</header>
<main>
  <!-- Stats strip -->
  <div class="stats-strip">
    <div class="stat-card">
      <div class="label">Phases Complete</div>
      <div class="value" id="phases-done">—</div>
      <div class="sub" id="phases-sub">of 21</div>
    </div>
    <div class="stat-card">
      <div class="label">Active Phase</div>
      <div class="value" style="font-size:18px;padding-top:6px" id="active-phase">—</div>
      <div class="sub">&nbsp;</div>
    </div>
    <div class="stat-card">
      <div class="label">Episodes</div>
      <div class="value" id="stat-episodes">—</div>
      <div class="sub">imported</div>
    </div>
    <div class="stat-card">
      <div class="label">Edges</div>
      <div class="value" id="stat-edges">—</div>
      <div class="sub">graph edges</div>
    </div>
    <div class="stat-card">
      <div class="label">Embeddings</div>
      <div class="value" id="stat-embed">—</div>
      <div class="sub" id="stat-embed-sub">/ 3108 backfilled</div>
    </div>
    <div class="stat-card">
      <div class="label">State DB Total</div>
      <div class="value" id="stat-db-total">—</div>
      <div class="sub">across all DBs</div>
    </div>
  </div>

  <!-- Phase timeline -->
  <div class="panel">
    <h2>Phase Timeline</h2>
    <div class="phases" id="phases-list"></div>
    <div class="progress-wrap">
      <div class="progress-label">
        <span>Overall Progress</span>
        <span id="overall-pct">0%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill fill-green" id="overall-bar" style="width:0%"></div>
      </div>
    </div>
    <div class="progress-wrap" style="margin-top:10px">
      <div class="progress-label">
        <span>Embedding Backfill</span>
        <span id="embed-pct">0%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill fill-yellow" id="embed-bar" style="width:0%"></div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="panel">
      <h2>Database Sizes</h2>
      <div class="chart-wrap"><canvas id="dbChart"></canvas></div>
    </div>
    <div class="panel">
      <h2>Row Counts</h2>
      <div class="chart-wrap"><canvas id="rowChart"></canvas></div>
    </div>
  </div>

  <!-- Log -->
  <div class="panel">
    <h2>Live Log (non-HTTP)</h2>
    <div class="log-box" id="log-box"></div>
  </div>
</main>

<script>
let dbChart = null, rowChart = null;

function fmt(n) {
  if (n >= 1073741824) return (n/1073741824).toFixed(1) + ' GB';
  if (n >= 1048576)    return (n/1048576).toFixed(1) + ' MB';
  if (n >= 1024)       return (n/1024).toFixed(0) + ' KB';
  return n + ' B';
}
function fmtNum(n) {
  return n >= 1000 ? n.toLocaleString() : String(n);
}

const PALETTE = [
  '#58a6ff','#3fb950','#d29922','#bc8cff','#f85149',
  '#39d353','#ffa657','#ff7b72','#79c0ff','#56d364',
  '#e3b341','#a5d6ff','#7ee787','#d2a8ff','#ffa198',
];

function mkChart(id, labels, data, fmtFn) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: PALETTE.slice(0, labels.length),
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => fmtFn ? fmtFn(ctx.raw) : fmtNum(ctx.raw)
          }
        }
      },
      scales: {
        x: { ticks: { color: '#8b949e', callback: v => fmtFn ? fmtFn(v) : fmtNum(v) }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#e6edf3', font: { size: 11 } }, grid: { display: false } },
      }
    }
  });
}

function updateChart(chart, labels, data) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.data.datasets[0].backgroundColor = PALETTE.slice(0, labels.length);
  chart.update('none');
}

function classifyLog(line) {
  if (/ERROR|Exception|Traceback/.test(line)) return 'log-error';
  if (/WARNING|WARN/.test(line)) return 'log-warning';
  return 'log-info';
}

async function refresh() {
  let data;
  try {
    const r = await fetch('/api/status');
    data = await r.json();
  } catch (e) { return; }

  // Header
  document.getElementById('elapsed').textContent = data.elapsed;
  document.getElementById('last-ts').textContent = data.last_ts || '—';
  const badge = document.getElementById('status-badge');
  if (data.active_phase === 'done') {
    badge.textContent = 'COMPLETE';
    badge.className = 'badge badge-done';
  }

  // Stats
  document.getElementById('phases-done').textContent = data.completed_count;
  document.getElementById('phases-sub').textContent = `of ${data.total_phases}`;
  document.getElementById('active-phase').textContent = data.active_phase.replace(/_/g,' ');
  document.getElementById('stat-episodes').textContent = fmtNum(data.import_counts.episodes || data.row_counts.episodes || 0);
  document.getElementById('stat-edges').textContent = fmtNum(data.import_counts.edges || data.row_counts.episode_edges || 0);
  document.getElementById('stat-embed').textContent = fmtNum(data.import_counts.embed_calls);
  document.getElementById('stat-embed-sub').textContent = `/ ${fmtNum(data.import_counts.embed_total)} backfilled`;

  const dbTotal = Object.values(data.db_sizes).reduce((a,b) => a+b, 0);
  document.getElementById('stat-db-total').textContent = fmt(dbTotal);

  // Phase chips
  const list = document.getElementById('phases-list');
  list.innerHTML = data.phases.map(p =>
    `<span class="phase-chip phase-${p.status}">${p.name.replace(/_/g,' ')}</span>`
  ).join('');

  // Progress bars
  document.getElementById('overall-bar').style.width = data.pct_phases + '%';
  document.getElementById('overall-pct').textContent = data.pct_phases + '%';
  document.getElementById('embed-bar').style.width = data.import_counts.embed_pct + '%';
  document.getElementById('embed-pct').textContent = data.import_counts.embed_pct + '%';

  // DB sizes chart
  const dbEntries = Object.entries(data.db_sizes).sort((a,b) => b[1]-a[1]);
  const dbLabels = dbEntries.map(e => e[0]);
  const dbValues = dbEntries.map(e => e[1]);
  if (!dbChart) dbChart = mkChart('dbChart', dbLabels, dbValues, fmt);
  else updateChart(dbChart, dbLabels, dbValues);

  // Row counts chart
  const rowEntries = Object.entries(data.row_counts).sort((a,b) => b[1]-a[1]);
  const rowLabels = rowEntries.map(e => e[0]);
  const rowValues = rowEntries.map(e => e[1]);
  if (!rowChart) rowChart = mkChart('rowChart', rowLabels, rowValues, null);
  else updateChart(rowChart, rowLabels, rowValues);

  // Log
  const logBox = document.getElementById('log-box');
  const wasBottom = logBox.scrollHeight - logBox.clientHeight <= logBox.scrollTop + 5;
  logBox.innerHTML = data.recent_log.map(line =>
    `<div class="log-line ${classifyLog(line)}">${line.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</div>`
  ).join('');
  if (wasBottom) logBox.scrollTop = logBox.scrollHeight;
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, "text/html", HTML.encode())
        elif self.path == "/api/status":
            try:
                data = build_status()
                body = json.dumps(data).encode()
                self._send(200, "application/json", body)
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                self._send(500, "application/json", body)
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Import dashboard running at http://localhost:{PORT}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
