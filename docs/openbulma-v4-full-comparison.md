# OpenCAS vs OpenBulma-v4: Comprehensive Feature Comparison

**Generated:** 2026-04-10
**Scope:** Deep architectural analysis across 10 expert domains

---

## Executive Summary

| Dimension | OpenCAS | OpenBulma-v4 |
|------------|--------|--------------|
| **Language** | Python (async) | TypeScript (Node.js) |
| **Architecture** | Local-first, embedded | Distributed, multi-service |
| **Philosophy** | Persistent autonomous agent | High-trust autonomous partner |
| **Target** | Self-operating partner | Creative collaborator |
| **Complexity** | ~100+ Python modules | ~60+ TypeScript modules |

---

## Table 1: Core Architecture

| Feature | OpenCAS | OpenBulma-v4 | Better |
|---------|--------|--------------|-------|
| **Runtime Loop** | AgentScheduler (4 lanes) | BulmaAssistantAgent (queue) | **OpenCAS** (lane-based) |
| **Persistence** | SQLite (single file) | Postgres + Qdrant | **OpenBulma-v4** (vector search) |
| **Memory Model** | Episode + Memory + Edges | MemoryFabric | **OpenBulma-v4** (richer) |
| **Tool Registry** | Centralized ToolRegistry | Decentralized per-adapter | **OpenCAS** (centralized) |
| **Checkpointing** | Git-based (tags) | Git-based (incremental) | **OpenBulma-v4** (incremental) |
| **API Server** | FastAPI | Express.js | Tie |
| **Dashboard** | htmx + Alpine.js | React + Tailwind + D3 | **OpenBulma-v4** (visualization) |
| **Execution Model** | Lane-based BAA | Task queue + lifecycle | **OpenCAS** (lanes + dependency) |

---

## Table 2: Memory & Embedding Systems

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Storage Backend** | SQLite | Postgres + Qdrant | **OpenBulma-v4** |
| **Default Embedding** | Local 256-dim hash | Hash 64-dim + full 3072 | **OpenBulma-v4** |
| **Vector Index** | Tiered (Q→HNSW→SQL) | Qdrant (ANN) | **OpenBulma-v4** |
| **Retrieval Signals** | 8 weighted | 8+ with profiles | **OpenBulma-v4** |
| **Semantic Fusion** | RRF + weighted | Weighted + penalties | **OpenBulma-v4** |
| **Personal Recall** | Intent detection | Intent + different weights | **OpenBulma-v4** |
| **MMR/Diversity** | Yes | Yes (diversity penalty) | Tie |
| **Consolidation** | Cluster-based | O(n²) pair-wise | **OpenBulma-v4** |
| **Confidence Evolution** | Via salience | Usage counters | **OpenBulma-v4** |
| **Narrative Threads** | No | Yes | **OpenBulma-v4** |
| **Emotion Timeline** | No | Yes | **OpenBulma-v4** |
| **Edge Indexing** | Basic traversal | O(degree) lookups | **OpenBulma-v4** |
| **Deployment** | Zero-config | Requires 3 services | **OpenCAS** |
| **Local Fallback** | SQLite brute-force | Lexical (Jaccard) | **OpenCAS** |

**OpenCAS Advantages:**
- Zero-config embedded operation
- HNSW fallback without Qdrant
- Clean tiered architecture

---

## Table 3: Agent Autonomy & Loops

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Main Loop** | AgentRuntime (4 modes) | BulmaAssistantAgent | Tie |
| **Scheduler** | 4-lane (CHAT/BAA/CONSOL/CRON) | Single queue | **OpenCAS** |
| **Lane Separation** | Yes | No | **OpenCAS** |
| **Task Dependencies** | Yes | No | **OpenCAS** |
| **Daydream Engine** | SparkRouter + Evaluator | DaydreamService | Tie |
| **Work Promotion** | CreativeLadder (7 stages) | WorkProductStore | **OpenCAS** (formal) |
| **Executive Queue** | ExecutiveState + TaskQueue | ExecutiveWorkspace | Tie |
| **Recovery Loop** | 10-attempt cap + backoff | Convergence guard | **OpenCAS** (explicit cap) |
| **Tool Loop Guard** | Circuit breaker (24 rounds) | No | **OpenCAS** |
| **Somatic State** | 4 dimensions | 8+ dimensions | **OpenBulma-v4** |
| **Musubi Tracking** | 4 dimensions | 8+ + micro-gain | **OpenBulma-v4** |

---

## Table 4: Tool Registry & Execution

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Registry Type** | Centralized | Decentralized | **OpenCAS** |
| **Risk Tiers** | 6 tiers (READONLY→DESTRUCTIVE) | Family-based | **OpenCAS** |
| **Tool Count** | 30+ (11 categories) | ~10 (3 categories) | **OpenCAS** |
| **Filesystem Tools** | fs_read/write/list | fs_* | Tie |
| **Shell Tools** | bash + Docker sandbox | bash | **OpenCAS** (Docker) |
| **Browser Tools** | Full control (8 actions) | Limited (3 actions) | **OpenCAS** |
| **PTY Tools** | Full PTY management | No | **OpenCAS** |
| **Process Tools** | Background proc management | No | **OpenCAS** |
| **Workflow Tools** | Composite (8 high-level) | No | **OpenCAS** |
| **MCP Support** | Protocol bridge | No | **OpenCAS** |
| **Validation Pipeline** | 5 validators + learning | 4 validators | **OpenCAS** |
| **Smart Command** | Learning classifier | No | **OpenCAS** |
| **Path Validator** | Allowed roots | Allowed roots | Tie |
| **Watchlist** | Credentials files | Same | Tie |

---

## Table 5: Identity & Self-Model

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Self-Model** | SelfModel (narrative, values, traits, goals) | IdentityProfile (richer) | **OpenBulma-v4** |
| **Narrative** | Manual | Auto-generated | **OpenBulma-v4** |
| **Memory Anchors** | No | Yes (8 most salient) | **OpenBulma-v4** |
| **Activity Types** | Boot/import only | 6 types + rolling | **OpenBulma-v4** |
| **Theme Extraction** | No | Yes (term frequency) | **OpenBulma-v4** |
| **Formal ToM** | Belief/Intention + confidence | No | **OpenCAS** |
| **Belief Confidence** | Yes (0.0-1.0) | No | **OpenCAS** |
| **Evidence Tracking** | Yes | No | **OpenCAS** |
| **Consistency Check** | Yes (boundary, pref, polarity) | No | **OpenCAS** |
| **Identity Rebuild** | LLM synthesis + graph walk | Theme extraction + quota | Tie |
| **Rebuild Audit** | No | Yes | **OpenBulma-v4** |
| **Workspace Policy** | No | Yes (quarantine) | **OpenBulma-v4** |

---

## Table 6: Governance & Safety

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Risk Tiers** | 6 tiers (score-based) | Config-driven | **OpenCAS** |
| **Approval Logic** | Multi-signal scoring | Regex-based blocking | **OpenCAS** |
| **Trust Tracking** | Per-session level | OwnerTrustPolicy | **OpenBulma-v4** |
| **Approval Levels** | DENIED / ESCALATE / CONDITIONAL / APPROVED | ApprovalManager | **OpenCAS** |
| **Historical Evidence** | Yes (self-beliefs) | No | **OpenCAS** |
| **Somatic Modulation** | Yes | Yes | Tie |
| **Musubi Modulation** | Yes | Yes | Tie |
| **Governance Ledger** | Yes (durable) | No | **OpenCAS** |
| **Hook Bus** | PRE_TOOL/COMMAND/FILE_WRITE | pre_tool_use | **OpenCAS** |
| **Refusal Gate** | ConversationalRefusalGate | No | **OpenCAS** |
| **Policy Enforcement** | ToolValidationPipeline | SafetyPolicy | Tie |

---

## Table 7: API & Dashboard

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **API Framework** | FastAPI | Express.js | Tie |
| **REST Endpoints** | 50+ (10 categories) | 30+ (8 categories) | **OpenCAS** |
| **Operations API** | Full (sessions, receipts, tasks, qualification) | Basic tasks only | **OpenCAS** |
| **WebSocket** | Yes (bridge + events) | WebChat server | Tie |
| **Dashboard Stack** | htmx + Alpine + Chart.js | React + Tailwind + D3 | **OpenBulma-v4** |
| **Graph Viz** | No | D3.js | **OpenBulma-v4** |
| **Real-time Polling** | HTMX triggers | React state | **OpenBulma-v4** |
| **Session Management** | PTY/browser/process control | No | **OpenCAS** |
| **Operator Actions** | Kill, input, navigate, click | No | **OpenCAS** |
| **Qualification System** | Live validation + rerun tracking | No | **OpenCAS** |
| **Retrieval Debug** | Fusion weights exposed | No | **OpenCAS** |
| **Memory Endpoints** | Moderate | Rich (backfill, reindex, timeline) | **OpenBulma-v4** |

---

## Table 8: Somatic & Relational

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **State Dimensions** | 4 (trust, resonance, presence, attunement) | 8+ (valence, arousal, stress, fatigue, focus, certainty, intensity, musubi) | **OpenBulma-v4** |
| **Musubi Composite** | Yes (-1 to 1) | Yes (0-1) | **OpenBulma-v4** |
| **Micro-gain Dynamics** | No | Yes (+0.01/message) | **OpenBulma-v4** |
| **Absence Decay** | -0.03/heartbeat | -0.02/day | Tie |
| **Event Updates** | Yes | Yes | Tie |
| **Runtime Modulation** | Memory, creative, approval | Daydream, context | Tie |
| **Homeostatic Tracking** | No | Yes (declining/steady/rising/bouncing) | **OpenBulma-v4** |
| **Persistence** | SQLite MusubiStore | JSON + memory | Tie |

---

## Table 9: Execution Receipts & Audit

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Receipt Store** | ExecutionReceiptStore (SQLite) | ExecutionReceiptStore | Tie |
| **Task History** | TaskTransitionRecord | TaskLifecycleMachine | Tie |
| **Lifecycle Stages** |queued→planning→executing→verifying→done | Same | Tie |
| **Stage Timestamps** | Yes | Yes | Tie |
| **Auto-resume** | Tasks held until ready | Tasks auto-resumed | **OpenBulma-v4** |
| **Approval Ledger** | Yes (governance/) | No | **OpenCAS** |
| **Operator Actions** | jsonl history | No | **OpenCAS** |

---

## Table 10: Deployment & Dependencies

| Feature | OpenCAS | OpenBulma-v4 | Winner |
|---------|--------|--------------|-------|
| **Runtime** | Python 3.11+ | Node.js 18+ | Tie |
| **Database** | SQLite (built-in) | Postgres (external) | **OpenCAS** |
| **Vector Index** | Optional Qdrant | Qdrant (required for embeddings) | **OpenCAS** |
| **LLM Gateway** | open_llm_auth | embedFn injection | Tie |
| **Docker** | Optional | Not built-in | N/A |
| **Configuration** | BootstrapConfig | AppConfig + RuntimeConfigManager | Tie |
| **Skill Registry** | Yes (plugin-based) | Yes | Tie |
| **Single Instance** | RuntimeInstanceLock | RuntimeInstanceLock | Tie |

---

## Summary: Winner by Category

| Category | Winner | Reason |
|----------|--------|--------|
| **Memory Systems** | OpenBulma-v4 | True vector search, narrative threads, richer retrieval |
| **Tool Execution** | OpenCAS | More tools, lane-based, centralized registry |
| **Identity & ToM** | OpenCAS | Formal beliefs/intentions, consistency checking |
| **Governance** | OpenCAS | Multi-signal scoring, ledger, hooks |
| **Somatic/Relational** | OpenBulma-v4 | 8+ dimensions, micro-gain dynamics |
| **API Coverage** | OpenCAS | Operations, qualification, retrieval debug |
| **Dashboard** | OpenBulma-v4 | D3 visualization, React ecosystem |
| **Deployment** | OpenCAS | Zero-config embedded |

---

## Cross-Pollination Recommendations

### OpenCAS Could Adopt from OpenBulma-v4

1. **Confidence Evolution** — Persist usage counters on Episode, use in retrieval scoring
2. **Recall-Specific Weight Profiles** — Different fusion weights for personal recall intent
3. **Narrative Thread Detection** — Cluster temporally-adjacent related episodes
4. **Emotion History** — Track emotion shifts per episode over time
5. **Incremental Checkpoints** — Git-based with file-specific deltas
6. **D3 Visualization** — Add graph/network visualization to dashboard
7. **Activity Types** — Expand tracking beyond boot/import events

### OpenBulma-v4 Could Adopt from OpenCAS

1. **Centralized ToolRegistry** — Unify decentralized tool definitions
2. **Lane-Based Queue** — Separate CHAT/BAA/CONSOLIDATION/CRON execution
3. **Formal ToM** — Belief confidence, evidence, consistency checking
4. **Governance Ledger** — Durable approval audit trail
5. **ToolLoopGuard** — Circuit breaker for infinite loops
6. **Operations Control Plane** — PTY/browser/process session management
7. **Qualification System** — Live validation with rerun tracking

---

## Overlap Analysis

### Shared Patterns (High Overlap)

| Pattern | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| **Risk Tier Classification** | READONLY→DESTRUCTIVE | Same concept |
| **Git Checkpointing** | tags | tags + incremental |
| **Tool Validation Pipeline** | chain of validators | chain of validators |
| **BAA Queue** | asyncio.Future queue | task array |
| **Memory Edges** | 10 weight types | same 10 types |
| **Nightly Consolidation** | cluster-based | O(n²) pair-wise |
| **Musubi Tracking** | trust/resonance/presence/attunement | musubi in somatic state |

### Different Approaches

| Pattern | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| **Embedding** | Local-first fallback | LLM-powered primary |
| **Storage** | Embedded SQLite | Postgres + Qdrant |
| **Dashboard** | Server-driven (HTMX) | Client-driven (React) |
| **Beliefs** | Explicit ToM store | Implicit in identity |
| **Tool Scope** | 30+ tools | ~10 tools |

---

## Final Assessment

**OpenCAS is the more capable operational system:**
- Full PTY/browser/process control plane
- Formal Theory of Mind with beliefs/intentions
- Lane-based execution with dependencies
- Comprehensive qualification system
- Zero-config embedded deployment

**OpenBulma-v4 has richer memory and identity:**
- True vector search via Qdrant
- Narrative thread detection
- Emotion timeline
- D3 graph visualization
- More sophisticated identity extraction

**Both represent mature autonomous agent architectures** with the same core philosophy of high-trust self-approval, but they optimize for different use cases. OpenCAS is built for operational autonomy with extensive tooling, while OpenBulma-v4 is built for creative collaboration with deep memory.
