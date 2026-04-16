# OpenCAS vs OpenBulma-v4: Quick Comparison Tables

## Executive Overview

| | **OpenCAS** | **OpenBulma-v4** |
|---|-------------|------------------|
| **Language** | Python 3.11+ | TypeScript / Node.js 18+ |
| **Architecture** | Local-first, embedded | Distributed, multi-service |
| **Primary Storage** | SQLite (single file) | PostgreSQL + Qdrant |
| **API Framework** | FastAPI | Express.js |
| **Dashboard** | htmx + Alpine.js + Chart.js | React + Tailwind + D3 |
| **Philosophy** | Persistent autonomous agent | High-trust creative partner |

---

## Core Systems Comparison

### 1. Memory & Embeddings

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| Storage | SQLite (embedded) | Postgres + Qdrant (external) |
| Default embedding | Local 256-dim hash | Hash 64-dim + optional 3072-dim |
| Vector index | Tiered fallback (Q→HNSW→SQL) | Qdrant ANN |
| Retrieval signals | 8 weighted | 8+ profiles |
| Consolidation | Cluster-based | O(n²) pair-wise |
| **Winner** | Deployment simplicity | Richer memory |

### 2. Tool Execution

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| Registry | Centralized ToolRegistry | Decentralized |
| Tool count | 30+ (11 categories) | ~10 (3 categories) |
| Execution lanes | 4 lanes (CHAT/BAA/CONSOL/CRON) | Single queue |
| PTY/Process | Full management | Not available |
| **Winner** | OpenCAS | — |

### 3. Identity & Self-Model

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| ToM | Belief/Intention + confidence | Implicit in identity |
| Consistency check | Formal verification | Not available |
| Identity rebuild | LLM synthesis | Theme extraction |
| Self-knowledge | Registry + versioned | Procedural |
| **Winner** | OpenCAS | — |

### 4. Governance & Safety

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| Risk tiers | 6 score-based tiers | Regex-based blocking |
| Approval | Multi-signal scoring | Config-driven |
| Ledger | Durable audit trail | Not available |
| Refusal gate | ConversationalRefusalGate | Not available |
| **Winner** | OpenCAS | — |

### 5. Somatic & Relational

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| State dimensions | 4 (trust/resonance/presence/attunement) | 8+ (valence/arousal/stress/fatigue/etc.) |
| Musubi | Composite score | + micro-gain dynamics |
| **Winner** | — | OpenBulma-v4 |

### 6. Operations & Control Plane

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| PTY sessions | Full control | Not available |
| Browser sessions | Full control | Basic |
| Process sessions | Management | Not available |
| Qualification system | Live validation + rerun tracking | Not available |
| **Winner** | OpenCAS | — |

### 7. Dashboard & Visualization

| Feature | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| Graph visualization | Not available | D3.js |
| Tech stack | htmx + Alpine.js | React + Tailwind |
| Real-time | HTMX polling | React state |
| **Winner** | — | OpenBulma-v4 |

---

## Category Winners

| Category | Winner | Key Advantage |
|----------|--------|-------------|
| **Memory Systems** | OpenBulma-v4 | True vector search, narrative threads |
| **Tool Execution** | OpenCAS | 30+ tools, lane-based execution |
| **Identity & ToM** | OpenCAS | Formal beliefs, consistency checking |
| **Governance** | OpenCAS | Multi-signal, ledger, hooks |
| **Somatic** | OpenBulma-v4 | 8+ dimensions, micro-gain |
| **Operations** | OpenCAS | Full control plane |
| **Dashboard** | OpenBulma-v4 | D3 visualization, React |
| **Deployment** | OpenCAS | Zero-config embedded |

---

## Key Overlaps

| Pattern | Both Implement |
|---------|---------------|
| Risk tier classification | READONLY → DESTRUCTIVE |
| Git checkpointing | Commit + tags |
| Tool validation pipeline | Chain of validators |
| BAA queue | Task lifecycle |
| Memory edges | 10 weight types |
| Nightly consolidation | Scheduled reprocessing |
| Musubi tracking | Trust/connection scoring |

---

## Key Differences

| Pattern | OpenCAS | OpenBulma-v4 |
|---------|--------|--------------|
| Embedding approach | Local-first fallback | LLM-powered primary |
| Storage | Embedded SQLite | External Postgres+Qdrant |
| Dashboard | Server-driven (HTMX) | Client-driven (React) |
| Belief tracking | Explicit ToM store | Implicit in identity |
| Tool scope | 30+ comprehensive | ~10 minimal |
| Identity rebuild | LLM synthesis | Theme extraction |

---

## Cross-Pollination Opportunities

### Adopt from OpenBulma-v4 → OpenCAS
- Confidence evolution (usage counters)
- Recall-specific weight profiles
- Narrative thread detection
- D3 graph visualization

### Adopt from OpenCAS → OpenBulma-v4
- Centralized ToolRegistry
- Lane-based execution
- Formal ToM with beliefs
- PTY/process management
- Operations control plane
