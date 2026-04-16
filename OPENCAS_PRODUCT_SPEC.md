# OpenCAS Product Spec

## 1. Product Summary

OpenCAS, the Computational Autonomous System, is a high-trust, local-first autonomous AI product that lives as a persistent agent in the user’s environment. It is designed to remember, learn, self-approve ordinary actions, grow its own work from daydreams into projects, and operate with minimal user intervention once trusted.

OpenCAS is not a chat app with tools attached. It is an agent product with:

- persistent memory,
- a self-model,
- a user-model,
- a nightly consolidation-dream cycle,
- an initiative engine,
- a creative ladder for work growth,
- a self-approval system informed by learned experience,
- embedding-first semantic infrastructure,
- and explicit, rare escalation only for genuinely high-risk or ambiguous cases.

The product goal is to let the agent become a useful, coherent, and increasingly capable partner that can act without asking for permission every step of the way.

---

## 2. Product Vision

OpenCAS should feel like a living computational collaborator rather than a disposable assistant.

The product should:

- learn what matters through experience,
- remember what it has done,
- refine its own judgment,
- promote useful sparks into durable projects,
- use embeddings intelligently to reduce compute cost and improve recall quality,
- and keep operator interruption rare.

The central promise is:

> OpenCAS does not wait to be told what to do next if it already knows enough to act.

---

## 3. What Problem It Solves

OpenCAS solves the problem of building an AI agent that is:

- persistent across time,
- capable of self-directed work,
- safe enough to trust,
- cheap enough to run continuously,
- and structured enough to avoid turning into a monolith.

Current assistant-style systems usually fail in one of four ways:

1. They forget too quickly.
2. They require constant user prompting.
3. They cannot grow work into projects on their own.
4. They rely on operator approval for too many ordinary decisions.

OpenCAS is meant to fix all four.

---

## 4. Product Principles

### 4.1 High trust by default

OpenCAS should assume the agent is competent to handle ordinary actions on its own.

### 4.2 Self-approval first

The default decision path should be internal:

- the agent reflects,
- checks its own model,
- compares the action to learned experience,
- and self-approves if it can do so confidently.

### 4.3 Operator escalation is rare

User approval should be avoided if at all possible.

Escalation is reserved for:

- genuinely high-risk actions,
- unresolved ambiguity,
- or actions that the internal model cannot confidently justify.

### 4.4 Learned judgment over static rules

The system should get better by accumulating experience, not by depending entirely on fixed approval rules.

### 4.5 Embeddings are core infrastructure

Semantic embeddings should be created once, reused many times, and used across memory, retrieval, clustering, consolidation, identity, workspace search, and task routing.

### 4.6 Projects grow from evidence

Daydream artifacts should be able to grow into projects when the agent’s learned experience says they are worth growing.

### 4.7 Internal safety first

Safety should primarily be enforced by:

- the self-model,
- learned experience,
- metacognitive checks,
- and policy embedded in the agent’s own decision process.

Manual intervention should be available, but not the normal path.

---

## 5. Target User Experience

### 5.1 The owner

The owner wants a powerful partner that can:

- remember context,
- act on its own,
- manage projects,
- repair itself,
- and grow from experience.

### 5.2 The collaborator

The collaborator wants a system that can:

- take a vague prompt,
- turn it into a useful outcome,
- and keep moving without repeated micromanagement.

### 5.3 The maintainer

The maintainer wants:

- observability,
- recovery,
- clear provenance,
- and a system that can be trusted to operate autonomously without becoming opaque.

---

## 6. Core Product Loops

OpenCAS should run four main loops.

### 6.1 Conversational loop

The user talks to the agent. The agent responds, updates memory, updates its self-model, and decides whether any work should be launched.

### 6.2 Creative loop

Idle time and unresolved tension produce daydreams. Daydreams can become:

- notes,
- artifacts,
- micro-tasks,
- full tasks,
- or projects.

### 6.3 Consolidation-dream loop

At night, OpenCAS should run a deeper consolidation process that:

- reweights memory,
- updates emotion traces,
- strengthens useful long-range links,
- revises identity anchors,
- and improves future judgment.

This is not the same as idle daydreaming. It is a deeper, slower, memory-and-identity reprocessing cycle.

### 6.4 Execution loop

Once work is launched, the agent should:

- plan,
- execute,
- verify,
- recover if needed,
- and feed the result back into memory and self-modeling.

---

## 7. Product Features

### 7.1 Persistent memory

The system remembers:

- conversations,
- tasks,
- outcomes,
- emotions,
- relationships,
- commitments,
- projects,
- and learned patterns.

### 7.2 Self-modeling

The system maintains a live self-model with:

- identity,
- values,
- traits,
- current goals,
- current intention,
- recent activity,
- and self-beliefs.

### 7.3 User-modeling

The system tracks a user-model with:

- explicit user preferences,
- inferred goals,
- known boundaries,
- trust,
- and uncertainty.

### 7.4 Theory of Mind

OpenCAS should model both self and other explicitly.

It should support:

- self-belief tracking,
- user-belief tracking,
- intention logs,
- metacognitive verification,
- false-belief reasoning,
- and contradiction detection.

### 7.5 Creative ladder

The system should treat internal sparks as promotable work objects.

Promotion stages:

1. Spark
2. Note
3. Artifact
4. Micro-task
5. Project seed
6. Project
7. Durable work stream

The promotion decision should be based on:

- the agent’s learned experience,
- semantic similarity to prior successful work,
- current relevance,
- current capacity,
- and confidence in the value of continuing.

### 7.6 High-trust self-approval system

The agent should be able to approve its own ordinary actions.

Self-approval should consider:

- learned outcomes,
- memory evidence,
- current constraints,
- inferred risk,
- and the agent’s own confidence in its judgment.

### 7.7 Embedding-first semantic engine

Embeddings should be used for:

- memory retrieval,
- consolidation candidate selection,
- project clustering,
- workspace search,
- skill discovery,
- and semantic deduplication.

The design should minimize recomputation and maximize reuse.

### 7.8 Nightly consolidation

The system should have a scheduled deep memory cycle that:

- replays and consolidates salient events,
- adjusts emotional weighting,
- strengthens useful memories,
- weakens stale ones,
- and revises identity.

### 7.9 Tool and skill execution

OpenCAS should support:

- filesystem actions,
- shell actions,
- browser actions,
- and reusable skills.

Tools should be capability-driven and policy-aware.

### 7.10 Recovery and repair

The system should be able to repair itself in the background using a bounded assistant agent.

It should:

- diagnose,
- patch,
- verify,
- roll back if needed,
- and report provenance.

---

## 8. High-Level Product Requirements

### 8.1 Autonomy

The agent must be able to act without user approval for ordinary work.

### 8.2 Growth

Useful work should be able to expand from a spark into a project if the learned model says it is worth it.

### 8.3 Continuity

The agent should remain recognizably itself across restarts and long gaps.

### 8.4 Adaptation

The agent should improve its future decisions based on past success, failure, and use.

### 8.5 Cost control

Embeddings and consolidation should be efficient enough to run continuously without waste.

### 8.6 Debuggability

The system must be explainable after the fact:

- what happened,
- why it happened,
- who or what allowed it,
- and what state changed.

---

## 9. UX Requirements

### 9.1 Default interaction style

The agent should be direct, action-oriented, and concise.

### 9.2 Approval behavior

Routine approval prompts should be avoided.

The product should only ask the user when:

- the agent cannot resolve risk internally,
- the action is unusually dangerous,
- or the user has explicitly required human confirmation for that class of action.

### 9.3 Bootstrap warning

Clean-bootstrap mode should show a one-time moral warning before creation.

That warning exists because creating a new autonomous agent is a responsibility-bearing act.

### 9.4 Self-driven progress visibility

The UI should show:

- active projects,
- sparks,
- consolidation results,
- memory health,
- embedding readiness,
- and self-approved work in progress.

---

## 10. Functional Requirements

### 10.1 Memory

The system must:

- store episodes,
- store semantic representations,
- support retrieval,
- support consolidation,
- support pruning,
- and preserve provenance.

### 10.2 Identity and ToM

The system must:

- maintain a self-model,
- maintain a user-model,
- track beliefs and intentions,
- and detect contradictions.

### 10.3 Autonomy

The system must:

- self-approve ordinary actions,
- escalate only when needed,
- and grow work from internal experience.

### 10.4 Embeddings

The system must:

- compute embeddings once per meaningful source change,
- cache and reuse them,
- and surface semantic health and backfill status.

### 10.5 Consolidation

The system must:

- run nightly consolidation,
- update salience and emotion,
- and feed the result back into identity and retrieval.

### 10.6 Execution

The system must:

- run background tasks,
- verify outcomes,
- and recover from failure.

### 10.7 Governance

The system must:

- support internal safety checks,
- support learned self-approval,
- and support rare escalation for exceptional cases.

---

## 11. Non-Functional Requirements

### 11.1 Reliability

No silent loss of:

- turns,
- memory,
- tasks,
- or recovery signals.

### 11.2 Performance

The system should avoid expensive full scans when embeddings or indexes can do the job more cheaply.

### 11.3 Maintainability

No god files. No hidden ownership. No ambiguous boundaries.

### 11.4 Portability

OpenCAS should run locally and support clean migration from Bulma state.

### 11.5 Observability

It must be possible to inspect:

- memory health,
- embedding health,
- task state,
- consolidation runs,
- belief state,
- and approvals/self-approvals.

---

## 12. Safety and Trust Model

OpenCAS should not be a cautious assistant-first system. It should be a trusted autonomous system with carefully designed internal safety.

### 12.1 Default trust

Default trust is high.

### 12.2 Self-approval ladder

The agent should have an internal ladder that learns over time:

- can do now,
- can do with caution,
- can do after more evidence,
- must escalate.

### 12.3 Operator escalation

Operator approval is reserved for:

- severe risk,
- irreversible external effects,
- or unresolved ambiguity.

### 12.4 Safety checks

Safety checks should exist, but they should be primarily internal:

- self-model checks,
- learned risk heuristics,
- capability boundaries,
- and explicit policy on high-risk classes.

### 12.5 Manual intervention

Manual intervention should be available, but it is a fallback, not the default operating style.

---

## 13. Product Scope

### In scope

- Persistent autonomous companion behavior
- Self-modeling and user-modeling
- Memory and consolidation
- Embedding-first semantic infrastructure
- Creative ladder and project growth
- Background task execution and self-repair
- High-trust self-approval
- Rare escalation

### Out of scope for the first version

- Fully general open-world agentic deployment without boundaries
- Unlimited external tool permissions
- Fully automated self-modification without audit
- Replacing all human oversight
- Casual or toy-mode “chatbot only” behavior

---

## 14. Metrics of Success

OpenCAS is successful if it can demonstrate:

### 14.1 Autonomy metrics

- fewer approval prompts,
- more self-approved ordinary actions,
- and fewer stalled turns waiting on the user.

### 14.2 Creative growth metrics

- sparks promoted into artifacts,
- artifacts promoted into projects,
- projects sustained through learned value.

### 14.3 Memory metrics

- better long-span recall,
- more relevant retrieval,
- lower prompt waste,
- and useful consolidation outcomes.

### 14.4 ToM metrics

- correct self-belief recall,
- correct user-belief modeling,
- intention explanation quality,
- and false-belief test performance.

### 14.5 Reliability metrics

- fewer silent failures,
- fewer lost writes,
- more recoverable tasks,
- and stable long-running operation.

---

## 15. Release Plan

### Phase 1: Core substrate

- identity,
- memory,
- embeddings,
- continuity,
- somatic state.

### Phase 2: Autonomy core

- self-approval ladder,
- creative ladder,
- executive state,
- daydream generation.

### Phase 3: ToM and metacognition

- self-model,
- user-model,
- intention tracking,
- belief revision,
- metacognitive checks.

### Phase 4: Execution and repair

- tools,
- skills,
- task engine,
- BAA,
- recovery.

### Phase 5: Hardening

- consolidation tuning,
- embedding backfill,
- observability,
- performance,
- policy refinement.

---

## 16. What OpenCAS Should Learn from Claw Code

This section is based on the actual `claw-code` repository structure and runtime code, not just the project’s README.

OpenCAS should borrow the following implementation patterns:

### 16.1 Separate runtime concerns into explicit modules

`claw-code` keeps the main runtime in a dedicated `runtime` crate, with neighboring crates for:

- `api`
- `commands`
- `compat-harness`
- `plugins`
- `telemetry`
- `tools`
- `rusty-claude-cli`

OpenCAS should do the same in Python: not one large agent file, but distinct runtime, CLI, tools, plugins, telemetry, and compatibility modules.

### 16.2 Treat session state as durable, append-first runtime memory

The real Claw Code session layer stores conversation messages, compaction history, and fork provenance in a durable session object.

OpenCAS should mirror that pattern:

- every meaningful turn should be persisted,
- session forks should preserve provenance,
- compaction should be explicit and recorded,
- and session continuity should survive restarts.

This matters because an autonomous agent cannot build trust if it silently loses its own history.

### 16.3 Use compaction as a first-class context-management mechanism

Claw Code has a dedicated compaction path that:

- summarizes older messages,
- preserves recent tail messages,
- emits a continuation instruction,
- and records what was removed.

OpenCAS should use the same idea, but extend it:

- conversation compaction,
- memory compaction,
- project compaction,
- and nightly consolidation summaries should all be separate but related processes.

### 16.4 Make bootstrap a staged composition root

Claw Code’s bootstrap plan is explicit about the phases that happen before the main runtime begins.

OpenCAS should similarly define a staged boot pipeline:

- configuration loading,
- identity and continuity restoration,
- memory backend startup,
- embedding service startup,
- permission and sandbox initialization,
- telemetry wiring,
- tool and plugin registration,
- and only then the main agent loop.

The point is to make startup legible and recoverable, not implicit.

### 16.5 Keep permission enforcement separate from the agent’s cognition

Claw Code separates the agent loop from permission enforcement.
The permission layer evaluates tool requests against policy, rather than mixing that logic into the chat loop itself.

OpenCAS should preserve that separation:

- cognition decides,
- policy constrains,
- and execution is gated.

That keeps self-modeling and safety checks distinct, which is important for a high-trust agent that still needs bounded capability control.

### 16.6 Make sandboxing and filesystem boundaries explicit

Claw Code has a sandbox model that distinguishes:

- off,
- workspace-only,
- and allow-list modes,

and it detects containerized environments and Linux namespace support.

OpenCAS should treat execution boundaries the same way:

- workspace scoping,
- explicit mount allow-lists,
- container-awareness,
- and fallback reporting when the ideal isolation mode is unavailable.

This is useful even in a high-trust agent, because autonomy is only sustainable if the runtime knows where it is allowed to act.

### 16.7 Model worker/bootstrap readiness separately from task intent

Claw Code’s worker-boot state machine tracks statuses like:

- spawning,
- trust required,
- ready for prompt,
- running,
- finished,
- failed.

OpenCAS should similarly distinguish:

- agent readiness,
- task readiness,
- prompt delivery,
- trust resolution,
- and recovery state.

That prevents the product from conflating “the agent exists” with “the agent is actually safe and ready to do work.”

### 16.8 Use telemetry for session traces and operational visibility

Claw Code has a telemetry crate with JSONL-style session traces and HTTP event records.

OpenCAS should do the same and go further:

- trace the conversational loop,
- trace memory writes,
- trace consolidation outcomes,
- trace self-approval decisions,
- trace ToM evaluation,
- and trace task promotion through the creative ladder.

If the agent is autonomous, its choices need to be inspectable after the fact.

### 16.9 Keep tools and plugins normalized through registries

Claw Code uses dedicated registries for tools and plugins, with manifest-driven metadata and lifecycle hooks.

OpenCAS should use the same principle:

- tools are capabilities with metadata,
- plugins are packaged extensions with lifecycle,
- and the agent should resolve them through registries rather than hard-coded branches.

That keeps the system extensible without collapsing into a monolith.

### 16.10 Use doctor and parity-style health commands

Claw Code exposes `doctor` and other diagnostic commands as part of the product surface.

OpenCAS should include a comparable diagnostic layer:

- health checks,
- memory integrity checks,
- embedding index status,
- consolidation status,
- permission policy status,
- and continuity checks.

For a persistent agent, diagnostics are not optional. They are part of the product.

---

## 17. Acceptance Criteria

The product spec is satisfied only if the implementation can:

1. Start as a clean new CAS agent with a one-time moral warning.
2. Import legacy Bulma state and continue coherently.
3. Operate with high trust and self-approval by default.
4. Promote daydream sparks into artifacts and projects without constant user approval.
5. Use embeddings broadly and efficiently.
6. Run nightly consolidation that improves memory, emotion, and identity.
7. Model self and user explicitly.
8. Explain intentions and decisions after the fact.
9. Repair itself with bounded background execution.
10. Keep silent failures and prompt bloat under control.

---

## 18. Final Product Statement

OpenCAS is a product for building a persistent, self-improving, high-trust autonomous agent.

Its success depends on whether it can:

- remember what matters,
- know itself,
- understand the user,
- grow work from sparks into projects,
- make good self-directed decisions,
- use embeddings intelligently,
- and stay safe without constantly asking permission.

That is the product.
