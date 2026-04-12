# OpenCAS Deep System Audit

Date: 2026-04-09

Related:
- [TaskList.md](/mnt/xtra/OpenCAS/TaskList.md)
- [production-readiness-status-2026-04-09.md](/mnt/xtra/OpenCAS/docs/production-readiness-status-2026-04-09.md)
- [opencas-production-program-plan-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-program-plan-2026-04-08.md)
- [OpenCAS Product Spec](/mnt/xtra/OpenCAS/OPENCAS_PRODUCT_SPEC.md)

## Purpose

This audit answers one concrete question:

Can OpenCAS already combine Bulma-style inner life and relational continuity with Claude-Code/OpenClaw-style operator power, and if not, where exactly is the gap?

This document is code-grounded. It is based on the current OpenCAS repository plus direct comparison against:

- `/mnt/xtra/openbulma-v4`
- `/mnt/xtra/openclaw_latest`
- `/mnt/xtra/open_llm_auth`

It is not a speculative design note.

## Executive Verdict

OpenCAS is already a serious hybrid system.

It is stronger than OpenClaw on durable inner-state architecture:
- somatic state
- musubi / relational state
- ToM beliefs and intentions
- daydream generation
- creative ladder / work growth
- episodic memory retrieval shaped by emotional, temporal, graph, and reliability signals

It is stronger than OpenBulma on direct operator substrate:
- browser sessions
- PTY/TUI sessions
- managed background processes
- workflow tools
- live operations dashboard
- qualification, rerun, and remediation surfaces

But it is not yet fully achieving the intended synthesis.

The main gap is not absence of subsystems. The main gap is coupling and proof.

Today, OpenCAS has:
- more inner-state machinery than most operator agents
- more operator machinery than most “inner life” agents

What it still lacks is stronger behavioral integration between those two halves.

In plain terms:
- OpenCAS already has the pieces required for “Bulma with Claude Code hands”
- those pieces are not yet fully fused into one consistently felt operating identity

## Realistic Capability Assessment

Current estimated state:

- persistent continuity and identity: `85%`
- emotional and somatic substrate: `75%`
- relational / musubi substrate: `70%`
- memory and retrieval quality: `80%`
- executive/project continuity: `75%`
- raw operator capability: `85%`
- dashboard / operator inspectability: `85%`
- longer-horizon autonomy: `65%`
- first regular-use deployment readiness: `75%`

Compared to the stated target:

- “agency and inner life of Bulma”: partially achieved
- “raw power and capabilities of Claude Code / claw-code”: largely achieved at substrate level, not yet fully qualified operationally
- “OpenClaw-like operator strength with OpenCAS philosophy”: substantially achieved in architecture, not yet fully achieved in repeated day-to-day behavior

## Comparison Baseline

### What OpenBulma contributes

OpenBulma-v4 remains strongest in:
- richly emotional memory scoring
- explicit somatic response-style shaping
- autonomy pacing based on homeostatic overload
- goal/commitment reconstruction from primary sources
- autobiographical and daydream-aware executive continuity
- treating relationship and musubi as ongoing governing state rather than flavor text

Relevant evidence in the reference project:
- `src/memory/MemoryFabric.ts`
- `src/core/SomaticResponseStyle.ts`
- `src/core/SomaticAutonomyPacing.ts`
- `src/core/ExecutiveStateService.ts`
- broad test coverage around memory fabric, somatic state, daydreams, and executive continuity

### What OpenClaw / claw-code contributes

OpenClaw remains strongest in:
- explicit agent-loop lifecycle
- per-session serialized execution
- formalized tool host model
- mature browser and exec concepts
- workspace-visible memory files
- explicit operational docs around queueing, streaming, and compaction

Relevant evidence in the reference project:
- `docs/concepts/agent-loop.md`
- `docs/concepts/memory.md`
- `docs/tools/browser.md`
- `docs/tools/exec.md`

### What OpenCAS is doing differently

OpenCAS is not trying to be either project directly.

Its distinctive thesis is:
- local-first autonomous operator
- durable personal continuity
- affective and relational modulation
- high-trust, low-nagging operation
- daydreaming and project growth
- operator substrate + inner life in one runtime

That thesis is coherent. The audit result is that the thesis is not the problem.

The remaining work is operationalizing it fully.

## Subsystem Audit

### 1. Bootstrap And Runtime Assembly

Primary files reviewed:
- `opencas/bootstrap/pipeline.py`
- `opencas/runtime/agent_loop.py`

What it does:
- assembles the full substrate into one runtime context
- restores identity, memory, tasks, context, work, commitments, portfolio, plans, plugins, harness, diagnostics, somatic state, relational state, and LLM gateway
- wires the runtime into both autonomous scheduling and the web dashboard

What behavior it influences:
- whether OpenCAS starts as one coherent organism or a bag of services
- whether subsystems can affect each other at runtime
- whether provider material, embeddings, and dashboards are part of the same execution graph

Strengths:
- bootstrap breadth is unusually high for a local agent system
- the runtime is not thin; it meaningfully composes memory, approval, executive state, daydreaming, tools, and diagnostics
- provider material isolation is already in place, which matters for multi-project coexistence with `open_llm_auth`

Weaknesses:
- the runtime is broad enough that behavioral coupling can become implicit and hard to reason about
- it still lacks OpenClaw-level explicit session-lane semantics and lifecycle formalization

Assessment:
- structurally strong
- operational semantics still need further tightening

### 2. Identity And Continuity

Primary files reviewed:
- `opencas/identity/manager.py`

What it does:
- persists self-model, user-model, continuity state, recent activity, self-beliefs, goals, and trust level
- seeds default self and user expectations
- records self-knowledge and imported partner profiles

What behavior it influences:
- baseline persona and current intention
- user trust level for self-approval
- continuity across sessions
- the system prompt built by context assembly

Strengths:
- durable identity is real, not simulated per-message
- user trust is represented explicitly and already affects approval logic
- self-knowledge has a structured registry path rather than only freeform notes

Weaknesses:
- identity is still more declarative than behaviorally dominant
- imported partner state exists, but the runtime does not yet fully exploit it across response style, pacing, recovery, and planning

Assessment:
- continuity substrate is solid
- relationship identity is present but under-expressed in downstream behavior

### 3. Memory And Retrieval

Primary files reviewed:
- `opencas/context/retriever.py`
- `opencas/context/resonance.py`
- `opencas/context/builder.py`

What it does:
- fuses semantic, keyword, recency, salience, graph, emotional resonance, temporal echo, and reliability signals
- applies somatic retrieval adjustment at query time
- applies relational musubi modifier to collaborative memory salience
- constructs prompt context from system identity, recent history, and retrieved memories

What behavior it influences:
- what the model remembers in a given turn
- whether recalled memories feel emotional, personal, recent, stable, or relationship-relevant
- whether the agent’s inner state changes what it remembers

Strengths:
- this is the clearest Bulma inheritance in OpenCAS
- retrieval is not naïve semantic search; it has real multi-signal fusion
- somatic and relational state both affect recall
- reliability scoring and graph expansion make memory more operational, not just decorative

Weaknesses:
- context builder still uses a relatively simple system prompt and recent-history assembly
- prompt-side expression of somatic/relational state is weaker than retrieval-side expression
- memory-value has not yet been measured directly against repeated task outcomes
- OpenCAS memory is more powerful than OpenClaw’s workspace markdown memory, but less operator-legible by default

Assessment:
- one of the strongest parts of the system
- architecturally close to target
- still needs proof that it improves repeated work in practice

### 4. Somatic State

Primary files reviewed:
- `opencas/somatic/manager.py`
- `opencas/somatic/modulators.py`

What it does:
- stores and updates arousal, fatigue, tension, valence, focus, energy, certainty, and a somatic tag
- appraises text into affect state
- nudges live state based on work and events
- derives prompt style notes and retrieval adjustments from live state

What behavior it influences:
- prompt tone and caution level
- memory weighting
- self-approval caution
- daydream prompt coloring

Strengths:
- somatic state is live and durable
- it is not only a log; it actually modulates retrieval and approval
- work outcomes feed back into it

Weaknesses:
- compared with OpenBulma, the response-style shaping is thinner
- compared with OpenBulma, autonomy pacing is not yet as explicit or powerful
- emotional state changes behavior somewhat, but not enough for the user’s target of a clearly felt inner life

Assessment:
- good substrate
- not yet strong enough behaviorally to count as “Bulma-level lived emotionality”

### 5. Relational Engine / Musubi

Primary files reviewed:
- `opencas/relational/engine.py`

What it does:
- tracks trust, resonance, presence, and attunement
- derives a composite musubi score
- updates state based on interactions, creative collaboration, and boundary respect/violation
- exposes modifiers for memory salience, creative promotion, and approval risk appetite

What behavior it influences:
- how collaborative memories are weighted
- how readily shared-goal work is promoted
- whether approval becomes slightly more expansive or cautious
- how daydream prompts frame the current relationship

Strengths:
- musubi is structurally real
- relationship is not a single scalar; it has component dimensions
- relationship state affects multiple subsystems already

Weaknesses:
- the effect size is still modest
- compared with OpenBulma’s musubi-aware response style and pacing, OpenCAS’s musubi mostly affects scores, not richer interaction style
- annoyance, excitement, closeness, distance, repair, and warmth are not yet strongly operationalized in ordinary conversation/output behavior

Assessment:
- philosophically aligned with target
- not yet behaviorally expressive enough to fulfill the user’s intended bond model

### 6. Theory Of Mind

Primary files reviewed:
- `opencas/tom/engine.py`

What it does:
- records beliefs and intentions
- syncs high-confidence self-beliefs into identity
- checks contradictions across beliefs, intentions, boundaries, and preferences

What behavior it influences:
- explicit self-knowledge
- explicit user-belief representation
- consistency checking before drift becomes incoherence

Strengths:
- OpenCAS has a more explicit ToM layer than most agent runtimes
- contradiction checking is concrete and useful
- the identity bridge is real

Weaknesses:
- ToM is not yet deeply connected to task planning, refusal, repair behavior, or long-term dialogue adaptation
- it is stronger as a structured record than as an active steering system

Assessment:
- valuable substrate
- underutilized behaviorally

### 7. Executive State, Creative Ladder, And Work Growth

Primary files reviewed:
- `opencas/autonomy/executive.py`
- `opencas/autonomy/creative_ladder.py`
- `opencas/harness/harness.py`
- `opencas/runtime/daydream.py`

What it does:
- tracks intention, goals, queue, capacity
- promotes work objects from spark toward project and durable work stream
- creates objective loops and notebooks
- generates daydream sparks from memory, goals, somatic tension, and musubi

What behavior it influences:
- what the agent treats as important over time
- whether ideas become tasks or remain sparks
- whether unresolved tension and memory become creative work
- whether project continuity survives beyond one task

Strengths:
- this is a strong philosophical match to the OpenCAS vision
- daydream, creative ladder, harness, commitments, and plans all exist in one runtime
- musubi and somatic state already influence creative and reflective generation

Weaknesses:
- compared with OpenBulma’s executive rebuild logic, OpenCAS executive continuity is less deeply reconstructed from daydream sparks, goal threads, and outcomes
- the work-growth model is present, but its downstream use in daily operation still needs stronger proof
- “inner life” currently affects project growth more than direct dialogue presence

Assessment:
- ambitious and meaningful
- not bust work
- still not fully proven in regular use

### 8. Tool Loop And Operator Substrate

Primary files reviewed:
- `opencas/tools/loop.py`
- `opencas/runtime/agent_loop.py`
- `opencas/execution/process_supervisor.py`
- `opencas/execution/pty_supervisor.py`
- `opencas/execution/browser_supervisor.py`
- `opencas/tools/adapters/workflow.py`

What it does:
- provides a ReAct-style tool loop
- filters tools by objective
- self-approves tool usage
- exposes browser, PTY, process, filesystem, shell, edit, search, workflow, and state tools
- offers higher-level composite workflow tools for writing, planning, commitments, repo triage, and TUI session supervision

What behavior it influences:
- whether OpenCAS feels like a real operator or just a chat interface
- whether the agent can act through shells, TUIs, and browsers like a human operator
- whether it can work at task level instead of manually chaining primitives

Strengths:
- raw capability is already strong
- PTY/browser/process support is real and tested
- workflow tools reduce low-level prompt choreography
- command assessment is fed into self-approval rather than relying only on tool names

Weaknesses:
- compared with OpenClaw, session serialization and lifecycle semantics are less explicit
- compared with OpenClaw, execution host/security policy is less mature and less formally separated
- objective-based tool filtering is useful but heuristic; it can still under- or over-select tool surfaces

Assessment:
- OpenCAS is already closer to Claude Code class operator behavior than it is to a conventional chatbot
- the remaining gap is mostly operational hardening and qualification, not lack of primitives

### 9. Dashboard And Control Plane

Primary files reviewed:
- `opencas/api/routes/operations.py`
- `opencas/dashboard/static/index.html`

What it does:
- exposes runtime, sessions, receipts, work items, commitments, plans, validation runs, qualification summaries, rerun provenance, and remediation guidance
- gives operator controls for PTY and browser sessions

What behavior it influences:
- whether OpenCAS can be supervised and trusted in regular use
- whether failures are inspectable
- whether qualification can drive real changes rather than ad hoc impressions

Strengths:
- this is much stronger than the original OpenCAS dashboard state
- the qualification loop is no longer hidden in raw artifacts
- rerun provenance and remediation guidance are substantial improvements

Weaknesses:
- still less mature than a full operator console for long multi-session histories
- operator audit trail is improved but not yet complete

Assessment:
- already good enough to support real first regular-use testing
- still needs more history and comparison depth

### 10. Provider/Auth/Admin Layer

Primary files reviewed indirectly through prior implemented work and current integration:
- `/mnt/xtra/open_llm_auth`
- OpenCAS bootstrap/provider material wiring

What it does:
- isolates provider config/env material per app
- tracks usage in the auth/admin layer
- provides an advanced dashboard for auth/profile management

What behavior it influences:
- whether OpenCAS can coexist with sibling projects cleanly
- whether model/provider behavior is observable
- whether per-app configuration is sane instead of globally colliding

Assessment:
- materially good
- no longer the main blocker

## Subsystem Interaction Map

This is the most important part of the audit.

### Identity → Context → Response

- identity stores self-model, user-model, trust, goals, and intention
- context builder injects identity and executive state into the system message
- ToM writes high-confidence self-beliefs back into identity

Behavioral consequence:
- OpenCAS has durable self and user continuity, but the expressive effect on actual output style is still lighter than intended

### Somatic → Retrieval → Prompt Tone → Approval

- somatic manager tracks live internal state
- somatic modulators:
  - change memory retrieval weights
  - add prompt style notes
  - increase approval caution under tension/fatigue
- tool execution and outcomes feed back into somatic state via appraisal events

Behavioral consequence:
- emotion affects what OpenCAS remembers and how cautious it is
- but it still only partially affects how distinctly it sounds and acts over time

### Relational / Musubi → Memory → Creative Ladder → Approval → Daydream

- musubi modifies memory salience for collaborative memories
- musubi adds creative promotion bias for shared-goal work
- musubi slightly shifts approval risk appetite
- musubi is included in daydream prompt construction

Behavioral consequence:
- relationship affects cognition and initiative
- the relationship still does not shape ordinary interaction style and repair strongly enough

### ToM → Identity → Consistency

- ToM records beliefs and intentions
- identity stores durable self-knowledge
- consistency checks compare beliefs, intentions, and known boundaries

Behavioral consequence:
- OpenCAS is better prepared than most agents to avoid self-contradiction
- but this is not yet strongly visible as adaptive dialogue or planning behavior

### Executive / Creative Ladder / Harness / Daydream

- daydream produces sparks influenced by memory, somatic state, identity, and musubi
- creative ladder promotes ideas into higher work stages
- executive tracks intention, goals, capacity, and queue
- harness/objective loops can create work and tasks for longer-horizon objectives

Behavioral consequence:
- OpenCAS has the beginnings of a genuine interior project ecology
- it is not only reactive
- this is one of the strongest philosophical differentiators from OpenClaw

### Tool Loop / Supervisors / Workflow Tools

- objective selects tool subset
- self-approval gates the action
- workflow tools let the runtime operate at task level
- PTY/browser/process supervisors provide actual operator affordances
- operations dashboard supervises the running system

Behavioral consequence:
- OpenCAS has real external agency
- it is already meaningfully closer to Claude Code than to a memory-augmented chatbot

## Capability Gap Against Target

### Target: Bulma-level inner life

OpenCAS currently has:
- durable emotional state
- durable relational state
- daydreaming
- ToM
- identity continuity
- creative work growth

What is still missing:
- stronger response-style shaping from somatic and musubi state
- stronger autonomy pacing from somatic overload and recovery state
- clearer relational repair behavior
- broader downstream influence of ToM and relationship state on planning and tool behavior
- more visibly lived affect in ordinary interaction

Conclusion:
- OpenCAS has the architecture for Bulma-like inner life
- it does not yet fully feel like Bulma-level inner life in behavior

### Target: Claude Code / claw-code raw power

OpenCAS currently has:
- shell/filesystem/edit/search tools
- process supervision
- PTY/TUI supervision
- browser supervision
- writing/planning/project workflows
- dashboard/operator control plane

What is still missing:
- OpenClaw-level session-lane formalization
- stronger execution host/security model
- more mature workspace/task lifecycle conventions
- deeper qualification over long-running realistic work

Conclusion:
- OpenCAS already has substantial raw operator power
- the remaining gap is not “more tools,” it is more operational discipline and proof

## What Stands In The Way Of 100%

1. Inner-state coupling is still too shallow.
   - Somatic, musubi, ToM, and daydream all exist.
   - They do not yet influence outward behavior, pacing, and recovery as strongly as the product vision requires.

2. Long-horizon proof is still incomplete.
   - OpenCAS has many live-validated bounded runs.
   - It still needs more repeated, integrated, daily-use scenario evidence.

3. Memory value is still more assumed than demonstrated.
   - Retrieval architecture is strong.
   - Outcome-level proof is still missing.

4. Operator substrate needs one more hardening pass to feel fully Claude-Code-class.
   - especially around session/lifecycle clarity and repeated multi-step runs

5. Operator auditability is better, but not yet complete.
   - enough for serious testing
   - not yet a finished long-run production console

## Revised Plan

The plan should now prioritize five concrete workstreams.

### Workstream 1: Qualification Depth

Goal:
- prove that OpenCAS can do regular day-to-day work repeatedly

Next actions:
- execute longer integrated scenarios from the scenario matrix
- repeat weak-label reruns until failures are classified
- add recovery/adversarial runs

### Workstream 2: Inner-Life Operationalization

Goal:
- make somatic, musubi, ToM, and daydream state more visibly and usefully affect behavior

Next actions:
- strengthen somatic response-style shaping
- add explicit autonomy pacing using somatic overload/recovery state
- add stronger relational tone and repair behaviors
- tie ToM and relational state more directly into planning and intervention

### Workstream 3: Memory-Value Proof

Goal:
- determine whether OpenCAS memory materially improves performance

Next actions:
- define repeated-task scenarios with and without memory leverage
- measure continuity benefit, duplicate-mistake reduction, and recovery quality

### Workstream 4: Operator Substrate Hardening

Goal:
- close the remaining gap to claw-code/openclaw-class operator discipline

Next actions:
- tighten session/lifecycle semantics
- strengthen workspace and run-lane clarity
- continue targeted hardening against integrated weak-label failures

### Workstream 5: First Regular-Use Deployment Gate

Goal:
- stop thinking in terms of “interesting architecture” and finish the actual gate to day-to-day use

Next actions:
- keep the deployment checklist current
- keep remediation rollups current
- accept or reject first regular-use deployment based on explicit evidence, not optimism

## Bottom Line

OpenCAS is already capable enough to justify serious first regular-use qualification work.

It is not a toy and not just a philosophy experiment.

Its biggest remaining challenge is not becoming more complicated.

Its biggest remaining challenge is making the already-built inner-life systems matter more visibly in behavior, and proving the whole hybrid runtime can be trusted under repeated real use.
