# Enterprise Capabilities — duxx-ai + DuxxDB

This document maps every enterprise concern an agentic / multi-agent platform
has to face (build, govern, observe, evaluate, improve, secure, cost-control,
fine-tune, replay) to a concrete primitive in either `duxx-ai` (the Python
agent framework) or `DuxxDB` (the Rust storage + retrieval engine that
underpins it).

It is meant for two audiences:

1. Architects deciding whether the Duxx stack can replace a glue of
   LangChain + LangSmith + Portkey + Pinecone + a separate prompt-management
   SaaS.
2. Engineers about to run the two enterprise reference projects in
   `examples/enterprise/` and wanting to know what each moving part is for.

We use the term **"Duxx Stack"** for the combined product:
`duxx-ai` (orchestrates) + `DuxxDB` (persists, retrieves, scores, replays).

---

## 1. Why the Duxx Stack is unusual

Most agent stacks are a runtime (LangChain, CrewAI, AutoGen) plus an external
observability/eval SaaS (LangSmith, Braintrust) plus an external vector DB
(Pinecone, Qdrant) plus a separate prompt manager (PromptLayer) plus a
separate gateway (Portkey). The pieces share nothing — there is no atomic
view across them.

The Duxx Stack collapses these into **one storage engine with one wire
protocol** (RESP2/3, gRPC, MCP) that the agent runtime treats as its single
source of truth. Concretely, that means:

| What every enterprise wants                              | Who supplies it          | Backed by which primitive                              |
|----------------------------------------------------------|--------------------------|--------------------------------------------------------|
| Multi-step agent / tool loop                             | `duxx-ai`                | `core.Agent`, `core.AutonomousAgent`                   |
| DAG workflows + human-in-the-loop                        | `duxx-ai`                | `orchestration.Graph`, `GraphInterrupt`                |
| Role-based crews (sequential / parallel / hierarchical)  | `duxx-ai`                | `orchestration.Crew`                                   |
| Hierarchical memory (working / episodic / semantic)      | `duxx-ai` + `DuxxDB`     | `memory.MemoryManager` → `MEM.*` on DuxxDB             |
| Hybrid retrieval (BM25 + HNSW + RRF)                     | `DuxxDB`                 | All `*.SEARCH` commands                                |
| Versioned prompts + canary routing                       | `DuxxDB` + `duxx-ai`     | `PROMPT.*` + `self_improving.PromptRouter`             |
| Continuous online eval + failure clustering              | `DuxxDB` + `duxx-ai`     | `EVAL.*` + `self_improving.ImprovementLoop`            |
| Deterministic causal trace replay (with overrides)       | `DuxxDB` + `duxx-ai`     | `REPLAY.*` + `debug.TraceReplayer`                     |
| Cost ledger + per-tenant budgets                         | `DuxxDB`                 | `COST.*`                                               |
| Dataset registry (golden + mined)                        | `DuxxDB`                 | `DATASET.*` + `DATASET.FROM_RECALL`                    |
| OpenTelemetry-compatible span export                     | `duxx-ai`                | `observability.DuxxExporter`                           |
| Guardrails (PII / prompt-injection / toxicity)           | `duxx-ai`                | `governance.GuardrailChain`                            |
| RBAC + audit log                                         | `duxx-ai`                | `governance.rbac`, `governance.audit`                  |
| Cost+quality adaptive routing                            | `duxx-ai`                | `router.AdaptiveRouter`                                |
| RAG (loaders, splitters, embeddings, contextual rerank)  | `duxx-ai`                | `rag.*`                                                |
| Fine-tune dataset prep + connector to trainers           | `duxx-ai`                | `finetune.pipeline`                                    |
| Reinforcement-learning from agent traces                 | `duxx-ai`                | `rl.training`                                          |

A single transaction can `PROMPT.GET` the active version, `MEM.SEARCH` a
tenant's episodic memory, score the result with `EVAL.SCORE`, debit the
tenant via `COST.RECORD`, capture the span via `TRACE.SPAN`, and capture
the model call via `REPLAY.CAPTURE`. None of that requires cross-system
2PC or eventual-consistency reasoning — it is one server.

---

## 2. The eight capability pillars

### 2.1 Agent runtime

* `duxx_ai.core.Agent` — single-agent tool-use loop with retry, fallback
  LLM, max-iteration guard, structured output parsing, approval callbacks
  for high-risk tools.
* `duxx_ai.core.AutonomousAgent` — ReAct / plan-execute / self-critique
  reasoning strategies; emits an `ExecutionTrace` so every step is
  auditable.
* `duxx_ai.core.DeepAgent` — long-horizon planning with sub-goal stack.
* `duxx_ai.core.middleware` — pre/post hooks (rate limit, redaction,
  caching) registered globally.

**Why it matters for enterprise:** every agent is the same `Agent` shape,
so middleware (audit, redaction, cost meter) attaches uniformly. There is
no "framework A vs framework B" tax inside the same fleet.

### 2.2 Orchestration

* `orchestration.Graph` — typed DAG with conditional edges, state
  reducers (`append`, `sum`, `merge_dict`), checkpointing, cycle
  control. `GraphInterrupt` pauses a node so a human approver supplies
  input out-of-band, then resume.
* `orchestration.Crew` — role-based, three execution strategies:
  - **sequential** (analyst → writer → reviewer),
  - **parallel** (N specialists same input, aggregate),
  - **hierarchical** (supervisor delegates).
* `orchestration.channels` — typed pub/sub between nodes for streaming
  partial results.
* `orchestration.analytics` — built-in latency / token / step counters
  for every run.

### 2.3 Memory & retrieval

Memory is a thin Python facade over DuxxDB's `MEM.*`, `MEM.SEARCH_HYBRID`.

* `memory.MemoryManager` with pluggable backends:
  - `InProcessBackend` — embedded, no server. Good for dev.
  - `DuxxBackend` — embedded DuxxDB via Python bindings (PyO3).
  - `DuxxServerBackend` — talks to a DuxxDB server over RESP/gRPC.
* Tier model: working (token-window), episodic (per-session, per-tenant),
  semantic (long-term, embeddings).
* Hybrid recall = BM25 + HNSW fused via RRF, served by DuxxDB in a
  single command — no manual fusion needed.

**Enterprise hook:** every recall request carries `tenant=` so the
retrieval index is automatically partitioned. The same backend hosts a
million tenants without cross-leakage.

### 2.4 Observability

* `observability.Tracer` — wraps agent steps, tool calls, LLM calls in
  OpenTelemetry spans.
* `observability.DuxxExporter` — exports spans straight into
  `TRACE.SPAN` on DuxxDB. Same shape as OTLP, but durable in your
  storage tier so you can query traces alongside the prompts that
  produced them and the evals that scored them.
* `observability.AgentEvaluator` — pluggable scorers (exact, contains,
  regex, LLM-judge) for offline eval.

### 2.5 Prompt management + canary routing

* `DuxxDB PROMPT.*` — versioned prompts with `tag` ("prod", "canary",
  "preview"), atomic version assignment, history search.
* `duxx_ai.self_improving.PromptRouter` — deterministic SHA1-based split:
  the same `(tenant, message_id)` always hits the same variant. Stable
  canary measurement without flapping.
* `duxx_ai.self_improving.ImprovementLoop` — background daemon that:
  1. Reads `EVAL.SCORES` on the canary run,
  2. Calls `EVAL.CLUSTER_FAILURES` for the patterns,
  3. Asks an LLM rewriter for a candidate,
  4. Writes the candidate as a new prompt version,
  5. Promotes canary → prod via `EVAL.COMPARE` once thresholds clear.

### 2.6 Cost ledger

* `DuxxDB COST.*` — per-call records with `(tenant, model, prompt_tokens,
  completion_tokens, cents)`, plus `COST.BUDGET_SET` and `COST.BUDGET_CHECK`.
* `COST.CLUSTER_EXPENSIVE` — finds clusters of similar expensive calls
  so you can replace the prompt or downgrade the model precisely there.

### 2.7 Governance

* `governance.GuardrailChain` — composable input/output filters: PII
  detector, prompt-injection sniffer, toxicity, regex deny-lists,
  schema enforcement.
* `governance.RBAC` — role + tool ACL; "support agents can call refunds
  ≤ $50, supervisors ≤ $5000".
* `governance.AuditLog` — append-only; every refused action, every
  approved tool call, every guardrail trip.

### 2.8 Replay & debugging

* `DuxxDB REPLAY.*` — captures every model call + tool call as
  `(trace_id, step_id, input, output)` blobs.
* `duxx_ai.debug.TraceReplayer` — deterministic re-execution with
  per-step overrides:
  - `swap_model("gpt-4")` — counterfactual model audit,
  - `swap_prompt(name="...", version=N)` — try a different version,
  - `set_temperature(0.0)` — collapse randomness,
  - `inject_output("...")` — pin a tool reply (great for testing
    "what if this API had returned X?"),
  - `skip()` — drop a step entirely.
* `duxx_ai.debug.CapturingChat` — drop-in wrapper around any chat
  callable that emits `REPLAY.CAPTURE` on every invocation.

This is the differentiator that lets you do compliance-grade
post-incident review: pick a `trace_id`, replay, see exactly which step
went wrong, then run a `swap_prompt` counterfactual to prove a
candidate prompt would have handled it.

---

## 3. Unique capabilities (what nobody else has put under one roof)

1. **Self-improving prompts in one server-side feedback loop.** The
   loop reads scores, clusters failures, mutates prompts, ramps canaries
   — without leaving DuxxDB. There is no other open-source system that
   does the full circle with atomicity.

2. **Causal replay with override-injection.** Tracing frameworks
   (LangSmith, Arize) show you what happened. `TraceReplayer` re-runs
   it with new prompts / models / pinned outputs so you can prove a
   fix works on the exact failed turn before shipping it.

3. **Cross-primitive joins in one store.** `EVAL.SCORES` rows carry a
   `notes` blob that includes the `input_text` and `idempotency_key`,
   joinable to `PROMPT.GET_BY_VERSION` to `COST.RECORDS_FOR_TENANT` to
   `TRACE.GET`. You write SQL-shaped questions ("which prompt version
   produced the most expensive low-score turns last week, grouped by
   tenant?") as RESP commands, not a 5-system ETL.

4. **One wire protocol.** RESP2/3 means every Valkey/Redis client in
   every language already talks to DuxxDB. Combined with gRPC and MCP
   stdio, you do not need an SDK to integrate from Go, Rust, .NET, or
   Bun.

5. **Single-binary deploy.** DuxxDB ships as a Rust binary; duxx-ai
   ships as a Python package. The reference enterprise projects in
   `examples/enterprise/` run on a developer laptop with `pip install
   duxx-ai duxxdb` and one `duxxdb-server` process.

6. **Per-tenant everything.** Memory, prompts, evals, costs, replay
   are all partitioned by tenant tag. Multi-tenant SaaS is a default,
   not a project.

7. **Atomic GDPR delete.** Because every artifact is in DuxxDB and
   tagged with `tenant=...`, a `TENANT.PURGE <id>` walks all primitives
   under one write lock. (Server-side primitive — see the v0.3 roadmap.)

8. **Fine-tune feedback loop.** `DATASET.FROM_RECALL` mines high-score
   memories of a tenant into a SFT-ready dataset, which
   `duxx_ai.finetune.pipeline` then ships to a trainer (QLoRA,
   transformers SFT). The loop closes without leaving the stack.

---

## 4. Reference deployment

```
┌──────────────────────────────────────────────────────────────────────┐
│  Edge / API                                                          │
│    FastAPI service hosting duxx_ai.Crew / Graph / Agent              │
│    Guardrails, RBAC, audit log applied per request                   │
└──────────┬────────────────────────────────────────────┬──────────────┘
           │ RESP / gRPC / MCP                          │ OTel
           ▼                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DuxxDB cluster                                                      │
│    MEM.*    PROMPT.*    EVAL.*    DATASET.*    COST.*                │
│    TRACE.*  REPLAY.*    SESSION.* TOOLCACHE.*                        │
│    Persistent HNSW + redb tables; cold-start ≪ 1s at 1M rows         │
└──────────┬───────────────────────────────────────────────────────────┘
           │ Background loops
           ▼
   ImprovementLoop (continuous prompt self-improvement)
   TraceReplayer (on-demand audits + counterfactual)
   Finetune Pipeline (DATASET.FROM_RECALL → trainer)
```

The two enterprise reference projects in `examples/enterprise/` exercise
every box on this diagram in under 30 seconds of demo.

---

## 5. Capability checklist (one-line per item)

- [x] Single-agent loop with retry + fallback LLM
- [x] DAG orchestration with conditional edges + checkpoints
- [x] Multi-agent crews (sequential / parallel / hierarchical)
- [x] Working / episodic / semantic memory
- [x] Hybrid retrieval (BM25 + HNSW + RRF)
- [x] Versioned prompts with canary tag + atomic promotion
- [x] Continuous online eval + failure clustering
- [x] LLM-as-judge default scorer + pluggable Scorer Protocol
- [x] Deterministic causal trace replay w/ overrides
- [x] OpenTelemetry-compatible span export
- [x] Cost ledger + per-tenant budgets + expensive-call clustering
- [x] Dataset registry + DATASET.FROM_RECALL mining
- [x] PII / prompt-injection / toxicity guardrails
- [x] RBAC + append-only audit log
- [x] Cost+quality adaptive router
- [x] RAG retrievers (loaders, splitters, embeddings, contextual rerank)
- [x] Fine-tune dataset prep + trainer connector
- [x] RL training from agent traces
- [x] Multi-tenant partitioning everywhere
- [x] One binary deploy (DuxxDB) + one Python package (duxx-ai)

The two reference projects below light up >90% of this list.
