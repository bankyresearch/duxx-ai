<p align="center">
  <strong>Duxx AI</strong><br>
  <em>The Python framework for building agents on the Duxx Stack</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/duxx-ai/"><img src="https://img.shields.io/pypi/v/duxx-ai?color=d6336c&label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/bankyresearch/duxx-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python"></a>
</p>

---

Duxx AI is a Python framework for building, orchestrating, and operating
AI agents. It's the upper half of the **Duxx Stack** — the Python code
your agent runs in. The lower half is [**DuxxDB**](https://github.com/bankyresearch/duxxdb),
a Rust engine that gives every agent persistent hybrid memory, agent
observability, prompt versioning, eval runs, deterministic replay,
and a cost ledger — all through one RESP-compatible daemon.

```
   ┌─ duxx-ai (Python) ────────────────────────────────────────┐
   │  agents · graphs · tools · governance · routing · RAG     │
   └──────────────────────┬────────────────────────────────────┘
                          │  pluggable backends
   ┌──────────────────────▼────────────────────────────────────┐
   │  DuxxDB (Rust)                                            │
   │  hybrid recall (HNSW + BM25) · trace store · prompts      │
   │  · datasets · eval · replay · cost · RESP / gRPC / MCP    │
   └───────────────────────────────────────────────────────────┘
```

Pair them or use just the framework half. They ship and version
independently.

---

## Quick start

```bash
pip install duxx-ai
```

```python
from duxx_ai import Agent, AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import tool

@tool(name="search", description="Search the web")
def search(query: str) -> str:
    return f"results for: {query}"

agent = Agent(
    config=AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
    ),
    llm_config=LLMConfig(provider="openai", model="gpt-4o-mini"),
    tools=[search],
)

result = await agent.run("Find recent papers on RLHF.")
```

For a single-process script you're done. For multi-agent fleets or
production deployment, plug the DuxxDB backends in — see
[**Wiring DuxxDB in**](#wiring-duxxdb-in) below.

---

## Module status

Duxx AI is alpha software. Some modules are wired end-to-end, some are
scaffolds, some are experimental. Honest status per module:

| Module | Status | What it does |
|---|---|---|
| `core` | **Stable** | Agent, AgentConfig, Tool decorator, Message, Role |
| `memory` | **Stable** with `DuxxBackend` / experimental in-process | Five-tier memory (working, episodic, semantic, procedural, shared) over a pluggable backend |
| `observability.tracer` | **Stable** with `DuxxExporter` | OTel-shape spans + a DuxxDB exporter for persistence and tree-aware queries |
| `orchestration.graph` | **Beta** | DAG graphs with conditional edges and reducers |
| `orchestration.crew` | **Beta** | Multi-agent runners (sequential, parallel, hierarchical) |
| `tools` | **Beta** | 40+ built-in tools across email / calendar / DB / API / docs / finance / security / devops / analytics |
| `router` | **Beta** | Adaptive model routing with budget caps |
| `governance` | **Experimental** | Guardrails + RBAC + audit log |
| `rag` | **Experimental** | Loaders, splitters, embedders, retrievers (BM25 / vector / hybrid / SVM / reranker) |
| `observability.evaluator` | **Experimental** | In-process eval runner; for persistent runs use DuxxDB |
| `agents.qa_agent` | **Experimental** | QA-test-generation agent |
| `templates` | **Experimental** | Pre-configured agents (researcher, banker, CFO, etc.) |
| `importers.n8n` | **Experimental** | Convert n8n workflow JSON into Duxx AI graphs |
| `finetune` | **Experimental** | QLoRA / LoRA fine-tuning pipelines (heavy optional deps) |
| `rl` | **Experimental** | Reinforcement-learning training loop |

**Stable** = wired end-to-end, covered by tests, recommended for use today.
**Beta** = works, lightly tested, API may shift before 1.0.
**Experimental** = scaffolded; expect rough edges and surface-area changes.

Optional dependency groups follow the same shape:

```bash
pip install duxx-ai                          # core framework only
pip install duxx-ai[duxxdb]                  # + embedded DuxxDB backend
pip install duxx-ai[duxxdb-server]           # + remote DuxxDB daemon client
pip install duxx-ai[tools]                   # + finance/PDF/etc. tool deps
pip install duxx-ai[finetune]                # + torch/transformers/peft/trl
pip install duxx-ai[dev]                     # + pytest/ruff/mypy
```

---

## Wiring DuxxDB in

The two integration points where DuxxDB gives you depth the framework
can't deliver on its own:

### Pluggable memory backend

By default `MemoryManager` keeps everything in a Python dict. Swap in
the `DuxxBackend` and you get sub-millisecond hybrid recall (HNSW +
BM25 + Reciprocal Rank Fusion), persistence, importance-decay
eviction, and the option to share memory across worker processes.

```python
# pip install duxx-ai[duxxdb]
from duxx_ai.memory import MemoryManager
from duxx_ai.memory.storage import DuxxBackend

memory = MemoryManager(
    backend=DuxxBackend(
        dim=1536,
        storage="dir:./.duxxdb",   # persistent; "memory:_" for in-memory
    ),
    agent_id="alice",
)

memory.remember(
    "I lost my wallet at the cafe",
    memory_type="episodic",
    embedding=embed("I lost my wallet at the cafe"),
)

hits = memory.recall("wallet", k=5, query_embedding=embed("wallet"))
```

For multi-worker fleets that share state, swap `DuxxBackend` for
`DuxxServerBackend` and point it at a running `duxx-server` daemon.

### Trace exporter

`Tracer` is the local span collector. Adding `DuxxExporter` makes
every finished trace land in DuxxDB, where any RESP client can query
it (full trees, subtrees, multi-turn threads) or live-tail it via
`PSUBSCRIBE`.

```python
# pip install redis>=5  (or duxx-ai[duxxdb-server])
from duxx_ai.observability import Tracer
from duxx_ai.observability.duxx_exporter import DuxxExporter

exporter = DuxxExporter(
    url=f"redis://:{DUXX_TOKEN}@localhost:6379",
    thread_id="user-42-session",  # optional; enables TRACE.THREAD
)
tracer = Tracer(exporters=[exporter])
```

Then from any RESP client:

```bash
redis-cli TRACE.GET       <trace_id>           # full tree
redis-cli TRACE.SUBTREE   <span_id>            # descendants
redis-cli TRACE.THREAD    user-42-session      # multi-turn convo
redis-cli PSUBSCRIBE      "trace.*"            # live tail
```

### Other server-side primitives

The `duxx-server` daemon also exposes **prompts**, **datasets**, **eval
runs**, **deterministic replay**, and a **token + cost ledger** as
first-class RESP commands. Today you can reach these from Python via
`redis-py`; native Python facades land in duxx-ai 0.32.

```bash
redis-cli PROMPT.PUT          support-greeting "Hello, how can I help?"
redis-cli PROMPT.SEARCH       "warm greeting" 3
redis-cli DATASET.FROM_RECALL alice "refund" 5 regression-set eval
redis-cli EVAL.CLUSTER_FAILURES <run_id>
redis-cli COST.SET_BUDGET     acme daily 50.00 0.8
redis-cli COST.CLUSTER_EXPENSIVE - 0.7 5 50
redis-cli REPLAY.START        <trace_id> live
```

Every primitive shares one HNSW vector space, so semantic queries
cross primitive boundaries — find prompts that look like a failing
eval row, find expensive queries that match a memory cluster, replay
a failure with a different prompt version and compare scores. Full
details: [DuxxDB Phase 7](https://github.com/bankyresearch/duxxdb/blob/master/docs/ROADMAP.md).

---

## Core surface — by example

### Agent + tools

```python
from duxx_ai.core import Agent, AgentConfig, tool
from duxx_ai.core.llm import LLMConfig

@tool(name="db_query", description="Run a SQL query against the warehouse")
def db_query(sql: str) -> str:
    ...

agent = Agent(
    config=AgentConfig(name="analyst", system_prompt="..."),
    llm_config=LLMConfig(provider="openai", model="gpt-4o-mini"),
    tools=[db_query],
)
```

### Graph orchestration

```python
from duxx_ai.orchestration import Graph

graph = Graph("research-pipeline")
graph.add_node("gather",   gather_data)
graph.add_node("analyze",  analyze_data)
graph.add_node("review",   human_review, node_type="HUMAN")
graph.add_node("report",   generate_report)

graph.add_edge("gather", "analyze")
graph.add_conditional_edge("analyze", route_fn, {
    "needs_review": "review",
    "approved":     "report",
})

result = await graph.run()
```

### Multi-agent crew

```python
from duxx_ai.orchestration import Crew

crew = Crew(
    name="research-team",
    agents=[researcher, writer, reviewer],
    strategy="hierarchical",   # also: "sequential", "parallel"
)
result = await crew.run("Write a market analysis report")
```

### Memory (five tiers)

```python
from duxx_ai.memory import MemoryManager

memory = MemoryManager()
memory.store("working",    "current_task", "Analyzing Q4 data")
memory.store("episodic",   "last_meeting", {"topic": "Q4 review"})
memory.store("semantic",   "company_info", {"revenue": "10M"})
memory.store("procedural", "report_steps", ["gather", "analyze", "write"])
memory.store("shared",     "team_context", {"project": "Alpha"})
```

The five tiers map onto any backend — the in-process dict default, or
DuxxDB via `MemoryManager(backend=DuxxBackend(...))`.

### Observability

```python
from duxx_ai.observability import Tracer, AgentEvaluator

tracer = Tracer(exporters=["console", "json", "otlp"])
agent  = Agent(config=cfg, tracer=tracer)

evaluator = AgentEvaluator()
metrics   = evaluator.evaluate(agent, test_cases=[
    {"input": "What is 2+2?", "expected": "4"},
])
```

### Adaptive routing with budget

```python
from duxx_ai.router import AdaptiveRouter

router = AdaptiveRouter(
    providers=["openai", "anthropic", "local"],
    budget_limit=10.0,   # USD ceiling per session
)
response = await router.route("Summarize this paragraph.")
```

### Governance

```python
from duxx_ai.governance import GuardrailChain, RBACManager

guardrails = GuardrailChain([
    {"type": "pii_filter"},
    {"type": "prompt_injection"},
    {"type": "content_filter"},
    {"type": "hallucination_check"},
    {"type": "token_budget", "max_tokens": 10000},
])

rbac = RBACManager()
rbac.assign_role("user@company.com", "developer")
```

### Templates

```python
from duxx_ai.templates import DeepResearcherAgent

agent  = DeepResearcherAgent.create()
result = await agent.run("Analyze Q4 market trends")
```

Templates ship as starting points; expect to customize the system
prompt and tool list for any real use case.

---

## Layout

```
duxx_ai/
├── core/                 Agent, AgentConfig, Tool, Message, LLM config
├── memory/
│   ├── manager.py        MemoryManager + the five tier classes
│   └── storage/          MemoryBackend Protocol + in-process and
│                         DuxxBackend / DuxxServerBackend implementations
├── observability/
│   ├── tracer.py         Tracer + console/json/otlp exporters
│   ├── duxx_exporter.py  DuxxExporter for RESP-backed trace persistence
│   └── evaluator.py      Local AgentEvaluator (use DuxxDB for persistent runs)
├── orchestration/
│   ├── graph.py          Graph + Node + Edge + reducers
│   ├── crew.py           Crew + CrewAgent (sequential/parallel/hierarchical)
│   ├── channels.py       Channel + state reducers
│   ├── state_graph.py    Typed FlowGraph with streaming + checkpointing
│   └── analytics.py      Optional graph metrics (requires networkx)
├── governance/           Guardrails, RBACManager, AuditLog
├── router/               AdaptiveRouter with budget caps
├── rag/                  Loaders, splitters, embedders, vector store,
│                         retrievers (BM25, vector, hybrid, SVM, reranker)
├── tools/                40+ built-in tools across 9 domains
├── templates/            Pre-configured agents (Researcher, Banker,
│                         CFO, CMO, CHRO, Email, Finance, Compliance, …)
├── importers/n8n.py      n8n workflow JSON → Duxx AI Graph code
├── agents/qa_agent.py    QA test-case generator
├── finetune/             QLoRA / LoRA pipelines (optional)
├── rl/                   RL training loop (optional)
└── cli/                  duxx-ai CLI
```

---

## What duxx-ai is — and is not

**Is:** a Python framework that gives you agent runtime primitives
(Agent, Tool, Graph, Crew, Memory, Tracer) and an integration
surface for the DuxxDB engine. The interesting work lives at the
seams — pluggable memory backends, exporter-style observability,
deterministic replay over a captured trace.

**Is not:** a self-contained substitute for every piece of
infrastructure a production agent needs. The depth — sub-ms recall,
versioned prompts, eval clustering, cost-ledger semantics — lives in
DuxxDB. Use them together when you need that depth; use the
framework alone when you don't.

---

## Roadmap

The active work for the framework half of the stack:

- **0.32** — native Python facade for the Phase 7 RESP commands
  (prompts, datasets, eval, replay, cost) so calls look like
  `client.prompts.put(...)` instead of `redis_client.execute_command(...)`.
- **0.33** — async variants of `MemoryBackend` and `TracerExporter`.
- **0.34** — narrow the framework surface: split fine-tune / rl / n8n
  importer into separate `duxx-ai-*` packages.
- **1.0** — stable APIs; module status promotions ratchet up as test
  coverage lands.

DuxxDB roadmap (the engine half): [DuxxDB/docs/ROADMAP.md](https://github.com/bankyresearch/duxxdb/blob/master/docs/ROADMAP.md).

---

## License

Apache 2.0. See [LICENSE](LICENSE).

DuxxDB is also Apache 2.0. There is no "open core" tier in either
project — the same license applies to every module in this repo and
every crate in the engine.
