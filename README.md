<p align="center">
  <strong>Duxx AI</strong><br>
  <em>Enterprise Agentic AI SDK</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/duxx-ai/"><img src="https://img.shields.io/pypi/v/duxx-ai?color=d6336c&label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/duxxai/duxx-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python"></a>
</p>

---

Build, fine-tune, orchestrate, and govern AI agents at scale. The only open-source SDK that unifies **agent orchestration**, **fine-tuning pipelines**, **enterprise governance**, and **adaptive model routing** in a single framework.

## Quick Start

```bash
pip install duxx-ai
```

```python
from duxx_ai import Agent, AgentConfig
from duxx_ai.templates import DeepResearcherAgent

# Create an agent in 3 lines
agent = DeepResearcherAgent.create()
result = await agent.run("Analyze Q4 market trends")
print(result)
```

## Features

### Core Agent Framework
```python
from duxx_ai.core import Agent, AgentConfig, Tool, tool
from duxx_ai.core.llm import LLMConfig

# Custom tool
@tool(name="search", description="Search the web")
def search(query: str) -> str:
    return f"Results for: {query}"

# Agent with tools
agent = Agent(
    config=AgentConfig(name="researcher", system_prompt="You are a research assistant."),
    llm_config=LLMConfig(provider="openai", model="gpt-4"),
    tools=[search],
)
result = await agent.run("Find latest AI papers")
```

### Graph Orchestration (like LangGraph)
```python
from duxx_ai.orchestration import Graph

graph = Graph("research-pipeline")
graph.add_node("gather", gather_data)
graph.add_node("analyze", analyze_data)
graph.add_node("review", human_review, node_type="HUMAN")
graph.add_node("report", generate_report)

graph.add_edge("gather", "analyze")
graph.add_conditional_edge("analyze", route_fn, {
    "needs_review": "review",
    "approved": "report"
})

# State reducers for parallel merge
graph.set_state_reducer("results", append_reducer)

result = await graph.run()
```

### Multi-Agent Crews (like CrewAI)
```python
from duxx_ai.orchestration import Crew

crew = Crew(
    name="research-team",
    agents=[researcher, writer, reviewer],
    strategy="hierarchical",  # or "sequential", "parallel"
)
result = await crew.run("Write a market analysis report")
```

### 14 Enterprise Agent Templates
```python
from duxx_ai.templates import (
    DeepResearcherAgent,    # Research & analysis
    InvestmentBankerAgent,  # DCF, comps, due diligence
    PortfolioManagerAgent,  # Portfolio optimization
    CallCenterAgent,        # Customer service
    CodeBuilderAgent,       # Software engineering
    SecurityAgent,          # Vulnerability analysis
    DevOpsAgent,            # Infrastructure automation
    ComplianceAgent,        # GDPR, HIPAA, SOX
    VirtualCFO,             # Financial strategy
    VirtualCMO,             # Marketing strategy
    VirtualCHRO,            # HR & workforce planning
    EmailAgent,             # Email automation
    MarketingAgent,         # Campaign analysis
    FinanceManagerAgent,    # Budgeting & forecasting
)

# One-line agent creation
agent = VirtualCFO.create()
result = await agent.run("Analyze our Q4 cash flow projections")
```

### 40+ Tools Across 9 Domains
```python
from duxx_ai.tools.registry import get_tools

# Get tools by domain
finance_tools = get_tools(domains=["financial"])
security_tools = get_tools(domains=["security"])
all_tools = get_tools(domains=["email", "calendar", "database", "api",
                                "document", "financial", "security",
                                "devops", "analytics"])
```

| Domain | Tools |
|--------|-------|
| Email | send_email, read_inbox, search_email, reply_email |
| Calendar | schedule_meeting, check_availability, list_events |
| Database | sql_query, nosql_query, describe_table |
| API | rest_call, graphql_query |
| Document | parse_pdf, extract_tables, summarize_document |
| Financial | stock_price, market_analysis, portfolio_metrics |
| Security | scan_vulnerabilities, check_compliance, check_ssl |
| DevOps | deploy_service, check_status, rollback, scale_service |
| Analytics | track_metrics, generate_report, query_metrics |

### 5-Tier Memory System
```python
from duxx_ai.memory import MemoryManager

memory = MemoryManager()
memory.store("working", "current_task", "Analyzing market data")
memory.store("episodic", "last_meeting", {"topic": "Q4 review"})
memory.store("semantic", "company_info", {"revenue": "10M"})
memory.store("procedural", "report_steps", ["gather", "analyze", "write"])
memory.store("shared", "team_context", {"project": "Alpha"})
```

### Pluggable memory backend (v0.31+)

As of v0.31, `MemoryManager` accepts a pluggable storage backend.
Default is the same in-process dict + JSON-file store you've always
used. Two new options ship for production:

| Backend | Install | Use when |
|---|---|---|
| `InProcessBackend` (default) | (no extras) | Dev, tests, single-process scripts |
| `DuxxBackend` | `pip install duxx-ai[duxxdb]` | Embedded, sub-ms hybrid recall, persistence — same Python process |
| `DuxxServerBackend` | `pip install duxx-ai[duxxdb-server]` | Multi-worker fleets sharing memory through a `duxx-server` daemon |

```python
from duxx_ai.memory import MemoryManager, MemoryEntry
from duxx_ai.memory.storage import DuxxBackend          # embedded
# from duxx_ai.memory.storage import DuxxServerBackend  # remote daemon

memory = MemoryManager(
    backend=DuxxBackend(dim=1536),
    agent_id="alice",
)

memory.remember(
    "I lost my wallet at the cafe",
    memory_type="episodic",
    embedding=embedding_fn("I lost my wallet at the cafe"),
)

hits = memory.recall("wallet", k=5,
                     query_embedding=embedding_fn("wallet"))
for h in hits:
    print(h.content)
```

The pair forms the **Duxx Stack** — duxx-ai is the framework half,
[DuxxDB](https://github.com/bankyresearch/duxxdb) is the storage
engine half, both Apache 2.0. See the
[Duxx Stack integration design](https://github.com/bankyresearch/duxxdb/blob/master/docs/DUXX_STACK_INTEGRATION.md).

### Enterprise Governance
```python
from duxx_ai.governance import GuardrailChain, RBACManager

# 5 guardrail types
guardrails = GuardrailChain([
    {"type": "pii_filter"},           # Mask PII
    {"type": "prompt_injection"},     # Block injections
    {"type": "content_filter"},       # Filter harmful content
    {"type": "hallucination_check"},  # Verify claims
    {"type": "token_budget", "max_tokens": 10000},
])

# RBAC with 4 roles
rbac = RBACManager()
rbac.assign_role("user@company.com", "developer")  # admin, developer, operator, viewer
```

### RAG Pipeline
```python
from duxx_ai.rag import FileLoader, RecursiveTextSplitter, LocalEmbedder, InMemoryVectorStore

# Load -> Split -> Embed -> Store -> Retrieve
loader = FileLoader()
docs = loader.load("reports/q4.pdf")
chunks = RecursiveTextSplitter(chunk_size=512).split(docs)
embedder = LocalEmbedder()
store = InMemoryVectorStore(embedder)
store.add(chunks)
results = store.search("revenue growth", top_k=5)
```

### Observability & Evaluation
```python
from duxx_ai.observability import Tracer, AgentEvaluator

# Distributed tracing
tracer = Tracer(exporters=["console", "json", "otlp"])
agent = Agent(config=config, tracer=tracer)

# Agent evaluation
evaluator = AgentEvaluator()
metrics = evaluator.evaluate(agent, test_cases=[
    {"input": "What is 2+2?", "expected": "4"},
])
print(f"Accuracy: {metrics.accuracy}, Latency: {metrics.avg_latency}ms")
```

### Adaptive Model Router
```python
from duxx_ai.router import AdaptiveRouter

router = AdaptiveRouter(
    providers=["openai", "anthropic", "local"],
    budget_limit=10.0,  # $10 max
)
# Automatically routes simple tasks to cheap models, complex to powerful ones
response = await router.route("Summarize this paragraph")
```

### Local Fine-Tuning
```python
from duxx_ai.finetune import FineTunePipeline, FineTuneConfig

pipeline = FineTunePipeline(FineTuneConfig(
    base_model="unsloth/Qwen2.5-7B",
    method="qlora",
    epochs=3,
    learning_rate=2e-4,
    dataset_path="./training_data.jsonl",
))
await pipeline.run()
```

### n8n Workflow Import
```python
from duxx_ai.importers.n8n import N8nImporter

importer = N8nImporter()
result = importer.convert(n8n_json)
print(result["python_code"])  # Ready-to-run Duxx AI code
```

## Architecture

```
duxx_ai/
├── core/          # Agent, Tool, Message, LLM (OpenAI/Anthropic/Local)
├── orchestration/ # Graph (DAG + HITL + map-reduce), Crew (3 strategies)
├── memory/        # 5-tier memory system
├── governance/    # Guardrails (5 types), RBAC (4 roles), Audit
├── observability/ # Tracer (OTel), AgentEvaluator, cost tracking
├── router/        # Adaptive complexity-based routing
├── rag/           # Loaders, splitters, embeddings, vector store, retriever
├── finetune/      # Unsloth/PEFT pipeline, LoRA/QLoRA, dataset tools
├── tools/         # 8 builtin + 9 domain libraries (40+ tools)
├── templates/     # 14 enterprise agent templates
├── importers/     # n8n workflow converter
└── cli/           # Click CLI
```

> **Duxx AI Cloud** — Visual Studio dashboard, workflow builder, cloud fine-tuning, and fleet management are available at [duxxai.com](https://duxxai.com). [Join the waitlist](https://duxxai.com/waitlist).

## Requirements

- Python 3.10+
- API keys for LLM providers (OpenAI, Anthropic, or local models)

## Installation

```bash
# Core SDK
pip install duxx-ai

# With fine-tuning support
pip install duxx-ai[finetune]

# Everything
pip install duxx-ai[all]
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://duxxai.com/docs)
- [PyPI](https://pypi.org/project/duxx-ai/)
- [GitHub](https://github.com/duxxai/duxx-ai)
