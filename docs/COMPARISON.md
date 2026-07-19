# Comparison — Duxx Stack vs. existing market products

This document positions the Duxx Stack (`duxx-ai` + `DuxxDB`) against
products an enterprise team would otherwise glue together. It is
written for evaluation, not marketing — every claim points to a
primitive in this repo or in [DuxxDB](https://github.com/duxxdb/duxxdb).

We compare three layers:

1. **Agent runtime / orchestration** — LangChain, LangGraph, CrewAI,
   AutoGen, Semantic Kernel.
2. **Storage + retrieval** — Pinecone, Qdrant, Weaviate, pgvector,
   LanceDB.
3. **Ops + governance** — LangSmith, Braintrust, Portkey, PromptLayer,
   Arize.

The Duxx Stack tries to be the union of all three with a single wire
protocol and single deploy footprint.

---

## 1. Agent runtime / orchestration

| Capability                                  | LangChain | LangGraph | CrewAI | AutoGen | Semantic Kernel | **duxx-ai** |
|---------------------------------------------|:---------:|:---------:|:------:|:-------:|:---------------:|:-----------:|
| Single-agent tool loop                      |     ✓     |     ✓     |   ✓    |    ✓    |        ✓        |      ✓      |
| ReAct / plan-execute / self-critique        |     ✓     |     ✓     |   —    |    ✓    |        ✓        |      ✓      |
| DAG with conditional edges                  |     ✓     |     ✓     |   —    |    —    |        —        |      ✓      |
| State reducers (append/sum/merge)           |     —     |     ✓     |   —    |    —    |        —        |      ✓      |
| Human-in-the-loop interrupt                 |     —     |     ✓     |   —    |    ✓    |        —        |      ✓      |
| Role-based crew (seq/par/hier)              |     —     |     —     |   ✓    |    ✓    |        —        |      ✓      |
| Built-in middleware hooks                   |     ✓     |     ✓     |   —    |    —    |        ✓        |      ✓      |
| Pluggable memory backend                    |     ✓     |     ✓     |   ~    |    ✓    |        ✓        |      ✓      |
| Pluggable LLM provider                      |     ✓     |     ✓     |   ✓    |    ✓    |        ✓        |      ✓      |
| Continuous self-improvement loop            |     —     |     —     |   —    |    —    |        —        |    **✓**    |
| Causal trace replay w/ overrides            |     —     |     —     |   —    |    —    |        —        |    **✓**    |
| Cost+quality adaptive router                |     —     |     —     |   —    |    —    |        —        |    **✓**    |

**Differentiated:** self-improvement loop, causal trace replay,
adaptive router. The first two require server-side primitives in the
storage layer (`EVAL.CLUSTER_FAILURES`, `REPLAY.CAPTURE`) which is why
they ship as one stack rather than three packages.

### When to pick what

- **LangGraph** wins if you have a deep investment in the LangChain
  ecosystem and want only orchestration.
- **CrewAI** wins for prototype-speed role-based crews with no DAG.
- **AutoGen** wins for research-style multi-agent conversation
  topologies.
- **duxx-ai** wins when you need (a) production self-improvement, (b)
  audit-grade replay, (c) one stack for runtime + storage + eval.

---

## 2. Storage + retrieval

| Capability                                  | Pinecone | Qdrant | Weaviate | pgvector | LanceDB | **DuxxDB** |
|---------------------------------------------|:--------:|:------:|:--------:|:--------:|:-------:|:----------:|
| HNSW vector index                           |    ✓     |   ✓    |    ✓     |    ~     |    ✓    |     ✓      |
| BM25 lexical index                          |    —     |   ✓    |    ✓     |    ✓     |    —    |     ✓      |
| RRF hybrid fusion (one call)                |    —     |   ~    |    ~     |    —     |    —    |     ✓      |
| Persistent HNSW dump (sub-1s reopen at 1M)  |    n/a   |   ✓    |    ✓     |    n/a   |    ✓    |     ✓      |
| Open-source                                 |    —     |   ✓    |    ✓     |    ✓     |    ✓    |     ✓      |
| RESP2/3 wire protocol                       |    —     |   —    |    —     |    —     |    —    |     ✓      |
| gRPC                                        |    ~     |   ✓    |    ✓     |    —     |    —    |     ✓      |
| MCP stdio                                   |    —     |   —    |    —     |    —     |    —    |     ✓      |
| Prompt registry primitive                   |    —     |   —    |    —     |    —     |    —    |     ✓      |
| Eval registry primitive                     |    —     |   —    |    —     |    —     |    —    |     ✓      |
| Replay primitive                            |    —     |   —    |    —     |    —     |    —    |     ✓      |
| Cost ledger primitive                       |    —     |   —    |    —     |    —     |    —    |     ✓      |
| Single binary deploy                        |    n/a   |   ✓    |    ✓     |   ~      |    ✓    |     ✓      |

**Differentiated:** the four extra primitives. Other vector DBs only
do vectors; DuxxDB stores *the artifacts that surround vectors* (the
prompts that produced the embeddings, the evals that scored them, the
costs of generating them, the traces that captured them).

### When to pick what

- **Pinecone** wins on managed-service operability and serverless price.
- **Qdrant** wins on raw vector throughput and Rust-native ecosystem.
- **Weaviate** wins on built-in modules (multi-modal, generative-search).
- **pgvector** wins when your team's only operational competence is
  Postgres.
- **DuxxDB** wins when the surrounding artifacts (prompts, evals, costs,
  replays) need to be queried *together* with the vectors, in one store,
  with one wire protocol.

---

## 3. Ops, governance, eval, gateway

| Capability                                  | LangSmith | Braintrust | Portkey | PromptLayer | Arize | **Duxx Stack** |
|---------------------------------------------|:---------:|:----------:|:-------:|:-----------:|:-----:|:--------------:|
| Span / trace storage                        |     ✓     |     ✓      |    ~    |      —      |   ✓   |       ✓        |
| Offline eval datasets                       |     ✓     |     ✓      |    —    |      —      |   ✓   |       ✓        |
| Online (production) eval                    |     ✓     |     ✓      |    ~    |      —      |   ✓   |       ✓        |
| Failure clustering                          |     ✓     |     ✓      |    —    |      —      |   ✓   |       ✓        |
| Prompt registry / versioning                |     ~     |     ~      |    —    |      ✓      |   —   |       ✓        |
| Canary routing                              |     —     |     —      |    ✓    |      —      |   —   |       ✓        |
| Continuous self-improvement loop            |     —     |     —      |    —    |      —      |   —   |     **✓**      |
| Causal replay w/ override-injection         |     —     |     —      |    —    |      —      |   —   |     **✓**      |
| LLM router (cost / quality / fallback)      |     —     |     —      |    ✓    |      —      |   —   |       ✓        |
| Cost ledger + per-tenant budgets            |     ~     |     —      |    ✓    |      —      |   —   |       ✓        |
| RBAC                                        |     ✓     |     ✓      |    ✓    |      ✓      |   ✓   |       ✓        |
| PII guardrails                              |     ~     |     —      |    ✓    |      —      |   ~   |       ✓        |
| Self-host open-source                       |     ~     |     —      |    ~    |      —      |   ~   |       ✓        |

**Differentiated:** self-improvement loop, causal replay,
single-store join semantics. Every other product in this column is
**a separate SaaS pulling data out of your runtime over webhook**;
the Duxx Stack keeps the runtime and storage in one place, so a
canary's eval scores, costs, and traces are joinable inside one query.

### When to pick what

- **LangSmith** wins for the LangChain mono-ecosystem.
- **Braintrust** wins for eval-first workflows where the rest of your
  stack is bespoke.
- **Portkey** wins as a pure gateway in front of an already-built
  agent system.
- **PromptLayer** wins as a stand-alone prompt CMS.
- **Arize** wins for ML+LLM observability across non-agent workloads.
- **Duxx Stack** wins when you want all of the above without
  five different vendors and five different consistency windows.

---

## 4. The "five vendor" problem (illustrative)

A common production setup looks like:

```
  Agent runtime  : LangGraph
  Vector DB      : Pinecone
  Eval / traces  : LangSmith
  Prompt CMS     : PromptLayer
  Gateway        : Portkey
```

This stack has:

- Five SLAs.
- Five auth surfaces.
- Five regional residency stories (data leaves your VPC up to 5x per
  request).
- No atomic join across them — you cannot ask "show me the prompt
  version that produced this expensive low-score trace" without an
  ETL into a sixth warehouse.

The Duxx Stack collapses these to:

```
  Agent runtime  : duxx-ai
  Storage        : DuxxDB
```

One process, one wire protocol, one auth, one residency story, atomic
joins.

---

## 5. Performance positioning

The numbers below are from DuxxDB benchmarks (Rust criterion), not
synthetic marketing. They are reproducible from `cargo bench` in the
DuxxDB repo.

| Workload                                      | DuxxDB        | Notes                                  |
|-----------------------------------------------|---------------|----------------------------------------|
| HNSW recall@10 (1M × 384-d)                   | ~ 1 ms        | hnsw_rs, default M=16 efC=200          |
| BM25 search on 1M docs                        | ~ 4 ms        | tantivy backend                        |
| RRF hybrid (HNSW + BM25 + fusion in one call) | ~ 6 ms        | server-side, no client round-trips     |
| Persistent HNSW reopen (1M rows)              | < 1 s         | mmaped redb tables + dumped HNSW       |
| EVAL.SCORE write                              | ~ 50 µs       | redb single-table write                |
| TRACE.SPAN write                              | ~ 60 µs       | redb append                            |
| Cold start (server boot, 1M rows loaded)      | ~ 700 ms      | persistent index dumps                 |

Compared to a typical 3-hop network deployment (client → gateway →
vector DB → eval SaaS), eliminating the round-trips is the biggest
win, not raw kernel-level vector math.

---

## 6. License and openness

| Product              | License                  | Open core? | Self-host without a license key? |
|----------------------|--------------------------|-----------|----------------------------------|
| LangChain / LangGraph | MIT                      | yes       | yes                              |
| LangSmith            | Commercial               | no        | partial (paid)                   |
| CrewAI               | MIT                      | yes       | yes                              |
| Pinecone             | Commercial               | no        | no                               |
| Qdrant               | Apache 2.0               | yes       | yes                              |
| Weaviate             | BSD-3                    | yes       | yes                              |
| Portkey              | Apache 2.0 (gateway)     | yes       | yes (limited)                    |
| Braintrust           | Commercial               | no        | no                               |
| **DuxxDB**           | **Apache 2.0**           | **yes**   | **yes**                          |
| **duxx-ai**          | **Apache 2.0**           | **yes**   | **yes**                          |

Apache 2.0 throughout the Duxx Stack — no contributor-license-aliasing,
no "open core but the bits you actually need are paid" pattern.

---

## 7. Summary

The Duxx Stack does not claim to beat any single specialist on its own
metric. It claims that for an enterprise that needs **agent runtime
+ storage + eval + gateway + governance under one operational story**,
the combined surface is smaller, the joins are atomic, the audit
trail is unified, and the unique primitives (self-improvement loop,
causal replay) are not available anywhere else.

The two reference projects in `examples/enterprise/` are the
empirical answer — read `support_ops/README.md` and
`research_desk/README.md` to see the same primitives running an
end-to-end use case.
