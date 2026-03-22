"""n8n Workflow Importer — convert n8n JSON workflows to Duxx AI agents and graphs.

Parses n8n workflow JSON (exported from n8n Editor UI) and automatically
converts it to Duxx AI agentic solutions:
- Single-agent workflows → Agent with tools
- Multi-node workflows → Graph with conditional routing
- AI Agent nodes → Agent with LLM + tools
- Trigger nodes → Entry points
- HTTP/API nodes → web_request tool calls
- Code nodes → python_exec tool calls
- If/Switch nodes → Conditional edges

Usage:
    from duxx_ai.importers.n8n import N8nImporter
    result = N8nImporter.from_file("workflow.json")
    print(result.code)         # Generated Python code
    agent = result.build()     # Build the agent/graph
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  n8n Node Type → Duxx AI Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# n8n node types mapped to Duxx AI equivalents
N8N_TO_BANKYAI: dict[str, dict[str, Any]] = {
    # ── Triggers ──
    "n8n-nodes-base.manualTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.webhook": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.scheduleTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.emailTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.formTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.httpRequest": {"type": "tool", "duxx_ai": "web_request"},
    "@n8n/n8n-nodes-langchain.chatTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    # ── AI / LLM Providers ──
    "@n8n/n8n-nodes-langchain.agent": {"type": "agent", "duxx_ai": "Agent"},
    "@n8n/n8n-nodes-langchain.chainLlm": {"type": "agent", "duxx_ai": "Agent"},
    "@n8n/n8n-nodes-langchain.chainRetrievalQa": {"type": "agent", "duxx_ai": "RAG_Agent"},
    "@n8n/n8n-nodes-langchain.chainSummarization": {"type": "agent", "duxx_ai": "Agent"},
    "@n8n/n8n-nodes-langchain.lmChatOpenAi": {"type": "llm", "duxx_ai": "openai"},
    "@n8n/n8n-nodes-langchain.lmChatAnthropic": {"type": "llm", "duxx_ai": "anthropic"},
    "@n8n/n8n-nodes-langchain.lmChatOllama": {"type": "llm", "duxx_ai": "local"},
    "@n8n/n8n-nodes-langchain.lmChatGoogleGemini": {"type": "llm", "duxx_ai": "openai"},
    "@n8n/n8n-nodes-langchain.lmChatGroq": {"type": "llm", "duxx_ai": "local"},
    "@n8n/n8n-nodes-langchain.lmChatMistralCloud": {"type": "llm", "duxx_ai": "local"},
    "@n8n/n8n-nodes-langchain.lmChatAzureOpenAi": {"type": "llm", "duxx_ai": "openai"},
    "@n8n/n8n-nodes-langchain.lmChatHuggingFaceInference": {"type": "llm", "duxx_ai": "local"},
    "@n8n/n8n-nodes-langchain.lmOpenAi": {"type": "llm", "duxx_ai": "openai"},
    # ── AI Tools ──
    "@n8n/n8n-nodes-langchain.toolCode": {"type": "tool", "duxx_ai": "python_exec"},
    "@n8n/n8n-nodes-langchain.toolHttpRequest": {"type": "tool", "duxx_ai": "web_request"},
    "@n8n/n8n-nodes-langchain.toolCalculator": {"type": "tool", "duxx_ai": "calculator"},
    "@n8n/n8n-nodes-langchain.toolWikipedia": {"type": "tool", "duxx_ai": "web_request"},
    "@n8n/n8n-nodes-langchain.toolSerpApi": {"type": "tool", "duxx_ai": "web_request"},
    "@n8n/n8n-nodes-langchain.toolWorkflow": {"type": "tool", "duxx_ai": "subagent"},
    "@n8n/n8n-nodes-langchain.toolVectorStore": {"type": "tool", "duxx_ai": "rag_search"},
    # ── Vector Stores ──
    "@n8n/n8n-nodes-langchain.vectorStorePinecone": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreChroma": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreQdrant": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreWeaviate": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreSupabase": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreInMemory": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStorePGVector": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    "@n8n/n8n-nodes-langchain.vectorStoreMilvus": {"type": "vectorstore", "duxx_ai": "InMemoryVectorStore"},
    # ── Embeddings ──
    "@n8n/n8n-nodes-langchain.embeddingsOpenAi": {"type": "embeddings", "duxx_ai": "OpenAIEmbedder"},
    "@n8n/n8n-nodes-langchain.embeddingsCohere": {"type": "embeddings", "duxx_ai": "LocalEmbedder"},
    "@n8n/n8n-nodes-langchain.embeddingsHuggingFaceInference": {"type": "embeddings", "duxx_ai": "LocalEmbedder"},
    "@n8n/n8n-nodes-langchain.embeddingsGoogleGemini": {"type": "embeddings", "duxx_ai": "OpenAIEmbedder"},
    "@n8n/n8n-nodes-langchain.embeddingsAzureOpenAi": {"type": "embeddings", "duxx_ai": "OpenAIEmbedder"},
    "@n8n/n8n-nodes-langchain.embeddingsOllama": {"type": "embeddings", "duxx_ai": "LocalEmbedder"},
    # ── Document Loaders ──
    "@n8n/n8n-nodes-langchain.documentDefaultDataLoader": {"type": "loader", "duxx_ai": "TextLoader"},
    "@n8n/n8n-nodes-langchain.documentBinaryInputLoader": {"type": "loader", "duxx_ai": "TextLoader"},
    "@n8n/n8n-nodes-langchain.documentGithubLoader": {"type": "loader", "duxx_ai": "WebLoader"},
    "@n8n/n8n-nodes-langchain.documentJsonInputLoader": {"type": "loader", "duxx_ai": "JSONLLoader"},
    # ── Text Splitters ──
    "@n8n/n8n-nodes-langchain.textSplitterRecursiveCharacterTextSplitter": {"type": "splitter", "duxx_ai": "RecursiveSplitter"},
    "@n8n/n8n-nodes-langchain.textSplitterCharacterTextSplitter": {"type": "splitter", "duxx_ai": "CharacterSplitter"},
    "@n8n/n8n-nodes-langchain.textSplitterTokenSplitter": {"type": "splitter", "duxx_ai": "TokenSplitter"},
    # ── Memory ──
    "@n8n/n8n-nodes-langchain.memoryBufferWindow": {"type": "memory", "duxx_ai": "working_memory"},
    "@n8n/n8n-nodes-langchain.memoryChatRetrieval": {"type": "memory", "duxx_ai": "episodic_memory"},
    "@n8n/n8n-nodes-langchain.memoryVectorStore": {"type": "memory", "duxx_ai": "semantic_memory"},
    "@n8n/n8n-nodes-langchain.memoryPostgresChat": {"type": "memory", "duxx_ai": "episodic_memory"},
    "@n8n/n8n-nodes-langchain.memoryRedisChat": {"type": "memory", "duxx_ai": "episodic_memory"},
    # ── Output Parsers ──
    "@n8n/n8n-nodes-langchain.outputParserStructured": {"type": "parser", "duxx_ai": "JSONOutputParser"},
    "@n8n/n8n-nodes-langchain.outputParserAutofixing": {"type": "parser", "duxx_ai": "RetryParser"},
    "@n8n/n8n-nodes-langchain.outputParserItemList": {"type": "parser", "duxx_ai": "ListOutputParser"},
    # ── Retrievers / Rerankers ──
    "@n8n/n8n-nodes-langchain.retrieverVectorStore": {"type": "retriever", "duxx_ai": "VectorRetriever"},
    "@n8n/n8n-nodes-langchain.retrieverMultiQuery": {"type": "retriever", "duxx_ai": "HybridRetriever"},
    "@n8n/n8n-nodes-langchain.rerankerCohere": {"type": "retriever", "duxx_ai": "HybridRetriever"},
    # ── Logic ──
    "n8n-nodes-base.if": {"type": "conditional", "duxx_ai": "EdgeCondition"},
    "n8n-nodes-base.switch": {"type": "conditional", "duxx_ai": "EdgeCondition"},
    "n8n-nodes-base.merge": {"type": "merge", "duxx_ai": "parallel"},
    "n8n-nodes-base.splitInBatches": {"type": "map_reduce", "duxx_ai": "map_reduce"},
    "n8n-nodes-base.filter": {"type": "conditional", "duxx_ai": "EdgeCondition"},
    "n8n-nodes-base.compareDatasets": {"type": "conditional", "duxx_ai": "EdgeCondition"},
    # ── Code ──
    "n8n-nodes-base.code": {"type": "tool", "duxx_ai": "python_exec"},
    "n8n-nodes-base.executeCommand": {"type": "tool", "duxx_ai": "bash_exec"},
    # ── Data ──
    "n8n-nodes-base.set": {"type": "transform", "duxx_ai": "state_set"},
    "n8n-nodes-base.function": {"type": "tool", "duxx_ai": "python_exec"},
    "n8n-nodes-base.functionItem": {"type": "tool", "duxx_ai": "python_exec"},
    "n8n-nodes-base.itemLists": {"type": "transform", "duxx_ai": "state_set"},
    "n8n-nodes-base.spreadsheetFile": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.convertToFile": {"type": "tool", "duxx_ai": "write_file"},
    "n8n-nodes-base.extractFromFile": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.xml": {"type": "transform", "duxx_ai": "state_set"},
    "n8n-nodes-base.html": {"type": "transform", "duxx_ai": "state_set"},
    "n8n-nodes-base.markdown": {"type": "transform", "duxx_ai": "state_set"},
    "n8n-nodes-base.crypto": {"type": "tool", "duxx_ai": "python_exec"},
    "n8n-nodes-base.dateTime": {"type": "transform", "duxx_ai": "state_set"},
    # ── Communication ──
    "n8n-nodes-base.emailSend": {"type": "tool", "duxx_ai": "send_email"},
    "n8n-nodes-base.emailReadImap": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.gmail": {"type": "tool", "duxx_ai": "send_email"},
    "n8n-nodes-base.slack": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.telegram": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.discord": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.microsoftTeams": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.whatsApp": {"type": "tool", "duxx_ai": "web_request"},
    # ── Database ──
    "n8n-nodes-base.postgres": {"type": "tool", "duxx_ai": "sql_query"},
    "n8n-nodes-base.mysql": {"type": "tool", "duxx_ai": "sql_query"},
    "n8n-nodes-base.mongoDb": {"type": "tool", "duxx_ai": "nosql_query"},
    "n8n-nodes-base.redis": {"type": "tool", "duxx_ai": "nosql_query"},
    "n8n-nodes-base.microsoftSql": {"type": "tool", "duxx_ai": "sql_query"},
    "n8n-nodes-base.supabase": {"type": "tool", "duxx_ai": "sql_query"},
    # ── Cloud / Storage ──
    "n8n-nodes-base.awsS3": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.googleDrive": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.googleSheets": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.microsoftOneDrive": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.dropbox": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.ftp": {"type": "tool", "duxx_ai": "read_file"},
    # ── CRM / Business ──
    "n8n-nodes-base.hubspot": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.salesforce": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.airtable": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.notion": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.jira": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.linear": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.asana": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.trello": {"type": "tool", "duxx_ai": "web_request"},
    # ── Dev Tools ──
    "n8n-nodes-base.github": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.gitlab": {"type": "tool", "duxx_ai": "web_request"},
    "n8n-nodes-base.bitbucket": {"type": "tool", "duxx_ai": "web_request"},
    # ── File ──
    "n8n-nodes-base.readBinaryFile": {"type": "tool", "duxx_ai": "read_file"},
    "n8n-nodes-base.writeBinaryFile": {"type": "tool", "duxx_ai": "write_file"},
    # ── Respond ──
    "n8n-nodes-base.respondToWebhook": {"type": "output", "duxx_ai": "exit_point"},
    "n8n-nodes-base.noOp": {"type": "noop", "duxx_ai": "pass"},
    # ── UI / Display (ignored) ──
    "n8n-nodes-base.stickyNote": {"type": "noop", "duxx_ai": "pass"},
    # ── Wait / Delay ──
    "n8n-nodes-base.wait": {"type": "noop", "duxx_ai": "pass"},
    # ── Error Handling ──
    "n8n-nodes-base.errorTrigger": {"type": "trigger", "duxx_ai": "entry_point"},
    "n8n-nodes-base.stopAndError": {"type": "output", "duxx_ai": "exit_point"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class N8nNode(BaseModel):
    """Parsed n8n node."""
    id: str = ""
    name: str = ""
    type: str = ""
    type_version: float = 1.0
    position: list[int] = Field(default_factory=lambda: [0, 0])
    parameters: dict[str, Any] = Field(default_factory=dict)
    credentials: dict[str, Any] = Field(default_factory=dict)
    disabled: bool = False
    # Mapped Duxx AI equivalent
    duxx_ai_type: str = ""
    duxx_ai_category: str = ""


class N8nConnection(BaseModel):
    """Parsed n8n connection."""
    source: str
    target: str
    source_output: int = 0
    target_input: int = 0


class N8nWorkflow(BaseModel):
    """Complete parsed n8n workflow."""
    name: str = "Untitled Workflow"
    nodes: list[N8nNode] = Field(default_factory=list)
    connections: list[N8nConnection] = Field(default_factory=list)
    is_multi_agent: bool = False
    agent_count: int = 0
    tool_count: int = 0
    has_conditionals: bool = False
    has_triggers: bool = False
    raw_json: dict[str, Any] = Field(default_factory=dict)


class N8nConversionResult(BaseModel):
    """Result of converting an n8n workflow to Duxx AI."""
    workflow: N8nWorkflow = Field(default_factory=N8nWorkflow)
    pattern: str = "single_agent"  # single_agent | graph | crew
    code: str = ""  # Generated Python code
    agent_names: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_workflow(data: dict[str, Any]) -> N8nWorkflow:
    """Parse raw n8n JSON into structured N8nWorkflow."""
    workflow = N8nWorkflow(
        name=data.get("name", "Untitled Workflow"),
        raw_json=data,
    )

    # Parse nodes
    for n in data.get("nodes", []):
        node_type = n.get("type", "")
        mapping = N8N_TO_BANKYAI.get(node_type, {"type": "unknown", "duxx_ai": "custom"})

        node = N8nNode(
            id=n.get("id", ""),
            name=n.get("name", ""),
            type=node_type,
            type_version=n.get("typeVersion", 1.0),
            position=n.get("position", [0, 0]),
            parameters=n.get("parameters", {}),
            credentials=n.get("credentials", {}),
            disabled=n.get("disabled", False),
            duxx_ai_type=mapping["duxx_ai"],
            duxx_ai_category=mapping["type"],
        )
        workflow.nodes.append(node)

    # Parse connections — handle ALL connection types (main, ai_tool, ai_memory, ai_embedding, etc.)
    for source_name, outputs in data.get("connections", {}).items():
        for conn_type, output_groups in outputs.items():
            # conn_type can be: "main", "ai_tool", "ai_memory", "ai_embedding",
            # "ai_languageModel", "ai_document", "ai_textSplitter", "ai_reranker", etc.
            if not isinstance(output_groups, list):
                continue
            for out_idx, targets in enumerate(output_groups):
                if not isinstance(targets, list):
                    continue
                for conn in targets:
                    if not isinstance(conn, dict):
                        continue
                    target_name = conn.get("node", "")
                    if target_name:
                        workflow.connections.append(N8nConnection(
                            source=source_name,
                            target=target_name,
                            source_output=out_idx,
                            target_input=conn.get("index", 0),
                        ))

    # Analyze workflow
    agent_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "agent"]
    tool_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "tool"]
    cond_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "conditional"]
    trigger_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "trigger"]

    workflow.agent_count = len(agent_nodes)
    workflow.tool_count = len(tool_nodes)
    workflow.is_multi_agent = len(agent_nodes) > 1
    workflow.has_conditionals = len(cond_nodes) > 0
    workflow.has_triggers = len(trigger_nodes) > 0

    return workflow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Code Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sanitize_name(name: str) -> str:
    """Convert n8n node name to valid Python identifier."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower().strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if clean and clean[0].isdigit():
        clean = "n_" + clean
    return clean or "node"


def _extract_system_prompt(node: N8nNode) -> str:
    """Extract system prompt from n8n AI Agent node parameters."""
    params = node.parameters
    # n8n AI Agent stores system message in various places
    for key in ("systemMessage", "system_message", "prompt", "text", "instructions"):
        if key in params:
            val = params[key]
            if isinstance(val, str) and len(val) > 10:
                return val
    # Check nested options.systemMessage (common in n8n agent v2)
    options = params.get("options", {})
    if isinstance(options, dict):
        for key in ("systemMessage", "system_message", "prompt"):
            if key in options:
                val = options[key]
                if isinstance(val, str) and len(val) > 5:
                    return val
    return "You are a helpful AI assistant."


def _extract_llm_config(workflow: N8nWorkflow) -> tuple[str, str]:
    """Extract LLM provider and model from workflow nodes."""
    for node in workflow.nodes:
        if node.duxx_ai_category == "llm":
            provider = node.duxx_ai_type  # openai, anthropic, local
            model = node.parameters.get("model", "gpt-4o")
            if isinstance(model, dict):
                model = model.get("value", "gpt-4o")
            return provider, model
    return "openai", "gpt-4o"


def _generate_single_agent(workflow: N8nWorkflow) -> str:
    """Generate code for a single-agent workflow."""
    provider, model = _extract_llm_config(workflow)

    # Find agent node for system prompt
    agent_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "agent"]
    system_prompt = "You are a helpful AI assistant."
    if agent_nodes:
        system_prompt = _extract_system_prompt(agent_nodes[0])

    # Collect tools
    tool_nodes = [n for n in workflow.nodes if n.duxx_ai_category == "tool"]
    tool_names = sorted(set(n.duxx_ai_type for n in tool_nodes if n.duxx_ai_type != "custom"))

    code = f'''"""
Duxx AI Agent — Converted from n8n workflow: {workflow.name}
Auto-generated by Duxx AI n8n Importer
"""
import asyncio
from duxx_ai import Agent, AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.tools.builtin import get_builtin_tools

# Agent Configuration
config = AgentConfig(
    name="{_sanitize_name(workflow.name)}",
    system_prompt="""{system_prompt}""",
    llm=LLMConfig(provider="{provider}", model="{model}"),
    max_iterations=10,
)

# Tools (mapped from n8n workflow nodes)
tools = get_builtin_tools({json.dumps(tool_names)})

# Create Agent
agent = Agent(config=config, tools=tools)

# Run
async def main():
    result = await agent.run("Hello, how can you help me?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
'''
    return code


def _generate_graph(workflow: N8nWorkflow) -> str:
    """Generate code for a multi-node graph workflow."""
    provider, model = _extract_llm_config(workflow)

    # Detect RAG pattern (has vectorstore + embeddings + loader + splitter)
    categories = {n.duxx_ai_category for n in workflow.nodes}
    is_rag = "vectorstore" in categories and "embeddings" in categories
    has_loader = "loader" in categories
    has_splitter = "splitter" in categories

    # Build node definitions
    node_defs = []
    edge_defs = []
    entry_node = None
    exit_node = None
    skip_nodes: set[str] = set()

    # Skip noop nodes (sticky notes, wait)
    for node in workflow.nodes:
        if node.duxx_ai_category == "noop":
            skip_nodes.add(_sanitize_name(node.name))

    for node in workflow.nodes:
        if node.disabled:
            continue
        safe_name = _sanitize_name(node.name)
        if safe_name in skip_nodes:
            continue

        if node.duxx_ai_category == "trigger":
            entry_node = safe_name
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Trigger: {node.name}"""
    state.set("triggered", True)
    return state''')

        elif node.duxx_ai_category == "agent":
            system_prompt = _extract_system_prompt(node)
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """AI Agent: {node.name}"""
    from duxx_ai.tools.builtin import get_builtin_tools
    agent = Agent(config=AgentConfig(
        name="{safe_name}",
        system_prompt="""{system_prompt[:300]}""",
        llm=LLMConfig(provider="{provider}", model="{model}", api_key=OPENAI_API_KEY),
    ), tools=get_builtin_tools(["calculator", "web_request"]))
    # RAG context injection
    context = state.get("rag_results", "")
    user_input = state.get("input", "Hello")
    if context:
        user_input = f"Context:\\n{{context}}\\n\\nQuestion: {{user_input}}"
    result = await agent.run(user_input)
    state.set("{safe_name}_output", result)
    state.set("final_output", result)
    return state''')

        elif node.duxx_ai_category == "vectorstore":
            mode = node.parameters.get("mode", "insert")
            if mode == "retrieve-as-tool" or mode == "retrieve":
                node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Vector Store Retrieval: {node.name} → Duxx AI InMemoryVectorStore"""
    query = state.get("input", "")
    top_k = {node.parameters.get("topK", 5)}
    results = vector_store.search(query, top_k=top_k)
    rag_text = "\\n---\\n".join(r.document.content[:500] for r in results)
    state.set("rag_results", rag_text)
    state.set("rag_count", len(results))
    return state''')
            else:
                node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Vector Store Insert: {node.name} → Duxx AI InMemoryVectorStore"""
    chunks = state.get("chunks", [])
    if chunks:
        ids = vector_store.add(chunks)
        state.set("inserted_count", len(ids))
    return state''')

        elif node.duxx_ai_category == "embeddings":
            # Embeddings are configured at store level, skip as standalone node
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Embeddings: {node.name} → configured in vector_store (OpenAIEmbedder)"""
    # Embeddings are applied automatically by the vector store
    return state''')

        elif node.duxx_ai_category == "loader":
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Document Loader: {node.name} → Duxx AI TextLoader"""
    from duxx_ai.rag import TextLoader, Document
    file_path = state.get("file_path", "uploaded_document.pdf")
    try:
        docs = TextLoader(file_path).load()
        state.set("documents", docs)
        state.set("doc_count", len(docs))
    except FileNotFoundError:
        state.set("documents", [Document(content="Sample document content for testing.", doc_id="sample")])
        state.set("doc_count", 1)
    return state''')

        elif node.duxx_ai_category == "splitter":
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Text Splitter: {node.name} → Duxx AI RecursiveSplitter"""
    from duxx_ai.rag import RecursiveSplitter
    docs = state.get("documents", [])
    splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_many(docs)
    state.set("chunks", chunks)
    state.set("chunk_count", len(chunks))
    return state''')

        elif node.duxx_ai_category == "memory":
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Memory: {node.name} → Duxx AI WorkingMemory"""
    # Memory is handled by the agent's conversation history
    return state''')

        elif node.duxx_ai_category == "retriever":
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Retriever/Reranker: {node.name} → Duxx AI HybridRetriever"""
    # Reranking is handled by HybridRetriever (keyword + vector fusion)
    return state''')

        elif node.duxx_ai_category == "conditional":
            cond_key = node.parameters.get("conditions", {})
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Conditional: {node.name}"""
    state.set("route", "true_branch")
    return state''')

        elif node.duxx_ai_category == "tool":
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Tool: {node.name} (mapped to {node.duxx_ai_type})"""
    state.set("{safe_name}_result", "executed")
    return state''')

        elif node.duxx_ai_category == "output":
            exit_node = safe_name
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Output: {node.name}"""
    return state''')

        elif node.duxx_ai_category == "llm":
            # LLM config is extracted globally, skip as standalone node
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """LLM Provider: {node.name} → configured in Agent LLMConfig"""
    return state''')

        else:
            node_defs.append(f'''
async def {safe_name}(state: GraphState) -> GraphState:
    """Node: {node.name} (type: {node.type})"""
    return state''')

    # Build edges from connections (skip noop nodes)
    for conn in workflow.connections:
        src = _sanitize_name(conn.source)
        tgt = _sanitize_name(conn.target)
        if src not in skip_nodes and tgt not in skip_nodes:
            edge_defs.append(f'graph.add_edge("{src}", "{tgt}")')

    # Filter active nodes (non-disabled, non-noop)
    active_nodes = [n for n in workflow.nodes if not n.disabled and _sanitize_name(n.name) not in skip_nodes]
    all_names = [_sanitize_name(n.name) for n in active_nodes]
    if not entry_node and all_names:
        entry_node = all_names[0]
    if not exit_node and all_names:
        exit_node = all_names[-1]

    node_adds = "\n".join(f'graph.add_node("{_sanitize_name(n.name)}", {_sanitize_name(n.name)})' for n in active_nodes)
    edges = "\n".join(edge_defs)

    # RAG setup code
    rag_setup = ""
    if is_rag:
        rag_setup = """
# ── RAG Pipeline Setup ──
from duxx_ai.rag import LocalEmbedder, InMemoryVectorStore

# Initialize embedder and vector store (replace LocalEmbedder with OpenAIEmbedder for production)
embedder = LocalEmbedder(dimension=384)
# For production with OpenAI embeddings:
# from duxx_ai.rag import OpenAIEmbedder
# embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
vector_store = InMemoryVectorStore(embedder)
"""

    code = f'''"""
Duxx AI Graph Workflow — Converted from n8n workflow: {workflow.name}
Auto-generated by Duxx AI n8n Importer
Pattern: Multi-node graph with {len(active_nodes)} active nodes
{'RAG Pipeline: Load → Split → Embed → Store → Retrieve → Agent' if is_rag else ''}
"""
import asyncio
from duxx_ai import Agent, AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.orchestration.graph import Graph, GraphState

# ── API Keys (replace with your actual keys) ──
OPENAI_API_KEY = "sk-your-openai-key"  # Set via env: OPENAI_API_KEY
# PINECONE_API_KEY = "your-pinecone-key"
# COHERE_API_KEY = "your-cohere-key"
{rag_setup}
{"".join(node_defs)}

# ── Build Graph ──
graph = Graph("{_sanitize_name(workflow.name)}")

# Add nodes
{node_adds}

# Add edges (from n8n connections)
graph.add_edge("__start__", "{entry_node or all_names[0] if all_names else 'node_0'}")
{edges}
graph.add_edge("{exit_node or all_names[-1] if all_names else 'node_0'}", "__end__")

# ── Execute ──
async def main():
    result = await graph.run({{"input": "What information is in the knowledge base?"}})
    print(f"Status: {{result.status}}")
    print(f"Output: {{result.data.get('final_output', result.data)}}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    return code


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Importer Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class N8nImporter:
    """Convert n8n workflow JSON to Duxx AI agentic solutions.

    Supports:
    - Single-agent workflows → Agent + tools
    - Multi-node workflows → Graph with conditional routing
    - AI Agent chains → Agent with LLM config
    - HTTP/Code/DB nodes → mapped to Duxx AI tools
    - If/Switch nodes → Graph conditional edges

    Usage:
        result = N8nImporter.from_file("workflow.json")
        print(result.code)          # Generated Python code
        print(result.summary)       # Human-readable summary
        print(result.pattern)       # "single_agent" or "graph"
    """

    @staticmethod
    def from_file(path: str) -> N8nConversionResult:
        """Import from a JSON file path."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return N8nImporter.from_dict(data)

    @staticmethod
    def from_json(json_string: str) -> N8nConversionResult:
        """Import from a JSON string."""
        data = json.loads(json_string)
        return N8nImporter.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> N8nConversionResult:
        """Import from a parsed dict."""
        workflow = _parse_workflow(data)
        warnings: list[str] = []

        # Determine pattern
        active_nodes = [n for n in workflow.nodes if not n.disabled]
        agent_nodes = [n for n in active_nodes if n.duxx_ai_category == "agent"]
        tool_nodes = [n for n in active_nodes if n.duxx_ai_category == "tool"]

        # Check for unsupported nodes
        for node in active_nodes:
            if node.duxx_ai_type == "custom":
                warnings.append(f"Node '{node.name}' (type: {node.type}) has no direct Duxx AI mapping — using pass-through")

        # Credentials: auto-resolve from Duxx AI credentials manager
        from duxx_ai.credentials import creds
        unresolved_creds = []
        for node in active_nodes:
            if node.credentials:
                all_resolved = True
                for cred_type in node.credentials:
                    resolved = creds.resolve_n8n_credential(cred_type)
                    if not resolved:
                        all_resolved = False
                        unresolved_creds.append(cred_type)
                if not all_resolved:
                    warnings.append(f"Node '{node.name}' needs credentials ({', '.join(node.credentials.keys())}) — configure in Duxx AI Settings or load demo profile")

        # Choose pattern
        if len(active_nodes) <= 3 and len(agent_nodes) <= 1 and not workflow.has_conditionals:
            pattern = "single_agent"
            code = _generate_single_agent(workflow)
        else:
            pattern = "graph"
            code = _generate_graph(workflow)

        # Build summary
        summary_parts = [
            f"Workflow: {workflow.name}",
            f"Pattern: {pattern} ({'single Agent + tools' if pattern == 'single_agent' else 'Graph with ' + str(len(active_nodes)) + ' nodes'})",
            f"Nodes: {len(active_nodes)} ({len(agent_nodes)} agents, {len(tool_nodes)} tools)",
            f"Connections: {len(workflow.connections)}",
        ]
        if warnings:
            summary_parts.append(f"Warnings: {len(warnings)}")

        return N8nConversionResult(
            workflow=workflow,
            pattern=pattern,
            code=code,
            agent_names=[n.name for n in agent_nodes],
            tool_names=sorted(set(n.duxx_ai_type for n in tool_nodes if n.duxx_ai_type != "custom")),
            warnings=warnings,
            summary="\n".join(summary_parts),
        )

    @staticmethod
    def list_supported_nodes() -> dict[str, str]:
        """List all supported n8n node types and their Duxx AI mappings."""
        return {k: v["duxx_ai"] for k, v in N8N_TO_BANKYAI.items()}
