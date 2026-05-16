"""Workflow Code Generator — Convert React Flow canvas to Duxx AI Python code.

Takes the React Flow JSON (nodes + edges from the visual workflow builder)
and generates runnable Duxx AI Python code using Agent, Graph, Tool, etc.
"""

from __future__ import annotations

from typing import Any


def generate_code(workflow: dict[str, Any]) -> str:
    """Generate Duxx AI Python code from a React Flow workflow JSON.

    Args:
        workflow: Dict with 'name', 'nodes' (list), 'edges' (list)

    Returns:
        Complete Python script as a string.
    """
    name = workflow.get("name", "workflow")
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    if not nodes:
        return "# Empty workflow — add nodes to generate code."

    # Build adjacency map
    adj: dict[str, list[str]] = {}
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)

    # Classify nodes
    node_map: dict[str, dict] = {}
    triggers: list[dict] = []
    agents: list[dict] = []
    tools: list[dict] = []
    logic: list[dict] = []
    outputs: list[dict] = []
    others: list[dict] = []

    for n in nodes:
        nid = n.get("id", "")
        ntype = n.get("type", "tool")
        data = n.get("data", {})
        config = data.get("config", {})
        label = data.get("label", "Node")
        node_map[nid] = {**n, "_label": label, "_type": ntype, "_config": config}

        if ntype == "trigger":
            triggers.append(node_map[nid])
        elif ntype in ("agent", "subagent"):
            agents.append(node_map[nid])
        elif ntype in ("conditional", "merge", "human"):
            logic.append(node_map[nid])
        elif ntype == "output":
            outputs.append(node_map[nid])
        elif ntype in ("tool", "http", "code", "database", "email", "rag", "integration"):
            tools.append(node_map[nid])
        else:
            others.append(node_map[nid])

    # Determine pattern
    is_single_agent = len(agents) == 1 and not logic
    is_graph = len(agents) > 1 or logic

    # Generate code
    lines: list[str] = []

    # ── Header ──
    safe_name = _safe_var(name)
    lines.append(f'"""Duxx AI Workflow: {name}')
    lines.append("")
    lines.append("Auto-generated from Duxx AI Workflow Builder.")
    lines.append(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    lines.append('"""')
    lines.append("")
    lines.append("import asyncio")
    lines.append("")

    # ── Imports ──
    imports = set()
    imports.add("from duxx_ai.core.agent import Agent, AgentConfig")
    imports.add("from duxx_ai.core.llm import LLMConfig")

    if tools:
        imports.add("from duxx_ai.core.tool import Tool, tool")
        imports.add("from duxx_ai.tools.builtin import get_builtin_tools")

    if is_graph:
        imports.add("from duxx_ai.orchestration.graph import Graph")

    for t in tools:
        tc = t["_config"]
        if t["_type"] == "http":
            imports.add("from duxx_ai.tools.builtin import get_builtin_tools")
        if t["_type"] == "database":
            imports.add("from duxx_ai.tools.database import get_database_tools")
        if t["_type"] == "email":
            imports.add("from duxx_ai.tools.email import get_email_tools")
        if t["_type"] == "rag":
            imports.add("from duxx_ai.rag.retriever import SimpleRetriever")

    for imp in sorted(imports):
        lines.append(imp)
    lines.append("")
    lines.append("")

    # ── Tool Definitions ──
    custom_tool_vars: list[str] = []
    for i, t in enumerate(tools):
        tc = t["_config"]
        tvar = _safe_var(t["_label"])
        custom_tool_vars.append(tvar)

        if t["_type"] == "http":
            url = tc.get("url", "https://api.example.com")
            method = tc.get("method", "GET")
            lines.append("@tool")
            lines.append(f'def {tvar}(input_data: str) -> str:')
            lines.append(f'    """HTTP {method} request to {url}"""')
            lines.append("    import httpx")
            lines.append(f'    resp = httpx.request("{method}", "{url}")')
            lines.append("    return resp.text")
            lines.append("")
            lines.append("")

        elif t["_type"] == "code":
            code = tc.get("code", "result = 'Hello'")
            lines.append("@tool")
            lines.append(f'def {tvar}(input_data: str) -> str:')
            lines.append('    """Execute custom Python code"""')
            for cline in code.split("\n"):
                lines.append(f"    {cline}")
            lines.append("    return str(result) if 'result' in dir() else 'done'")
            lines.append("")
            lines.append("")

        elif t["_type"] == "database":
            db_type = tc.get("dbType", "postgresql")
            query = tc.get("query", "SELECT 1")
            conn = tc.get("connectionString", "")
            lines.append("@tool")
            lines.append(f'def {tvar}(query: str = "{query}") -> str:')
            lines.append(f'    """Execute {db_type} database query"""')
            lines.append(f'    # Connection: {conn or "configure connection string"}')
            lines.append('    return f"Query result for: {query}"')
            lines.append("")
            lines.append("")

        elif t["_type"] == "email":
            to = tc.get("to", "")
            subject = tc.get("subject", "")
            lines.append("@tool")
            lines.append(f'def {tvar}(body: str) -> str:')
            lines.append(f'    """Send email to {to}"""')
            lines.append(f'    # Subject: {subject}')
            lines.append(f'    return f"Email sent to {to}"')
            lines.append("")
            lines.append("")

        elif t["_type"] == "rag":
            store = tc.get("vectorStore", "memory")
            topk = tc.get("topK", "5")
            lines.append("@tool")
            lines.append(f'def {tvar}(query: str) -> str:')
            lines.append(f'    """RAG retrieval from {store} vector store (top_k={topk})"""')
            lines.append(f'    retriever = SimpleRetriever(top_k={topk})')
            lines.append('    results = retriever.retrieve(query)')
            lines.append('    return "\\n".join(r.content for r in results)')
            lines.append("")
            lines.append("")

        else:
            lines.append("@tool")
            lines.append(f'def {tvar}(input_data: str) -> str:')
            lines.append(f'    """{t["_label"]}"""')
            lines.append('    return f"Processed: {input_data}"')
            lines.append("")
            lines.append("")

    # ── Agent Definitions ──
    if is_single_agent and agents:
        a = agents[0]
        ac = a["_config"]
        provider = ac.get("provider", "openai")
        model = ac.get("model", "gpt-4o")
        sys_prompt = ac.get("systemPrompt", "You are a helpful AI assistant.")
        temp = ac.get("temperature", "0.7")
        max_tokens = ac.get("maxTokens", "4096")
        agent_var = _safe_var(a["_label"])

        # Collect tools connected to this agent
        agent_tools = []
        agent_id = a.get("id", "")
        for tid in adj.get(agent_id, []):
            if tid in node_map and node_map[tid]["_type"] in ("tool", "http", "code", "database", "email", "rag"):
                agent_tools.append(_safe_var(node_map[tid]["_label"]))
        # Also tools connected TO this agent
        for e in edges:
            if e.get("target") == agent_id and e.get("source") in node_map:
                src_node = node_map[e["source"]]
                if src_node["_type"] in ("tool", "http", "code", "database", "email", "rag"):
                    tvar = _safe_var(src_node["_label"])
                    if tvar not in agent_tools:
                        agent_tools.append(tvar)

        # Use custom + builtin tools
        tool_list = ", ".join(agent_tools) if agent_tools else ""
        builtin_str = 'get_builtin_tools(["calculator", "web_request"])'

        lines.append(f"# ── Agent: {a['_label']} ──")
        lines.append("agent = Agent(")
        lines.append("    config=AgentConfig(")
        lines.append(f'        name="{agent_var}",')
        lines.append(f'        system_prompt="""{sys_prompt}""",')
        lines.append("    ),")
        lines.append("    llm_config=LLMConfig(")
        lines.append(f'        provider="{provider}",')
        lines.append(f'        model="{model}",')
        lines.append(f"        temperature={temp},")
        lines.append(f"        max_tokens={max_tokens},")
        lines.append("    ),")
        if agent_tools:
            lines.append(f"    tools=[{tool_list}] + {builtin_str},")
        else:
            lines.append(f"    tools={builtin_str},")
        lines.append(")")
        lines.append("")
        lines.append("")

        # ── Main entry ──
        lines.append('async def main():')
        lines.append(f'    """Run the {name} workflow."""')
        lines.append('    result = await agent.run("Hello! How can you help me?")')
        lines.append('    print(result)')
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    asyncio.run(main())")

    elif is_graph:
        # ── Graph-based workflow ──
        lines.append(f'# ── Graph Workflow: {name} ──')
        lines.append(f'graph = Graph("{safe_name}")')
        lines.append("")

        # Add agent nodes
        for i, a in enumerate(agents):
            ac = a["_config"]
            avar = _safe_var(a["_label"])
            provider = ac.get("provider", "openai")
            model = ac.get("model", "gpt-4o")
            sys_prompt = ac.get("systemPrompt", "You are a helpful assistant.")
            temp = ac.get("temperature", "0.7")

            lines.append(f"# Agent: {a['_label']}")
            lines.append(f"{avar}_agent = Agent(")
            lines.append("    config=AgentConfig(")
            lines.append(f'        name="{avar}",')
            lines.append(f'        system_prompt="""{sys_prompt}""",')
            lines.append("    ),")
            lines.append("    llm_config=LLMConfig(")
            lines.append(f'        provider="{provider}",')
            lines.append(f'        model="{model}",')
            lines.append(f"        temperature={temp},")
            lines.append("    ),")

            # Find connected tools
            agent_id = a.get("id", "")
            connected_tools = []
            for tid in adj.get(agent_id, []):
                if tid in node_map and node_map[tid]["_type"] in ("tool", "http", "code", "database", "email", "rag"):
                    connected_tools.append(_safe_var(node_map[tid]["_label"]))
            for e in edges:
                if e.get("target") == agent_id and e.get("source") in node_map:
                    src_node = node_map[e["source"]]
                    if src_node["_type"] in ("tool", "http", "code", "database", "email", "rag"):
                        tv = _safe_var(src_node["_label"])
                        if tv not in connected_tools:
                            connected_tools.append(tv)

            if connected_tools:
                lines.append(f"    tools=[{', '.join(connected_tools)}],")
            lines.append(")")
            lines.append("")

            # Add as graph node
            lines.append(f"async def {avar}_handler(state):")
            lines.append(f'    result = await {avar}_agent.run(state.get("input", ""))')
            lines.append('    state["output"] = result')
            lines.append("    return state")
            lines.append("")
            lines.append(f'graph.add_node("{avar}", {avar}_handler)')
            lines.append("")

        # Add tool nodes as graph nodes
        for t in tools:
            tvar = _safe_var(t["_label"])
            lines.append(f"async def {tvar}_handler(state):")
            lines.append(f'    result = {tvar}(state.get("input", ""))')
            lines.append(f'    state["{tvar}_output"] = result')
            lines.append("    return state")
            lines.append(f'graph.add_node("{tvar}", {tvar}_handler)')
            lines.append("")

        # Add conditional nodes
        for c in logic:
            if c["_type"] == "conditional":
                cvar = _safe_var(c["_label"])
                cc = c["_config"]
                key = cc.get("conditionKey", "status")
                val = cc.get("conditionValue", "true")
                lines.append(f'def {cvar}_router(state):')
                lines.append(f'    if state.get("{key}") == "{val}":')
                lines.append('        return "true"')
                lines.append('    return "false"')
                lines.append("")

            elif c["_type"] == "human":
                hvar = _safe_var(c["_label"])
                lines.append(f'graph.add_node("{hvar}", None, node_type="HUMAN")')
                lines.append("")

        # Add edges from React Flow edges
        lines.append("# ── Edges ──")
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src in node_map and tgt in node_map:
                src_label = _safe_var(node_map[src]["_label"])
                tgt_label = _safe_var(node_map[tgt]["_label"])
                src_type = node_map[src]["_type"]

                if src_type == "trigger":
                    lines.append(f'graph.set_entry_point("{tgt_label}")')
                elif src_type == "conditional":
                    edge_id = e.get("sourceHandle", "")
                    if edge_id == "true":
                        lines.append('# Conditional true branch')
                    elif edge_id == "false":
                        lines.append('# Conditional false branch')
                    lines.append(f'graph.add_edge("{src_label}", "{tgt_label}")')
                else:
                    lines.append(f'graph.add_edge("{src_label}", "{tgt_label}")')
        lines.append("")

        # Set end node
        if outputs:
            out_var = _safe_var(outputs[0]["_label"])
            lines.append(f'graph.set_finish_point("{out_var}")')
        lines.append("")

        # Main
        lines.append('async def main():')
        lines.append(f'    """Run the {name} workflow."""')
        lines.append('    result = await graph.run({"input": "Hello! Process this workflow."})')
        lines.append('    print(result)')
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    asyncio.run(main())")

    else:
        # Fallback: simple script
        lines.append("# Simple workflow")
        lines.append('async def main():')
        lines.append(f'    agent = Agent(config=AgentConfig(name="{safe_name}"))')
        lines.append('    result = await agent.run("Hello!")')
        lines.append('    print(result)')
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    asyncio.run(main())")

    return "\n".join(lines)


def _safe_var(name: str) -> str:
    """Convert a display name to a safe Python variable name."""
    import re
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "node_" + s
    # Avoid Python keywords
    if s in ("if", "else", "for", "while", "return", "import", "from", "class", "def", "try", "except", "with", "as", "in", "not", "and", "or", "is", "None", "True", "False"):
        s = s + "_node"
    return s
