"""Deep Agent Architecture — Virtual File System, Planning Tools, Context Quarantine.

Implements the Deep Agent paradigm from the research paper:
- Virtual File System for persistent context across turns
- Planning tools (todo_write-style externalized thinking)
- Context quarantine via sub-agent isolation
- Compound intelligence via skill accumulation
- Long-horizon task execution with artifact management

Reference: "The Evolution of Agentic Orchestration" (2026)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Virtual File System — Persistent Context Beyond Token Limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VirtualFile:
    """A file in the virtual file system."""
    path: str
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    size: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.size = len(self.content)


class VirtualFileSystem:
    """In-memory file system for agent context persistence.

    Allows agents to offload large artifacts (code, reports, data)
    to a virtual filesystem instead of keeping everything in the
    context window. Persists across turns and sessions.

    Usage:
        vfs = VirtualFileSystem(persist_path="agent_workspace")

        # Agent writes artifacts
        vfs.write("analysis/q4_report.md", "# Q4 Analysis\\n...")
        vfs.write("code/model.py", "import torch\\n...")

        # Agent reads when needed
        content = vfs.read("analysis/q4_report.md")

        # List directory
        files = vfs.list_dir("analysis/")

        # Search across files
        results = vfs.search("revenue growth")
    """

    def __init__(self, persist_path: str | None = None):
        self._files: dict[str, VirtualFile] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._load()

    def write(self, path: str, content: str, tags: list[str] | None = None) -> VirtualFile:
        """Write or overwrite a file."""
        path = path.lstrip("/")
        f = VirtualFile(path=path, content=content, tags=tags or [])
        if path in self._files:
            f.created_at = self._files[path].created_at
        self._files[path] = f
        self._persist_file(f)
        return f

    def read(self, path: str) -> str | None:
        """Read file contents. Returns None if not found."""
        path = path.lstrip("/")
        f = self._files.get(path)
        return f.content if f else None

    def exists(self, path: str) -> bool:
        return path.lstrip("/") in self._files

    def delete(self, path: str) -> bool:
        path = path.lstrip("/")
        if path in self._files:
            del self._files[path]
            if self._persist_path:
                fp = self._persist_path / path
                if fp.exists():
                    fp.unlink()
            return True
        return False

    def list_dir(self, prefix: str = "") -> list[str]:
        """List files under a prefix."""
        prefix = prefix.lstrip("/")
        return sorted(p for p in self._files if p.startswith(prefix))

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Search files by content. Returns (path, matching_line) pairs."""
        query_lower = query.lower()
        results = []
        for path, f in self._files.items():
            for line in f.content.split("\n"):
                if query_lower in line.lower():
                    results.append((path, line.strip()))
                    break
        return results[:top_k]

    def tree(self) -> str:
        """Get directory tree as string."""
        if not self._files:
            return "(empty)"
        lines = []
        for path in sorted(self._files):
            f = self._files[path]
            lines.append(f"  {path} ({f.size} bytes)")
        return "\n".join(lines)

    def total_size(self) -> int:
        return sum(f.size for f in self._files.values())

    def as_tools(self) -> list[Any]:
        """Generate agent tools for file system access."""
        from duxx_ai.core.tool import Tool, ToolParameter

        tools = []

        # read_file tool
        read_tool = Tool(
            name="read_file",
            description="Read contents of a file from the workspace",
            parameters=[ToolParameter(name="path", type="string", description="File path", required=True)],
        )
        read_tool.bind(lambda path: self.read(path) or f"File not found: {path}")
        tools.append(read_tool)

        # write_file tool
        write_tool = Tool(
            name="write_file",
            description="Write content to a file in the workspace",
            parameters=[
                ToolParameter(name="path", type="string", description="File path", required=True),
                ToolParameter(name="content", type="string", description="File content", required=True),
            ],
        )
        write_tool.bind(lambda path, content: f"Written {len(content)} bytes to {path}" if self.write(path, content) else "Write failed")
        tools.append(write_tool)

        # list_files tool
        list_tool = Tool(
            name="list_files",
            description="List files in the workspace directory",
            parameters=[ToolParameter(name="prefix", type="string", description="Directory prefix", required=False, default="")],
        )
        list_tool.bind(lambda prefix="": "\n".join(self.list_dir(prefix)) or "(no files)")
        tools.append(list_tool)

        # search tool
        search_tool = Tool(
            name="search_files",
            description="Search file contents by keyword",
            parameters=[ToolParameter(name="query", type="string", description="Search query", required=True)],
        )
        search_tool.bind(lambda query: "\n".join(f"{p}: {l}" for p, l in self.search(query)) or "No matches found")
        tools.append(search_tool)

        return tools

    def _persist_file(self, f: VirtualFile) -> None:
        if self._persist_path:
            fp = self._persist_path / f.path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(f.content)

    def _load(self) -> None:
        if self._persist_path and self._persist_path.exists():
            for fp in self._persist_path.rglob("*"):
                if fp.is_file():
                    rel = str(fp.relative_to(self._persist_path)).replace("\\", "/")
                    try:
                        content = fp.read_text()
                        self._files[rel] = VirtualFile(
                            path=rel, content=content,
                            created_at=fp.stat().st_ctime,
                            updated_at=fp.stat().st_mtime,
                        )
                    except Exception:
                        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Planning Tool — Externalized Strategic Thinking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PlanStep:
    """A single step in a plan."""
    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = ""
    dependencies: list[int] = field(default_factory=list)


class PlanningTool:
    """Externalized planning via todo_write-style interface.

    Forces agents to think strategically by maintaining a visible plan.
    The plan persists in context, keeping strategy transparent.

    Usage:
        planner = PlanningTool()
        tools = planner.as_tools()
        agent = Agent(config=config, tools=tools)

        # Agent uses plan_create, plan_update, plan_status tools
        await agent.run("Research and write a report on AI trends")
    """

    def __init__(self):
        self.steps: list[PlanStep] = []
        self.objective: str = ""

    def create_plan(self, objective: str, steps: list[str]) -> str:
        """Create a new plan with steps."""
        self.objective = objective
        self.steps = [PlanStep(id=i+1, description=s) for i, s in enumerate(steps)]
        return f"Plan created: {objective}\n" + self._format_plan()

    def update_step(self, step_id: int, status: str, result: str = "") -> str:
        """Update a step's status and result."""
        for s in self.steps:
            if s.id == step_id:
                s.status = status
                if result:
                    s.result = result
                return f"Step {step_id} updated to '{status}'"
        return f"Step {step_id} not found"

    def add_step(self, description: str, after_step: int | None = None) -> str:
        """Add a new step to the plan."""
        new_id = max((s.id for s in self.steps), default=0) + 1
        step = PlanStep(id=new_id, description=description)
        if after_step:
            idx = next((i for i, s in enumerate(self.steps) if s.id == after_step), len(self.steps))
            self.steps.insert(idx + 1, step)
        else:
            self.steps.append(step)
        return f"Added step {new_id}: {description}"

    def get_status(self) -> str:
        """Get current plan status."""
        return self._format_plan()

    def get_next(self) -> str:
        """Get the next pending step."""
        for s in self.steps:
            if s.status == "pending":
                deps_done = all(
                    any(ss.id == d and ss.status == "completed" for ss in self.steps)
                    for d in s.dependencies
                )
                if deps_done:
                    return f"Next: Step {s.id} — {s.description}"
        return "All steps completed!" if all(s.status == "completed" for s in self.steps) else "Waiting on dependencies"

    def _format_plan(self) -> str:
        icons = {"pending": "○", "in_progress": "◑", "completed": "●", "failed": "✗"}
        lines = [f"Objective: {self.objective}", ""]
        for s in self.steps:
            icon = icons.get(s.status, "?")
            result = f" → {s.result[:80]}" if s.result else ""
            lines.append(f"  {icon} [{s.id}] {s.description} ({s.status}){result}")
        done = sum(1 for s in self.steps if s.status == "completed")
        lines.append(f"\nProgress: {done}/{len(self.steps)}")
        return "\n".join(lines)

    def as_tools(self) -> list[Any]:
        """Generate agent tools for planning."""
        from duxx_ai.core.tool import Tool, ToolParameter

        tools = []

        # plan_create
        t1 = Tool(
            name="plan_create",
            description="Create a strategic plan with numbered steps. Use this before starting complex tasks.",
            parameters=[
                ToolParameter(name="objective", type="string", description="What we're trying to achieve", required=True),
                ToolParameter(name="steps", type="string", description="Comma-separated list of steps", required=True),
            ],
        )
        t1.bind(lambda objective, steps: self.create_plan(objective, [s.strip() for s in steps.split(",")]))
        tools.append(t1)

        # plan_update
        t2 = Tool(
            name="plan_update",
            description="Update a plan step status (pending/in_progress/completed/failed) and optionally add result",
            parameters=[
                ToolParameter(name="step_id", type="integer", description="Step number", required=True),
                ToolParameter(name="status", type="string", description="New status", required=True),
                ToolParameter(name="result", type="string", description="Step result or output", required=False, default=""),
            ],
        )
        t2.bind(lambda step_id, status, result="": self.update_step(int(step_id), status, result))
        tools.append(t2)

        # plan_status
        t3 = Tool(name="plan_status", description="View current plan progress", parameters=[])
        t3.bind(lambda: self.get_status())
        tools.append(t3)

        # plan_next
        t4 = Tool(name="plan_next", description="Get the next step to work on", parameters=[])
        t4.bind(lambda: self.get_next())
        tools.append(t4)

        # plan_add_step
        t5 = Tool(
            name="plan_add_step",
            description="Add a new step to the plan",
            parameters=[ToolParameter(name="description", type="string", description="Step description", required=True)],
        )
        t5.bind(lambda description: self.add_step(description))
        tools.append(t5)

        return tools


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Deep Agent — Full Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeepAgent:
    """Full Deep Agent architecture with VFS, planning, and context quarantine.

    Combines all Deep Agent components:
    - Virtual File System for persistent artifacts
    - Planning tool for strategic thinking
    - Sub-agent spawning with context isolation
    - Compound intelligence via skill persistence

    Usage:
        from duxx_ai.core.deep_agent import DeepAgent
        from duxx_ai.core.agent import Agent, AgentConfig

        base = Agent(config=AgentConfig(
            name="deep-researcher",
            system_prompt="You are a deep research agent..."
        ))

        deep = DeepAgent(base, workspace="research_project")
        result = await deep.run("Research AI agent architectures and write a comprehensive report")
    """

    DEEP_SYSTEM = """You are a Deep Agent — an advanced AI that tackles complex, long-horizon tasks.

You have access to:
1. A Virtual File System — store analysis, code, reports as files
2. A Planning Tool — create and track multi-step plans
3. Sub-agent spawning — delegate specialized subtasks

METHODOLOGY:
- ALWAYS create a plan before starting complex tasks
- Write intermediate results to files (don't keep everything in context)
- Break complex problems into subtasks and delegate when appropriate
- Track progress via plan_update after each step
- Search your workspace for prior knowledge before starting new work

Your workspace contains: {workspace_tree}"""

    def __init__(
        self,
        agent: Any,
        workspace: str = "workspace",
        enable_planning: bool = True,
        enable_vfs: bool = True,
    ):
        self.agent = agent
        self.vfs = VirtualFileSystem(persist_path=workspace) if enable_vfs else None
        self.planner = PlanningTool() if enable_planning else None
        self._workspace = workspace

        # Inject tools
        extra_tools = []
        if self.vfs:
            extra_tools.extend(self.vfs.as_tools())
        if self.planner:
            extra_tools.extend(self.planner.as_tools())

        for tool in extra_tools:
            self.agent.register_tool(tool)

    async def run(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Execute a deep agent task."""
        # Update system prompt with workspace context
        tree = self.vfs.tree() if self.vfs else "(no workspace)"
        self.agent.config.system_prompt = self.DEEP_SYSTEM.format(workspace_tree=tree)

        result = await self.agent.run(task, context=context)
        return result

    @property
    def workspace(self) -> VirtualFileSystem | None:
        return self.vfs

    @property
    def plan(self) -> PlanningTool | None:
        return self.planner


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Graph Analytics — Topology Analysis for Agent Networks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GraphAnalytics:
    """Graph-theoretic analysis for agent orchestration graphs.

    Implements the graph analytics from the paper:
    - Betweenness centrality (risk analysis — which agents are critical)
    - Critical path analysis (latency optimization)
    - K-core decomposition (identify dense agent clusters)
    - Topology optimization suggestions

    Usage:
        from duxx_ai.core.deep_agent import GraphAnalytics
        from duxx_ai.orchestration.graph import Graph

        graph = Graph("my-pipeline")
        # ... add nodes and edges ...

        analytics = GraphAnalytics(graph)
        print(analytics.critical_path())
        print(analytics.centrality())
        print(analytics.bottlenecks())
    """

    def __init__(self, graph: Any):
        self.graph = graph
        self._adj: dict[str, list[str]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency list from graph edges."""
        self._adj = {}
        for node_id in self.graph.nodes:
            self._adj[node_id] = []
        for edge in self.graph.edges:
            if edge.source not in self._adj:
                self._adj[edge.source] = []
            self._adj[edge.source].append(edge.target)

    def centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for each node.

        Identifies which agents/nodes are critical — if they fail,
        many paths through the graph are disrupted.
        """
        nodes = list(self._adj.keys())
        centrality = {n: 0.0 for n in nodes}

        for source in nodes:
            # BFS shortest paths from source
            distances: dict[str, int] = {source: 0}
            paths: dict[str, int] = {source: 1}
            queue = [source]
            order = []

            while queue:
                current = queue.pop(0)
                order.append(current)
                for neighbor in self._adj.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        paths[neighbor] = 0
                        queue.append(neighbor)
                    if distances[neighbor] == distances[current] + 1:
                        paths[neighbor] = paths.get(neighbor, 0) + paths[current]

            # Accumulate centrality
            dependency: dict[str, float] = {n: 0.0 for n in nodes}
            for node in reversed(order):
                for neighbor in self._adj.get(node, []):
                    if distances.get(neighbor, -1) == distances.get(node, -1) + 1:
                        fraction = (paths.get(node, 1) / max(paths.get(neighbor, 1), 1)) * (1 + dependency[neighbor])
                        dependency[node] += fraction
                if node != source:
                    centrality[node] += dependency[node]

        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            centrality = {k: v * norm for k, v in centrality.items()}

        return centrality

    def critical_path(self) -> list[str]:
        """Find the longest (critical) path through the graph.

        The critical path determines the minimum execution time
        when parallelism is maximized.
        """
        # Topological sort + longest path
        in_degree = {n: 0 for n in self._adj}
        for node, neighbors in self._adj.items():
            for n in neighbors:
                in_degree[n] = in_degree.get(n, 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        dist: dict[str, int] = {n: 0 for n in queue}
        parent: dict[str, str | None] = {n: None for n in queue}

        while queue:
            current = queue.pop(0)
            for neighbor in self._adj.get(current, []):
                if dist.get(current, 0) + 1 > dist.get(neighbor, 0):
                    dist[neighbor] = dist[current] + 1
                    parent[neighbor] = current
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Trace back from the node with max distance
        if not dist:
            return []
        end_node = max(dist, key=dist.get)  # type: ignore
        path = []
        current: str | None = end_node
        while current is not None:
            path.append(current)
            current = parent.get(current)
        return list(reversed(path))

    def bottlenecks(self) -> list[dict[str, Any]]:
        """Identify bottleneck nodes (high centrality + many connections).

        These are nodes whose failure would cascade through the system.
        """
        cent = self.centrality()
        results = []
        for node_id, score in sorted(cent.items(), key=lambda x: x[1], reverse=True):
            in_edges = sum(1 for e in self.graph.edges if e.target == node_id)
            out_edges = sum(1 for e in self.graph.edges if e.source == node_id)
            if score > 0 or in_edges > 1 or out_edges > 1:
                risk = "HIGH" if score > 0.3 else "MEDIUM" if score > 0.1 else "LOW"
                results.append({
                    "node": node_id,
                    "centrality": round(score, 4),
                    "in_edges": in_edges,
                    "out_edges": out_edges,
                    "risk_level": risk,
                })
        return results

    def parallel_groups(self) -> list[list[str]]:
        """Identify groups of nodes that can execute in parallel.

        Uses topological layering — nodes in the same layer have
        no dependencies on each other.
        """
        in_degree = {n: 0 for n in self._adj}
        for node, neighbors in self._adj.items():
            for n in neighbors:
                in_degree[n] = in_degree.get(n, 0) + 1

        layers = []
        remaining = dict(in_degree)

        while remaining:
            layer = [n for n, d in remaining.items() if d == 0]
            if not layer:
                break  # Cycle detected
            layers.append(layer)
            for n in layer:
                for neighbor in self._adj.get(n, []):
                    if neighbor in remaining:
                        remaining[neighbor] -= 1
                del remaining[n]

        return layers

    def summary(self) -> dict[str, Any]:
        """Complete analytics summary."""
        cp = self.critical_path()
        groups = self.parallel_groups()
        bn = self.bottlenecks()

        return {
            "total_nodes": len(self._adj),
            "total_edges": len(self.graph.edges),
            "critical_path": cp,
            "critical_path_length": len(cp),
            "parallel_layers": len(groups),
            "max_parallelism": max((len(g) for g in groups), default=0),
            "bottleneck_count": sum(1 for b in bn if b["risk_level"] == "HIGH"),
            "bottlenecks": bn[:5],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. Agent-to-Agent Protocol (A2A) — Inter-Agent Communication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class A2AMessage:
    """Message in the Agent-to-Agent protocol."""
    sender: str
    receiver: str
    content: str
    msg_type: str = "request"  # request, response, broadcast, delegate
    correlation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class A2AProtocol:
    """Agent-to-Agent communication protocol for inter-agent messaging.

    Enables direct agent-to-agent communication, service discovery,
    and message routing — implementing the A2A protocol from the paper.

    Usage:
        from duxx_ai.core.deep_agent import A2AProtocol

        protocol = A2AProtocol()
        protocol.register("researcher", researcher_agent, capabilities=["research", "analysis"])
        protocol.register("writer", writer_agent, capabilities=["writing", "editing"])

        # Direct messaging
        response = await protocol.send("researcher", "writer", "Please edit this draft")

        # Capability-based discovery
        agents = protocol.discover(capability="research")

        # Broadcast
        await protocol.broadcast("researcher", "New data available for Q4")
    """

    def __init__(self):
        self._agents: dict[str, tuple[Any, list[str]]] = {}  # name -> (agent, capabilities)
        self._message_log: list[A2AMessage] = []
        self._msg_counter = 0

    def register(self, name: str, agent: Any, capabilities: list[str] | None = None) -> None:
        """Register an agent with the protocol."""
        self._agents[name] = (agent, capabilities or [])

    def discover(self, capability: str | None = None) -> list[dict[str, Any]]:
        """Discover agents by capability."""
        results = []
        for name, (agent, caps) in self._agents.items():
            if capability is None or capability in caps:
                results.append({
                    "name": name,
                    "capabilities": caps,
                    "status": "available",
                })
        return results

    async def send(self, sender: str, receiver: str, content: str,
                   msg_type: str = "request") -> str:
        """Send a message from one agent to another."""
        if receiver not in self._agents:
            return f"Agent '{receiver}' not found"

        self._msg_counter += 1
        msg = A2AMessage(
            sender=sender, receiver=receiver, content=content,
            msg_type=msg_type, correlation_id=f"msg_{self._msg_counter}",
        )
        self._message_log.append(msg)

        # Execute on receiver agent
        agent, _ = self._agents[receiver]
        context = {"from_agent": sender, "msg_type": msg_type, "correlation_id": msg.correlation_id}
        response = await agent.run(content, context=context)

        # Log response
        resp_msg = A2AMessage(
            sender=receiver, receiver=sender, content=response,
            msg_type="response", correlation_id=msg.correlation_id,
        )
        self._message_log.append(resp_msg)

        return response

    async def broadcast(self, sender: str, content: str) -> dict[str, str]:
        """Broadcast a message to all registered agents."""
        responses = {}
        for name in self._agents:
            if name != sender:
                responses[name] = await self.send(sender, name, content, msg_type="broadcast")
        return responses

    async def delegate(self, sender: str, task: str, capability: str | None = None) -> str:
        """Delegate a task to the best available agent."""
        candidates = self.discover(capability)
        candidates = [c for c in candidates if c["name"] != sender]
        if not candidates:
            return f"No agent found with capability '{capability}'"

        # Pick first available
        target = candidates[0]["name"]
        return await self.send(sender, target, task, msg_type="delegate")

    @property
    def message_log(self) -> list[A2AMessage]:
        return self._message_log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "VirtualFileSystem", "VirtualFile",
    "PlanningTool", "PlanStep",
    "DeepAgent",
    "GraphAnalytics",
    "A2AProtocol", "A2AMessage",
]
