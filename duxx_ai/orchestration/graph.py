"""Graph-based orchestration engine for complex agent workflows.

Features:
- DAG execution with conditional routing and parallel branches
- State reducers for intelligent parallel merge (append, sum, merge_dict)
- Human-in-the-loop nodes with interrupt/resume
- Map-reduce fan-out/fan-in pattern
- Checkpointing for recovery and debugging
- Cycle detection with optional controlled loops
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ── State Reducers ──

def replace_reducer(current: Any, incoming: Any) -> Any:
    """Default: last-write-wins."""
    return incoming


def append_reducer(current: Any, incoming: Any) -> Any:
    """Append incoming to a list."""
    if not isinstance(current, list):
        current = [current] if current is not None else []
    if isinstance(incoming, list):
        return current + incoming
    return current + [incoming]


def sum_reducer(current: Any, incoming: Any) -> Any:
    """Sum numeric values."""
    return (current or 0) + (incoming or 0)


def merge_dict_reducer(current: Any, incoming: Any) -> Any:
    """Deep merge dictionaries."""
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = dict(current)
        merged.update(incoming)
        return merged
    return incoming


# ── Graph Interrupt ──

class GraphInterrupt(Exception):
    """Raised when a HUMAN node needs external input to continue."""

    def __init__(self, node_id: str, prompt: str = "", metadata: dict[str, Any] | None = None):
        self.node_id = node_id
        self.prompt = prompt
        self.metadata = metadata or {}
        super().__init__(f"Graph interrupted at node '{node_id}': {prompt}")


# ── Core Types ──

class NodeType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    HUMAN = "human"
    MAP_REDUCE = "map_reduce"
    START = "start"
    END = "end"


class Node(BaseModel):
    id: str
    type: NodeType = NodeType.AGENT
    config: dict[str, Any] = Field(default_factory=dict)
    _handler: Callable[..., Awaitable[Any]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_handler(self, fn: Callable[..., Awaitable[Any]]) -> Node:
        object.__setattr__(self, "_handler", fn)
        return self

    async def execute(self, state: GraphState) -> GraphState:
        handler = object.__getattribute__(self, "_handler")
        if handler:
            result = await handler(state)
            if isinstance(result, GraphState):
                return result
            state.data[self.id] = result
        return state


class EdgeCondition(BaseModel):
    """Condition for conditional routing."""
    key: str = ""
    value: Any = None
    operator: str = "eq"  # eq, neq, gt, lt, contains, exists


class Edge(BaseModel):
    source: str
    target: str
    condition: EdgeCondition | None = None
    priority: int = 0

    def evaluate(self, state: GraphState) -> bool:
        if self.condition is None:
            return True
        val = state.data.get(self.condition.key)
        op = self.condition.operator
        expected = self.condition.value
        if op == "eq":
            return val == expected
        elif op == "neq":
            return val != expected
        elif op == "gt":
            return val is not None and val > expected
        elif op == "lt":
            return val is not None and val < expected
        elif op == "contains":
            return expected in val if val else False
        elif op == "exists":
            return val is not None
        return False


class GraphState(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    current_node: str = "__start__"
    history: list[str] = Field(default_factory=list)
    iteration: int = 0
    status: str = "running"

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class Graph:
    """Directed graph orchestration engine with conditional routing, parallel execution,
    state reducers, human-in-the-loop, map-reduce, and checkpointing."""

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {
            "__start__": Node(id="__start__", type=NodeType.START),
            "__end__": Node(id="__end__", type=NodeType.END),
        }
        self.edges: list[Edge] = []
        self.max_iterations: int = 50
        self.checkpoints: list[GraphState] = []
        self._reducers: dict[str, Callable[[Any, Any], Any]] = {}
        self._cycle_allowed_edges: set[tuple[str, str]] = set()

    # ── Graph Construction ──

    def add_node(
        self,
        node_id: str,
        handler: Callable[..., Awaitable[Any]] | None = None,
        node_type: NodeType = NodeType.AGENT,
        **config: Any,
    ) -> Graph:
        node = Node(id=node_id, type=node_type, config=config)
        if handler:
            node.set_handler(handler)
        self.nodes[node_id] = node
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: EdgeCondition | None = None,
        priority: int = 0,
        allow_cycle: bool = False,
    ) -> Graph:
        self.edges.append(Edge(source=source, target=target, condition=condition, priority=priority))
        if allow_cycle:
            self._cycle_allowed_edges.add((source, target))
        elif self._detect_cycle():
            self.edges.pop()
            raise ValueError(
                f"Adding edge '{source}' -> '{target}' would create a cycle in the graph. "
                f"Use allow_cycle=True for intentional loops."
            )
        return self

    def set_entry_point(self, node_id: str) -> Graph:
        return self.add_edge("__start__", node_id)

    def set_exit_point(self, node_id: str) -> Graph:
        return self.add_edge(node_id, "__end__")

    def set_state_reducer(self, key: str, reducer: Callable[[Any, Any], Any]) -> Graph:
        """Register a reducer function for a state key used during parallel merges."""
        self._reducers[key] = reducer
        return self

    def add_map_reduce_node(
        self,
        node_id: str,
        map_handler: Callable[[Any, int], Awaitable[Any]],
        reduce_handler: Callable[[list[Any]], Awaitable[Any]] | None = None,
        items_key: str = "items",
        result_key: str | None = None,
        max_concurrency: int = 10,
    ) -> Graph:
        """Add a map-reduce node that fans out over items and collects results."""

        async def _map_reduce_handler(state: GraphState) -> GraphState:
            items = state.get(items_key, [])
            if not isinstance(items, list):
                items = [items]

            semaphore = asyncio.Semaphore(max_concurrency)

            async def _run_one(item: Any, idx: int) -> Any:
                async with semaphore:
                    return await map_handler(item, idx)

            results = await asyncio.gather(
                *[_run_one(item, i) for i, item in enumerate(items)]
            )

            if reduce_handler:
                final = await reduce_handler(list(results))
            else:
                final = list(results)

            state.set(result_key or f"{node_id}_results", final)
            return state

        node = Node(id=node_id, type=NodeType.MAP_REDUCE, config={
            "items_key": items_key,
            "max_concurrency": max_concurrency,
        })
        node.set_handler(_map_reduce_handler)
        self.nodes[node_id] = node
        return self

    # ── Execution ──

    async def run(self, initial_state: dict[str, Any] | None = None) -> GraphState:
        state = GraphState(data=initial_state or {})
        state.current_node = "__start__"
        return await self._execute_loop(state)

    async def resume(
        self,
        checkpoint_index: int = -1,
        human_input: dict[str, Any] | None = None,
    ) -> GraphState:
        """Resume graph execution from a checkpoint after human input."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available to resume from")

        state = GraphState(**self.checkpoints[checkpoint_index].model_dump())

        if human_input:
            state.data.update(human_input)

        # Clean up interrupt markers
        interrupt_node = state.data.pop("__interrupt_node__", None)
        state.data.pop("__interrupt_prompt__", None)

        # Execute the interrupted node's handler with updated state
        if interrupt_node and interrupt_node in self.nodes:
            node = self.nodes[interrupt_node]
            handler = object.__getattribute__(node, "_handler")
            if handler:
                state = await handler(state)
            state.history.append(interrupt_node)

        state.status = "running"
        return await self._execute_loop(state)

    async def _execute_loop(self, state: GraphState) -> GraphState:
        """Core execution loop shared by run() and resume()."""
        while state.status == "running" and state.iteration < self.max_iterations:
            state.iteration += 1
            next_nodes = self._get_next_nodes(state.current_node, state)

            if not next_nodes:
                state.status = "completed"
                break

            # Check for parallel execution
            if len(next_nodes) > 1 and all(
                self.nodes.get(n, Node(id=n)).type != NodeType.CONDITIONAL for n in next_nodes
            ):
                results = await asyncio.gather(
                    *[self.nodes[n].execute(GraphState(**state.model_dump())) for n in next_nodes]
                )
                # Merge with reducers
                for r in results:
                    for key, value in r.data.items():
                        if key in self._reducers:
                            state.data[key] = self._reducers[key](state.data.get(key), value)
                        else:
                            state.data[key] = value

                state.history.extend(next_nodes)
                all_next: set[str] = set()
                for n in next_nodes:
                    all_next.update(self._get_next_nodes(n, state))
                if all_next:
                    state.current_node = sorted(all_next)[0]
                else:
                    state.status = "completed"
            else:
                next_node_id = next_nodes[0]
                if next_node_id == "__end__":
                    state.status = "completed"
                    break

                node = self.nodes.get(next_node_id)
                if node is None:
                    state.status = "error"
                    state.data["error"] = f"Node '{next_node_id}' not found"
                    break

                # Human-in-the-loop interrupt
                if node.type == NodeType.HUMAN:
                    state.status = "interrupted"
                    state.current_node = next_node_id
                    state.data["__interrupt_node__"] = next_node_id
                    state.data["__interrupt_prompt__"] = node.config.get("prompt", "Human input required")
                    self.checkpoints.append(GraphState(**state.model_dump()))
                    raise GraphInterrupt(
                        node_id=next_node_id,
                        prompt=node.config.get("prompt", "Human input required"),
                        metadata=node.config,
                    )

                state.current_node = next_node_id
                state.history.append(next_node_id)
                state = await node.execute(state)

            # Checkpoint
            self.checkpoints.append(GraphState(**state.model_dump()))

        return state

    # ── Internal Helpers ──

    def _get_next_nodes(self, current: str, state: GraphState) -> list[str]:
        candidates = [e for e in self.edges if e.source == current]
        candidates.sort(key=lambda e: e.priority, reverse=True)
        return [edge.target for edge in candidates if edge.evaluate(state)]

    def _detect_cycle(self) -> bool:
        """Detect cycles, excluding edges marked with allow_cycle."""
        adjacency: dict[str, list[str]] = {}
        for edge in self.edges:
            if (edge.source, edge.target) in self._cycle_allowed_edges:
                continue
            adjacency.setdefault(edge.source, []).append(edge.target)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {}
        for node_id in self.nodes:
            color[node_id] = WHITE
        for edge in self.edges:
            color.setdefault(edge.source, WHITE)
            color.setdefault(edge.target, WHITE)

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adjacency.get(node, []):
                if color.get(neighbor, WHITE) == GRAY:
                    return True
                if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node_id in list(color):
            if color[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False

    # ── Subgraph Composition ──

    def add_subgraph(
        self,
        node_id: str,
        subgraph: Graph,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        isolated: bool = True,
    ) -> Graph:
        """Add a nested subgraph as a node.

        Args:
            node_id: ID for this subgraph node
            subgraph: The nested Graph to execute
            input_mapping: Parent state key → subgraph state key mapping
            output_mapping: Subgraph state key → parent state key mapping
            isolated: If True, subgraph gets a fresh state (only mapped inputs)
        """
        async def _subgraph_handler(state: GraphState) -> GraphState:
            # Prepare subgraph initial state
            if isolated:
                sub_state = {}
                if input_mapping:
                    for parent_key, sub_key in input_mapping.items():
                        if parent_key in state.data:
                            sub_state[sub_key] = state.data[parent_key]
            else:
                sub_state = dict(state.data)
                if input_mapping:
                    for parent_key, sub_key in input_mapping.items():
                        if parent_key in state.data:
                            sub_state[sub_key] = state.data[parent_key]

            # Execute subgraph
            result = await subgraph.run(sub_state)

            # Map outputs back to parent state
            if output_mapping:
                for sub_key, parent_key in output_mapping.items():
                    if sub_key in result.data:
                        state.data[parent_key] = result.data[sub_key]
            elif not isolated:
                state.data.update(result.data)

            state.data[f"{node_id}_status"] = result.status
            return state

        node = Node(id=node_id, type=NodeType.AGENT, config={
            "subgraph": subgraph.name, "isolated": isolated,
        })
        node.set_handler(_subgraph_handler)
        self.nodes[node_id] = node
        return self

    # ── Streaming Modes ──

    async def stream(
        self,
        initial_state: dict[str, Any] | None = None,
        mode: str = "updates",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream graph execution events.

        Modes:
            "values" — yield full state after each node
            "updates" — yield only changed state keys per node
            "debug" — yield detailed execution info (node, duration, state diff)
        """
        import time as _time

        state = GraphState(data=initial_state or {})
        state.current_node = "__start__"
        prev_data = dict(state.data)

        while state.status == "running" and state.iteration < self.max_iterations:
            state.iteration += 1
            next_nodes = self._get_next_nodes(state.current_node, state)

            if not next_nodes:
                state.status = "completed"
                break

            next_node_id = next_nodes[0]
            if next_node_id == "__end__":
                state.status = "completed"
                break

            node = self.nodes.get(next_node_id)
            if node is None:
                state.status = "error"
                break

            if node.type == NodeType.HUMAN:
                state.status = "interrupted"
                yield {"type": "interrupt", "node": next_node_id, "state": dict(state.data)}
                break

            start = _time.time()
            state.current_node = next_node_id
            state.history.append(next_node_id)

            # Check node cache
            cache_key = f"{next_node_id}:{hash(str(sorted(state.data.items())))}"
            if next_node_id in self._node_cache and cache_key in self._node_cache:
                state = self._node_cache[cache_key]
            else:
                state = await node.execute(state)
                if self._caching_enabled:
                    self._node_cache[cache_key] = GraphState(**state.model_dump())

            elapsed = _time.time() - start

            # Compute diff
            changed = {k: v for k, v in state.data.items() if prev_data.get(k) != v}

            if mode == "values":
                yield {"type": "values", "node": next_node_id, "state": dict(state.data)}
            elif mode == "updates":
                yield {"type": "updates", "node": next_node_id, "changes": changed}
            elif mode == "debug":
                yield {
                    "type": "debug", "node": next_node_id, "node_type": node.type.value,
                    "duration_ms": round(elapsed * 1000, 1),
                    "changes": changed, "full_state": dict(state.data),
                    "iteration": state.iteration,
                }

            prev_data = dict(state.data)
            self.checkpoints.append(GraphState(**state.model_dump()))

        yield {"type": "complete", "status": state.status, "final_state": dict(state.data)}

    # ── Node Caching ──

    _node_cache: dict[str, GraphState] = {}
    _caching_enabled: bool = False

    def enable_node_caching(self) -> Graph:
        """Enable caching of node results to skip redundant execution."""
        self._caching_enabled = True
        self._node_cache = {}
        return self

    def clear_cache(self) -> None:
        self._node_cache.clear()

    # ── Checkpoint Persistence ──

    def save_checkpoints(self, path: str) -> int:
        """Persist all checkpoints to a JSON file."""
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [cp.model_dump() for cp in self.checkpoints]
        p.write_text(json.dumps(data, indent=2, default=str))
        return len(data)

    def load_checkpoints(self, path: str) -> int:
        """Load checkpoints from a JSON file."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return 0
        data = json.loads(p.read_text())
        self.checkpoints = [GraphState(**cp) for cp in data]
        return len(self.checkpoints)

    # ── Time-Travel Debugging ──

    def get_checkpoint(self, index: int = -1) -> GraphState | None:
        """Get a specific checkpoint by index."""
        if not self.checkpoints:
            return None
        return GraphState(**self.checkpoints[index].model_dump())

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with metadata."""
        return [
            {
                "index": i,
                "node": cp.current_node,
                "iteration": cp.iteration,
                "status": cp.status,
                "keys": list(cp.data.keys()),
            }
            for i, cp in enumerate(self.checkpoints)
        ]

    async def replay(self, from_checkpoint: int = 0) -> list[GraphState]:
        """Replay graph execution from a checkpoint, returning all states."""
        if not self.checkpoints:
            raise ValueError("No checkpoints to replay from")

        state = GraphState(**self.checkpoints[from_checkpoint].model_dump())
        state.status = "running"

        replay_states = [GraphState(**state.model_dump())]
        original_checkpoints = list(self.checkpoints)
        self.checkpoints = []

        try:
            result = await self._execute_loop(state)
            replay_states.extend(self.checkpoints)
            replay_states.append(result)
        except GraphInterrupt:
            replay_states.extend(self.checkpoints)
        finally:
            self.checkpoints = original_checkpoints

        return replay_states

    async def fork(
        self,
        checkpoint_index: int,
        state_overrides: dict[str, Any],
    ) -> GraphState:
        """Fork execution from a checkpoint with modified state.

        Creates a new branch of execution from the given checkpoint
        with overridden state values, without affecting the original checkpoints.
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints to fork from")

        state = GraphState(**self.checkpoints[checkpoint_index].model_dump())
        state.data.update(state_overrides)
        state.status = "running"

        # Run fork without modifying original checkpoints
        original = list(self.checkpoints)
        self.checkpoints = []
        try:
            result = await self._execute_loop(state)
        except GraphInterrupt:
            result = self.checkpoints[-1] if self.checkpoints else state
        finally:
            fork_checkpoints = list(self.checkpoints)
            self.checkpoints = original
            # Append fork checkpoints with metadata
            for cp in fork_checkpoints:
                cp.data["__forked_from__"] = checkpoint_index
                self.checkpoints.append(cp)

        return result

    # ── Deferred Nodes ──

    def add_deferred_node(
        self,
        node_id: str,
        handler: Callable[..., Awaitable[Any]],
        wait_for: list[str] | None = None,
    ) -> Graph:
        """Add a node that waits until all specified upstream nodes complete.

        The node's handler only executes once all `wait_for` nodes
        have appeared in the execution history.
        """
        async def _deferred_handler(state: GraphState) -> GraphState:
            if wait_for:
                missing = [n for n in wait_for if n not in state.history]
                if missing:
                    state.data[f"_deferred_{node_id}_waiting"] = missing
                    return state
            return await handler(state)

        node = Node(id=node_id, type=NodeType.AGENT, config={"deferred": True, "wait_for": wait_for or []})
        node.set_handler(_deferred_handler)
        self.nodes[node_id] = node
        return self

    def visualize(self) -> str:
        lines = [f"Graph: {self.name}", "=" * 40]
        for edge in self.edges:
            cond = f" [if {edge.condition.key} {edge.condition.operator} {edge.condition.value}]" if edge.condition else ""
            lines.append(f"  {edge.source} -> {edge.target}{cond}")
        return "\n".join(lines)

    # ── Advanced Graph Features ──

    def add_conditional_edge(
        self, source: str, condition_fn: Callable[[GraphState], str],
        path_map: dict[str, str] | None = None,
    ) -> Graph:
        """Add a conditional edge with a routing function ().

        The condition_fn receives the state and returns a string key.
        The path_map maps keys to target node IDs.

        Usage:
            def route(state):
                if state.get("needs_review"):
                    return "review"
                return "publish"

            graph.add_conditional_edge("analyze", route, {
                "review": "human_review",
                "publish": "publish_node",
            })
        """
        if path_map is None:
            path_map = {}

        async def _conditional_handler(state: GraphState) -> GraphState:
            key = condition_fn(state) if not asyncio.iscoroutinefunction(condition_fn) else await condition_fn(state)
            target = path_map.get(key, key)  # fallback to key as node_id
            state.data["_next_node"] = target
            return state

        cond_node_id = f"_cond_{source}_{id(condition_fn)}"
        self.add_node(cond_node_id, _conditional_handler, NodeType.CONDITIONAL)
        self.add_edge(source, cond_node_id)

        for _, target in path_map.items():
            self.add_edge(cond_node_id, target, EdgeCondition(
                key="_next_node", operator="eq", value=target,
            ))

        return self

    def compile(self, checkpointer: Any = None, store: Any = None) -> Graph:
        """Compile the graph for execution ().

        Validates the graph, sets up checkpointing backend, and optimizes.

        Args:
            checkpointer: Checkpoint backend (None=in-memory, or CheckpointBackend)
            store: Cross-thread store for persistent memory

        Usage:
            graph = Graph("my-graph")
            graph.add_node("a", handler_a)
            graph.add_node("b", handler_b)
            graph.add_edge("a", "b")
            compiled = graph.compile(checkpointer=SqliteCheckpointer("state.db"))
        """
        # Validate graph
        if not self.entry_point:
            # Auto-detect entry point
            targets = {e.target for e in self.edges}
            sources = {e.source for e in self.edges}
            roots = sources - targets
            if roots:
                self.entry_point = next(iter(roots))

        # Set checkpointer
        if checkpointer is not None:
            self._checkpointer = checkpointer

        # Set store
        if store is not None:
            self._store = store

        # Validate edges reference existing nodes
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in self.nodes:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")

        self._compiled = True
        return self

    def update_state(self, config: dict[str, Any], values: dict[str, Any], as_node: str | None = None) -> GraphState:
        """Update graph state externally ().

        Modifies the current state and creates a new checkpoint.
        Used for human-in-the-loop state modifications.

        Args:
            config: {"configurable": {"thread_id": "...", "checkpoint_id": "..."}}
            values: State values to update
            as_node: Pretend this update came from this node

        Usage:
            graph.update_state(
                config={"configurable": {"thread_id": "t1"}},
                values={"approved": True, "reviewer_notes": "Looks good"},
                as_node="human_review",
            )
        """
        # Find the relevant checkpoint
        cp_id = config.get("configurable", {}).get("checkpoint_id")
        if cp_id is not None and isinstance(cp_id, int) and cp_id < len(self.checkpoints):
            state = self.checkpoints[cp_id].model_copy(deep=True)
        elif self.checkpoints:
            state = self.checkpoints[-1].model_copy(deep=True)
        else:
            state = GraphState()

        # Apply updates with reducers
        for key, value in values.items():
            if key in self._state_reducers:
                state.data[key] = self._state_reducers[key](state.data.get(key), value)
            else:
                state.data[key] = value

        # Record the update source
        if as_node:
            state.current_node = as_node
            state.history.append(as_node)

        # Save as new checkpoint
        self.checkpoints.append(state.model_copy(deep=True))

        return state

    def get_state(self, config: dict[str, Any] | None = None) -> GraphState | None:
        """Get current graph state ().

        Args:
            config: Optional config with thread_id/checkpoint_id

        Returns:
            Current GraphState or None if no checkpoints
        """
        if not self.checkpoints:
            return None

        if config:
            cp_id = config.get("configurable", {}).get("checkpoint_id")
            if cp_id is not None and isinstance(cp_id, int) and cp_id < len(self.checkpoints):
                return self.checkpoints[cp_id]

        return self.checkpoints[-1]

    def get_state_history(self, config: dict[str, Any] | None = None) -> list[GraphState]:
        """Get checkpoint history in reverse chronological order ().

        Returns:
            List of GraphState checkpoints, most recent first
        """
        return list(reversed(self.checkpoints))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Command — Resume with goto/update ()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Command:
    """Control flow command for resuming interrupted graphs.

    Usage:
        # Resume with a value
        Command(resume="approved")

        # Resume and jump to a specific node
        Command(resume="approved", goto="publish")

        # Resume with state updates
        Command(resume=True, update={"approved": True})
    """

    def __init__(
        self,
        resume: Any = None,
        goto: str | None = None,
        update: dict[str, Any] | None = None,
    ):
        self.resume = resume
        self.goto = goto
        self.update = update or {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Checkpoint Backends ()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseCheckpointSaver:
    """Base class for checkpoint persistence backends."""

    async def put(self, config: dict[str, Any], state: GraphState) -> None:
        raise NotImplementedError

    async def get(self, config: dict[str, Any]) -> GraphState | None:
        raise NotImplementedError

    async def list(self, config: dict[str, Any]) -> list[GraphState]:
        raise NotImplementedError

    async def delete(self, config: dict[str, Any]) -> None:
        raise NotImplementedError


class InMemorySaver(BaseCheckpointSaver):
    """In-memory checkpoint saver ."""

    def __init__(self):
        self._storage: dict[str, list[GraphState]] = {}

    async def put(self, config: dict[str, Any], state: GraphState) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if thread_id not in self._storage:
            self._storage[thread_id] = []
        self._storage[thread_id].append(state.model_copy(deep=True))

    async def get(self, config: dict[str, Any]) -> GraphState | None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        states = self._storage.get(thread_id, [])
        cp_id = config.get("configurable", {}).get("checkpoint_id")
        if cp_id is not None and isinstance(cp_id, int) and cp_id < len(states):
            return states[cp_id]
        return states[-1] if states else None

    async def list(self, config: dict[str, Any]) -> list[GraphState]:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return list(reversed(self._storage.get(thread_id, [])))

    async def delete(self, config: dict[str, Any]) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        self._storage.pop(thread_id, None)


class SqliteCheckpointer(BaseCheckpointSaver):
    """SQLite-based checkpoint persistence.

    Usage:
        checkpointer = SqliteCheckpointer("checkpoints.db")
        graph.compile(checkpointer=checkpointer)
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
        """)
        conn.commit()
        conn.close()

    async def put(self, config: dict[str, Any], state: GraphState) -> None:
        import sqlite3
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        conn = sqlite3.connect(self.db_path)
        # Get next checkpoint_id
        cursor = conn.execute(
            "SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        next_id = (row[0] or -1) + 1
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, state_json) VALUES (?, ?, ?)",
            (thread_id, next_id, state.model_dump_json()),
        )
        conn.commit()
        conn.close()

    async def get(self, config: dict[str, Any]) -> GraphState | None:
        import sqlite3
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        cp_id = config.get("configurable", {}).get("checkpoint_id")
        conn = sqlite3.connect(self.db_path)
        if cp_id is not None:
            cursor = conn.execute(
                "SELECT state_json FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, cp_id),
            )
        else:
            cursor = conn.execute(
                "SELECT state_json FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            )
        row = cursor.fetchone()
        conn.close()
        if row:
            return GraphState.model_validate_json(row[0])
        return None

    async def list(self, config: dict[str, Any]) -> list[GraphState]:
        import sqlite3
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT state_json FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC",
            (thread_id,),
        )
        states = [GraphState.model_validate_json(row[0]) for row in cursor.fetchall()]
        conn.close()
        return states

    async def delete(self, config: dict[str, Any]) -> None:
        import sqlite3
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @task Decorator — Durable Execution ()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task(fn: Callable | None = None, *, retries: int = 0) -> Callable:
    """Decorator for durable task execution within graph nodes.

    Wraps non-deterministic or side-effect operations to ensure
    they are properly checkpointed and can be retried.

    Usage:
        @task
        async def call_api(url: str) -> dict:
            response = await httpx.get(url)
            return response.json()

        @task(retries=3)
        async def send_email(to: str, body: str) -> bool:
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            for attempt in range(max(1, retries + 1)):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
            raise last_error  # type: ignore
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._is_task = True  # type: ignore
        wrapper._retries = retries  # type: ignore
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  interrupt() function —
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def interrupt(value: Any = None) -> None:
    """Pause graph execution for human input ().

    Call this inside a node handler to pause execution and wait for
    human input. The graph state is checkpointed automatically.

    Args:
        value: JSON-serializable payload to send to the human
               (e.g., a question, approval request, or data to review)

    Usage:
        async def review_node(state: GraphState) -> GraphState:
            answer = interrupt("Do you approve this action?")
            state.set("approved", answer)
            return state
    """
    raise GraphInterrupt(
        node_id="__interrupt__",
        prompt=str(value) if value is not None else "Interrupt requested",
        metadata={"value": value},
    )
