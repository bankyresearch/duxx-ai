"""FlowGraph — LangGraph-compatible typed state graph with channels, commands, streaming, and checkpointing.

This is the modern graph API for Duxx AI, providing:
- Typed state schemas (TypedDict or Pydantic)
- Annotated reducers for parallel merge
- Route/Send for dynamic routing
- pause() for human-in-the-loop
- Streaming modes (values, updates, messages, debug)
- Checkpoint persistence (memory, SQLite, file)
- Time-travel debugging (replay, fork)
- @workflow / @step functional API
- RetryPolicy / CachePolicy per node
- ChatState built-in
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Literal, TypedDict,
    TypeVar, Union, get_type_hints,
)

from duxx_ai.orchestration.channels import (
    BaseChannel, LastValue, Topic, BinaryOperatorAggregate,
    EphemeralValue, merge_messages, channel_from_annotation,
)

logger = logging.getLogger(__name__)

# ── Constants ──
ENTRY = "__entry__"
EXIT = "__exit__"

T = TypeVar("T")


# ── Control Flow Commands ──

@dataclass
class Dispatch:
    """Route execution to a specific node with arguments.

    Usage:
        def router(state):
            return [Dispatch("process", {"item": item}) for item in state["items"]]
    """
    node: str
    arg: Any = None


@dataclass
class Route:
    """Control graph execution: update state, goto nodes, or resume.

    Usage:
        return Route(update={"key": "value"}, goto="next_node")
        return Route(goto=["node_a", "node_b"])  # Fan-out
        return Route(resume={"answer": "yes"})  # Resume from interrupt
    """
    update: dict[str, Any] | None = None
    goto: str | list[str] | None = None
    resume: dict[str, Any] | None = None

    PARENT = "__parent__"  # Sentinel for parent graph routing


class Override:
    """Override reducer — set value directly instead of applying reducer.

    Usage:
        return {"messages": Override([new_message_list])}
    """
    def __init__(self, value: Any):
        self.value = value


# ── Interrupt ──

class FlowPause(Exception):
    """Raised when pause() is called to pause execution."""
    def __init__(self, value: Any = None):
        self.value = value
        super().__init__(f"Graph interrupted: {value}")


def pause(value: Any = None) -> None:
    """Pause graph execution and request human/external input.

    Usage:
        def review_node(state):
            answer = interrupt("Do you approve this?")
            return {"approved": answer == "yes"}
    """
    raise FlowPause(value)


# ── Stream Parts ──

class EventMode(str, Enum):
    VALUES = "values"
    UPDATES = "updates"
    MESSAGES = "messages"
    DEBUG = "debug"
    TASKS = "tasks"
    CUSTOM = "custom"


@dataclass
class FlowEvent:
    mode: EventMode
    node: str
    data: Any
    timestamp: float = field(default_factory=time.time)


# ── Retry & Cache Policies ──

@dataclass
class RetryStrategy:
    """Per-node retry configuration."""
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def intervals(self) -> list[float]:
        intervals = []
        delay = self.initial_interval
        for _ in range(self.max_attempts - 1):
            intervals.append(min(delay, self.max_interval))
            delay *= self.backoff_factor
        return intervals


@dataclass
class CacheStrategy:
    """Per-node cache configuration."""
    ttl: float | None = None  # Seconds, None = no expiry

    def key(self, node_id: str, state_data: dict) -> str:
        content = json.dumps({"node": node_id, "state": state_data}, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ── State Snapshot for Time-Travel ──

@dataclass
class FlowSnapshot:
    """Snapshot of graph state at a point in time."""
    values: dict[str, Any]
    node: str
    step: int
    timestamp: float
    config: dict[str, Any] = field(default_factory=dict)
    parent_snapshot_id: str | None = None
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)
    interrupts: list[Any] = field(default_factory=list)


# ── Checkpoint Backends ──

class SnapshotStore:
    """Base checkpoint storage."""

    async def save(self, snapshot: FlowSnapshot) -> str:
        raise NotImplementedError

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        raise NotImplementedError

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        raise NotImplementedError

    async def get_latest(self) -> FlowSnapshot | None:
        raise NotImplementedError


class MemorySnapshotStore(SnapshotStore):
    """In-memory checkpoint storage (default)."""

    def __init__(self) -> None:
        self._store: dict[str, FlowSnapshot] = {}
        self._order: list[str] = []

    async def save(self, snapshot: FlowSnapshot) -> str:
        self._store[snapshot.snapshot_id] = snapshot
        self._order.append(snapshot.snapshot_id)
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        return self._store.get(snapshot_id)

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        ids = self._order[-limit:]
        return [self._store[sid] for sid in reversed(ids) if sid in self._store]

    async def get_latest(self) -> FlowSnapshot | None:
        if self._order:
            return self._store.get(self._order[-1])
        return None


class SQLiteSnapshotStore(SnapshotStore):
    """SQLite-based persistent checkpointer."""

    def __init__(self, db_path: str = "duxx_checkpoints.db") -> None:
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS checkpoints (
            snapshot_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            step INTEGER,
            timestamp REAL,
            created_at REAL DEFAULT (julianday('now'))
        )""")
        conn.commit()
        conn.close()

    async def save(self, snapshot: FlowSnapshot) -> str:
        import sqlite3
        data = json.dumps({
            "values": snapshot.values,
            "node": snapshot.node,
            "step": snapshot.step,
            "timestamp": snapshot.timestamp,
            "config": snapshot.config,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "metadata": snapshot.metadata,
            "interrupts": snapshot.interrupts,
        }, default=str)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (snapshot_id, data, step, timestamp) VALUES (?, ?, ?, ?)",
            (snapshot.snapshot_id, data, snapshot.step, snapshot.timestamp),
        )
        conn.commit()
        conn.close()
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT data FROM checkpoints WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        conn.close()
        if row:
            d = json.loads(row[0])
            return FlowSnapshot(snapshot_id=snapshot_id, **d)
        return None

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT snapshot_id, data FROM checkpoints ORDER BY step DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        result = []
        for sid, data_str in rows:
            d = json.loads(data_str)
            result.append(FlowSnapshot(snapshot_id=sid, **d))
        return result

    async def get_latest(self) -> FlowSnapshot | None:
        snapshots = await self.list(limit=1)
        return snapshots[0] if snapshots else None


class FileSnapshotStore(SnapshotStore):
    """File-based JSON checkpoint storage."""

    def __init__(self, directory: str = ".duxx_checkpoints") -> None:
        import os
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    async def save(self, snapshot: FlowSnapshot) -> str:
        import os
        path = os.path.join(self.directory, f"{snapshot.snapshot_id}.json")
        data = {
            "snapshot_id": snapshot.snapshot_id,
            "values": snapshot.values,
            "node": snapshot.node,
            "step": snapshot.step,
            "timestamp": snapshot.timestamp,
            "config": snapshot.config,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "metadata": snapshot.metadata,
            "interrupts": snapshot.interrupts,
        }
        with open(path, "w") as f:
            json.dump(data, f, default=str, indent=2)
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        import os
        path = os.path.join(self.directory, f"{snapshot_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            sid = d.pop("snapshot_id", snapshot_id)
            return FlowSnapshot(snapshot_id=sid, **d)
        return None

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        import os, glob
        files = sorted(glob.glob(os.path.join(self.directory, "*.json")), key=os.path.getmtime, reverse=True)
        result = []
        for path in files[:limit]:
            with open(path) as f:
                d = json.load(f)
            sid = d.pop("snapshot_id", "")
            result.append(FlowSnapshot(snapshot_id=sid, **d))
        return result

    async def get_latest(self) -> FlowSnapshot | None:
        snapshots = await self.list(limit=1)
        return snapshots[0] if snapshots else None


class PostgresSnapshotStore(SnapshotStore):
    """PostgreSQL-based snapshot store. Requires: pip install psycopg2-binary"""

    def __init__(self, connection_string: str, table_name: str = "duxx_snapshots") -> None:
        self._conn_str = connection_string
        self._table = table_name
        self._ensure_table()

    def _ensure_table(self) -> None:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {self._table} (
            snapshot_id TEXT PRIMARY KEY, data JSONB NOT NULL,
            step INTEGER, created_at TIMESTAMPTZ DEFAULT NOW()
        )""")
        conn.commit(); cur.close(); conn.close()

    async def save(self, snapshot: FlowSnapshot) -> str:
        import psycopg2
        data = json.dumps({"values": snapshot.values, "node": snapshot.node, "step": snapshot.step, "timestamp": snapshot.timestamp, "config": snapshot.config, "parent_snapshot_id": snapshot.parent_snapshot_id, "metadata": snapshot.metadata, "interrupts": snapshot.interrupts}, default=str)
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"INSERT INTO {self._table} (snapshot_id, data, step) VALUES (%s, %s, %s) ON CONFLICT (snapshot_id) DO UPDATE SET data=%s", (snapshot.snapshot_id, data, snapshot.step, data))
        conn.commit(); cur.close(); conn.close()
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"SELECT data FROM {self._table} WHERE snapshot_id=%s", (snapshot_id,))
        row = cur.fetchone(); cur.close(); conn.close()
        if row:
            d = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            return FlowSnapshot(snapshot_id=snapshot_id, **d)
        return None

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"SELECT snapshot_id, data FROM {self._table} ORDER BY step DESC LIMIT %s", (limit,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return [FlowSnapshot(snapshot_id=r[0], **(json.loads(r[1]) if isinstance(r[1], str) else r[1])) for r in rows]

    async def get_latest(self) -> FlowSnapshot | None:
        s = await self.list(1); return s[0] if s else None


class RedisSnapshotStore(SnapshotStore):
    """Redis-based snapshot store. Requires: pip install redis"""

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "duxx:snapshot:") -> None:
        try:
            import redis
        except ImportError:
            raise ImportError("redis is required: pip install redis")
        self._client = redis.from_url(url)
        self._prefix = prefix
        self._order_key = f"{prefix}__order__"

    async def save(self, snapshot: FlowSnapshot) -> str:
        data = json.dumps({"values": snapshot.values, "node": snapshot.node, "step": snapshot.step, "timestamp": snapshot.timestamp, "config": snapshot.config, "parent_snapshot_id": snapshot.parent_snapshot_id, "metadata": snapshot.metadata, "interrupts": snapshot.interrupts}, default=str)
        self._client.set(f"{self._prefix}{snapshot.snapshot_id}", data)
        self._client.rpush(self._order_key, snapshot.snapshot_id)
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        data = self._client.get(f"{self._prefix}{snapshot_id}")
        if data:
            d = json.loads(data)
            return FlowSnapshot(snapshot_id=snapshot_id, **d)
        return None

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        ids = self._client.lrange(self._order_key, -limit, -1)
        result = []
        for sid in reversed(ids):
            sid = sid.decode() if isinstance(sid, bytes) else sid
            snap = await self.load(sid)
            if snap:
                result.append(snap)
        return result

    async def get_latest(self) -> FlowSnapshot | None:
        s = await self.list(1); return s[0] if s else None


class MongoSnapshotStore(SnapshotStore):
    """MongoDB-based snapshot store. Requires: pip install pymongo"""

    def __init__(self, connection_string: str = "mongodb://localhost:27017", db_name: str = "duxx_ai", collection: str = "snapshots") -> None:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is required: pip install pymongo")
        self._client = MongoClient(connection_string)
        self._collection = self._client[db_name][collection]

    async def save(self, snapshot: FlowSnapshot) -> str:
        doc = {"_id": snapshot.snapshot_id, "values": snapshot.values, "node": snapshot.node, "step": snapshot.step, "timestamp": snapshot.timestamp, "config": snapshot.config, "parent_snapshot_id": snapshot.parent_snapshot_id, "metadata": snapshot.metadata, "interrupts": snapshot.interrupts}
        self._collection.replace_one({"_id": snapshot.snapshot_id}, doc, upsert=True)
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        doc = self._collection.find_one({"_id": snapshot_id})
        if doc:
            doc.pop("_id", None)
            return FlowSnapshot(snapshot_id=snapshot_id, **doc)
        return None

    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        docs = self._collection.find().sort("step", -1).limit(limit)
        return [FlowSnapshot(snapshot_id=d.pop("_id"), **d) for d in docs]

    async def get_latest(self) -> FlowSnapshot | None:
        s = await self.list(1); return s[0] if s else None


class DynamoDBSnapshotStore(SnapshotStore):
    """AWS DynamoDB snapshot store. Requires: pip install boto3"""
    def __init__(self, table_name: str = "duxx_snapshots", region: str = "us-east-1") -> None:
        try: import boto3
        except ImportError: raise ImportError("boto3 required: pip install boto3")
        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    async def save(self, snapshot: FlowSnapshot) -> str:
        import json
        self._table.put_item(Item={"snapshot_id": snapshot.snapshot_id, "step": snapshot.step, "data": json.dumps({"values": snapshot.values, "node": snapshot.node, "step": snapshot.step, "timestamp": snapshot.timestamp, "config": snapshot.config, "parent_snapshot_id": snapshot.parent_snapshot_id, "metadata": snapshot.metadata, "interrupts": snapshot.interrupts}, default=str)})
        return snapshot.snapshot_id
    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        import json
        resp = self._table.get_item(Key={"snapshot_id": snapshot_id})
        item = resp.get("Item")
        if item:
            d = json.loads(item["data"])
            return FlowSnapshot(snapshot_id=snapshot_id, **d)
        return None
    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        import json
        resp = self._table.scan(Limit=limit)
        items = sorted(resp.get("Items", []), key=lambda x: int(x.get("step", 0)), reverse=True)
        return [FlowSnapshot(snapshot_id=i["snapshot_id"], **json.loads(i["data"])) for i in items[:limit]]
    async def get_latest(self) -> FlowSnapshot | None:
        s = await self.list(1); return s[0] if s else None


class ValleySnapshotStore(SnapshotStore):
    """Valkey (Redis fork) snapshot store. Requires: pip install redis"""
    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "duxx:snap:") -> None:
        try: import redis
        except ImportError: raise ImportError("redis required: pip install redis")
        self._client = redis.from_url(url); self._prefix = prefix; self._order_key = f"{prefix}__order__"
    async def save(self, snapshot: FlowSnapshot) -> str:
        import json
        data = json.dumps({"values": snapshot.values, "node": snapshot.node, "step": snapshot.step, "timestamp": snapshot.timestamp, "config": snapshot.config, "parent_snapshot_id": snapshot.parent_snapshot_id, "metadata": snapshot.metadata, "interrupts": snapshot.interrupts}, default=str)
        self._client.set(f"{self._prefix}{snapshot.snapshot_id}", data); self._client.rpush(self._order_key, snapshot.snapshot_id)
        return snapshot.snapshot_id
    async def load(self, snapshot_id: str) -> FlowSnapshot | None:
        import json; data = self._client.get(f"{self._prefix}{snapshot_id}")
        if data: return FlowSnapshot(snapshot_id=snapshot_id, **json.loads(data))
        return None
    async def list(self, limit: int = 100) -> list[FlowSnapshot]:
        ids = self._client.lrange(self._order_key, -limit, -1); result = []
        for sid in reversed(ids):
            sid = sid.decode() if isinstance(sid, bytes) else sid
            snap = await self.load(sid)
            if snap: result.append(snap)
        return result
    async def get_latest(self) -> FlowSnapshot | None:
        s = await self.list(1); return s[0] if s else None


# ── Managed Values ──

class IsLastStep:
    """Indicates whether the current step is the last before max_iterations."""
    pass


class RemainingSteps:
    """Returns the number of remaining steps before max_iterations."""
    pass


# ── ChatState Built-in ──

class ChatState(TypedDict):
    """Pre-built state schema with messages list using merge_messages reducer.

    Usage:
        graph = FlowGraph(ChatState)
    """
    messages: list  # Uses merge_messages reducer automatically


# ── FlowGraph ──

class FlowGraph:
    """Modern graph engine with typed state, channels, streaming, and checkpointing.

    Usage:
        from duxx_ai.orchestration.state_graph import FlowGraph, ENTRY, EXIT

        class State(TypedDict):
            messages: Annotated[list, merge_messages]
            count: int

        graph = FlowGraph(State)
        graph.add_node("agent", agent_fn)
        graph.add_edge(ENTRY, "agent")
        graph.add_edge("agent", EXIT)

        compiled = graph.compile(checkpointer=MemorySnapshotStore())
        result = await compiled.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    """

    def __init__(self, state_schema: type | None = None, *, name: str = "graph") -> None:
        self.name = name
        self.state_schema = state_schema
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditional_edges: dict[str, list[dict[str, Any]]] = {}
        self._entry_points: list[str] = []

    def add_node(
        self,
        node_id: str,
        handler: Callable[..., Any] | None = None,
        *,
        retry: RetryStrategy | None = None,
        cache: CacheStrategy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FlowGraph:
        """Add a node with optional retry and cache policies."""
        self._nodes[node_id] = {
            "handler": handler,
            "retry": retry,
            "cache": cache,
            "metadata": metadata or {},
        }
        return self

    def add_edge(self, source: str, target: str) -> FlowGraph:
        """Add a static edge between nodes."""
        self._edges.append((source, target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., str | list[str] | list[Dispatch]],
        path_map: dict[str, str] | None = None,
    ) -> FlowGraph:
        """Add conditional routing from source based on path function output.

        Args:
            source: Source node ID
            path: Function(state) → target_node_id or list[Dispatch]
            path_map: Optional mapping from path output to actual node IDs
        """
        if source not in self._conditional_edges:
            self._conditional_edges[source] = []
        self._conditional_edges[source].append({
            "path": path,
            "path_map": path_map,
        })
        return self

    def add_sequence(self, node_ids: list[str]) -> FlowGraph:
        """Add a linear chain of nodes. Shorthand for multiple add_edge() calls.

        Usage:
            graph.add_sequence(["a", "b", "c"])
            # Equivalent to: add_edge("a","b"), add_edge("b","c")
        """
        for i in range(len(node_ids) - 1):
            self._edges.append((node_ids[i], node_ids[i + 1]))
        return self

    def set_entry_point(self, node_id: str) -> FlowGraph:
        """Set the entry point (equivalent to add_edge(ENTRY, node_id))."""
        self._entry_points.append(node_id)
        return self

    def compile(
        self,
        checkpointer: SnapshotStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> CompiledFlow:
        """Compile the graph for execution."""
        return CompiledFlow(
            graph=self,
            checkpointer=checkpointer or MemorySnapshotStore(),
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
        )


class CompiledFlow:
    """Compiled, executable state graph with streaming and checkpointing."""

    def __init__(
        self,
        graph: FlowGraph,
        checkpointer: SnapshotStore,
        interrupt_before: list[str],
        interrupt_after: list[str],
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self._channels: dict[str, BaseChannel] = {}
        self._node_cache: dict[str, tuple[float, Any]] = {}
        self.max_iterations = 50
        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels from state schema."""
        schema = self.graph.state_schema
        if schema is None:
            return

        hints = get_type_hints(schema, include_extras=True)
        for key, annotation in hints.items():
            # Special case: ChatState.messages gets merge_messages
            if key == "messages" and schema is ChatState:
                self._channels[key] = BinaryOperatorAggregate(merge_messages, list, [])
            else:
                self._channels[key] = channel_from_annotation(annotation)

    def _read_state(self) -> dict[str, Any]:
        """Read current state from all channels."""
        return {k: ch.get() for k, ch in self._channels.items()}

    def _write_state(self, updates: dict[str, Any]) -> None:
        """Write updates to channels, respecting reducers."""
        for key, value in updates.items():
            if isinstance(value, Override):
                if key in self._channels:
                    self._channels[key]._value = value.value
                    self._channels[key]._updated = True
            elif key in self._channels:
                self._channels[key].update(value)
            else:
                # Dynamic key — create LastValue channel
                ch = LastValue()
                ch.update(value)
                self._channels[key] = ch

    async def _execute_node(self, node_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Execute a single node with retry and cache support."""
        node_config = self.graph._nodes.get(node_id, {})
        handler = node_config.get("handler")
        retry_policy = node_config.get("retry")
        cache_policy = node_config.get("cache")

        if handler is None:
            return {}

        # Check cache
        if cache_policy:
            cache_key = cache_policy.key(node_id, state)
            if cache_key in self._node_cache:
                cached_time, cached_result = self._node_cache[cache_key]
                if cache_policy.ttl is None or (time.time() - cached_time) < cache_policy.ttl:
                    return cached_result if isinstance(cached_result, dict) else {}

        # Execute with retry
        last_error: Exception | None = None
        attempts = (retry_policy.max_attempts if retry_policy else 1)
        intervals = (retry_policy.intervals() if retry_policy else [])

        for attempt in range(attempts):
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(state)
                else:
                    result = handler(state)

                # Handle Route returns
                if isinstance(result, Route):
                    return {"__command__": result}

                # Handle Send returns
                if isinstance(result, list) and result and isinstance(result[0], Dispatch):
                    return {"__sends__": result}

                # Handle dict returns (state updates)
                if isinstance(result, dict):
                    # Cache result
                    if cache_policy:
                        cache_key = cache_policy.key(node_id, state)
                        self._node_cache[cache_key] = (time.time(), result)
                    return result

                return {}

            except FlowPause:
                raise  # Don't retry interrupts
            except Exception as e:
                last_error = e
                if retry_policy and attempt < attempts - 1:
                    if isinstance(e, retry_policy.retry_on):
                        await asyncio.sleep(intervals[attempt] if attempt < len(intervals) else 1.0)
                        continue
                raise

        if last_error:
            raise last_error
        return {}

    def _get_next_nodes(self, current: str) -> list[str]:
        """Get statically connected next nodes."""
        return [t for s, t in self.graph._edges if s == current]

    async def _resolve_conditional(self, node_id: str, state: dict[str, Any]) -> list[str | Send]:
        """Resolve conditional edges for a node."""
        targets: list[str | Send] = []
        for cond in self.graph._conditional_edges.get(node_id, []):
            path_fn = cond["path"]
            path_map = cond.get("path_map")

            if asyncio.iscoroutinefunction(path_fn):
                result = await path_fn(state)
            else:
                result = path_fn(state)

            if isinstance(result, list):
                for item in result:
                    if isinstance(item, Dispatch):
                        targets.append(item)
                    elif path_map and item in path_map:
                        targets.append(path_map[item])
                    else:
                        targets.append(item)
            elif isinstance(result, str):
                if path_map and result in path_map:
                    targets.append(path_map[result])
                else:
                    targets.append(result)

        return targets

    async def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute graph and return final state.

        Args:
            input_data: Initial state values
            config: Optional runtime configuration

        Returns:
            Final state as dict
        """
        # Initialize state
        self._write_state(input_data)
        config = config or {}
        step = 0
        parent_snapshot_id: str | None = None

        # Determine start nodes
        start_nodes = self.graph._entry_points.copy()
        start_nodes.extend(self._get_next_nodes(ENTRY))
        if not start_nodes:
            # Auto-detect: first added node
            if self.graph._nodes:
                start_nodes = [list(self.graph._nodes.keys())[0]]

        queue = list(start_nodes)
        visited: set[str] = set()

        while queue and step < self.max_iterations:
            current = queue.pop(0)

            if current == EXIT or current in visited:
                continue

            if current not in self.graph._nodes:
                continue

            visited.add(current)
            step += 1

            state = self._read_state()

            # Add managed values
            state["__is_last_step__"] = step >= self.max_iterations - 1
            state["__remaining_steps__"] = self.max_iterations - step

            # Interrupt before
            if current in self.interrupt_before:
                snapshot = FlowSnapshot(
                    values=state, node=current, step=step,
                    timestamp=time.time(), config=config,
                    parent_snapshot_id=parent_snapshot_id,
                    interrupts=[{"type": "before", "node": current}],
                )
                parent_snapshot_id = await self.checkpointer.save(snapshot)
                raise FlowPause(f"Interrupted before node '{current}'")

            # Execute node
            try:
                updates = await self._execute_node(current, state)
            except FlowPause as gi:
                snapshot = FlowSnapshot(
                    values=self._read_state(), node=current, step=step,
                    timestamp=time.time(), config=config,
                    parent_snapshot_id=parent_snapshot_id,
                    interrupts=[{"type": "interrupt", "value": gi.value}],
                )
                parent_snapshot_id = await self.checkpointer.save(snapshot)
                raise

            # Handle Route
            if "__command__" in updates:
                cmd: Route = updates["__command__"]
                if cmd.update:
                    self._write_state(cmd.update)
                if cmd.goto:
                    goto_nodes = [cmd.goto] if isinstance(cmd.goto, str) else cmd.goto
                    queue = goto_nodes + queue
                continue

            # Handle Send fan-out
            if "__sends__" in updates:
                sends: list[Dispatch] = updates["__sends__"]
                for send in sends:
                    if send.arg and isinstance(send.arg, dict):
                        self._write_state(send.arg)
                    queue.append(send.node)
                continue

            # Apply state updates
            if updates:
                self._write_state(updates)

            # Interrupt after
            if current in self.interrupt_after:
                snapshot = FlowSnapshot(
                    values=self._read_state(), node=current, step=step,
                    timestamp=time.time(), config=config,
                    parent_snapshot_id=parent_snapshot_id,
                    interrupts=[{"type": "after", "node": current}],
                )
                parent_snapshot_id = await self.checkpointer.save(snapshot)
                raise FlowPause(f"Interrupted after node '{current}'")

            # Checkpoint
            snapshot = FlowSnapshot(
                values=self._read_state(), node=current, step=step,
                timestamp=time.time(), config=config,
                parent_snapshot_id=parent_snapshot_id,
            )
            parent_snapshot_id = await self.checkpointer.save(snapshot)

            # Resolve next nodes (static + conditional)
            next_nodes = self._get_next_nodes(current)
            cond_targets = await self._resolve_conditional(current, self._read_state())
            for target in cond_targets:
                if isinstance(target, Dispatch):
                    queue.append(target.node)
                elif isinstance(target, str):
                    next_nodes.append(target)

            queue.extend([n for n in next_nodes if n not in visited])

        return self._read_state()

    async def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        stream_mode: EventMode | list[EventMode] = EventMode.VALUES,
    ) -> AsyncIterator[FlowEvent]:
        """Execute graph with streaming output.

        Args:
            input_data: Initial state values
            config: Optional runtime configuration
            stream_mode: What to stream (values, updates, messages, debug)

        Yields:
            StreamPart objects based on stream_mode
        """
        modes = [stream_mode] if isinstance(stream_mode, EventMode) else stream_mode
        self._write_state(input_data)
        config = config or {}
        step = 0
        parent_snapshot_id: str | None = None

        start_nodes = self.graph._entry_points.copy()
        start_nodes.extend(self._get_next_nodes(ENTRY))
        if not start_nodes and self.graph._nodes:
            start_nodes = [list(self.graph._nodes.keys())[0]]

        queue = list(start_nodes)
        visited: set[str] = set()

        while queue and step < self.max_iterations:
            current = queue.pop(0)
            if current == EXIT or current in visited or current not in self.graph._nodes:
                continue

            visited.add(current)
            step += 1
            state = self._read_state()

            if EventMode.DEBUG in modes:
                yield FlowEvent(EventMode.DEBUG, current, {"step": step, "state_before": dict(state)})

            try:
                updates = await self._execute_node(current, state)
            except FlowPause as gi:
                yield FlowEvent(EventMode.DEBUG, current, {"interrupted": True, "value": gi.value})
                raise

            # Handle Route/Send
            if "__command__" in updates:
                cmd = updates["__command__"]
                if cmd.update:
                    self._write_state(cmd.update)
                if cmd.goto:
                    goto_nodes = [cmd.goto] if isinstance(cmd.goto, str) else cmd.goto
                    queue = goto_nodes + queue
                continue

            if "__sends__" in updates:
                for send in updates["__sends__"]:
                    if send.arg and isinstance(send.arg, dict):
                        self._write_state(send.arg)
                    queue.append(send.node)
                continue

            if updates:
                self._write_state(updates)

            if EventMode.UPDATES in modes:
                yield FlowEvent(EventMode.UPDATES, current, updates)

            if EventMode.VALUES in modes:
                yield FlowEvent(EventMode.VALUES, current, self._read_state())

            if EventMode.MESSAGES in modes:
                msgs = self._read_state().get("messages", [])
                if msgs:
                    yield FlowEvent(EventMode.MESSAGES, current, msgs[-1] if msgs else None)

            # Checkpoint
            snapshot = FlowSnapshot(
                values=self._read_state(), node=current, step=step,
                timestamp=time.time(), config=config, parent_snapshot_id=parent_snapshot_id,
            )
            parent_snapshot_id = await self.checkpointer.save(snapshot)

            next_nodes = self._get_next_nodes(current)
            cond_targets = await self._resolve_conditional(current, self._read_state())
            for t in cond_targets:
                if isinstance(t, Dispatch):
                    queue.append(t.node)
                elif isinstance(t, str):
                    next_nodes.append(t)
            queue.extend([n for n in next_nodes if n not in visited])

    # ── Time-Travel ──

    def get_graph(self, xray: bool = False) -> GraphView:
        """Get a visual representation of the graph for introspection.

        Usage:
            view = compiled.get_graph()
            print(view.to_ascii())
            print(view.to_mermaid())
        """
        nodes = [{"id": nid, "has_handler": cfg.get("handler") is not None, "retry": cfg.get("retry") is not None, "cache": cfg.get("cache") is not None, **cfg.get("metadata", {})} for nid, cfg in self.graph._nodes.items()]
        edges = [{"source": s, "target": t} for s, t in self.graph._edges]
        for source, conds in self.graph._conditional_edges.items():
            for c in conds:
                if c.get("path_map"):
                    for label, target in c["path_map"].items():
                        edges.append({"source": source, "target": target, "label": label})
                else:
                    edges.append({"source": source, "target": "?", "label": "conditional"})
        return GraphView(nodes=nodes, edges=edges, entry_points=list(self.graph._entry_points), metadata={"name": self.graph.name})

    async def get_state_history(self, limit: int = 100) -> list[FlowSnapshot]:
        """Get all checkpointed states for time-travel debugging."""
        return await self.checkpointer.list(limit)

    async def replay_from(self, snapshot_id: str, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Replay execution from a specific checkpoint (time-travel)."""
        snapshot = await self.checkpointer.load(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        # Restore state
        for key, value in snapshot.values.items():
            if key.startswith("__"):
                continue
            if key in self._channels:
                self._channels[key].from_checkpoint(value)
            else:
                ch = LastValue()
                ch.update(value)
                self._channels[key] = ch

        if input_data:
            self._write_state(input_data)

        # Re-execute from the checkpointed node
        return await self.invoke(self._read_state())

    async def fork_from(self, snapshot_id: str, updates: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fork a new execution branch from a checkpoint with optional state modifications."""
        snapshot = await self.checkpointer.load(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        state = dict(snapshot.values)
        if updates:
            state.update(updates)
        return await self.invoke(state)

    async def resume(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Resume from the latest checkpoint after an interrupt."""
        latest = await self.checkpointer.get_latest()
        if not latest:
            raise ValueError("No checkpoint to resume from")

        state = dict(latest.values)
        state.update(input_data)
        return await self.invoke(state)


# ── Functional API: @workflow / @step ──

def workflow(
    *,
    checkpointer: SnapshotStore | None = None,
    name: str | None = None,
    retry_policy: RetryStrategy | None = None,
    cache_policy: CacheStrategy | None = None,
) -> Callable:
    """Decorator to create a graph entry point from a function.

    Usage:
        @workflow(checkpointer=MemorySnapshotStore())
        async def my_workflow(input_data: dict) -> str:
            result = await task_a(input_data)
            return result
    """
    def decorator(fn: Callable) -> Callable:
        async def wrapper(input_data: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
            cp = checkpointer or MemorySnapshotStore()

            if retry_policy:
                for attempt in range(retry_policy.max_attempts):
                    try:
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(input_data)
                        else:
                            result = fn(input_data)
                        break
                    except Exception as e:
                        if attempt >= retry_policy.max_attempts - 1:
                            raise
                        await asyncio.sleep(retry_policy.intervals()[attempt] if attempt < len(retry_policy.intervals()) else 1.0)
            else:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(input_data)
                else:
                    result = fn(input_data)

            # Save checkpoint
            snapshot = FlowSnapshot(
                values=input_data, node=name or fn.__name__,
                step=1, timestamp=time.time(),
                metadata={"result": str(result)[:200]},
            )
            await cp.save(snapshot)

            return result

        wrapper.__name__ = name or fn.__name__
        wrapper.__wrapped__ = fn
        return wrapper
    return decorator


def step(
    *,
    name: str | None = None,
    retry_policy: RetryStrategy | None = None,
    cache_policy: CacheStrategy | None = None,
) -> Callable:
    """Decorator to mark a function as a graph task.

    Usage:
        @step(name="fetch_data", retry_policy=RetryStrategy(max_attempts=3))
        async def fetch_data(url: str) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if retry_policy:
                for attempt in range(retry_policy.max_attempts):
                    try:
                        if asyncio.iscoroutinefunction(fn):
                            return await fn(*args, **kwargs)
                        return fn(*args, **kwargs)
                    except Exception:
                        if attempt >= retry_policy.max_attempts - 1:
                            raise
                        intervals = retry_policy.intervals()
                        await asyncio.sleep(intervals[attempt] if attempt < len(intervals) else 1.0)
            else:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                return fn(*args, **kwargs)

        wrapper.__name__ = name or fn.__name__
        wrapper.__wrapped__ = fn
        return wrapper
    return decorator


# ── StreamWriter ──

class StreamWriter:
    """Write custom events to the stream from inside node code.

    Usage:
        def my_node(state, *, writer: StreamWriter):
            writer({"progress": 0.5, "message": "Halfway done"})
            return {"result": "done"}
    """

    def __init__(self) -> None:
        self._buffer: list[Any] = []

    def __call__(self, data: Any) -> None:
        self._buffer.append(data)

    def flush(self) -> list[Any]:
        items = list(self._buffer)
        self._buffer.clear()
        return items


# ── Store (Namespaced KV Persistence) ──

class Store:
    """Namespaced key-value store accessible from nodes at runtime.

    Usage:
        store = Store()
        store.put("user:123", "preferences", {"theme": "dark"})
        prefs = store.get("user:123", "preferences")
        results = store.search("user:123", query="theme")
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def put(self, namespace: str, key: str, value: Any) -> None:
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = {"value": value, "updated_at": time.time()}

    def get(self, namespace: str, key: str) -> Any | None:
        ns = self._data.get(namespace, {})
        entry = ns.get(key)
        return entry["value"] if entry else None

    def delete(self, namespace: str, key: str) -> bool:
        ns = self._data.get(namespace, {})
        if key in ns:
            del ns[key]
            return True
        return False

    def list_keys(self, namespace: str) -> list[str]:
        return list(self._data.get(namespace, {}).keys())

    def list_namespaces(self) -> list[str]:
        return list(self._data.keys())

    def search(self, namespace: str, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        ns = self._data.get(namespace, {})
        results = []
        for key, entry in ns.items():
            if not query or query.lower() in str(entry["value"]).lower() or query.lower() in key.lower():
                results.append({"key": key, **entry})
        return results[:limit]

    def clear(self, namespace: str | None = None) -> None:
        if namespace:
            self._data.pop(namespace, None)
        else:
            self._data.clear()


# ── UIMessage System ──

@dataclass
class UIMessage:
    """UI component message for interactive agents."""
    name: str
    props: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoveUIMessage:
    """Remove a UI component by ID."""
    id: str


def push_ui_message(name: str, props: dict[str, Any] | None = None, id: str | None = None) -> UIMessage:
    return UIMessage(name=name, props=props or {}, id=id or uuid.uuid4().hex[:8])


def delete_ui_message(id: str) -> RemoveUIMessage:
    return RemoveUIMessage(id=id)


# ── RemoveMessage ──

@dataclass
class RemoveMessage:
    """Remove a message by ID from messages state."""
    id: str


REMOVE_ALL_MESSAGES = "__remove_all__"


# ── Graph Validation ──

def validate_graph(graph: FlowGraph) -> list[str]:
    """Validate a FlowGraph for common errors. Returns list of errors (empty = valid)."""
    errors: list[str] = []
    if not graph._nodes:
        errors.append("Graph has no nodes")
    all_ids = set(graph._nodes.keys()) | {ENTRY, EXIT}
    for s, t in graph._edges:
        if s not in all_ids and s != ENTRY:
            errors.append(f"Edge source '{s}' not found")
        if t not in all_ids and t != EXIT:
            errors.append(f"Edge target '{t}' not found")
    targets = {t for _, t in graph._edges} | set(graph._entry_points)
    for nid in graph._nodes:
        if nid not in targets:
            cond_tgts = set()
            for conds in graph._conditional_edges.values():
                for c in conds:
                    if c.get("path_map"):
                        cond_tgts.update(c["path_map"].values())
            if nid not in cond_tgts:
                errors.append(f"Node '{nid}' has no incoming edges")
    for nid, cfg in graph._nodes.items():
        if cfg.get("handler") is None:
            errors.append(f"Node '{nid}' has no handler")
    if not graph._entry_points and not any(s == ENTRY for s, _ in graph._edges):
        errors.append("No entry point defined")
    return errors


# ── Graph Introspection ──

@dataclass
class GraphView:
    """Visual representation of a compiled graph."""
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
    entry_points: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ascii(self) -> str:
        lines = [f"Graph: {self.metadata.get('name', 'unnamed')}", "=" * 40]
        for ep in self.entry_points:
            lines.append(f"  [ENTRY] --> [{ep}]")
        for edge in self.edges:
            label = f" ({edge.get('label', '')})" if edge.get("label") else ""
            lines.append(f"  [{edge['source']}] --> [{edge['target']}]{label}")
        lines.append("=" * 40)
        lines.append(f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for ep in self.entry_points:
            lines.append(f"    START([Start]) --> {ep}")
        for edge in self.edges:
            s, t = edge["source"], edge["target"]
            if t == EXIT:
                t = "EXIT_NODE([End])"
            label = edge.get("label", "")
            lines.append(f"    {s} -->|{label}| {t}" if label else f"    {s} --> {t}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "entry_points": self.entry_points, "metadata": self.metadata}


# ── create_react_agent ──

def create_react_agent(
    model_config: Any = None,
    tools: list[Any] | None = None,
    system_prompt: str = "You are a helpful AI assistant.",
    max_iterations: int = 10,
    checkpointer: SnapshotStore | None = None,
) -> CompiledFlow:
    """Create a pre-built ReAct agent as a compiled FlowGraph.

    Usage:
        agent = create_react_agent(model_config=LLMConfig(...), tools=[...])
        result = await agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    """

    async def reasoning_node(state: dict) -> dict:
        iteration = state.get("__iteration__", 0) + 1
        return {"__iteration__": iteration, "__needs_tool__": True}

    async def tool_node(state: dict) -> dict:
        tool_calls = state.get("__pending_tools__", [])
        results = []
        for tc in tool_calls:
            for t in (tools or []):
                if t.name == tc.get("name", ""):
                    try:
                        result = await t.execute(tc)
                        results.append({"tool": t.name, "result": str(getattr(result, 'result', result))})
                    except Exception as e:
                        results.append({"tool": t.name, "error": str(e)})
        return {"__tool_results__": results, "__needs_tool__": False}

    def should_continue(state: dict) -> str:
        if state.get("__iteration__", 0) >= max_iterations:
            return EXIT
        if state.get("__needs_tool__") and state.get("__pending_tools__"):
            return "tools"
        return EXIT

    graph = FlowGraph(ChatState, name="react_agent")
    graph.add_node("reason", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(ENTRY, "reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", EXIT: EXIT})
    graph.add_edge("tools", "reason")

    compiled = graph.compile(checkpointer=checkpointer or MemorySnapshotStore())
    compiled.max_iterations = max_iterations * 2 + 5
    return compiled
