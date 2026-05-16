"""Tests for FlowGraph — the modern typed state graph engine."""

from typing import Annotated, TypedDict

import pytest

from duxx_ai.orchestration.channels import (
    BinaryOperatorAggregate,
    Topic,
    merge_messages,
)
from duxx_ai.orchestration.state_graph import (
    ENTRY,
    EXIT,
    ChatState,
    Dispatch,
    EventMode,
    FlowGraph,
    FlowPause,
    MemorySnapshotStore,
    Override,
    RetryStrategy,
    Route,
    step,
    workflow,
)

# ── State Schemas ──

class CounterState(TypedDict):
    count: int
    log: list


class MessageState(TypedDict):
    messages: Annotated[list, merge_messages]
    summary: str


# ── Basic Graph Tests ──

@pytest.mark.asyncio
async def test_basic_flow():
    """Test a simple 2-node graph."""
    def add_one(state):
        return {"count": state.get("count", 0) + 1}

    def add_two(state):
        return {"count": state.get("count", 0) + 2}

    g = FlowGraph(CounterState)
    g.add_node("a", add_one)
    g.add_node("b", add_two)
    g.add_edge(ENTRY, "a")
    g.add_edge("a", "b")
    g.add_edge("b", EXIT)

    compiled = g.compile()
    result = await compiled.invoke({"count": 0, "log": []})
    assert result["count"] == 3  # 0 + 1 + 2


@pytest.mark.asyncio
async def test_conditional_edges():
    """Test conditional routing."""
    def classifier(state):
        return {"category": "high" if state.get("score", 0) > 50 else "low"}

    def high_handler(state):
        return {"result": "premium"}

    def low_handler(state):
        return {"result": "basic"}

    def route(state):
        return "high" if state.get("category") == "high" else "low"

    g = FlowGraph()
    g.add_node("classify", classifier)
    g.add_node("high", high_handler)
    g.add_node("low", low_handler)
    g.add_edge(ENTRY, "classify")
    g.add_conditional_edges("classify", route)

    compiled = g.compile()
    result = await compiled.invoke({"score": 80})
    assert result["result"] == "premium"


@pytest.mark.asyncio
async def test_route_command():
    """Test Route for dynamic routing."""
    def router_node(state):
        if state.get("urgent"):
            return Route(update={"priority": "high"}, goto="fast_track")
        return Route(goto="normal")

    def fast_track(state):
        return {"handled": "fast"}

    def normal(state):
        return {"handled": "normal"}

    g = FlowGraph()
    g.add_node("router", router_node)
    g.add_node("fast_track", fast_track)
    g.add_node("normal", normal)
    g.add_edge(ENTRY, "router")

    compiled = g.compile()
    result = await compiled.invoke({"urgent": True})
    assert result["handled"] == "fast"
    assert result["priority"] == "high"


@pytest.mark.asyncio
async def test_dispatch_fan_out():
    """Test Dispatch for fan-out execution."""
    def splitter(state):
        return [Dispatch("worker", {"item": item}) for item in state.get("items", [])]

    def worker(state):
        return {"processed": state.get("item", "") + "_done"}

    g = FlowGraph()
    g.add_node("split", splitter)
    g.add_node("worker", worker)
    g.add_edge(ENTRY, "split")

    compiled = g.compile()
    result = await compiled.invoke({"items": ["a", "b", "c"]})
    assert "processed" in result


# ── Channel Tests ──

@pytest.mark.asyncio
async def test_merge_messages_reducer():
    """Test merge_messages channel."""
    msgs1 = [{"id": "1", "content": "hello"}]
    msgs2 = [{"id": "2", "content": "world"}]
    msgs3 = [{"id": "1", "content": "updated hello"}]  # Replace by ID

    result = merge_messages(msgs1, msgs2)
    assert len(result) == 2

    result = merge_messages(result, msgs3)
    assert len(result) == 2  # ID "1" replaced, not duplicated
    assert result[0]["content"] == "updated hello"


@pytest.mark.asyncio
async def test_topic_channel():
    """Test Topic accumulation."""
    ch = Topic()
    ch.update("a")
    ch.update("b")
    ch.update(["c", "d"])
    assert ch.get() == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_binary_operator_channel():
    """Test BinaryOperatorAggregate."""
    import operator
    ch = BinaryOperatorAggregate(operator.add, int, 0)
    ch.update(5)
    ch.update(3)
    assert ch.get() == 8


# ── Checkpoint Tests ──

@pytest.mark.asyncio
async def test_checkpointing():
    """Test that checkpoints are saved during execution."""
    def node_a(state):
        return {"value": 42}

    g = FlowGraph()
    g.add_node("a", node_a)
    g.add_edge(ENTRY, "a")
    g.add_edge("a", EXIT)

    store = MemorySnapshotStore()
    compiled = g.compile(checkpointer=store)
    await compiled.invoke({"value": 0})

    history = await compiled.get_state_history()
    assert len(history) >= 1
    assert history[0].values["value"] == 42


@pytest.mark.asyncio
async def test_interrupt_and_resume():
    """Test FlowPause and resume."""
    def review_node(state):
        if not state.get("approved"):
            raise FlowPause("Need approval")
        return {"status": "approved"}

    g = FlowGraph()
    g.add_node("review", review_node)
    g.add_edge(ENTRY, "review")
    g.add_edge("review", EXIT)

    store = MemorySnapshotStore()
    compiled = g.compile(checkpointer=store)

    with pytest.raises(FlowPause):
        await compiled.invoke({"approved": False})

    # Resume with approval
    result = await compiled.resume({"approved": True})
    assert result["status"] == "approved"


# ── Streaming Tests ──

@pytest.mark.asyncio
async def test_streaming_values():
    """Test streaming mode."""
    def node_a(state):
        return {"step": 1}

    def node_b(state):
        return {"step": 2}

    g = FlowGraph()
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(ENTRY, "a")
    g.add_edge("a", "b")
    g.add_edge("b", EXIT)

    compiled = g.compile()
    events = []
    async for event in compiled.stream({"step": 0}, stream_mode=EventMode.UPDATES):
        events.append(event)

    assert len(events) == 2
    assert events[0].node == "a"
    assert events[1].node == "b"


# ── Retry Tests ──

@pytest.mark.asyncio
async def test_retry_strategy():
    """Test per-node retry."""
    call_count = 0

    def flaky_node(state):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Flaky!")
        return {"result": "success"}

    g = FlowGraph()
    g.add_node("flaky", flaky_node, retry=RetryStrategy(max_attempts=3, initial_interval=0.01))
    g.add_edge(ENTRY, "flaky")
    g.add_edge("flaky", EXIT)

    compiled = g.compile()
    result = await compiled.invoke({})
    assert result["result"] == "success"
    assert call_count == 3


# ── Decorator Tests ──

@pytest.mark.asyncio
async def test_workflow_decorator():
    """Test @workflow decorator."""
    @workflow(name="test_wf")
    async def my_flow(data):
        return data.get("x", 0) * 2

    result = await my_flow({"x": 5})
    assert result == 10


@pytest.mark.asyncio
async def test_step_decorator():
    """Test @step decorator."""
    @step(name="double")
    async def double(x):
        return x * 2

    result = await double(7)
    assert result == 14


# ── Override Tests ──

@pytest.mark.asyncio
async def test_override():
    """Test Override to bypass reducer."""
    def reset_node(state):
        return {"messages": Override(["fresh_start"])}

    g = FlowGraph(MessageState)
    g.add_node("reset", reset_node)
    g.add_edge(ENTRY, "reset")
    g.add_edge("reset", EXIT)

    compiled = g.compile()
    result = await compiled.invoke({"messages": ["old1", "old2"], "summary": ""})
    assert result["messages"] == ["fresh_start"]


# ── ChatState Tests ──

@pytest.mark.asyncio
async def test_chat_state():
    """Test ChatState with merge_messages."""
    def echo(state):
        last = state.get("messages", [])[-1] if state.get("messages") else {}
        return {"messages": [{"id": "resp", "role": "assistant", "content": f"Echo: {last.get('content', '')}"}]}

    g = FlowGraph(ChatState)
    g.add_node("echo", echo)
    g.add_edge(ENTRY, "echo")
    g.add_edge("echo", EXIT)

    compiled = g.compile()
    result = await compiled.invoke({"messages": [{"id": "1", "role": "user", "content": "Hi"}]})
    assert len(result["messages"]) == 2
    assert "Echo: Hi" in result["messages"][-1]["content"]
