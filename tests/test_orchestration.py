"""Tests for orchestration engines."""

import pytest

from duxx_ai.orchestration.graph import Graph, GraphState, EdgeCondition
from duxx_ai.orchestration.crew import Crew, CrewAgent, Task


class TestGraph:
    @pytest.mark.asyncio
    async def test_simple_linear_graph(self):
        async def step1(state: GraphState) -> GraphState:
            state.set("result", "step1_done")
            return state

        async def step2(state: GraphState) -> GraphState:
            state.set("result", state.get("result") + "_step2_done")
            return state

        graph = Graph(name="test")
        graph.add_node("step1", handler=step1)
        graph.add_node("step2", handler=step2)
        graph.set_entry_point("step1")
        graph.add_edge("step1", "step2")
        graph.set_exit_point("step2")

        result = await graph.run()
        assert result.status == "completed"
        assert result.data["result"] == "step1_done_step2_done"

    @pytest.mark.asyncio
    async def test_conditional_routing(self):
        async def classifier(state: GraphState) -> GraphState:
            state.set("category", "complex" if len(state.get("input", "")) > 10 else "simple")
            return state

        async def simple_handler(state: GraphState) -> GraphState:
            state.set("output", "simple_path")
            return state

        async def complex_handler(state: GraphState) -> GraphState:
            state.set("output", "complex_path")
            return state

        graph = Graph()
        graph.add_node("classify", handler=classifier)
        graph.add_node("simple", handler=simple_handler)
        graph.add_node("complex", handler=complex_handler)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "simple", condition=EdgeCondition(key="category", value="simple"))
        graph.add_edge("classify", "complex", condition=EdgeCondition(key="category", value="complex"))
        graph.set_exit_point("simple")
        graph.set_exit_point("complex")

        # Short input -> simple path
        result = await graph.run({"input": "Hi"})
        assert result.data["output"] == "simple_path"

        # Long input -> complex path
        result = await graph.run({"input": "This is a much longer input text"})
        assert result.data["output"] == "complex_path"

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        async def loop(state: GraphState) -> GraphState:
            state.set("count", state.get("count", 0) + 1)
            return state

        graph = Graph()
        graph.max_iterations = 5
        graph.add_node("loop", handler=loop)
        graph.set_entry_point("loop")

        # Self-loops are now rejected by cycle detection
        import pytest
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("loop", "loop")

    def test_visualize(self):
        graph = Graph(name="test-graph")
        graph.add_node("a")
        graph.add_node("b")
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_exit_point("b")

        viz = graph.visualize()
        assert "test-graph" in viz
        assert "a" in viz
        assert "b" in viz


class TestCrew:
    @pytest.mark.asyncio
    async def test_sequential_crew(self):
        agents = [
            CrewAgent(name="a1", role="Researcher", goal="Research"),
            CrewAgent(name="a2", role="Writer", goal="Write"),
        ]
        tasks = [
            Task(id="t1", description="Research AI", assigned_to="a1"),
            Task(id="t2", description="Write report", assigned_to="a2", dependencies=["t1"]),
        ]

        crew = Crew(agents=agents, tasks=tasks, strategy="sequential")
        result = await crew.run()

        assert all(t.status == "completed" for t in result.tasks)
        assert result.final_output != ""

    @pytest.mark.asyncio
    async def test_parallel_crew(self):
        agents = [
            CrewAgent(name="a1", role="Analyst1", goal="Analyze sector A"),
            CrewAgent(name="a2", role="Analyst2", goal="Analyze sector B"),
        ]
        tasks = [
            Task(id="t1", description="Analyze sector A", assigned_to="a1"),
            Task(id="t2", description="Analyze sector B", assigned_to="a2"),
        ]

        crew = Crew(agents=agents, tasks=tasks, strategy="parallel")
        result = await crew.run()
        assert all(t.status == "completed" for t in result.tasks)
