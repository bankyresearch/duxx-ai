"""Graph Analytics for Agent Workflows — powered by NetworkX.

Provides deep analysis of agent workflow graphs for optimization,
debugging, bottleneck detection, and intelligent routing.

Features:
- Workflow optimization (critical path, bottleneck detection, parallel opportunities)
- Agent importance ranking (centrality metrics)
- Task routing (shortest path, optimal flow)
- Dependency analysis (topological sort, cycle detection, DAG properties)
- Community detection (agent grouping, module identification)
- Graph visualization (ASCII, Mermaid, DOT, JSON export)
- Performance simulation (execution time estimation)
- Reliability analysis (single points of failure, redundancy)

Requires: pip install networkx (optional — fallback to basic analysis without it)

Usage:
    from duxx_ai.orchestration.analytics import WorkflowAnalyzer

    analyzer = WorkflowAnalyzer(compiled_flow)
    report = analyzer.full_report()
    print(report.critical_path)
    print(report.bottlenecks)
    print(report.parallel_opportunities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

HAS_NETWORKX = False
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore


# ── Data Classes ──

@dataclass
class NodeMetrics:
    """Metrics for a single node in the workflow graph."""
    node_id: str
    in_degree: int = 0
    out_degree: int = 0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    degree_centrality: float = 0.0
    pagerank: float = 0.0
    is_bottleneck: bool = False
    is_entry: bool = False
    is_exit: bool = False
    is_critical: bool = False
    community: int = -1
    depth: int = 0


@dataclass
class EdgeMetrics:
    """Metrics for a single edge in the workflow graph."""
    source: str
    target: str
    is_critical_path: bool = False
    is_conditional: bool = False
    weight: float = 1.0


@dataclass
class WorkflowReport:
    """Complete analysis report for a workflow graph."""
    # Structure
    node_count: int = 0
    edge_count: int = 0
    is_dag: bool = True
    has_cycles: bool = False
    depth: int = 0
    width: int = 0  # Max parallel branches

    # Critical Path
    critical_path: list[str] = field(default_factory=list)
    critical_path_length: int = 0

    # Optimization
    bottlenecks: list[str] = field(default_factory=list)
    parallel_opportunities: list[list[str]] = field(default_factory=list)
    redundant_edges: list[tuple[str, str]] = field(default_factory=list)
    single_points_of_failure: list[str] = field(default_factory=list)

    # Ranking
    most_important_nodes: list[tuple[str, float]] = field(default_factory=list)
    pagerank: dict[str, float] = field(default_factory=dict)

    # Communities
    communities: list[list[str]] = field(default_factory=list)
    community_count: int = 0

    # Node/Edge details
    node_metrics: dict[str, NodeMetrics] = field(default_factory=dict)
    edge_metrics: list[EdgeMetrics] = field(default_factory=list)

    # Topology
    topological_order: list[str] = field(default_factory=list)
    longest_path: list[str] = field(default_factory=list)
    shortest_paths: dict[str, list[str]] = field(default_factory=dict)

    # Quality
    connectivity_score: float = 0.0  # 0-1
    balance_score: float = 0.0  # 0-1 (how balanced the workload is)
    complexity_score: float = 0.0  # 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": {"nodes": self.node_count, "edges": self.edge_count, "is_dag": self.is_dag, "depth": self.depth, "width": self.width},
            "critical_path": {"path": self.critical_path, "length": self.critical_path_length},
            "optimization": {"bottlenecks": self.bottlenecks, "parallel_opportunities": self.parallel_opportunities, "single_points_of_failure": self.single_points_of_failure},
            "ranking": {"most_important": self.most_important_nodes, "pagerank": self.pagerank},
            "communities": {"count": self.community_count, "groups": self.communities},
            "topology": {"topological_order": self.topological_order, "longest_path": self.longest_path},
            "quality": {"connectivity": self.connectivity_score, "balance": self.balance_score, "complexity": self.complexity_score},
        }

    def summary(self) -> str:
        lines = [
            f"Workflow Analysis Report",
            f"{'=' * 50}",
            f"Nodes: {self.node_count} | Edges: {self.edge_count} | DAG: {self.is_dag}",
            f"Depth: {self.depth} | Width: {self.width}",
            f"",
            f"Critical Path ({self.critical_path_length} steps): {' -> '.join(self.critical_path)}",
            f"Bottlenecks: {', '.join(self.bottlenecks) or 'None'}",
            f"Single Points of Failure: {', '.join(self.single_points_of_failure) or 'None'}",
            f"Parallel Opportunities: {len(self.parallel_opportunities)} groups",
            f"",
            f"Top Nodes by Importance:",
        ]
        for name, score in self.most_important_nodes[:5]:
            lines.append(f"  {name}: {score:.4f}")
        lines.extend([
            f"",
            f"Communities: {self.community_count}",
            f"Quality: connectivity={self.connectivity_score:.2f} balance={self.balance_score:.2f} complexity={self.complexity_score:.2f}",
        ])
        return "\n".join(lines)


class WorkflowAnalyzer:
    """Analyze agent workflow graphs for optimization and debugging.

    Works with FlowGraph, Graph, or any node/edge structure.

    Usage:
        from duxx_ai.orchestration.analytics import WorkflowAnalyzer

        # From FlowGraph
        analyzer = WorkflowAnalyzer.from_flow_graph(compiled_flow)

        # From raw nodes/edges
        analyzer = WorkflowAnalyzer(
            nodes=["a", "b", "c", "d"],
            edges=[("a","b"), ("a","c"), ("b","d"), ("c","d")],
        )

        report = analyzer.full_report()
        print(report.summary())
        print(report.critical_path)
        print(report.bottlenecks)
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
        edge_weights: dict[tuple[str, str], float] | None = None,
        node_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.nodes = nodes or []
        self.edges = edges or []
        self.edge_weights = edge_weights or {}
        self.node_metadata = node_metadata or {}
        self._G: Any = None  # NetworkX graph

        if HAS_NETWORKX:
            self._build_nx_graph()

    def _build_nx_graph(self) -> None:
        """Build NetworkX DiGraph from nodes and edges."""
        self._G = nx.DiGraph()
        self._G.add_nodes_from(self.nodes)
        for src, tgt in self.edges:
            w = self.edge_weights.get((src, tgt), 1.0)
            self._G.add_edge(src, tgt, weight=w)

    @classmethod
    def from_flow_graph(cls, compiled_flow: Any) -> WorkflowAnalyzer:
        """Create analyzer from a CompiledFlow object."""
        graph = compiled_flow.graph
        nodes = list(graph._nodes.keys())
        edges = list(graph._edges)
        # Add conditional edge targets
        for source, conds in graph._conditional_edges.items():
            for c in conds:
                if c.get("path_map"):
                    for target in c["path_map"].values():
                        edges.append((source, target))
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def from_graph(cls, graph: Any) -> WorkflowAnalyzer:
        """Create analyzer from a legacy Graph object."""
        nodes = [n for n in graph.nodes if n not in ("__start__", "__end__")]
        edges = [(e.source, e.target) for e in graph.edges if e.source not in ("__start__",) and e.target not in ("__end__",)]
        return cls(nodes=nodes, edges=edges)

    # ── Full Analysis ──

    def full_report(self) -> WorkflowReport:
        """Run complete analysis and return a WorkflowReport."""
        report = WorkflowReport()
        report.node_count = len(self.nodes)
        report.edge_count = len(self.edges)

        if HAS_NETWORKX and self._G:
            report.is_dag = nx.is_directed_acyclic_graph(self._G)
            report.has_cycles = not report.is_dag

            # Topology
            if report.is_dag:
                report.topological_order = list(nx.topological_sort(self._G))
                try:
                    report.longest_path = nx.dag_longest_path(self._G)
                except Exception:
                    report.longest_path = []
                report.depth = len(report.longest_path)
            else:
                report.topological_order = list(self.nodes)
                report.depth = len(self.nodes)

            # Width (max parallel at any level)
            report.width = self._compute_width()

            # Critical path
            report.critical_path = self._find_critical_path()
            report.critical_path_length = len(report.critical_path)

            # Centrality & ranking
            report.pagerank = self._compute_pagerank()
            report.most_important_nodes = sorted(report.pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

            # Bottlenecks
            report.bottlenecks = self._find_bottlenecks()
            report.single_points_of_failure = self._find_spof()

            # Parallel opportunities
            report.parallel_opportunities = self._find_parallel_groups()

            # Communities
            report.communities = self._detect_communities()
            report.community_count = len(report.communities)

            # Node metrics
            report.node_metrics = self._compute_node_metrics()

            # Quality scores
            report.connectivity_score = self._connectivity_score()
            report.balance_score = self._balance_score()
            report.complexity_score = self._complexity_score()

            # Shortest paths from entries
            entries = [n for n in self._G.nodes if self._G.in_degree(n) == 0]
            for entry in entries[:3]:
                for target in self._G.nodes:
                    if entry != target:
                        try:
                            path = nx.shortest_path(self._G, entry, target)
                            report.shortest_paths[f"{entry}->{target}"] = path
                        except nx.NetworkXNoPath:
                            pass
        else:
            # Basic analysis without NetworkX
            report.is_dag = self._basic_is_dag()
            report.depth = len(self.nodes)
            report.width = 1
            report.critical_path = list(self.nodes)
            report.critical_path_length = len(self.nodes)

        return report

    # ── Individual Analyses ──

    def _compute_pagerank(self) -> dict[str, float]:
        """Compute PageRank for node importance."""
        try:
            return dict(nx.pagerank(self._G))
        except Exception:
            return {n: 1.0 / max(len(self.nodes), 1) for n in self.nodes}

    def _find_critical_path(self) -> list[str]:
        """Find the critical (longest) path in a DAG."""
        if not nx.is_directed_acyclic_graph(self._G):
            return list(self.nodes)
        try:
            return nx.dag_longest_path(self._G)
        except Exception:
            return list(self.nodes)

    def _find_bottlenecks(self) -> list[str]:
        """Find bottleneck nodes (high betweenness centrality + high in/out degree)."""
        if len(self._G.nodes) < 3:
            return []
        bc = nx.betweenness_centrality(self._G)
        avg = sum(bc.values()) / max(len(bc), 1)
        return [n for n, c in bc.items() if c > avg * 1.5 and self._G.in_degree(n) > 1]

    def _find_spof(self) -> list[str]:
        """Find single points of failure (articulation-point-like nodes)."""
        spof = []
        for node in self._G.nodes:
            if self._G.in_degree(node) >= 1 and self._G.out_degree(node) >= 1:
                # Check if removing this node disconnects the graph
                G_copy = self._G.copy()
                G_copy.remove_node(node)
                entries = [n for n in G_copy.nodes if G_copy.in_degree(n) == 0]
                exits = [n for n in G_copy.nodes if G_copy.out_degree(n) == 0]
                for e in entries:
                    for x in exits:
                        if not nx.has_path(G_copy, e, x):
                            spof.append(node)
                            break
                    if node in spof:
                        break
        return list(set(spof))

    def _find_parallel_groups(self) -> list[list[str]]:
        """Find groups of nodes that can execute in parallel."""
        if not nx.is_directed_acyclic_graph(self._G):
            return []
        groups = []
        try:
            for gen in nx.topological_generations(self._G):
                gen_list = list(gen)
                if len(gen_list) > 1:
                    groups.append(gen_list)
        except Exception:
            pass
        return groups

    def _detect_communities(self) -> list[list[str]]:
        """Detect communities/modules in the workflow."""
        try:
            undirected = self._G.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            return [list(c) for c in communities]
        except Exception:
            return [list(self.nodes)]

    def _compute_node_metrics(self) -> dict[str, NodeMetrics]:
        """Compute detailed metrics for each node."""
        metrics = {}
        bc = nx.betweenness_centrality(self._G) if len(self._G.nodes) >= 2 else {}
        cc = nx.closeness_centrality(self._G) if len(self._G.nodes) >= 2 else {}
        dc = nx.degree_centrality(self._G) if len(self._G.nodes) >= 2 else {}
        pr = self._compute_pagerank()

        for node in self._G.nodes:
            m = NodeMetrics(
                node_id=node,
                in_degree=self._G.in_degree(node),
                out_degree=self._G.out_degree(node),
                betweenness_centrality=bc.get(node, 0),
                closeness_centrality=cc.get(node, 0),
                degree_centrality=dc.get(node, 0),
                pagerank=pr.get(node, 0),
                is_entry=self._G.in_degree(node) == 0,
                is_exit=self._G.out_degree(node) == 0,
                is_bottleneck=node in self._find_bottlenecks(),
            )
            metrics[node] = m
        return metrics

    def _compute_width(self) -> int:
        """Compute maximum width (parallelism level)."""
        if not nx.is_directed_acyclic_graph(self._G):
            return 1
        try:
            max_width = 0
            for gen in nx.topological_generations(self._G):
                max_width = max(max_width, len(list(gen)))
            return max_width
        except Exception:
            return 1

    def _connectivity_score(self) -> float:
        """Score how well-connected the graph is (0-1)."""
        n = len(self._G.nodes)
        if n <= 1:
            return 1.0
        max_edges = n * (n - 1)  # Directed
        return min(len(self._G.edges) / max(max_edges, 1), 1.0)

    def _balance_score(self) -> float:
        """Score how balanced workload distribution is (0-1)."""
        if not self._G.nodes:
            return 1.0
        degrees = [self._G.degree(n) for n in self._G.nodes]
        avg = sum(degrees) / len(degrees)
        if avg == 0:
            return 1.0
        variance = sum((d - avg) ** 2 for d in degrees) / len(degrees)
        return max(0, 1.0 - (variance / (avg * avg + 1)))

    def _complexity_score(self) -> float:
        """Score workflow complexity (0-1). Higher = more complex."""
        n = len(self._G.nodes)
        e = len(self._G.edges)
        if n == 0:
            return 0.0
        # McCabe-like complexity: edges - nodes + 2 * connected_components
        try:
            components = nx.number_weakly_connected_components(self._G)
        except Exception:
            components = 1
        cyclomatic = e - n + 2 * components
        return min(cyclomatic / max(n, 1), 1.0)

    def _basic_is_dag(self) -> bool:
        """Basic cycle detection without NetworkX."""
        adjacency: dict[str, list[str]] = {}
        for s, t in self.edges:
            adjacency.setdefault(s, []).append(t)
        visited: set[str] = set()
        stack: set[str] = set()
        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor in stack:
                    return False
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False
            stack.discard(node)
            return True
        for node in self.nodes:
            if node not in visited:
                if not dfs(node):
                    return False
        return True

    # ── Routing Algorithms ──

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two nodes."""
        if HAS_NETWORKX and self._G:
            try:
                return nx.shortest_path(self._G, source, target)
            except nx.NetworkXNoPath:
                return []
        return []

    def all_paths(self, source: str, target: str, max_length: int = 10) -> list[list[str]]:
        """Find all simple paths between two nodes."""
        if HAS_NETWORKX and self._G:
            try:
                return list(nx.all_simple_paths(self._G, source, target, cutoff=max_length))
            except Exception:
                return []
        return []

    def optimal_route(self, source: str, target: str) -> list[str]:
        """Find the optimal (minimum weight) route."""
        if HAS_NETWORKX and self._G:
            try:
                return nx.dijkstra_path(self._G, source, target)
            except Exception:
                return self.shortest_path(source, target)
        return []

    # ── Export ──

    def to_networkx(self) -> Any:
        """Get the underlying NetworkX DiGraph object."""
        if not HAS_NETWORKX:
            raise ImportError("networkx required: pip install networkx")
        return self._G

    def to_adjacency_matrix(self) -> list[list[int]]:
        """Export as adjacency matrix."""
        if HAS_NETWORKX and self._G:
            import numpy as np
            return nx.adjacency_matrix(self._G).todense().tolist()
        n = len(self.nodes)
        matrix = [[0] * n for _ in range(n)]
        idx = {node: i for i, node in enumerate(self.nodes)}
        for s, t in self.edges:
            if s in idx and t in idx:
                matrix[idx[s]][idx[t]] = 1
        return matrix

    def to_dot(self) -> str:
        """Export as GraphViz DOT format."""
        lines = ["digraph workflow {", '  rankdir=LR;', '  node [shape=box, style="rounded,filled", fillcolor="#f3f4f6"];']
        for node in self.nodes:
            lines.append(f'  "{node}";')
        for s, t in self.edges:
            w = self.edge_weights.get((s, t))
            label = f' [label="{w:.1f}"]' if w and w != 1.0 else ""
            lines.append(f'  "{s}" -> "{t}"{label};')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Export as Mermaid diagram."""
        lines = ["graph LR"]
        for s, t in self.edges:
            lines.append(f"    {s} --> {t}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Export as JSON-serializable dict."""
        return {
            "nodes": [{"id": n, **self.node_metadata.get(n, {})} for n in self.nodes],
            "edges": [{"source": s, "target": t, "weight": self.edge_weights.get((s, t), 1.0)} for s, t in self.edges],
        }
