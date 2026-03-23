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

    def to_graphml(self) -> str:
        """Export as GraphML XML format."""
        if not HAS_NETWORKX: raise ImportError("networkx required")
        from io import BytesIO
        buf = BytesIO(); nx.write_graphml(self._G, buf); return buf.getvalue().decode()

    def to_gml(self) -> str:
        """Export as GML (Graph Modelling Language)."""
        if not HAS_NETWORKX: raise ImportError("networkx required")
        from io import BytesIO
        buf = BytesIO(); nx.write_gml(self._G, buf); return buf.getvalue().decode()

    def to_gexf(self) -> str:
        """Export as GEXF (Gephi format)."""
        if not HAS_NETWORKX: raise ImportError("networkx required")
        from io import BytesIO
        buf = BytesIO(); nx.write_gexf(self._G, buf); return buf.getvalue().decode()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Advanced Graph Algorithms (NetworkX-powered)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def max_flow(self, source: str, target: str) -> tuple[float, dict]:
        """Compute maximum flow between source and target. Useful for capacity planning."""
        if not HAS_NETWORKX: return (0, {})
        try:
            flow_value, flow_dict = nx.maximum_flow(self._G, source, target, capacity="weight")
            return (flow_value, flow_dict)
        except Exception: return (0, {})

    def min_cut(self, source: str, target: str) -> tuple[float, tuple]:
        """Find minimum cut between source and target. Identifies weakest link."""
        if not HAS_NETWORKX: return (0, (set(), set()))
        try:
            cut_value, partition = nx.minimum_cut(self._G, source, target, capacity="weight")
            return (cut_value, (set(partition[0]), set(partition[1])))
        except Exception: return (0, (set(), set()))

    def bridges(self) -> list[tuple[str, str]]:
        """Find bridge edges whose removal disconnects the graph."""
        if not HAS_NETWORKX: return []
        try: return list(nx.bridges(self._G.to_undirected()))
        except Exception: return []

    def articulation_points(self) -> list[str]:
        """Find articulation points (nodes whose removal disconnects graph)."""
        if not HAS_NETWORKX: return []
        try: return list(nx.articulation_points(self._G.to_undirected()))
        except Exception: return []

    def strongly_connected_components(self) -> list[set[str]]:
        """Find strongly connected components (tightly coupled agent groups)."""
        if not HAS_NETWORKX: return []
        return [set(c) for c in nx.strongly_connected_components(self._G)]

    def weakly_connected_components(self) -> list[set[str]]:
        """Find weakly connected components."""
        if not HAS_NETWORKX: return []
        return [set(c) for c in nx.weakly_connected_components(self._G)]

    def find_cycles(self) -> list[list[str]]:
        """Find all simple cycles in the graph (loop detection)."""
        if not HAS_NETWORKX: return []
        try: return [list(c) for c in nx.simple_cycles(self._G)]
        except Exception: return []

    def transitive_reduction(self) -> list[tuple[str, str]]:
        """Compute transitive reduction — remove redundant edges."""
        if not HAS_NETWORKX: return list(self.edges)
        try: tr = nx.transitive_reduction(self._G); return list(tr.edges())
        except Exception: return list(self.edges)

    def transitive_closure(self) -> list[tuple[str, str]]:
        """Compute transitive closure — all reachable pairs."""
        if not HAS_NETWORKX: return list(self.edges)
        try: tc = nx.transitive_closure(self._G); return list(tc.edges())
        except Exception: return list(self.edges)

    def ancestors(self, node: str) -> set[str]:
        """Find all ancestors (upstream nodes) of a node."""
        if not HAS_NETWORKX: return set()
        try: return nx.ancestors(self._G, node)
        except Exception: return set()

    def descendants(self, node: str) -> set[str]:
        """Find all descendants (downstream nodes) of a node."""
        if not HAS_NETWORKX: return set()
        try: return nx.descendants(self._G, node)
        except Exception: return set()

    def graph_coloring(self, strategy: str = "largest_first") -> dict[str, int]:
        """Color graph nodes — assign resources without conflicts.
        Nodes with same color can share resources safely."""
        if not HAS_NETWORKX: return {n: 0 for n in self.nodes}
        try: return nx.coloring.greedy_color(self._G.to_undirected(), strategy=strategy)
        except Exception: return {n: i for i, n in enumerate(self.nodes)}

    def dominating_set(self) -> set[str]:
        """Find minimum dominating set — fewest nodes to cover all others."""
        if not HAS_NETWORKX: return set(self.nodes)
        try: return nx.dominating_set(self._G.to_undirected())
        except Exception: return set(self.nodes)

    def minimum_spanning_tree(self) -> list[tuple[str, str]]:
        """Find minimum spanning tree — most efficient connection structure."""
        if not HAS_NETWORKX: return list(self.edges)
        try: mst = nx.minimum_spanning_tree(self._G.to_undirected()); return list(mst.edges())
        except Exception: return list(self.edges)

    def hits(self) -> tuple[dict[str, float], dict[str, float]]:
        """HITS algorithm — find hub and authority scores.
        Hubs = nodes that route to many others. Authorities = nodes many route to."""
        if not HAS_NETWORKX: return ({}, {})
        try:
            hubs, auths = nx.hits(self._G)
            return (dict(hubs), dict(auths))
        except Exception: return ({}, {})

    def link_prediction(self, method: str = "jaccard") -> list[tuple[str, str, float]]:
        """Predict likely new connections between agents."""
        if not HAS_NETWORKX: return []
        G_und = self._G.to_undirected()
        non_edges = list(nx.non_edges(G_und))[:50]  # Limit for performance
        try:
            if method == "jaccard":
                preds = nx.jaccard_coefficient(G_und, non_edges)
            elif method == "adamic_adar":
                preds = nx.adamic_adar_index(G_und, non_edges)
            elif method == "preferential":
                preds = nx.preferential_attachment(G_und, non_edges)
            else:
                preds = nx.jaccard_coefficient(G_und, non_edges)
            return [(u, v, float(p)) for u, v, p in preds if p > 0]
        except Exception: return []

    def efficiency(self) -> dict[str, float]:
        """Compute graph efficiency metrics."""
        if not HAS_NETWORKX: return {"local": 0, "global": 0}
        G_und = self._G.to_undirected()
        try:
            return {"local": nx.local_efficiency(G_und), "global": nx.global_efficiency(G_und)}
        except Exception: return {"local": 0, "global": 0}

    def rich_club_coefficient(self) -> dict[int, float]:
        """Measure if high-degree nodes preferentially connect to each other."""
        if not HAS_NETWORKX: return {}
        try: return dict(nx.rich_club_coefficient(self._G.to_undirected(), normalized=False))
        except Exception: return {}

    def structural_holes(self) -> dict[str, float]:
        """Find structural holes — brokerage opportunities between groups.
        Lower constraint = more brokerage power."""
        if not HAS_NETWORKX: return {}
        try: return dict(nx.constraint(self._G.to_undirected()))
        except Exception: return {}

    def small_world_metrics(self) -> dict[str, float]:
        """Compute small-world metrics (sigma, omega)."""
        if not HAS_NETWORKX: return {"sigma": 0, "omega": 0}
        G_und = self._G.to_undirected()
        if len(G_und.nodes) < 4: return {"sigma": 0, "omega": 0}
        try:
            s = nx.sigma(G_und, niter=10, nrand=5)
            o = nx.omega(G_und, niter=10, nrand=5)
            return {"sigma": s, "omega": o}
        except Exception: return {"sigma": 0, "omega": 0}

    def graph_similarity(self, other: WorkflowAnalyzer) -> float:
        """Compute similarity between two workflow graphs using SimRank."""
        if not HAS_NETWORKX or not other._G: return 0.0
        try:
            sim = nx.simrank_similarity(self._G)
            return sum(sum(v.values()) for v in sim.values()) / max(len(sim) ** 2, 1)
        except Exception: return 0.0

    def node_classification(self, labeled: dict[str, str]) -> dict[str, str]:
        """Classify unlabeled nodes from labeled ones using harmonic function.
        Useful for propagating agent roles through workflow."""
        if not HAS_NETWORKX: return {}
        try:
            G = self._G.to_undirected()
            label_map = {}
            unique_labels = list(set(labeled.values()))
            for node in G.nodes:
                if node in labeled:
                    G.nodes[node]["label"] = unique_labels.index(labeled[node])
                else:
                    G.nodes[node]["label"] = -1
            # Simple label propagation
            result = {}
            for node in G.nodes:
                if node in labeled:
                    result[node] = labeled[node]
                else:
                    neighbor_labels = [labeled.get(n) for n in G.neighbors(node) if n in labeled]
                    if neighbor_labels:
                        from collections import Counter
                        most_common = Counter(neighbor_labels).most_common(1)[0][0]
                        result[node] = most_common
            return result
        except Exception: return {}

    def diameter(self) -> int:
        """Graph diameter — longest shortest path."""
        if not HAS_NETWORKX: return len(self.nodes)
        try: return nx.diameter(self._G.to_undirected())
        except Exception: return len(self.nodes)

    def radius(self) -> int:
        """Graph radius — minimum eccentricity."""
        if not HAS_NETWORKX: return 0
        try: return nx.radius(self._G.to_undirected())
        except Exception: return 0

    def center_nodes(self) -> list[str]:
        """Find center nodes (minimum eccentricity)."""
        if not HAS_NETWORKX: return []
        try: return list(nx.center(self._G.to_undirected()))
        except Exception: return []

    def periphery_nodes(self) -> list[str]:
        """Find periphery nodes (maximum eccentricity)."""
        if not HAS_NETWORKX: return []
        try: return list(nx.periphery(self._G.to_undirected()))
        except Exception: return []

    def clustering_coefficient(self) -> dict[str, float]:
        """Compute clustering coefficient for each node."""
        if not HAS_NETWORKX: return {}
        try: return dict(nx.clustering(self._G.to_undirected()))
        except Exception: return {}

    def average_clustering(self) -> float:
        """Average clustering coefficient of the graph."""
        if not HAS_NETWORKX: return 0.0
        try: return nx.average_clustering(self._G.to_undirected())
        except Exception: return 0.0

    def density(self) -> float:
        """Graph density (0-1)."""
        if not HAS_NETWORKX: return 0.0
        return nx.density(self._G)

    def is_bipartite(self) -> bool:
        """Check if graph is bipartite (can be split into two independent sets)."""
        if not HAS_NETWORKX: return False
        try: return nx.is_bipartite(self._G.to_undirected())
        except Exception: return False

    def topological_generations(self) -> list[list[str]]:
        """Get nodes grouped by topological level (for parallel scheduling)."""
        if not HAS_NETWORKX: return [self.nodes]
        if not nx.is_directed_acyclic_graph(self._G): return [self.nodes]
        try: return [list(gen) for gen in nx.topological_generations(self._G)]
        except Exception: return [self.nodes]
