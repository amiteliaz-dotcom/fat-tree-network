"""Parametric 3-tier Fat-Tree network topology generator."""
from __future__ import annotations

import networkx as nx
from src.models import FatTreeGraph, NodeType, TierType, EdgeKey


class FatTreeTopology:
    """Constructs and validates canonical 3-tier Fat-Tree graphs for radix k."""

    def __init__(self, k: int) -> None:
        """Initialize and validate switch radix k.

        Args:
            k: Switch port count (must be an even integer >= 2).

        Raises:
            ValueError: If k is not an integer, k < 2, or k is odd.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(f"Switch radix k must be an integer, got {type(k).__name__}.")
        if k < 2:
            raise ValueError(f"Switch radix k must be >= 2, got {k}.")
        if k % 2 != 0:
            raise ValueError(f"Switch radix k must be an even integer, got {k}.")

        self.k: int = k
        self.half_k: int = k // 2

    def build(self) -> FatTreeGraph:
        """Constructs the full unperturbed NetworkX graph G_0 and returns a FatTreeGraph dataclass.

        Returns:
            FatTreeGraph encapsulating G_0, categorized node lists, and sorted edge keys.
        """
        k = self.k
        half_k = self.half_k
        g = nx.Graph()

        core_switches: list[str] = []
        agg_switches: list[str] = []
        edge_switches: list[str] = []
        hosts: list[str] = []

        # 1. Instantiate Core Switches: (k/2)^2 switches arranged in k/2 groups of k/2 switches
        for j in range(half_k):
            for i in range(half_k):
                name = f"core_{j}_{i}"
                core_switches.append(name)
                g.add_node(name, type=NodeType.CORE, pod=None, tier=3)

        # 2. Instantiate Pod Components: k pods, each with k/2 agg, k/2 edge, and (k/2)^2 hosts
        for p in range(k):
            # Aggregation tier in pod p
            for a in range(half_k):
                name = f"agg_{p}_{a}"
                agg_switches.append(name)
                g.add_node(name, type=NodeType.AGGREGATION, pod=p, tier=2)

            # Edge tier in pod p
            for e in range(half_k):
                name = f"edge_{p}_{e}"
                edge_switches.append(name)
                g.add_node(name, type=NodeType.EDGE, pod=p, tier=1)

                # Host tier attached to edge switch e
                for h in range(half_k):
                    h_name = f"host_{p}_{e}_{h}"
                    hosts.append(h_name)
                    g.add_node(h_name, type=NodeType.HOST, pod=p, tier=0)

        # 3. Wire Host-to-Edge Links (k^3 / 4 links)
        for p in range(k):
            for e in range(half_k):
                edge_name = f"edge_{p}_{e}"
                for h in range(half_k):
                    host_name = f"host_{p}_{e}_{h}"
                    g.add_edge(host_name, edge_name, tier_type=TierType.HOST_EDGE, status="active")

        # 4. Wire Edge-to-Aggregation Links: Complete bipartite K_{k/2, k/2} within each pod (k^3 / 4 links)
        for p in range(k):
            for e in range(half_k):
                edge_name = f"edge_{p}_{e}"
                for a in range(half_k):
                    agg_name = f"agg_{p}_{a}"
                    g.add_edge(edge_name, agg_name, tier_type=TierType.EDGE_AGG, status="active")

        # 5. Wire Aggregation-to-Core Links (k^3 / 4 links)
        # In pod p, the i-th port of aggregation switch a connects to core switch (group a, index i)
        for p in range(k):
            for a in range(half_k):
                agg_name = f"agg_{p}_{a}"
                for i in range(half_k):
                    core_name = f"core_{a}_{i}"
                    g.add_edge(agg_name, core_name, tier_type=TierType.AGG_CORE, status="active")

        switches = core_switches + agg_switches + edge_switches

        # Build canonical sorted edge keys: (u, v) where u < v, sorted lexicographically
        raw_edges: set[EdgeKey] = set()
        for u, v in g.edges():
            u_str, v_str = str(u), str(v)
            edge_key: EdgeKey = (u_str, v_str) if u_str < v_str else (v_str, u_str)
            raw_edges.add(edge_key)

        sorted_edges = sorted(raw_edges)

        ft = FatTreeGraph(
            k=k,
            graph=g,
            hosts=hosts,
            switches=switches,
            core_switches=core_switches,
            agg_switches=agg_switches,
            edge_switches=edge_switches,
            sorted_edges=sorted_edges,
        )

        # Audit invariants immediately
        self.verify_invariants(ft)
        return ft

    def verify_invariants(self, ft: FatTreeGraph) -> None:
        """Verifies structural and degree invariants (MI-1, MI-2, MI-3, MI-4).

        Raises:
            AssertionError: If any conservation law or regularity invariant is violated.
        """
        k = ft.k
        half_k = k // 2
        g = ft.graph

        # MI-1: Node Counts
        expected_core = (half_k) ** 2
        expected_agg = k * half_k
        expected_edge = k * half_k
        expected_switches = expected_core + expected_agg + expected_edge
        expected_hosts = k * (half_k ** 2)

        assert len(ft.core_switches) == expected_core, (
            f"Core count mismatch: expected {expected_core}, got {len(ft.core_switches)}"
        )
        assert len(ft.agg_switches) == expected_agg, (
            f"Aggregation count mismatch: expected {expected_agg}, got {len(ft.agg_switches)}"
        )
        assert len(ft.edge_switches) == expected_edge, (
            f"Edge count mismatch: expected {expected_edge}, got {len(ft.edge_switches)}"
        )
        assert ft.num_switches == expected_switches, (
            f"Total switch count mismatch: expected {expected_switches}, got {ft.num_switches}"
        )
        assert ft.num_hosts == expected_hosts, (
            f"Total host count mismatch: expected {expected_hosts}, got {ft.num_hosts}"
        )
        assert ft.total_nodes == expected_switches + expected_hosts, (
            f"Total node count mismatch: expected {expected_switches + expected_hosts}, got {ft.total_nodes}"
        )

        # MI-2: Link Counts
        expected_links_per_tier = k * (half_k ** 2)
        expected_total_links = 3 * expected_links_per_tier

        assert ft.total_links == expected_total_links, (
            f"Total link count mismatch: expected {expected_total_links}, got {ft.total_links}"
        )
        assert len(ft.sorted_edges) == expected_total_links, (
            f"Sorted edge list length mismatch: expected {expected_total_links}, got {len(ft.sorted_edges)}"
        )

        # Count links per tier
        tier_counts = {TierType.HOST_EDGE: 0, TierType.EDGE_AGG: 0, TierType.AGG_CORE: 0}
        for _, _, data in g.edges(data=True):
            tier_type = data.get("tier_type")
            if tier_type in tier_counts:
                tier_counts[tier_type] += 1

        for tier, count in tier_counts.items():
            assert count == expected_links_per_tier, (
                f"Tier {tier} link count mismatch: expected {expected_links_per_tier}, got {count}"
            )

        # MI-3: Degree Regularity
        for h in ft.hosts:
            deg = g.degree(h)
            assert deg == 1, f"Host {h} degree mismatch: expected 1, got {deg}"

        for s in ft.switches:
            deg = g.degree(s)
            assert deg == k, f"Switch {s} degree mismatch: expected {k}, got {deg}"

        # Check internal switch wiring breakdowns
        for c in ft.core_switches:
            # Must connect only to agg switches across all k pods
            neighbors = list(g.neighbors(c))
            pods_connected = {g.nodes[n]["pod"] for n in neighbors}
            assert len(neighbors) == k
            assert len(pods_connected) == k, f"Core {c} not connected to all {k} pods: {pods_connected}"
            for n in neighbors:
                assert g.nodes[n]["type"] == NodeType.AGGREGATION

        for a in ft.agg_switches:
            neighbors = list(g.neighbors(a))
            edge_neighbors = [n for n in neighbors if g.nodes[n]["type"] == NodeType.EDGE]
            core_neighbors = [n for n in neighbors if g.nodes[n]["type"] == NodeType.CORE]
            assert len(edge_neighbors) == half_k, (
                f"Agg {a} edge neighbors mismatch: expected {half_k}, got {len(edge_neighbors)}"
            )
            assert len(core_neighbors) == half_k, (
                f"Agg {a} core neighbors mismatch: expected {half_k}, got {len(core_neighbors)}"
            )

        for e in ft.edge_switches:
            neighbors = list(g.neighbors(e))
            host_neighbors = [n for n in neighbors if g.nodes[n]["type"] == NodeType.HOST]
            agg_neighbors = [n for n in neighbors if g.nodes[n]["type"] == NodeType.AGGREGATION]
            assert len(host_neighbors) == half_k, (
                f"Edge {e} host neighbors mismatch: expected {half_k}, got {len(host_neighbors)}"
            )
            assert len(agg_neighbors) == half_k, (
                f"Edge {e} agg neighbors mismatch: expected {half_k}, got {len(agg_neighbors)}"
            )

