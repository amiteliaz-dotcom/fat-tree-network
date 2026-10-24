"""Unit tests for FatTreeTopology generation and structural invariants (AC-1, AC-2, AC-3, EC-1, EC-2)."""
from __future__ import annotations

import pytest
from src.topology import FatTreeTopology
from src.models import FatTreeGraph, NodeType, TierType


def test_invalid_radix_rejection():
    """Verify EC-1: Odd radix or k < 2 is rejected with ValueError."""
    # Radix < 2
    with pytest.raises(ValueError, match="Switch radix k must be >= 2"):
        FatTreeTopology(0)
    with pytest.raises(ValueError, match="Switch radix k must be >= 2"):
        FatTreeTopology(-4)

    # Odd radix
    with pytest.raises(ValueError, match="Switch radix k must be an even integer"):
        FatTreeTopology(3)
    with pytest.raises(ValueError, match="Switch radix k must be an even integer"):
        FatTreeTopology(5)

    # Non-integer
    with pytest.raises(ValueError, match="Switch radix k must be an integer"):
        FatTreeTopology("4")  # type: ignore
    with pytest.raises(ValueError, match="Switch radix k must be an integer"):
        FatTreeTopology(True)  # type: ignore


def test_boundary_radix_k2(topo_k2: FatTreeGraph):
    """Verify EC-2 & AC-1: Boundary radix k=2 generates valid structure."""
    ft = topo_k2
    assert ft.k == 2
    # k=2: core=1, agg=2, edge=2 (5 switches), hosts=2, links=6
    assert len(ft.core_switches) == 1
    assert len(ft.agg_switches) == 2
    assert len(ft.edge_switches) == 2
    assert ft.num_switches == 5
    assert ft.num_hosts == 2
    assert ft.total_links == 6
    assert ft.total_nodes == 7


@pytest.mark.parametrize(
    "k, expected_core, expected_agg, expected_edge, expected_switches, expected_hosts, expected_links",
    [
        (2, 1, 2, 2, 5, 2, 6),
        (4, 4, 8, 8, 20, 16, 48),
        (6, 9, 18, 18, 45, 54, 162),
        (8, 16, 32, 32, 80, 128, 384),
    ],
)
def test_structural_invariants_ac1(
    k: int,
    expected_core: int,
    expected_agg: int,
    expected_edge: int,
    expected_switches: int,
    expected_hosts: int,
    expected_links: int,
):
    """Verify AC-1: Structural formulas hold across all target radices."""
    ft = FatTreeTopology(k).build()
    assert len(ft.core_switches) == expected_core
    assert len(ft.agg_switches) == expected_agg
    assert len(ft.edge_switches) == expected_edge
    assert ft.num_switches == expected_switches
    assert ft.num_hosts == expected_hosts
    assert ft.total_links == expected_links

    # Formula checks
    half_k = k // 2
    assert ft.num_switches == (5 * k * k) // 4
    assert ft.num_hosts == (k ** 3) // 4
    assert ft.total_links == (3 * (k ** 3)) // 4


def test_core_bipartite_inter_pod_connectivity_ac2(topo_k4: FatTreeGraph):
    """Verify AC-2: Core switch bipartite inter-pod connectivity."""
    ft = topo_k4
    g = ft.graph
    half_k = 2

    for j in range(half_k):
        for i in range(half_k):
            core_name = f"core_{j}_{i}"
            assert core_name in g
            neighbors = list(g.neighbors(core_name))
            assert len(neighbors) == ft.k  # Degree is 4

            # Must connect to aggregation switch agg_{p}_{j} in each pod p in {0, 1, 2, 3}
            expected_neighbors = {f"agg_{p}_{j}" for p in range(ft.k)}
            assert set(neighbors) == expected_neighbors


def test_switch_radix_regularity_and_degrees_ac3(topo_k4: FatTreeGraph):
    """Verify AC-3: Switch radix regularity and host degree invariants."""
    ft = topo_k4
    g = ft.graph
    k = ft.k

    # All switches degree k
    for s in ft.switches:
        assert g.degree(s) == k

    # All hosts degree 1
    for h in ft.hosts:
        assert g.degree(h) == 1

    # Tier-specific link types
    for u, v, data in g.edges(data=True):
        tier_type = data["tier_type"]
        status = data["status"]
        assert status == "active"
        if tier_type == TierType.HOST_EDGE:
            # One must be host, other edge
            types = {g.nodes[u]["type"], g.nodes[v]["type"]}
            assert types == {NodeType.HOST, NodeType.EDGE}
        elif tier_type == TierType.EDGE_AGG:
            types = {g.nodes[u]["type"], g.nodes[v]["type"]}
            assert types == {NodeType.EDGE, NodeType.AGGREGATION}
        elif tier_type == TierType.AGG_CORE:
            types = {g.nodes[u]["type"], g.nodes[v]["type"]}
            assert types == {NodeType.AGGREGATION, NodeType.CORE}
        else:
            pytest.fail(f"Unknown tier_type: {tier_type}")

