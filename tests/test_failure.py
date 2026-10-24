"""Unit tests for LinkFailureEngine (AC-6, AC-7, EC-3, EC-4, EC-5, EC-7)."""
from __future__ import annotations

import pytest
from src.failure import LinkFailureEngine
from src.models import FatTreeGraph, FailureMode


def test_invalid_failure_rate_ec3(topo_k4: FatTreeGraph):
    """Verify EC-3: Out-of-bounds failure rates raise ValueError."""
    engine = LinkFailureEngine(seed=42)
    with pytest.raises(ValueError, match="Link failure rate must be in range"):
        engine.perturb(topo_k4, failure_rate=-0.01)
    with pytest.raises(ValueError, match="Link failure rate must be in range"):
        engine.perturb(topo_k4, failure_rate=1.01)
    with pytest.raises(ValueError, match="Link failure rate must be in range"):
        engine.perturb(topo_k4, failure_rate=50.0)


def test_invalid_seed_ec7():
    """Verify EC-7: Negative seeds raise ValueError."""
    with pytest.raises(ValueError, match="Random seed must be non-negative"):
        LinkFailureEngine(seed=-1)
    with pytest.raises(ValueError, match="Random seed must be non-negative"):
        LinkFailureEngine(seed=-999)


def test_deterministic_reproducibility_ac6(topo_k4: FatTreeGraph):
    """Verify AC-6: Deterministic failure injection reproducibility via seed."""
    engine1 = LinkFailureEngine(seed=42)
    engine2 = LinkFailureEngine(seed=42)

    res1 = engine1.perturb(topo_k4, failure_rate=0.2, mode=FailureMode.BERNOULLI)
    res2 = engine2.perturb(topo_k4, failure_rate=0.2, mode=FailureMode.BERNOULLI)

    assert res1.failed_edges == res2.failed_edges
    assert res1.failed_link_count == res2.failed_link_count
    assert res1.active_link_count == res2.active_link_count
    assert res1.active_graph.number_of_edges() == res2.active_graph.number_of_edges()

    # Exact mode determinism
    res_exact1 = engine1.perturb(topo_k4, failure_rate=0.25, mode=FailureMode.EXACT, seed=123)
    res_exact2 = engine2.perturb(topo_k4, failure_rate=0.25, mode=FailureMode.EXACT, seed=123)
    assert res_exact1.failed_edges == res_exact2.failed_edges
    # 25% of 48 links is exactly 12
    assert res_exact1.failed_link_count == 12


def test_boundary_failure_conditions_ac7(topo_k4: FatTreeGraph):
    """Verify AC-7: Boundary conditions p=0.0 (zero failures) and p=1.0 (all links failed)."""
    engine = LinkFailureEngine(seed=100)

    # p = 0.0
    res_zero = engine.perturb(topo_k4, failure_rate=0.0)
    assert res_zero.failed_link_count == 0
    assert len(res_zero.failed_edges) == 0
    assert res_zero.active_link_count == topo_k4.total_links
    assert res_zero.active_graph.number_of_nodes() == topo_k4.total_nodes

    # p = 1.0
    res_one = engine.perturb(topo_k4, failure_rate=1.0)
    assert res_one.failed_link_count == topo_k4.total_links
    assert res_one.failed_edges == set(topo_k4.sorted_edges)
    assert res_one.active_link_count == 0
    # Original nodes are preserved intact
    assert res_one.active_graph.number_of_nodes() == topo_k4.total_nodes


def test_non_mutating_invariant(topo_k4: FatTreeGraph):
    """Verify that perturbation does not mutate the baseline graph."""
    original_edge_count = topo_k4.graph.number_of_edges()
    original_node_count = topo_k4.graph.number_of_nodes()

    engine = LinkFailureEngine(seed=777)
    res = engine.perturb(topo_k4, failure_rate=0.5)

    # Baseline remains unaffected
    assert topo_k4.graph.number_of_edges() == original_edge_count
    assert topo_k4.graph.number_of_nodes() == original_node_count
    assert len(topo_k4.sorted_edges) == original_edge_count
    # Active graph has reduced edge count
    assert res.active_graph.number_of_edges() < original_edge_count

