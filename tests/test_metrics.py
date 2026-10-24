"""Unit tests for MetricAnalyzer (AC-4, AC-5, AC-8, AC-9, AC-10)."""
from __future__ import annotations

import math
import pytest
from src.topology import FatTreeTopology
from src.failure import LinkFailureEngine
from src.metrics import MetricAnalyzer
from src.models import FatTreeGraph, FailureMode


def test_baseline_aspl_ac4(topo_k4: FatTreeGraph):
    """Verify AC-4: Baseline unperturbed ASPL at p=0.0 matches closed-form 82/15."""
    engine = LinkFailureEngine(seed=42)
    pert = engine.perturb(topo_k4, failure_rate=0.0)
    metrics = MetricAnalyzer.compute_all_metrics(topo_k4, pert)

    expected_aspl = 82.0 / 15.0  # 5.4666666...
    expected_harmonic = 5.0  # Harmonic mean of distances: 120 / (8*1/2 + 16*1/4 + 96*1/6) = 120 / 24 = 5.0
    assert math.isclose(metrics.aspl_reachable, expected_aspl, rel_tol=1e-7)
    assert math.isclose(metrics.aspl_harmonic, expected_harmonic, rel_tol=1e-7)
    assert metrics.total_host_pairs == 120
    assert metrics.reachable_host_pairs == 120
    assert metrics.reachability_ratio == 1.0
    assert metrics.disconnection_ratio == 0.0

    # Also verify formula implementation
    assert math.isclose(MetricAnalyzer.theoretical_aspl_0(4), expected_aspl, rel_tol=1e-7)


def test_baseline_redundancy_ac5(topo_k4: FatTreeGraph):
    """Verify AC-5: Baseline path redundancy at p=0.0 matches closed-form 53/15."""
    engine = LinkFailureEngine(seed=42)
    pert = engine.perturb(topo_k4, failure_rate=0.0)
    metrics = MetricAnalyzer.compute_all_metrics(topo_k4, pert)

    expected_redundancy = 53.0 / 15.0  # 3.533333...
    assert math.isclose(metrics.mean_redundancy, expected_redundancy, rel_tol=1e-7)
    assert math.isclose(MetricAnalyzer.theoretical_redundancy_0(4), expected_redundancy, rel_tol=1e-7)


def test_reachability_disconnection_sum_ac8(topo_k4: FatTreeGraph):
    """Verify AC-8: Reachability ratio + Disconnection ratio == 1.0."""
    engine = LinkFailureEngine(seed=999)
    for p in [0.0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]:
        pert = engine.perturb(topo_k4, failure_rate=p)
        metrics = MetricAnalyzer.compute_all_metrics(topo_k4, pert)
        assert math.isclose(metrics.reachability_ratio + metrics.disconnection_ratio, 1.0, abs_tol=1e-9)
        assert 0.0 <= metrics.reachability_ratio <= 1.0
        assert 0.0 <= metrics.disconnection_ratio <= 1.0


def test_aspl_partitioning_reachable_vs_harmonic_ac9(topo_k4: FatTreeGraph):
    """Verify AC-9: ASPL behavior under network partitioning (Reachable vs Harmonic)."""
    # Seed 101 with p=0.25 induces network partitions
    engine = LinkFailureEngine(seed=101)
    pert = engine.perturb(topo_k4, failure_rate=0.25)
    metrics = MetricAnalyzer.compute_all_metrics(topo_k4, pert)

    assert metrics.reachable_host_pairs < metrics.total_host_pairs
    assert metrics.reachability_ratio < 1.0
    # Harmonic ASPL accounts for unreachable pairs (1/inf = 0), so aspl_harmonic > aspl_reachable
    assert metrics.aspl_harmonic > metrics.aspl_reachable


def test_switch_multi_failure_binomial_ac10():
    """Verify AC-10: Multi-failure switch counts against theoretical binomial formula."""
    ft = FatTreeTopology(4).build()
    p = 0.01

    _, _, expected_count_k4 = MetricAnalyzer.compute_multi_failure_switches(
        ft.switches, ft.graph, set(), ft.k, p
    )
    # k=4: 20 switches * [1 - (0.99)^4 - 4(0.01)(0.99)^3]
    manual_expected = 20 * (1.0 - (0.99 ** 4) - (4 * 0.01 * (0.99 ** 3)))
    assert math.isclose(expected_count_k4, manual_expected, rel_tol=1e-7)

    # Radix 8
    ft8 = FatTreeTopology(8).build()
    _, _, expected_count_k8 = MetricAnalyzer.compute_multi_failure_switches(
        ft8.switches, ft8.graph, set(), ft8.k, p
    )
    manual_expected_k8 = 80 * (1.0 - (0.99 ** 8) - (8 * 0.01 * (0.99 ** 7)))
    assert math.isclose(expected_count_k8, manual_expected_k8, rel_tol=1e-7)
