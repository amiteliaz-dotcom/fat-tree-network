"""Comprehensive mathematical invariant property tests across Fat-Tree radices."""
from __future__ import annotations

import math
import pytest
from src.topology import FatTreeTopology
from src.metrics import MetricAnalyzer
from src.failure import LinkFailureEngine
from src.models import FailureMode


@pytest.mark.parametrize("k", [2, 4, 6])
def test_baseline_theoretical_vs_empirical_aspl(k: int):
    """Verify that empirical unperturbed ASPL at p=0 matches theoretical closed form for k=2, 4, 6."""
    ft = FatTreeTopology(k).build()
    engine = LinkFailureEngine(seed=1)
    pert = engine.perturb(ft, failure_rate=0.0)
    metrics = MetricAnalyzer.compute_all_metrics(ft, pert)

    theoretical_aspl = MetricAnalyzer.theoretical_aspl_0(k)
    assert math.isclose(metrics.aspl_reachable, theoretical_aspl, rel_tol=1e-7)

    theoretical_red = MetricAnalyzer.theoretical_redundancy_0(k)
    assert math.isclose(metrics.mean_redundancy, theoretical_red, rel_tol=1e-7)


def test_bisection_link_count():
    """Verify that bisection cut links between any pod and the core equals (k^2 / 4)."""
    k = 4
    ft = FatTreeTopology(k).build()
    g = ft.graph

    # For pod 0, count links between pod 0 agg switches and core
    pod0_agg = [f"agg_0_{a}" for a in range(k // 2)]
    cut_links = 0
    for agg in pod0_agg:
        for neighbor in g.neighbors(agg):
            if g.nodes[neighbor]["type"] == "core":
                cut_links += 1

    # In pod 0, (k/2) agg switches each have (k/2) uplinks to core => (k/2)^2 = 4 links
    assert cut_links == (k // 2) ** 2

