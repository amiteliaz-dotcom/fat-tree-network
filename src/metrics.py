"""Analytical metric calculations for Fat-Tree topologies under failure."""
from __future__ import annotations

from collections import deque
import networkx as nx
from src.models import (
    FatTreeGraph,
    PerturbationResult,
    SimulationMetrics,
    ASPLMode,
    EdgeKey,
)


class MetricAnalyzer:
    """Computes ASPL, reachability, redundancy, and switch multi-failure metrics."""

    @staticmethod
    def compute_all_metrics(
        topology: FatTreeGraph,
        perturbation: PerturbationResult,
        aspl_mode: ASPLMode = ASPLMode.REACHABLE,
    ) -> SimulationMetrics:
        """Executes full metric suite on the perturbed network state.

        Args:
            topology: Baseline unperturbed FatTreeGraph.
            perturbation: PerturbationResult containing active subgraph and failed edge set.
            aspl_mode: Formulation for ASPL (REACHABLE or HARMONIC).

        Returns:
            SimulationMetrics dataclass capturing all computed metrics.
        """
        hosts = topology.hosts
        active_graph = perturbation.active_graph
        k = topology.k
        p = perturbation.failure_rate

        # 1. Reachability & ASPL
        (
            total_pairs,
            reachable_pairs,
            reach_ratio,
            disc_ratio,
            aspl_reach,
            aspl_harm,
        ) = MetricAnalyzer.compute_reachability_and_aspl(hosts, active_graph)

        # 2. Shortest Path Redundancy
        mean_redundancy = MetricAnalyzer.compute_shortest_path_multiplicity(hosts, active_graph)

        # 3. Switch Multi-Failure Analysis
        (
            multi_fail_count,
            multi_fail_ratio,
            expected_multi_count,
        ) = MetricAnalyzer.compute_multi_failure_switches(
            topology.switches,
            topology.graph,
            perturbation.failed_edges,
            k,
            p,
        )

        return SimulationMetrics(
            k=k,
            failure_rate=p,
            failure_mode=perturbation.failure_mode,
            seed=perturbation.seed,
            total_host_pairs=total_pairs,
            reachable_host_pairs=reachable_pairs,
            reachability_ratio=reach_ratio,
            disconnection_ratio=disc_ratio,
            aspl_reachable=aspl_reach,
            aspl_harmonic=aspl_harm,
            mean_redundancy=mean_redundancy,
            switches_multi_failure_count=multi_fail_count,
            switches_multi_failure_ratio=multi_fail_ratio,
            expected_multi_failure_count=expected_multi_count,
        )

    @staticmethod
    def compute_reachability_and_aspl(
        hosts: list[str],
        active_graph: nx.Graph,
    ) -> tuple[int, int, float, float, float, float]:
        """Runs unweighted BFS from each host to compute reachability and ASPL (reachable & harmonic).

        Args:
            hosts: List of host node identifiers.
            active_graph: Active NetworkX graph view.

        Returns:
            tuple of:
            (total_pairs, reachable_pairs, reachability_ratio, disconnection_ratio, aspl_reach, aspl_harm)
        """
        num_hosts = len(hosts)
        total_pairs = (num_hosts * (num_hosts - 1)) // 2

        if total_pairs == 0:
            return 0, 0, 1.0, 0.0, 0.0, 0.0

        reachable_pairs = 0
        total_distance = 0
        sum_harmonic_efficiency = 0.0

        # Run BFS from each host u
        for i in range(num_hosts):
            src = hosts[i]
            # Fast single-source unweighted BFS distances
            if src not in active_graph:
                continue

            dist: dict[str, int] = {src: 0}
            queue = deque([src])

            while queue:
                curr = queue.popleft()
                d_curr = dist[curr]
                for neighbor in active_graph.neighbors(curr):
                    if neighbor not in dist:
                        dist[neighbor] = d_curr + 1
                        queue.append(neighbor)

            # Accumulate metrics for pairs (src, dst) where j > i
            for j in range(i + 1, num_hosts):
                dst = hosts[j]
                if dst in dist:
                    d = dist[dst]
                    reachable_pairs += 1
                    total_distance += d
                    sum_harmonic_efficiency += 1.0 / d

        reachability_ratio = reachable_pairs / total_pairs
        disconnection_ratio = 1.0 - reachability_ratio

        aspl_reachable = (total_distance / reachable_pairs) if reachable_pairs > 0 else 0.0

        mean_harmonic_efficiency = sum_harmonic_efficiency / total_pairs
        if mean_harmonic_efficiency > 0.0:
            aspl_harmonic = 1.0 / mean_harmonic_efficiency
        else:
            aspl_harmonic = float('inf')

        return (
            total_pairs,
            reachable_pairs,
            reachability_ratio,
            disconnection_ratio,
            aspl_reachable,
            aspl_harmonic,
        )

    @staticmethod
    def compute_shortest_path_multiplicity(
        hosts: list[str],
        active_graph: nx.Graph,
    ) -> float:
        """Computes mean shortest path redundancy (ECMP path count) across all reachable host pairs.

        Args:
            hosts: List of host node identifiers.
            active_graph: Active NetworkX graph view.

        Returns:
            Average number of equal-cost shortest paths across reachable host pairs.
        """
        num_hosts = len(hosts)
        if num_hosts <= 1:
            return 0.0

        reachable_pairs = 0
        total_shortest_paths = 0

        for i in range(num_hosts):
            src = hosts[i]
            if src not in active_graph:
                continue

            # BFS tracking both shortest distance and number of shortest paths
            dist: dict[str, int] = {src: 0}
            path_count: dict[str, int] = {src: 1}
            queue = deque([src])

            while queue:
                curr = queue.popleft()
                d_curr = dist[curr]
                c_curr = path_count[curr]

                for neighbor in active_graph.neighbors(curr):
                    if neighbor not in dist:
                        dist[neighbor] = d_curr + 1
                        path_count[neighbor] = c_curr
                        queue.append(neighbor)
                    elif dist[neighbor] == d_curr + 1:
                        path_count[neighbor] += c_curr

            for j in range(i + 1, num_hosts):
                dst = hosts[j]
                if dst in path_count:
                    reachable_pairs += 1
                    total_shortest_paths += path_count[dst]

        if reachable_pairs == 0:
            return 0.0

        return total_shortest_paths / reachable_pairs

    @staticmethod
    def compute_multi_failure_switches(
        switches: list[str],
        base_graph: nx.Graph,
        failed_edges: set[EdgeKey],
        k: int,
        failure_rate: float,
    ) -> tuple[int, float, float]:
        """Identifies switches with >= 2 failed incident links and compares against binomial baseline.

        Args:
            switches: List of switch node identifiers.
            base_graph: Baseline unperturbed graph.
            failed_edges: Set of failed canonical edge keys (u, v) where u < v.
            k: Switch port radix.
            failure_rate: Failure probability p.

        Returns:
            tuple of (multi_failure_count, multi_failure_ratio, expected_multi_failure_count)
        """
        total_switches = len(switches)
        if total_switches == 0:
            return 0, 0.0, 0.0

        multi_fail_count = 0

        for s in switches:
            failed_incident_count = 0
            for neighbor in base_graph.neighbors(s):
                edge_key: EdgeKey = (s, neighbor) if s < neighbor else (neighbor, s)
                if edge_key in failed_edges:
                    failed_incident_count += 1
                    if failed_incident_count >= 2:
                        multi_fail_count += 1
                        break

        multi_fail_ratio = multi_fail_count / total_switches

        # Binomial theoretical expected value: P(X >= 2) = 1 - (1-p)^k - k*p*(1-p)^(k-1)
        p = float(failure_rate)
        if p <= 0.0:
            prob_multi = 0.0
        elif p >= 1.0:
            prob_multi = 1.0
        else:
            prob_multi = 1.0 - ((1.0 - p) ** k) - (k * p * ((1.0 - p) ** (k - 1)))
            prob_multi = max(0.0, min(1.0, prob_multi))

        expected_count = total_switches * prob_multi
        return multi_fail_count, multi_fail_ratio, expected_count

    @staticmethod
    def theoretical_aspl_0(k: int) -> float:
        """Returns closed-form baseline ASPL for radix k at p=0: 2(3k^3 - k^2 - 2k - 4)/(k^3 - 4).

        Args:
            k: Switch port count (even integer >= 2).

        Returns:
            Theoretical ASPL at failure rate p=0.0.
        """
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be an even integer >= 2, got {k}")

        half_k = k // 2
        num_hosts = k * (half_k ** 2)
        total_pairs = (num_hosts * (num_hosts - 1)) // 2

        # 1. Intra-edge pairs (dist = 2)
        intra_edge_pairs = k * half_k * ((half_k * (half_k - 1)) // 2)

        # 2. Intra-pod inter-edge pairs (dist = 4)
        intra_pod_inter_edge_pairs = k * ((half_k * (half_k - 1)) // 2) * (half_k ** 2)

        # 3. Inter-pod pairs (dist = 6)
        inter_pod_pairs = ((k * (k - 1)) // 2) * ((half_k ** 2) ** 2)

        total_hops = (
            (2 * intra_edge_pairs)
            + (4 * intra_pod_inter_edge_pairs)
            + (6 * inter_pod_pairs)
        )

        return total_hops / total_pairs

    @staticmethod
    def theoretical_redundancy_0(k: int) -> float:
        """Returns closed-form baseline path redundancy for radix k at p=0.

        Args:
            k: Switch port count (even integer >= 2).

        Returns:
            Theoretical mean number of shortest paths at p=0.0.
        """
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be an even integer >= 2, got {k}")

        half_k = k // 2
        num_hosts = k * (half_k ** 2)
        total_pairs = (num_hosts * (num_hosts - 1)) // 2

        intra_edge_pairs = k * half_k * ((half_k * (half_k - 1)) // 2)
        intra_pod_inter_edge_pairs = k * ((half_k * (half_k - 1)) // 2) * (half_k ** 2)
        inter_pod_pairs = ((k * (k - 1)) // 2) * ((half_k ** 2) ** 2)

        # Path counts: 1 for intra-edge, k/2 for intra-pod, (k/2)^2 for inter-pod
        total_paths = (
            (1 * intra_edge_pairs)
            + (half_k * intra_pod_inter_edge_pairs)
            + ((half_k ** 2) * inter_pod_pairs)
        )

        return total_paths / total_pairs

