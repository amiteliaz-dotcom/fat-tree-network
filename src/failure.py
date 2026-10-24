"""Uniform link failure perturbation engine supporting Bernoulli and Exact models."""
from __future__ import annotations

import numpy as np
import networkx as nx
from src.models import FatTreeGraph, PerturbationResult, FailureMode, EdgeKey


class LinkFailureEngine:
    """Injects link failures into a FatTreeGraph under seeded determinism."""

    def __init__(self, seed: int | None = None) -> None:
        """Initializes the failure engine with an optional 64-bit non-negative seed.

        Args:
            seed: Non-negative integer random seed for reproducibility, or None.

        Raises:
            ValueError: If seed is a negative integer.
        """
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"Seed must be an integer, got {type(seed).__name__}.")
            if seed < 0:
                raise ValueError(f"Random seed must be non-negative, got {seed}.")

        self.seed: int | None = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def perturb(
        self,
        topology: FatTreeGraph,
        failure_rate: float,
        mode: FailureMode = FailureMode.BERNOULLI,
        seed: int | None = None,
    ) -> PerturbationResult:
        """Applies link failures to create an active subgraph without mutating the baseline.

        Args:
            topology: Baseline unperturbed FatTreeGraph.
            failure_rate: Failure probability or proportion in [0.0, 1.0].
            mode: FailureMode (BERNOULLI or EXACT).
            seed: Per-call override seed. If None, uses engine's active RNG stream.

        Returns:
            PerturbationResult containing the active graph and failed edge set.

        Raises:
            ValueError: If failure_rate < 0.0 or failure_rate > 1.0.
            ValueError: If seed is negative.
        """
        if not isinstance(failure_rate, (int, float)):
            raise ValueError(f"failure_rate must be a float or int, got {type(failure_rate).__name__}.")
        if failure_rate < 0.0 or failure_rate > 1.0:
            raise ValueError(f"Link failure rate must be in range [0.0, 1.0], got {failure_rate}.")

        # Determine active RNG
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"Seed must be an integer, got {type(seed).__name__}.")
            if seed < 0:
                raise ValueError(f"Random seed must be non-negative, got {seed}.")
            rng = np.random.default_rng(seed)
            effective_seed = seed
        else:
            rng = self.rng
            effective_seed = self.seed

        sorted_edges = topology.sorted_edges
        total_edges = len(sorted_edges)
        failed_edges: set[EdgeKey] = set()

        if failure_rate == 0.0 or total_edges == 0:
            failed_edges = set()
        elif failure_rate == 1.0:
            failed_edges = set(sorted_edges)
        elif mode == FailureMode.BERNOULLI:
            # Each edge independently fails with probability p
            random_draws = rng.random(total_edges)
            for idx, draw in enumerate(random_draws):
                if draw < failure_rate:
                    failed_edges.add(sorted_edges[idx])
        elif mode == FailureMode.EXACT:
            # Exactly round(p * total_edges) links are removed uniformly without replacement
            num_failed = int(round(failure_rate * total_edges))
            num_failed = min(max(num_failed, 0), total_edges)
            if num_failed == total_edges:
                failed_edges = set(sorted_edges)
            elif num_failed > 0:
                sampled_indices = rng.choice(total_edges, size=num_failed, replace=False)
                for idx in sampled_indices:
                    failed_edges.add(sorted_edges[idx])
        else:
            raise ValueError(f"Unsupported failure mode: {mode}")

        # Construct non-mutating active subgraph
        active_graph: nx.Graph = topology.graph.copy()
        active_graph.remove_edges_from(failed_edges)

        return PerturbationResult(
            baseline=topology,
            active_graph=active_graph,
            failed_edges=failed_edges,
            failure_rate=float(failure_rate),
            failure_mode=mode,
            seed=effective_seed,
        )

