"""Domain models, type definitions, and immutable data contracts."""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypeAlias
import networkx as nx

EdgeKey: TypeAlias = tuple[str, str]  # Canonical lexicographically sorted: (u, v) where u < v


class NodeType(enum.StrEnum):
    """Canonical node type discriminator."""
    CORE = "core"
    AGGREGATION = "agg"
    EDGE = "edge"
    HOST = "host"


class TierType(enum.StrEnum):
    """Hierarchical link tier discriminator."""
    HOST_EDGE = "host-edge"
    EDGE_AGG = "edge-agg"
    AGG_CORE = "agg-core"


class FailureMode(enum.StrEnum):
    """Link failure sampling model."""
    BERNOULLI = "bernoulli"
    EXACT = "exact"


class ASPLMode(enum.StrEnum):
    """Average Shortest Path Length calculation formulation."""
    REACHABLE = "reachable"
    HARMONIC = "harmonic"


class MetricType(enum.StrEnum):
    """Available metric evaluations."""
    ASPL = "aspl"
    REACHABILITY = "reachability"
    REDUNDANCY = "redundancy"
    MULTI_FAILURE = "multi-failure"
    ALL = "all"


class SweepType(enum.StrEnum):
    """Parameter sweep modes."""
    NONE = "none"
    FAILURE_RATE = "failure-rate"
    PORT_COUNT = "port-count"


class OutputFormat(enum.StrEnum):
    """Serialization and export formats."""
    JSON = "json"
    CSV = "csv"
    PLOT = "plot"
    ALL = "all"


@dataclass(frozen=True)
class FatTreeGraph:
    """Immutable representation of an unperturbed 3-tier Fat-Tree topology."""
    k: int
    graph: nx.Graph
    hosts: list[str]
    switches: list[str]
    core_switches: list[str]
    agg_switches: list[str]
    edge_switches: list[str]
    sorted_edges: list[EdgeKey]

    @property
    def total_nodes(self) -> int:
        """Total node count in graph."""
        return self.graph.number_of_nodes()

    @property
    def total_links(self) -> int:
        """Total bidirectional link count in graph."""
        return self.graph.number_of_edges()

    @property
    def num_hosts(self) -> int:
        """Total terminal host count."""
        return len(self.hosts)

    @property
    def num_switches(self) -> int:
        """Total switch count across all tiers."""
        return len(self.switches)


@dataclass(frozen=True)
class PerturbationResult:
    """Represents a perturbed network state with failed edges removed."""
    baseline: FatTreeGraph
    active_graph: nx.Graph
    failed_edges: set[EdgeKey]
    failure_rate: float
    failure_mode: FailureMode
    seed: int | None

    @property
    def failed_link_count(self) -> int:
        """Number of failed links."""
        return len(self.failed_edges)

    @property
    def active_link_count(self) -> int:
        """Number of remaining active operational links."""
        return self.active_graph.number_of_edges()


@dataclass(frozen=True)
class SimulationMetrics:
    """Comprehensive performance and topological metrics for a single simulation run."""
    k: int
    failure_rate: float
    failure_mode: FailureMode
    seed: int | None
    total_host_pairs: int
    reachable_host_pairs: int
    reachability_ratio: float
    disconnection_ratio: float
    aspl_reachable: float
    aspl_harmonic: float
    mean_redundancy: float
    switches_multi_failure_count: int
    switches_multi_failure_ratio: float
    expected_multi_failure_count: float


@dataclass(frozen=True)
class SweepPointResult:
    """Aggregated statistical outcome across Monte Carlo trials for a single sweep point."""
    k: int
    parameter_value: float  # e.g., failure_rate or k
    trials: int
    reachability_mean: float
    reachability_std: float
    aspl_reachable_mean: float
    aspl_reachable_std: float
    aspl_harmonic_mean: float
    aspl_harmonic_std: float
    redundancy_mean: float
    redundancy_std: float
    multi_failure_mean: float
    multi_failure_std: float

