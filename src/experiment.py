"""Monte Carlo experimentation engine and parameter sweep runner."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from src.models import (
    FatTreeGraph,
    SimulationMetrics,
    SweepPointResult,
    FailureMode,
    ASPLMode,
)
from src.topology import FatTreeTopology
from src.failure import LinkFailureEngine
from src.metrics import MetricAnalyzer


class ExperimentRunner:
    """Coordinates Monte Carlo trials, parameter sweeps, and file serialization."""

    def __init__(self, output_dir: Path | str = "results") -> None:
        """Initialize experiment runner with default output root directory.

        Args:
            output_dir: Root directory for result datasets and exports.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_simulation(
        self,
        k: int,
        failure_rate: float = 0.0,
        failure_mode: FailureMode = FailureMode.BERNOULLI,
        seed: int | None = None,
        aspl_mode: ASPLMode = ASPLMode.REACHABLE,
    ) -> SimulationMetrics:
        """Executes a single deterministic simulation trial.

        Args:
            k: Switch port radix.
            failure_rate: Link failure probability or fraction in [0.0, 1.0].
            failure_mode: FailureMode (BERNOULLI or EXACT).
            seed: Deterministic random seed, or None.
            aspl_mode: ASPL calculation mode (REACHABLE or HARMONIC).

        Returns:
            SimulationMetrics containing all observed metrics.
        """
        topo = FatTreeTopology(k).build()
        engine = LinkFailureEngine(seed=seed)
        pert = engine.perturb(topo, failure_rate=failure_rate, mode=failure_mode, seed=seed)
        return MetricAnalyzer.compute_all_metrics(topo, pert, aspl_mode=aspl_mode)

    def run_failure_rate_sweep(
        self,
        k: int,
        p_min: float,
        p_max: float,
        p_step: float,
        trials: int = 10,
        failure_mode: FailureMode = FailureMode.BERNOULLI,
        aspl_mode: ASPLMode = ASPLMode.REACHABLE,
        base_seed: int | None = None,
    ) -> list[SweepPointResult]:
        """Executes failure rate sweep with M independent trials per point.

        Args:
            k: Switch port radix.
            p_min: Minimum failure probability.
            p_max: Maximum failure probability.
            p_step: Step size between failure probability evaluation points.
            trials: Number of Monte Carlo trials per evaluation point.
            failure_mode: FailureMode (BERNOULLI or EXACT).
            aspl_mode: ASPL calculation mode.
            base_seed: Base seed for reproducible trial streams, or None.

        Returns:
            list of SweepPointResult for each evaluation point.
        """
        if p_min < 0.0 or p_max > 1.0 or p_min > p_max:
            raise ValueError(f"Invalid sweep range [{p_min}, {p_max}]. Must be within [0.0, 1.0].")
        if p_step <= 0.0:
            raise ValueError(f"Sweep step must be positive, got {p_step}.")
        if trials < 1:
            raise ValueError(f"Trials must be >= 1, got {trials}.")

        p_vals = np.arange(p_min, p_max + (p_step / 2.0), p_step)
        p_vals = np.clip(p_vals, 0.0, 1.0)

        # Pre-build topology once
        topo = FatTreeTopology(k).build()
        engine = LinkFailureEngine(seed=base_seed)

        results: list[SweepPointResult] = []

        for p in p_vals:
            reach_list: list[float] = []
            aspl_reach_list: list[float] = []
            aspl_harm_list: list[float] = []
            redundancy_list: list[float] = []
            multi_fail_list: list[float] = []

            for t in range(trials):
                trial_seed = (
                    None
                    if base_seed is None
                    else (base_seed + t * 10007 + int(round(p * 100000))) % (2**63 - 1)
                )
                pert = engine.perturb(
                    topo,
                    failure_rate=float(p),
                    mode=failure_mode,
                    seed=trial_seed,
                )
                metrics = MetricAnalyzer.compute_all_metrics(topo, pert, aspl_mode=aspl_mode)

                reach_list.append(metrics.reachability_ratio)
                aspl_reach_list.append(metrics.aspl_reachable)
                aspl_harm_list.append(metrics.aspl_harmonic)
                redundancy_list.append(metrics.mean_redundancy)
                multi_fail_list.append(float(metrics.switches_multi_failure_count))

            ddof = 1 if trials > 1 else 0
            point_result = SweepPointResult(
                k=k,
                parameter_value=float(p),
                trials=trials,
                reachability_mean=float(np.mean(reach_list)),
                reachability_std=float(np.std(reach_list, ddof=ddof)),
                aspl_reachable_mean=float(np.mean(aspl_reach_list)),
                aspl_reachable_std=float(np.std(aspl_reach_list, ddof=ddof)),
                aspl_harmonic_mean=float(np.mean(aspl_harm_list)),
                aspl_harmonic_std=float(np.std(aspl_harm_list, ddof=ddof)),
                redundancy_mean=float(np.mean(redundancy_list)),
                redundancy_std=float(np.std(redundancy_list, ddof=ddof)),
                multi_failure_mean=float(np.mean(multi_fail_list)),
                multi_failure_std=float(np.std(multi_fail_list, ddof=ddof)),
            )
            results.append(point_result)

        return results

    def run_port_count_sweep(
        self,
        k_list: list[int],
        failure_rate: float,
        trials: int = 10,
        failure_mode: FailureMode = FailureMode.BERNOULLI,
        aspl_mode: ASPLMode = ASPLMode.REACHABLE,
        base_seed: int | None = None,
        include_host_metrics: bool = True,
    ) -> list[SweepPointResult]:
        """Executes radix sweep across specified k list at fixed failure rate.

        Args:
            k_list: List of switch radices k.
            failure_rate: Link failure rate.
            trials: Number of Monte Carlo trials per radix.
            failure_mode: FailureMode (BERNOULLI or EXACT).
            aspl_mode: ASPL calculation mode.
            base_seed: Base seed for reproducible trial streams, or None.
            include_host_metrics: Whether to run full all-pairs host BFS (set False for fast multi-failure sweeps).

        Returns:
            list of SweepPointResult for each evaluated radix.
        """
        results: list[SweepPointResult] = []

        for k in k_list:
            topo = FatTreeTopology(k).build()
            engine = LinkFailureEngine(seed=base_seed)

            reach_list: list[float] = []
            aspl_reach_list: list[float] = []
            aspl_harm_list: list[float] = []
            redundancy_list: list[float] = []
            multi_fail_list: list[float] = []

            for t in range(trials):
                trial_seed = (
                    None
                    if base_seed is None
                    else (base_seed + t * 10007 + k * 101) % (2**63 - 1)
                )
                pert = engine.perturb(
                    topo,
                    failure_rate=failure_rate,
                    mode=failure_mode,
                    seed=trial_seed,
                )

                if include_host_metrics and k <= 8:
                    metrics = MetricAnalyzer.compute_all_metrics(topo, pert, aspl_mode=aspl_mode)
                    reach_list.append(metrics.reachability_ratio)
                    aspl_reach_list.append(metrics.aspl_reachable)
                    aspl_harm_list.append(metrics.aspl_harmonic)
                    redundancy_list.append(metrics.mean_redundancy)
                    multi_fail_list.append(float(metrics.switches_multi_failure_count))
                else:
                    mf_count, _, _ = MetricAnalyzer.compute_multi_failure_switches(
                        topo.switches,
                        topo.graph,
                        pert.failed_edges,
                        k,
                        failure_rate,
                    )
                    multi_fail_list.append(float(mf_count))
                    reach_list.append(1.0)
                    aspl_reach_list.append(0.0)
                    aspl_harm_list.append(0.0)
                    redundancy_list.append(0.0)

            ddof = 1 if trials > 1 else 0
            point_result = SweepPointResult(
                k=k,
                parameter_value=float(k),
                trials=trials,
                reachability_mean=float(np.mean(reach_list)),
                reachability_std=float(np.std(reach_list, ddof=ddof)),
                aspl_reachable_mean=float(np.mean(aspl_reach_list)),
                aspl_reachable_std=float(np.std(aspl_reach_list, ddof=ddof)),
                aspl_harmonic_mean=float(np.mean(aspl_harm_list)),
                aspl_harmonic_std=float(np.std(aspl_harm_list, ddof=ddof)),
                redundancy_mean=float(np.mean(redundancy_list)),
                redundancy_std=float(np.std(redundancy_list, ddof=ddof)),
                multi_failure_mean=float(np.mean(multi_fail_list)),
                multi_failure_std=float(np.std(multi_fail_list, ddof=ddof)),
            )
            results.append(point_result)

        return results

    def export_metrics_to_json(self, metrics: SimulationMetrics, path: Path | str) -> Path:
        """Serializes single-run metrics to structured JSON matching docs/SPEC.md Section 4.3.1.

        Args:
            metrics: SimulationMetrics dataclass.
            path: Target JSON file path.

        Returns:
            Path of written JSON file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "topology": {
                "k": metrics.k,
                "switches": (5 * metrics.k * metrics.k) // 4,
                "hosts": (metrics.k ** 3) // 4,
                "links": (3 * (metrics.k ** 3)) // 4,
            },
            "perturbation": {
                "failure_rate": metrics.failure_rate,
                "failure_mode": str(metrics.failure_mode),
                "seed": metrics.seed,
            },
            "metrics": {
                "total_host_pairs": metrics.total_host_pairs,
                "reachable_host_pairs": metrics.reachable_host_pairs,
                "reachability_ratio": metrics.reachability_ratio,
                "disconnection_ratio": metrics.disconnection_ratio,
                "aspl_reachable": metrics.aspl_reachable,
                "aspl_harmonic": metrics.aspl_harmonic,
                "mean_redundancy": metrics.mean_redundancy,
                "switches_multi_failure_count": metrics.switches_multi_failure_count,
                "switches_multi_failure_ratio": metrics.switches_multi_failure_ratio,
                "expected_multi_failure_count": metrics.expected_multi_failure_count,
            },
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    def export_sweep_to_csv(self, results: list[SweepPointResult], path: Path | str) -> Path:
        """Serializes sweep results to standard CSV matching docs/SPEC.md Section 4.3.2.

        Args:
            results: List of SweepPointResult dataclasses.
            path: Target CSV file path.

        Returns:
            Path of written CSV file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "k",
            "parameter_value",
            "trials",
            "reachability_mean",
            "reachability_std",
            "aspl_reachable_mean",
            "aspl_reachable_std",
            "aspl_harmonic_mean",
            "aspl_harmonic_std",
            "redundancy_mean",
            "redundancy_std",
            "multi_failure_mean",
            "multi_failure_std",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "k": r.k,
                    "parameter_value": r.parameter_value,
                    "trials": r.trials,
                    "reachability_mean": f"{r.reachability_mean:.6f}",
                    "reachability_std": f"{r.reachability_std:.6f}",
                    "aspl_reachable_mean": f"{r.aspl_reachable_mean:.6f}",
                    "aspl_reachable_std": f"{r.aspl_reachable_std:.6f}",
                    "aspl_harmonic_mean": f"{r.aspl_harmonic_mean:.6f}",
                    "aspl_harmonic_std": f"{r.aspl_harmonic_std:.6f}",
                    "redundancy_mean": f"{r.redundancy_mean:.6f}",
                    "redundancy_std": f"{r.redundancy_std:.6f}",
                    "multi_failure_mean": f"{r.multi_failure_mean:.6f}",
                    "multi_failure_std": f"{r.multi_failure_std:.6f}",
                })

        return target
