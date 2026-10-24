"""Command Line Interface (CLI) entry point, input validation, and execution engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from src.models import (
    FailureMode,
    ASPLMode,
    MetricType,
    SweepType,
    OutputFormat,
    SimulationMetrics,
    SweepPointResult,
)
from src.experiment import ExperimentRunner
from src.visualizer import MetricVisualizer


def build_parser() -> argparse.ArgumentParser:
    """Builds the comprehensive CLI argument parser matching docs/SPEC.md Section 4.1."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Fat-Tree Network Topology Generation, Failure Modeling, and Metric Analysis Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core topology parameters
    topo_group = parser.add_argument_group("Topology Generation Options")
    topo_group.add_argument(
        "--k",
        type=int,
        default=4,
        help="Switch port count / radix (must be an even integer >= 2)",
    )

    # Failure modeling parameters
    failure_group = parser.add_argument_group("Failure Perturbation Options")
    failure_group.add_argument(
        "-p",
        "--failure-rate",
        type=float,
        default=0.0,
        help="Link failure probability in range [0.0, 1.0]",
    )
    failure_group.add_argument(
        "--failure-percentage",
        type=float,
        default=None,
        help="Alternative: Link failure percentage in range [0.0, 100.0]%%",
    )
    failure_group.add_argument(
        "--failure-mode",
        type=str,
        choices=[FailureMode.BERNOULLI.value, FailureMode.EXACT.value],
        default=FailureMode.BERNOULLI.value,
        help="Failure model: independent bernoulli or exact percentage sampling",
    )
    failure_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Non-negative 64-bit integer seed for deterministic pseudo-random execution",
    )

    # Metric selection
    metric_group = parser.add_argument_group("Metric Analysis Options")
    metric_group.add_argument(
        "--metric",
        type=str,
        choices=[
            MetricType.ASPL.value,
            MetricType.REACHABILITY.value,
            MetricType.REDUNDANCY.value,
            MetricType.MULTI_FAILURE.value,
            MetricType.ALL.value,
        ],
        default=MetricType.ALL.value,
        help="Specific metric to analyze and display",
    )
    metric_group.add_argument(
        "--aspl-mode",
        type=str,
        choices=[ASPLMode.REACHABLE.value, ASPLMode.HARMONIC.value],
        default=ASPLMode.REACHABLE.value,
        help="ASPL calculation formulation: reachable pairs mean or harmonic geodesic mean",
    )

    # Experiment sweep parameters
    sweep_group = parser.add_argument_group("Parameter Sweep Options")
    sweep_group.add_argument(
        "--sweep",
        type=str,
        choices=[SweepType.NONE.value, SweepType.FAILURE_RATE.value, SweepType.PORT_COUNT.value],
        default=SweepType.NONE.value,
        help="Parameter sweep mode: none, failure-rate, or port-count",
    )
    sweep_group.add_argument(
        "--sweep-min",
        type=float,
        default=0.0,
        help="Minimum value for failure rate sweep",
    )
    sweep_group.add_argument(
        "--sweep-max",
        type=float,
        default=0.3,
        help="Maximum value for failure rate sweep",
    )
    sweep_group.add_argument(
        "--sweep-step",
        type=float,
        default=0.05,
        help="Step increment for failure rate sweep",
    )
    sweep_group.add_argument(
        "--sweep-k-list",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="List of switch radices for port-count sweep",
    )
    sweep_group.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Monte Carlo trials per evaluation point in sweep mode",
    )

    # Output and visualization options
    output_group = parser.add_argument_group("Output and Export Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save experimental data, reports, and plots",
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=[OutputFormat.JSON.value, OutputFormat.CSV.value, OutputFormat.PLOT.value, OutputFormat.ALL.value],
        default=OutputFormat.ALL.value,
        help="Output serialization formats",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validates command-line arguments, raising ValueError on domain rule violations.

    Raises:
        ValueError: If any input violates specification boundaries.
    """
    # 1. Radix k validation
    if args.k < 2:
        raise ValueError(f"Switch radix k must be >= 2, got {args.k}.")
    if args.k % 2 != 0:
        raise ValueError(f"Switch radix k must be an even integer, got {args.k}.")

    # 2. Failure rate / percentage validation
    if args.failure_percentage is not None:
        if args.failure_percentage < 0.0 or args.failure_percentage > 100.0:
            raise ValueError(
                f"Failure percentage must be in [0.0, 100.0], got {args.failure_percentage}."
            )
        args.failure_rate = args.failure_percentage / 100.0
    else:
        if args.failure_rate < 0.0 or args.failure_rate > 1.0:
            raise ValueError(
                f"Link failure rate must be in range [0.0, 1.0], got {args.failure_rate}."
            )

    # 3. Seed validation
    if args.seed is not None and args.seed < 0:
        raise ValueError(f"Random seed must be a non-negative integer, got {args.seed}.")

    # 4. Sweep parameters validation
    if args.sweep == SweepType.FAILURE_RATE.value:
        if args.sweep_min < 0.0 or args.sweep_max > 1.0 or args.sweep_min > args.sweep_max:
            raise ValueError(
                f"Sweep range [{args.sweep_min}, {args.sweep_max}] invalid. Must be in [0.0, 1.0] with min <= max."
            )
        if args.sweep_step <= 0.0:
            raise ValueError(f"Sweep step must be positive, got {args.sweep_step}.")
        if args.trials < 1:
            raise ValueError(f"Trials must be >= 1, got {args.trials}.")

    if args.sweep == SweepType.PORT_COUNT.value:
        for k_val in args.sweep_k_list:
            if k_val < 2 or k_val % 2 != 0:
                raise ValueError(
                    f"All radices in --sweep-k-list must be even integers >= 2, got {k_val}."
                )


def print_single_summary(metrics: SimulationMetrics, args: argparse.Namespace) -> None:
    """Prints clean terminal summary for single simulation run."""
    print("=" * 65)
    print(" FAT-TREE TOPOLOGY SIMULATION REPORT")
    print("=" * 65)
    print(f" Switch Port Count (k)       : {metrics.k}")
    print(f" Total Switches Deployment   : {(5 * metrics.k * metrics.k) // 4}")
    print(f" Total Compute Hosts         : {(metrics.k ** 3) // 4}")
    print(f" Total Bi-directional Links  : {(3 * (metrics.k ** 3)) // 4}")
    print(f" Injected Link Failure Rate  : {metrics.failure_rate:.2%}")
    print(f" Failure Injection Model     : {metrics.failure_mode.upper()}")
    print(f" Deterministic Random Seed   : {metrics.seed if metrics.seed is not None else 'None (Stochastic)'}")
    print("-" * 65)
    print(" METRIC EVALUATION RESULTS:")
    print(f" - Total Host Pairs Evaluated : {metrics.total_host_pairs}")
    print(f" - Reachable Host Pairs       : {metrics.reachable_host_pairs}")
    print(f" - Pairwise Reachability Ratio: {metrics.reachability_ratio:.6f} ({metrics.reachability_ratio:.2%})")
    print(f" - Pairwise Disconnection Ratio: {metrics.disconnection_ratio:.6f} ({metrics.disconnection_ratio:.2%})")
    print(f" - Average Shortest Path (ASPL): {metrics.aspl_reachable:.6f} hops")
    print(f" - Harmonic Geodesic ASPL     : {metrics.aspl_harmonic:.6f} hops")
    print(f" - Mean ECMP Path Redundancy  : {metrics.mean_redundancy:.6f} shortest paths")
    print(f" - Switches with >= 2 Failures: {metrics.switches_multi_failure_count} ({metrics.switches_multi_failure_ratio:.2%})")
    print(f" - Expected Multi-Failures    : {metrics.expected_multi_failure_count:.4f} (Binomial Theory)")
    print("=" * 65)


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution handler mapping results to standard return codes.

    Returns:
        0: Successful execution.
        1: Input validation or command-line syntax error.
        2: Runtime or filesystem I/O error.
    """
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
        validate_args(args)
    except ValueError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except SystemExit as err:
        return 1 if err.code != 0 else 0

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        runner = ExperimentRunner(output_dir=output_dir)
        viz = MetricVisualizer(output_dir=plots_dir)

        # -------------------------------------------------------------
        # Mode 1: Parameter Sweep
        # -------------------------------------------------------------
        if args.sweep == SweepType.FAILURE_RATE.value:
            if not args.quiet:
                print(f"Executing Failure Rate Sweep: k={args.k}, p=[{args.sweep_min}, {args.sweep_max}], step={args.sweep_step}, trials={args.trials}...")

            results = runner.run_failure_rate_sweep(
                k=args.k,
                p_min=args.sweep_min,
                p_max=args.sweep_max,
                p_step=args.sweep_step,
                trials=args.trials,
                failure_mode=FailureMode(args.failure_mode),
                aspl_mode=ASPLMode(args.aspl_mode),
                base_seed=args.seed,
            )

            # Export CSV
            csv_path = output_dir / f"sweep_failure_rate_k{args.k}.csv"
            runner.export_sweep_to_csv(results, csv_path)
            if not args.quiet:
                print(f"Exported tabular sweep data to: {csv_path}")

            # Export Plots
            if args.format in (OutputFormat.PLOT.value, OutputFormat.ALL.value):
                p1, s1 = viz.plot_aspl_vs_failure_rate({args.k: results})
                p2, s2 = viz.plot_reachability_vs_failure_rate({args.k: results})
                if not args.quiet:
                    print(f"Generated visualization: {p1} and {s1}")
                    print(f"Generated visualization: {p2} and {s2}")

            return 0

        elif args.sweep == SweepType.PORT_COUNT.value:
            if not args.quiet:
                print(f"Executing Port Count Sweep: k_list={args.sweep_k_list}, failure_rate={args.failure_rate:.2%}, trials={args.trials}...")

            results = runner.run_port_count_sweep(
                k_list=args.sweep_k_list,
                failure_rate=args.failure_rate,
                trials=args.trials,
                failure_mode=FailureMode(args.failure_mode),
                aspl_mode=ASPLMode(args.aspl_mode),
                base_seed=args.seed,
            )

            csv_path = output_dir / f"sweep_port_count_p{int(args.failure_rate * 100)}.csv"
            runner.export_sweep_to_csv(results, csv_path)
            if not args.quiet:
                print(f"Exported tabular radix sweep data to: {csv_path}")

            if args.format in (OutputFormat.PLOT.value, OutputFormat.ALL.value):
                empirical_means = [r.multi_failure_mean for r in results]
                theoretical_means = []
                for k_val in args.sweep_k_list:
                    sw_count = (5 * k_val * k_val) // 4
                    p = args.failure_rate
                    exp_mf = sw_count * (1.0 - (1.0 - p)**k_val - k_val * p * (1.0 - p)**(k_val - 1)) if p > 0 else 0.0
                    theoretical_means.append(exp_mf)

                p3, s3 = viz.plot_multi_failure_vs_radix(
                    args.sweep_k_list,
                    empirical_means,
                    theoretical_means,
                    failure_rate=args.failure_rate,
                )
                p4, s4 = viz.plot_hosts_supported_vs_radix(args.sweep_k_list)
                if not args.quiet:
                    print(f"Generated visualization: {p3} and {s3}")
                    print(f"Generated visualization: {p4} and {s4}")

            return 0

        # -------------------------------------------------------------
        # Mode 2: Single Simulation Execution
        # -------------------------------------------------------------
        else:
            metrics = runner.run_single_simulation(
                k=args.k,
                failure_rate=args.failure_rate,
                failure_mode=FailureMode(args.failure_mode),
                seed=args.seed,
                aspl_mode=ASPLMode(args.aspl_mode),
            )

            if not args.quiet:
                print_single_summary(metrics, args)

            # Export JSON if requested
            if args.format in (OutputFormat.JSON.value, OutputFormat.ALL.value):
                seed_str = f"seed{args.seed}" if args.seed is not None else "stochastic"
                json_path = output_dir / f"metrics_k{args.k}_p{args.failure_rate}_{seed_str}.json"
                runner.export_metrics_to_json(metrics, json_path)
                if not args.quiet:
                    print(f"Exported JSON metrics to: {json_path}")

            return 0

    except Exception as err:
        sys.stderr.write(f"Runtime Error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())

