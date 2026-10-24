"""Automated execution script for all Homework 1 experimental campaigns."""
from __future__ import annotations

from pathlib import Path
from src.experiment import ExperimentRunner
from src.visualizer import MetricVisualizer
from src.models import SweepPointResult


def main() -> None:
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(output_dir=results_dir)
    viz = MetricVisualizer(output_dir=plots_dir)

    print("=" * 65)
    print(" EXECUTING HOMEWORK 1 EXPERIMENTAL EVALUATION SUITE")
    print("=" * 65)

    # -------------------------------------------------------------
    # Campaign 1: Multi-Radix Failure Rate Sweep (k = 4, 6, 8)
    # -------------------------------------------------------------
    print("\n[Campaign 1] Running Multi-Radix Failure Rate Sweeps (k=4, 6, 8)...")
    k_sweep_data: dict[int, list[SweepPointResult]] = {}
    radices = [4, 6, 8]
    p_min, p_max, p_step = 0.0, 0.35, 0.025
    trials = 10
    base_seed = 42

    all_sweep_results: list[SweepPointResult] = []

    for k in radices:
        print(f"  -> Simulating k={k} across p in [{p_min}, {p_max}] with {trials} trials per point...")
        pts = runner.run_failure_rate_sweep(
            k=k,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            trials=trials,
            base_seed=base_seed,
        )
        k_sweep_data[k] = pts
        all_sweep_results.extend(pts)
        csv_file = results_dir / f"sweep_failure_rate_k{k}.csv"
        runner.export_sweep_to_csv(pts, csv_file)
        print(f"     Saved {csv_file}")

    # Export combined multi-radix CSV
    combined_csv = results_dir / "sweep_multi_radix.csv"
    runner.export_sweep_to_csv(all_sweep_results, combined_csv)
    print(f"  -> Saved combined dataset: {combined_csv}")

    # Generate Campaign 1 plots
    print("  -> Generating ASPL vs. Failure Rate plot...")
    p1, s1 = viz.plot_aspl_vs_failure_rate(k_sweep_data, output_name="aspl_vs_failure_rate")
    print(f"     Rendered: {p1}")

    print("  -> Generating Reachability vs. Failure Rate plot...")
    p2, s2 = viz.plot_reachability_vs_failure_rate(k_sweep_data, output_name="reachability_vs_failure_rate")
    print(f"     Rendered: {p2}")

    # -------------------------------------------------------------
    # Campaign 2: Switch Multi-Failure Analysis at 1% Failure Rate
    # -------------------------------------------------------------
    print("\n[Campaign 2] Running Switch Multi-Failure Analysis at p=0.01 (Intermediate Metric)...")
    sweep_k_list = [4, 6, 8, 12, 16]
    trials_mf = 50
    mf_results = runner.run_port_count_sweep(
        k_list=sweep_k_list,
        failure_rate=0.01,
        trials=trials_mf,
        base_seed=2026,
        include_host_metrics=False,
    )
    mf_csv = results_dir / "sweep_multi_failure_p01.csv"
    runner.export_sweep_to_csv(mf_results, mf_csv)
    print(f"  -> Saved dataset: {mf_csv}")

    empirical_mf_means = [r.multi_failure_mean for r in mf_results]
    theoretical_mf_means: list[float] = []
    p_fail = 0.01
    for k_val in sweep_k_list:
        switches_count = (5 * k_val * k_val) // 4
        exp_val = switches_count * (1.0 - (1.0 - p_fail)**k_val - k_val * p_fail * (1.0 - p_fail)**(k_val - 1))
        theoretical_mf_means.append(exp_val)

    print("  -> Generating Multi-Failure vs. Radix comparison plot...")
    p3, s3 = viz.plot_multi_failure_vs_radix(
        sweep_k_list,
        empirical_mf_means,
        theoretical_mf_means,
        failure_rate=p_fail,
        output_name="multi_failure_vs_radix",
    )
    print(f"     Rendered: {p3}")

    # -------------------------------------------------------------
    # Campaign 3: Structural Scaling Capacity (Hosts vs. Radix)
    # -------------------------------------------------------------
    print("\n[Campaign 3] Generating Structural Capacity Scaling Plot (Basic Metric)...")
    capacity_radices = [2, 4, 6, 8, 12, 16, 24, 32, 48]
    p4, s4 = viz.plot_hosts_supported_vs_radix(capacity_radices, output_name="hosts_supported_vs_radix")
    print(f"  -> Rendered: {p4}")

    print("\n" + "=" * 65)
    print(" ALL EXPERIMENTS COMPLETE. ARTIFACTS READY IN results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
