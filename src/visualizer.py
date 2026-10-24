"""High-resolution publication figure generation using Matplotlib."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from src.models import SweepPointResult


class MetricVisualizer:
    """Renders 300 DPI raster PNG and vector SVG plots for academic publication."""

    def __init__(self, output_dir: Path | str = "results/plots") -> None:
        """Initialize visualizer with target export directory.

        Args:
            output_dir: Directory where figures will be saved.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        })

    def plot_aspl_vs_failure_rate(
        self,
        k_sweep_data: dict[int, list[SweepPointResult]],
        output_name: str = "aspl_vs_failure_rate",
    ) -> tuple[Path, Path]:
        """Renders ASPL as a function of failure rate p with shaded std dev bands.

        Args:
            k_sweep_data: Mapping from radix k to list of SweepPointResult.
            output_name: Base filename without extension.

        Returns:
            tuple of (png_path, svg_path).
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        markers = ["o", "s", "^", "D"]

        for idx, (k, points) in enumerate(sorted(k_sweep_data.items())):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            p_vals = np.array([pt.parameter_value for pt in points])
            aspl_means = np.array([pt.aspl_reachable_mean for pt in points])
            aspl_stds = np.array([pt.aspl_reachable_std for pt in points])

            ax.plot(
                p_vals,
                aspl_means,
                label=f"Fat-Tree (k={k})",
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
            )
            ax.fill_between(
                p_vals,
                aspl_means - aspl_stds,
                aspl_means + aspl_stds,
                color=color,
                alpha=0.18,
            )

        ax.set_title("Average Shortest Path Length (Reachable) vs. Link Failure Rate")
        ax.set_xlabel("Link Failure Probability ($p$)")
        ax.set_ylabel("Average Shortest Path Length (Hops)")
        ax.set_xlim(left=0.0)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", frameon=True)

        png_path = self.output_dir / f"{output_name}.png"
        svg_path = self.output_dir / f"{output_name}.svg"
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        return png_path, svg_path

    def plot_reachability_vs_failure_rate(
        self,
        k_sweep_data: dict[int, list[SweepPointResult]],
        output_name: str = "reachability_vs_failure_rate",
    ) -> tuple[Path, Path]:
        """Renders host pair reachability and disconnection ratios vs failure rate.

        Args:
            k_sweep_data: Mapping from radix k to list of SweepPointResult.
            output_name: Base filename without extension.

        Returns:
            tuple of (png_path, svg_path).
        """
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        markers = ["o", "s", "^", "D"]

        for idx, (k, points) in enumerate(sorted(k_sweep_data.items())):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            p_vals = np.array([pt.parameter_value for pt in points])
            reach_means = np.array([pt.reachability_mean for pt in points])
            reach_stds = np.array([pt.reachability_std for pt in points])

            ax1.plot(
                p_vals,
                reach_means,
                label=f"Reachability (k={k})",
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
            )
            ax1.fill_between(
                p_vals,
                np.clip(reach_means - reach_stds, 0.0, 1.0),
                np.clip(reach_means + reach_stds, 0.0, 1.0),
                color=color,
                alpha=0.15,
            )

        ax1.set_title("Host Pair Reachability vs. Link Failure Rate ($p$)")
        ax1.set_xlabel("Link Failure Probability ($p$)")
        ax1.set_ylabel("Host Pair Reachability Ratio ($r$)")
        ax2.set_ylabel("Disconnection Ratio ($d = 1 - r$)")

        ax1.set_ylim(-0.02, 1.02)
        ax2.set_ylim(1.02, -0.02)  # Inverted so 0 disconnection matches 1 reachability
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower left", frameon=True)

        png_path = self.output_dir / f"{output_name}.png"
        svg_path = self.output_dir / f"{output_name}.svg"
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        return png_path, svg_path

    def plot_multi_failure_vs_radix(
        self,
        radices: list[int],
        empirical_means: list[float],
        theoretical_means: list[float],
        failure_rate: float = 0.01,
        output_name: str = "multi_failure_vs_radix",
    ) -> tuple[Path, Path]:
        """Renders switch multi-failure counts vs radix with theoretical binomial baseline overlay.

        Args:
            radices: List of switch port counts.
            empirical_means: Mean empirical counts from simulations.
            theoretical_means: Closed-form binomial expected values.
            failure_rate: Fixed failure probability (default 1%).
            output_name: Base filename without extension.

        Returns:
            tuple of (png_path, svg_path).
        """
        fig, ax = plt.subplots(figsize=(7.5, 5))
        x = np.arange(len(radices))
        width = 0.35

        ax.bar(
            x - width / 2,
            empirical_means,
            width,
            label="Empirical Simulation",
            color="#1f77b4",
            alpha=0.85,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            theoretical_means,
            width,
            label="Theoretical Binomial $\\mathbb{E}[N_{multi}]$",
            color="#ff7f0e",
            alpha=0.85,
            edgecolor="black",
        )

        pct = int(round(failure_rate * 100))
        ax.set_title(f"Switches with $\\geq 2$ Failed Links at {pct}% Failure Rate vs. Switch Radix")
        ax.set_xlabel("Switch Port Count ($k$)")
        ax.set_ylabel("Expected Switch Count with $\\geq 2$ Failed Links")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in radices])
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        ax.legend(loc="upper left", frameon=True)

        png_path = self.output_dir / f"{output_name}.png"
        svg_path = self.output_dir / f"{output_name}.svg"
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        return png_path, svg_path

    def plot_hosts_supported_vs_radix(
        self,
        radices: list[int],
        output_name: str = "hosts_supported_vs_radix",
    ) -> tuple[Path, Path]:
        """Renders cubic host capacity scaling H(k) = k^3/4 vs radix k.

        Args:
            radices: List of switch radices k.
            output_name: Base filename without extension.

        Returns:
            tuple of (png_path, svg_path).
        """
        fig, ax = plt.subplots(figsize=(7.5, 5))
        k_dense = np.linspace(min(radices), max(radices), 100)
        h_dense = (k_dense ** 3) / 4.0

        ax.plot(k_dense, h_dense, color="#2ca02c", linewidth=2.0, label="Cubic Scaling: $H(k) = k^3 / 4$")

        k_pts = np.array(radices)
        h_pts = (k_pts ** 3) // 4
        ax.scatter(k_pts, h_pts, color="#d62728", s=60, zorder=5, label="Evaluation Radices")

        for k_val, h_val in zip(k_pts, h_pts):
            ax.annotate(
                f"{h_val:,}",
                (k_val, h_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title("Network Compute Capacity: Supported Hosts vs. Switch Port Count")
        ax.set_xlabel("Switch Port Count ($k$)")
        ax.set_ylabel("Supported Hosts ($N_{hosts}$)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)

        png_path = self.output_dir / f"{output_name}.png"
        svg_path = self.output_dir / f"{output_name}.svg"
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        return png_path, svg_path
