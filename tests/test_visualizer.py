"""Unit tests for MetricVisualizer headless rendering and image output generation."""
from __future__ import annotations

from pathlib import Path
import pytest
from src.visualizer import MetricVisualizer
from src.models import SweepPointResult


def test_visualizer_generation(tmp_path: Path):
    """Verify that MetricVisualizer generates valid non-empty PNG and SVG files."""
    viz = MetricVisualizer(output_dir=tmp_path)

    # Mock sweep data
    pts_k4 = [
        SweepPointResult(
            k=4,
            parameter_value=0.0,
            trials=5,
            reachability_mean=1.0,
            reachability_std=0.0,
            aspl_reachable_mean=5.467,
            aspl_reachable_std=0.0,
            aspl_harmonic_mean=5.0,
            aspl_harmonic_std=0.0,
            redundancy_mean=3.533,
            redundancy_std=0.0,
            multi_failure_mean=0.0,
            multi_failure_std=0.0,
        ),
        SweepPointResult(
            k=4,
            parameter_value=0.1,
            trials=5,
            reachability_mean=0.92,
            reachability_std=0.04,
            aspl_reachable_mean=5.8,
            aspl_reachable_std=0.15,
            aspl_harmonic_mean=6.5,
            aspl_harmonic_std=0.3,
            redundancy_mean=2.1,
            redundancy_std=0.2,
            multi_failure_mean=0.8,
            multi_failure_std=0.4,
        ),
    ]

    # 1. ASPL plot
    png_aspl, svg_aspl = viz.plot_aspl_vs_failure_rate({4: pts_k4})
    assert png_aspl.exists()
    assert svg_aspl.exists()
    assert png_aspl.stat().st_size > 1000
    assert svg_aspl.stat().st_size > 1000

    # 2. Reachability plot
    png_reach, svg_reach = viz.plot_reachability_vs_failure_rate({4: pts_k4})
    assert png_reach.exists()
    assert svg_reach.exists()
    assert png_reach.stat().st_size > 1000

    # 3. Multi-failure plot
    png_mf, svg_mf = viz.plot_multi_failure_vs_radix([4, 8], [0.012, 0.21], [0.0118, 0.215])
    assert png_mf.exists()
    assert svg_mf.exists()
    assert png_mf.stat().st_size > 1000

    # 4. Host capacity plot
    png_hc, svg_hc = viz.plot_hosts_supported_vs_radix([2, 4, 6, 8, 12, 16])
    assert png_hc.exists()
    assert svg_hc.exists()
    assert png_hc.stat().st_size > 1000

