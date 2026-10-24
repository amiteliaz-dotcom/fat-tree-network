"""Unit tests for ExperimentRunner and multi-format serialization (AC-10, AC-12, EC-8)."""
from __future__ import annotations

import json
import csv
from pathlib import Path
import pytest
from src.experiment import ExperimentRunner
from src.models import FailureMode


def test_single_simulation(tmp_path: Path):
    """Verify single deterministic simulation run."""
    runner = ExperimentRunner(output_dir=tmp_path)
    metrics = runner.run_single_simulation(k=4, failure_rate=0.1, seed=42)

    assert metrics.k == 4
    assert metrics.failure_rate == 0.1
    assert metrics.total_host_pairs == 120
    assert 0 <= metrics.reachable_host_pairs <= 120
    assert 0.0 <= metrics.reachability_ratio <= 1.0


def test_failure_rate_sweep_and_csv_export(tmp_path: Path):
    """Verify failure rate sweep execution and CSV export."""
    runner = ExperimentRunner(output_dir=tmp_path)
    results = runner.run_failure_rate_sweep(
        k=4,
        p_min=0.0,
        p_max=0.2,
        p_step=0.1,
        trials=3,
        base_seed=123,
    )

    assert len(results) == 3  # 0.0, 0.1, 0.2
    assert results[0].parameter_value == 0.0
    assert results[0].reachability_mean == 1.0
    assert results[0].reachability_std == 0.0

    csv_path = tmp_path / "sweep_test.csv"
    exported = runner.export_sweep_to_csv(results, csv_path)
    assert exported.exists()

    with open(exported, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert "k" in rows[0]
        assert "reachability_mean" in rows[0]
        assert "aspl_reachable_mean" in rows[0]


def test_json_export(tmp_path: Path):
    """Verify JSON export schema matching Section 4.3.1."""
    runner = ExperimentRunner(output_dir=tmp_path)
    metrics = runner.run_single_simulation(k=4, failure_rate=0.0, seed=1)

    json_path = tmp_path / "metrics_test.json"
    exported = runner.export_metrics_to_json(metrics, json_path)
    assert exported.exists()

    with open(exported, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "topology" in data
    assert data["topology"]["k"] == 4
    assert data["topology"]["hosts"] == 16
    assert data["topology"]["switches"] == 20
    assert data["topology"]["links"] == 48

    assert "metrics" in data
    assert data["metrics"]["total_host_pairs"] == 120
    assert data["metrics"]["reachable_host_pairs"] == 120
    assert data["metrics"]["reachability_ratio"] == 1.0

