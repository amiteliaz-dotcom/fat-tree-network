"""Unit tests for CLI parsing, validation rules, and exit codes (AC-11, AC-12)."""
from __future__ import annotations

from pathlib import Path
import pytest
from src.cli import main


def test_cli_help():
    """Verify --help returns exit code 0."""
    assert main(["--help"]) == 0


def test_cli_validation_errors_ac11(capsys: pytest.CaptureFixture[str]):
    """Verify AC-11: Invalid arguments return exit code 1 with error messages."""
    # 1. Odd radix k
    code = main(["--k", "3"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Switch radix k must be an even integer" in err

    # 2. Radix k < 2
    code = main(["--k", "0"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Switch radix k must be >= 2" in err

    # 3. Negative failure rate
    code = main(["--k", "4", "-p", "-0.1"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Link failure rate must be in range" in err

    # 4. Failure rate > 1.0
    code = main(["--k", "4", "-p", "1.5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Link failure rate must be in range" in err

    # 5. Negative seed
    code = main(["--k", "4", "--seed", "-5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Random seed must be a non-negative integer" in err


def test_cli_single_run_success(tmp_path: Path):
    """Verify single simulation execution via CLI returns 0 and outputs JSON."""
    code = main([
        "--k", "4",
        "-p", "0.05",
        "--seed", "42",
        "--output-dir", str(tmp_path),
        "--format", "json",
        "--quiet",
    ])
    assert code == 0

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    assert "metrics_k4_p0.05_seed42.json" in json_files[0].name


def test_cli_sweep_run_ac12(tmp_path: Path):
    """Verify AC-12: CLI sweep execution generates CSV and plots with exit code 0."""
    code = main([
        "--sweep", "failure-rate",
        "--k", "4",
        "--sweep-min", "0.0",
        "--sweep-max", "0.1",
        "--sweep-step", "0.05",
        "--trials", "3",
        "--seed", "100",
        "--output-dir", str(tmp_path),
        "--format", "all",
        "--quiet",
    ])
    assert code == 0

    csv_file = tmp_path / "sweep_failure_rate_k4.csv"
    assert csv_file.exists()

    plots_dir = tmp_path / "plots"
    assert (plots_dir / "aspl_vs_failure_rate.png").exists()
    assert (plots_dir / "reachability_vs_failure_rate.png").exists()

