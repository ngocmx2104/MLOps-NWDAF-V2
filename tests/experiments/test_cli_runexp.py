"""Tests for CLI subcommands: run-exp1, run-exp2, run-exp3, tables (T11)."""
from __future__ import annotations

from src.experiments.cli import main


def test_cli_run_exp1(tmp_path):
    """run-exp1 returns 0 and writes exp1_summary.json."""
    from tests.experiments._fixtures import write_tiny_labeled_parquet

    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")
    rc = main(
        [
            "run-exp1",
            "--dataset", str(ds),
            "--seeds", "1,2",
            "--output-root", str(tmp_path / "out"),
            "--backend", "noop",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "exp1_e2e" / "exp1_summary.json").exists()


def test_cli_run_exp1_missing_dataset(tmp_path):
    """run-exp1 with a non-existent dataset returns 1 (error, no crash)."""
    rc = main(
        [
            "run-exp1",
            "--dataset", str(tmp_path / "nope.parquet"),
            "--seeds", "1",
            "--output-root", str(tmp_path / "out"),
            "--backend", "noop",
        ]
    )
    assert rc == 1


def test_cli_run_exp2(tmp_path):
    """run-exp2 returns 0 and writes exp2_summary.json."""
    from tests.experiments._fixtures import write_tiny_labeled_parquet

    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")
    db_path = tmp_path / "mlflow.db"
    rc = main(
        [
            "run-exp2",
            "--dataset", str(ds),
            "--seeds", "1,2",
            "--output-root", str(tmp_path / "out"),
            "--c1-tracking-uri", f"sqlite:///{db_path}",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "exp2_ablation" / "exp2_summary.json").exists()


def test_cli_run_exp3(tmp_path):
    """run-exp3 returns 0 and writes exp3_summary.json."""
    from tests.experiments._fixtures import write_tiny_labeled_parquet

    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")
    rc = main(
        [
            "run-exp3",
            "--dataset", str(ds),
            "--seeds", "1,2",
            "--output-root", str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "exp3_framework" / "exp3_summary.json").exists()


def test_cli_tables(tmp_path):
    """tables subcommand generates a Markdown file from available summaries."""
    import json

    # Write a minimal exp1_summary.json (tables must not crash on partial input)
    exp1_dir = tmp_path / "out" / "exp1_e2e"
    exp1_dir.mkdir(parents=True)
    (exp1_dir / "exp1_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp1_e2e",
                "n_runs": 2,
                "model_perf": {
                    "roc_auc": {"n": 2, "mean": 0.88, "std": 0.01, "min": 0.87, "max": 0.89}
                },
            }
        )
    )
    out_file = tmp_path / "tables.md"
    rc = main(
        [
            "tables",
            "--output-root", str(tmp_path / "out"),
            "--output", str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.exists()
    text = out_file.read_text()
    assert "roc_auc" in text
    assert "|" in text  # Markdown pipe table format


def test_cli_tables_no_summaries(tmp_path):
    """tables with no summaries writes a file with N/A (does not crash)."""
    out_file = tmp_path / "tables.md"
    rc = main(
        [
            "tables",
            "--output-root", str(tmp_path / "empty"),
            "--output", str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.exists()


def test_existing_assess_still_works(tmp_path):
    """Existing assess subcommand still works after T11 extension."""
    (tmp_path / "real.txt").write_text("x")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "tests:\n"
        "  - id: d1\n    section: data\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n"
    )
    rc = main(["assess", "--manifest", str(manifest), "--repo-root", str(tmp_path)])
    assert rc == 0
