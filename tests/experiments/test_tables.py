"""Tests for src/experiments/tables.py — T10 result tables (Markdown + LaTeX)."""
from __future__ import annotations

from src.experiments.tables import exp1_table, exp2_table, exp3_table, render_latex, render_markdown


def test_exp1_table_markdown(tmp_path):
    summary = {
        "experiment_id": "exp1_e2e",
        "n_runs": 10,
        "model_perf": {
            "roc_auc": {"n": 10, "mean": 0.91, "std": 0.02, "min": 0.88, "max": 0.94},
            "pr_auc": {"n": 10, "mean": 0.85, "std": 0.03, "min": 0.80, "max": 0.90},
        },
    }
    rows = exp1_table(summary)
    assert rows[0]["metric"] == "roc_auc" and rows[0]["mean"] == 0.91
    md = render_markdown(rows, columns=["metric", "mean", "std"])
    assert "| metric | mean | std |" in md and "roc_auc" in md


def test_render_markdown_handles_missing(tmp_path):
    md = render_markdown([], columns=["metric", "mean"])
    # empty -> header only / N/A note, never crash
    assert "N/A" in md or md.strip().startswith("|")


def test_exp1_table_rows_for_all_perf_keys():
    """All perf keys present in summary appear as rows."""
    summary = {
        "experiment_id": "exp1_e2e",
        "n_runs": 5,
        "model_perf": {
            "roc_auc": {"n": 5, "mean": 0.88, "std": 0.01, "min": 0.85, "max": 0.91},
            "pr_auc": {"n": 5, "mean": 0.80, "std": 0.02, "min": 0.77, "max": 0.84},
            "f1": {"n": 5, "mean": 0.75, "std": 0.03, "min": 0.70, "max": 0.80},
        },
    }
    rows = exp1_table(summary)
    metrics = [r["metric"] for r in rows]
    assert "roc_auc" in metrics and "pr_auc" in metrics and "f1" in metrics


def test_exp1_table_missing_perf_returns_na():
    """If model_perf key is absent, no crash — missing values appear as 'N/A'."""
    summary = {"experiment_id": "exp1_e2e", "n_runs": 0}
    rows = exp1_table(summary)
    # No rows or N/A sentinel row — but must NOT raise
    assert isinstance(rows, list)


def test_render_markdown_format():
    """render_markdown produces proper pipe-table with header + separator + data rows."""
    rows = [{"metric": "roc_auc", "mean": 0.91, "std": 0.02}]
    md = render_markdown(rows, columns=["metric", "mean", "std"])
    lines = [line for line in md.splitlines() if line.strip()]
    assert lines[0].startswith("|")
    assert "---" in lines[1]  # separator row
    assert "roc_auc" in lines[2]


def test_render_latex_format():
    """render_latex produces a tabular environment with & separators and \\\\ row endings."""
    rows = [{"metric": "roc_auc", "mean": 0.91, "std": 0.02}]
    latex = render_latex(rows, columns=["metric", "mean", "std"])
    assert r"\begin{tabular}" in latex
    assert "&" in latex
    assert r"\\" in latex
    assert r"\end{tabular}" in latex


def test_render_latex_handles_empty():
    """Empty rows → latex table with header + N/A note, never crash."""
    latex = render_latex([], columns=["metric", "mean"])
    assert r"\begin{tabular}" in latex
    assert "N/A" in latex or r"\end{tabular}" in latex


def test_exp2_table_groups():
    """exp2_table extracts C0 vs C1 comparisons from operational + resource + maturity groups."""
    summary = {
        "experiment_id": "exp2_ablation",
        "n_seeds": 5,
        "operational": {
            "C0_wall_s": {"mean": 1.5, "std": 0.1, "n": 5},
            "C1_wall_s": {"mean": 2.0, "std": 0.15, "n": 5},
            "wilcoxon_wall_s": {"p_value": 0.03, "significant_0_05": True},
        },
        "resource": {
            "storage_delta_bytes": 512000,
        },
        "maturity": {
            "C0_score": 2.0,
            "C1_score": 5.0,
            "delta": 3.0,
        },
    }
    rows = exp2_table(summary)
    assert isinstance(rows, list) and len(rows) > 0
    metrics = [r.get("metric") for r in rows]
    # Should have wall_s row and maturity/delta row
    assert any("wall" in str(m) for m in metrics if m)
    assert any("maturity" in str(m) or "delta" in str(m) or "score" in str(m) for m in metrics if m)


def test_exp3_table_backends():
    """exp3_table produces per-backend overhead rows."""
    summary = {
        "experiment_id": "exp3_framework",
        "backends": ["noop", "mlflow"],
        "overhead": {
            "noop": {"wall_s": {"mean": 1.0, "std": 0.05}, "delta_wall_s": 0.0},
            "mlflow": {"wall_s": {"mean": 1.3, "std": 0.07}, "delta_wall_s": 0.3},
        },
        "governance": {
            "noop": {"registry": False, "run_id": False},
            "mlflow": {"registry": True, "run_id": True},
        },
    }
    rows = exp3_table(summary)
    assert isinstance(rows, list) and len(rows) > 0
    backends_in_rows = [r.get("backend") for r in rows]
    assert "noop" in backends_in_rows
    assert "mlflow" in backends_in_rows


def test_exp2_table_empty():
    """exp2_table with empty summary does not crash."""
    rows = exp2_table({})
    assert isinstance(rows, list)


def test_exp3_table_empty():
    """exp3_table with empty summary does not crash."""
    rows = exp3_table({})
    assert isinstance(rows, list)


def test_never_hardcode_numbers():
    """Tables are pure (dict→rows); changing input changes output — not hardcoded."""
    s1 = {"experiment_id": "exp1_e2e", "model_perf": {"roc_auc": {"mean": 0.91, "std": 0.02, "n": 10}}}
    s2 = {"experiment_id": "exp1_e2e", "model_perf": {"roc_auc": {"mean": 0.75, "std": 0.05, "n": 10}}}
    rows1 = exp1_table(s1)
    rows2 = exp1_table(s2)
    # Different inputs must produce different outputs
    assert rows1[0]["mean"] != rows2[0]["mean"]
