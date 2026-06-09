"""Result tables — Markdown and LaTeX renderers for experiment summaries.

All functions are pure (input dict → output string/list).
Numbers come exclusively from summary dicts — NEVER hard-coded.
Empty input → header-only table with an N/A sentinel row; never crash.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Summary → row extraction
# ---------------------------------------------------------------------------

def exp1_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-metric rows from an Exp-1 (E2E quality) summary dict.

    Each row has keys: ``metric``, ``n``, ``mean``, ``std``, ``min``, ``max``.
    Reads from ``summary["model_perf"]``; missing keys → value "N/A".
    """
    perf = summary.get("model_perf") or {}
    rows: list[dict[str, Any]] = []
    for metric, stats in perf.items():
        if not isinstance(stats, dict):
            continue
        rows.append({
            "metric": metric,
            "n": stats.get("n", "N/A"),
            "mean": stats.get("mean", "N/A"),
            "std": stats.get("std", "N/A"),
            "min": stats.get("min", "N/A"),
            "max": stats.get("max", "N/A"),
        })
    return rows


def exp2_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract C0 vs C1 comparison rows from an Exp-2 (ablation) summary dict.

    Covers: operational (wall_s), resource (storage_delta_bytes), maturity (score delta).
    Each row has keys: ``metric``, ``C0``, ``C1``, ``delta``, ``p_value`` (where applicable).
    """
    rows: list[dict[str, Any]] = []

    # 1) Operational: wall time
    operational = summary.get("operational") or {}
    c0_wall = operational.get("C0_wall_s") or {}
    c1_wall = operational.get("C1_wall_s") or {}
    wilcoxon = operational.get("wilcoxon_wall_s") or {}
    rows.append({
        "metric": "wall_s (mean)",
        "C0": c0_wall.get("mean", "N/A"),
        "C1": c1_wall.get("mean", "N/A"),
        "delta": (
            round(c1_wall["mean"] - c0_wall["mean"], 4)
            if isinstance(c1_wall.get("mean"), (int, float)) and isinstance(c0_wall.get("mean"), (int, float))
            else "N/A"
        ),
        "p_value": wilcoxon.get("p_value", "N/A"),
        "significant": wilcoxon.get("significant_0_05", "N/A"),
    })

    # 2) Resource: storage delta
    resource = summary.get("resource") or {}
    rows.append({
        "metric": "storage_delta_bytes",
        "C0": resource.get("storage_c0_bytes", "N/A"),
        "C1": resource.get("storage_c1_bytes", "N/A"),
        "delta": resource.get("storage_delta_bytes", "N/A"),
        "p_value": "N/A",
        "significant": "N/A",
    })

    # 3) Maturity: ML Test Score
    maturity = summary.get("maturity") or {}
    rows.append({
        "metric": "ml_test_score",
        "C0": maturity.get("C0_score", "N/A"),
        "C1": maturity.get("C1_score", "N/A"),
        "delta": maturity.get("delta", "N/A"),
        "p_value": "N/A",
        "significant": "N/A",
    })

    # 4) Maturity: traceability
    rows.append({
        "metric": "traceable",
        "C0": maturity.get("traceable_C0", "N/A"),
        "C1": maturity.get("traceable_C1", "N/A"),
        "delta": "N/A",
        "p_value": "N/A",
        "significant": "N/A",
    })

    return rows


def exp3_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-backend overhead + governance rows from an Exp-3 (framework) summary dict.

    Each row has keys: ``backend``, ``wall_s_mean``, ``delta_wall_s``, ``rss_mb_mean``,
    ``delta_rss_mb``, ``registry``, ``run_id``.
    """
    overhead = summary.get("overhead") or {}
    governance = summary.get("governance") or {}
    backends = list(overhead.keys()) or list(governance.keys())

    rows: list[dict[str, Any]] = []
    for name in backends:
        ov = overhead.get(name) or {}
        gov = governance.get(name) or {}
        wall_stats = ov.get("wall_s") or {}
        rss_stats = ov.get("rss_mb") or {}
        rows.append({
            "backend": name,
            "wall_s_mean": wall_stats.get("mean", "N/A"),
            "wall_s_std": wall_stats.get("std", "N/A"),
            "delta_wall_s": ov.get("delta_wall_s", "N/A"),
            "rss_mb_mean": rss_stats.get("mean", "N/A"),
            "delta_rss_mb": ov.get("delta_rss_mb", "N/A"),
            "registry": gov.get("registry", "N/A"),
            "run_id": gov.get("run_id", "N/A"),
        })
    return rows


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _na_sentinel_row(columns: list[str]) -> dict[str, Any]:
    """Return a single-row dict with 'N/A' for every column."""
    return {c: "N/A" for c in columns}


def render_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a list of row dicts as a GitHub-flavoured Markdown pipe table.

    Parameters
    ----------
    rows:
        List of dicts; each dict is one data row.  Missing keys → "N/A".
    columns:
        Ordered list of column names (determines both header and extraction order).

    Returns
    -------
    Multiline string.  Empty ``rows`` → header + separator + one "N/A" sentinel row.
    """
    if not rows:
        rows = [_na_sentinel_row(columns)]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v) if v is not None else "N/A"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    data_lines = [
        "| " + " | ".join(fmt(row.get(c)) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + data_lines) + "\n"


def render_latex(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a list of row dicts as a LaTeX ``tabular`` environment.

    Parameters
    ----------
    rows:
        List of dicts; each dict is one data row.  Missing keys → "N/A".
    columns:
        Ordered list of column names.

    Returns
    -------
    Multiline LaTeX string.  Empty ``rows`` → tabular with header + N/A row.
    """
    if not rows:
        rows = [_na_sentinel_row(columns)]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v) if v is not None else "N/A"

    col_spec = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\"
    midrule = r"\hline"
    data_lines = [
        " & ".join(fmt(row.get(c)) for c in columns) + r" \\"
        for row in rows
    ]
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        header,
        midrule,
    ] + data_lines + [
        r"\hline",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"
