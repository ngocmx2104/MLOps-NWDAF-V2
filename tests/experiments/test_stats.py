from src.experiments.stats import bootstrap_ci, summarize, wilcoxon_compare


def test_wilcoxon_detects_difference():
    c0 = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.0]
    c1 = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 2.1, 1.9, 2.0, 2.0]
    r = wilcoxon_compare(c0, c1)
    assert r["p_value"] < 0.05 and r["n"] == 10


def test_bootstrap_ci_and_summary():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["n"] == 5
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=0)["ci95"]
    assert lo <= 3.0 <= hi
