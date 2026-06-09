# tests/experiments/test_tables_rq4.py
from src.experiments.tables import exp4_table, exp5_table, exp6_table, render_markdown


def test_rq4_tables():
    e4 = {"scenarios": {"sudden": {"on": {"detection_latency_steps": 0, "retrain_count": 2,
                                          "drift_detected_any": True},
                                   "off": {"retrain_count": 0}}}}
    rows4 = exp4_table(e4)
    assert rows4[0]["scenario"] == "sudden" and rows4[0]["retrain_on"] == 2
    e5 = {"iforest": {"model_perf": {"roc_auc": {"mean": 0.9}}, "train_wall_s": {"mean": 1.0}},
          "lstm_ae": {"model_perf": {"roc_auc": {"mean": 0.8}}, "train_wall_s": {"mean": 5.0}},
          "swap_core_changes": 0}
    rows5 = exp5_table(e5)
    assert any(r["model"] == "iforest" for r in rows5)
    e6 = {"mods": [{"mod_id": "add_feature", "section": "data", "files_changed": 2,
                    "lines_changed": 10, "regression_count": 0, "pass": True}]}
    rows6 = exp6_table(e6)
    assert rows6[0]["regression_count"] == 0
    assert "| " in render_markdown(rows6, columns=["mod_id", "regression_count"])
