# tests/experiments/test_cost_sensitivity.py
from src.experiments.metrics.business import cost_sensitivity_curve


def test_cost_sensitivity_curve():
    # from confusion counts: 1 FP, 1 FN
    curve = cost_sensitivity_curve(fp=1, fn=1, c_fp=1.0, ratios=[1, 5, 10])
    # expected_cost = FP*c_fp + FN*(ratio*c_fp) = 1*1 + 1*ratio
    assert curve[0] == {"ratio": 1, "c_fp": 1.0, "c_fn": 1.0, "expected_cost": 2.0}
    assert curve[1]["expected_cost"] == 6.0    # 1 + 5
    assert curve[2]["expected_cost"] == 11.0   # 1 + 10
