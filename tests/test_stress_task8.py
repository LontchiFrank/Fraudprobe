"""Task 8: sweep the decision threshold. Before this, ScoredModel.threshold was
fixed at 0.5 throughout, so a reviewer could reasonably ask whether the measured
evasions survive a threshold a real fraud team would actually choose (tuned to a
false-positive budget, not a modelling convenience).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudprobe.stress import FPR_TARGETS, THRESHOLD_GRID, stress_test, threshold_sweep


class _FakeModel:
    """A ScoredModel stand-in that returns pre-set scores keyed by a 'score' column,
    so we can test threshold logic without training anything."""

    threshold = 0.5

    def score_rows(self, df: pd.DataFrame) -> np.ndarray:
        return df["score"].to_numpy(dtype=float)


def _df(scores, is_fraud):
    return pd.DataFrame({"score": scores, "isFraud": is_fraud, "mutation": ["amount_split"] * len(scores)})


def test_threshold_grid_spans_005_to_095():
    assert THRESHOLD_GRID[0] == pytest.approx(0.05)
    assert THRESHOLD_GRID[-1] == pytest.approx(0.95)
    assert len(THRESHOLD_GRID) == 19


def test_grid_detection_rates_match_manual_calculation():
    model = _FakeModel()
    clean_test = _df([0.9, 0.9, 0.3, 0.3], [1, 1, 0, 0])
    corpus = _df([0.4, 0.6], [1, 1])
    sweep = threshold_sweep(model, clean_test, corpus, thresholds=[0.5])
    point = sweep["grid"][0]
    # clean fraud: both score 0.9 -> both caught at t=0.5
    assert point["clean_detection_rate"] == 1.0
    # adversarial: 0.4 evades, 0.6 caught -> 50% detection
    assert point["adversarial_detection_rate"] == 0.5
    assert point["drop_off"] == pytest.approx(0.5)


def test_fixed_fpr_thresholds_achieve_approximately_target_rate():
    model = _FakeModel()
    rng = np.random.default_rng(0)
    n_legit = 2000
    legit_scores = rng.uniform(0, 0.3, size=n_legit)  # low scores, mostly correctly-scored legit
    fraud_scores = rng.uniform(0.6, 1.0, size=200)
    clean_test = _df(np.concatenate([legit_scores, fraud_scores]),
                      [0] * n_legit + [1] * 200)
    corpus = _df(rng.uniform(0.3, 0.7, size=50), [1] * 50)

    sweep = threshold_sweep(model, clean_test, corpus)
    for pt in sweep["fixed_fpr"]:
        # With 2000 legit rows the achievable granularity is 1/2000 = 0.05%; allow
        # a small margin for the target/achievable mismatch at low target rates.
        assert abs(pt["actual_fpr"] - pt["target_fpr"]) <= 0.003


def test_fixed_fpr_targets_are_001_005_01_percent():
    assert FPR_TARGETS == (0.001, 0.005, 0.01)


def test_higher_threshold_never_increases_detection():
    model = _FakeModel()
    rng = np.random.default_rng(1)
    clean_test = _df(rng.uniform(0, 1, size=500), rng.integers(0, 2, size=500))
    corpus = _df(rng.uniform(0, 1, size=200), [1] * 200)
    sweep = threshold_sweep(model, clean_test, corpus)
    clean_rates = [pt["clean_detection_rate"] for pt in sweep["grid"]]
    adv_rates = [pt["adversarial_detection_rate"] for pt in sweep["grid"]]
    assert all(a >= b for a, b in zip(clean_rates, clean_rates[1:]))
    assert all(a >= b for a, b in zip(adv_rates, adv_rates[1:]))


def test_empty_legit_population_yields_empty_fixed_fpr():
    model = _FakeModel()
    clean_test = _df([0.9, 0.9], [1, 1])  # no legit rows at all
    corpus = _df([0.4], [1])
    sweep = threshold_sweep(model, clean_test, corpus)
    assert sweep["fixed_fpr"] == []
    assert len(sweep["grid"]) == len(THRESHOLD_GRID)


def test_stress_test_includes_threshold_sweep():
    model = _FakeModel()
    clean_test = _df([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0])
    corpus = _df([0.4, 0.6], [1, 1])
    result = stress_test(model, clean_test, corpus)
    assert "threshold_sweep" in result
    assert "grid" in result["threshold_sweep"]
    assert "fixed_fpr" in result["threshold_sweep"]
