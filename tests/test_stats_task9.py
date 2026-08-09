"""Task 9: repeat across seeds and compute statistics. Every saved result before
this was a single run at seed=42 — neither the paired significance test nor the
95% confidence intervals the methodology promises are computable from one
observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudprobe.stats import (
    WIDE_INTERVAL_N_THRESHOLD,
    format_aggregate_report,
    mean_std_ci95,
    paired_test,
    wilson_ci,
)


def test_mean_std_ci95_basic():
    s = mean_std_ci95([0.8, 0.9, 1.0])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.9)
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]


def test_mean_std_ci95_single_value_has_zero_width_interval():
    s = mean_std_ci95([0.5])
    assert s["n"] == 1
    assert s["std"] == 0.0
    assert s["ci95_low"] == s["ci95_high"] == 0.5


def test_mean_std_ci95_empty():
    s = mean_std_ci95([])
    assert s["n"] == 0
    assert s["mean"] is None


def test_mean_std_ci95_clips_to_bounds_near_a_hard_limit():
    # Values clustered near 1.0 with some spread push the symmetric t-based
    # interval's upper bound past 1.0 without clipping.
    unclipped = mean_std_ci95([0.98, 0.99, 1.0, 0.97, 1.0])
    assert unclipped["ci95_high"] > 1.0
    clipped = mean_std_ci95([0.98, 0.99, 1.0, 0.97, 1.0], bounds=(0.0, 1.0))
    assert clipped["ci95_high"] == 1.0
    assert clipped["ci95_low"] >= 0.0


def test_mean_std_ci95_bounds_do_not_affect_interior_values():
    a = mean_std_ci95([0.4, 0.5, 0.6])
    b = mean_std_ci95([0.4, 0.5, 0.6], bounds=(0.0, 1.0))
    assert a["ci95_low"] == pytest.approx(b["ci95_low"])
    assert a["ci95_high"] == pytest.approx(b["ci95_high"])


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    clean = rng.normal(0.95, 0.02, size=20)
    adv = clean - rng.normal(0.4, 0.03, size=20)  # consistently, substantially lower
    result = paired_test(clean, adv)
    assert result["ttest"]["p_value"] < 0.01
    assert result["wilcoxon"]["p_value"] < 0.01
    assert result["cohens_d_z"] > 1.0  # large effect
    assert result["mean_diff"] == pytest.approx(np.mean(clean - adv))


def test_paired_test_too_few_runs():
    result = paired_test([0.9], [0.5])
    assert result["ttest"] is None
    assert "note" in result


def test_paired_test_zero_variance_differences():
    result = paired_test([0.9, 0.9, 0.9], [0.5, 0.5, 0.5])
    assert result["ttest"] is None
    assert "note" in result


def test_paired_test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_test([0.9, 0.8], [0.5])


def test_wilson_ci_basic():
    w = wilson_ci(successes=50, n=100)
    assert w["proportion"] == pytest.approx(0.5)
    assert w["ci95_low"] < 0.5 < w["ci95_high"]
    assert w["n"] == 100


def test_wilson_ci_zero_n():
    w = wilson_ci(0, 0)
    assert w["proportion"] is None
    assert w["wide"] is True


def test_wilson_ci_extreme_proportion_stays_within_bounds():
    w = wilson_ci(successes=20, n=20)  # 100% evasion
    assert 0.0 <= w["ci95_low"] <= w["ci95_high"] <= 1.0
    assert w["proportion"] == 1.0


def test_wilson_ci_flags_small_samples_as_wide():
    small = wilson_ci(10, 20)
    large = wilson_ci(500, 1000)
    assert small["n"] < WIDE_INTERVAL_N_THRESHOLD
    assert small["wide"] is True
    assert large["n"] >= WIDE_INTERVAL_N_THRESHOLD
    assert large["wide"] is False
    # And the small-sample interval must actually be wider in absolute terms.
    assert (small["ci95_high"] - small["ci95_low"]) > (large["ci95_high"] - large["ci95_low"])


def test_format_aggregate_report_contains_key_sections():
    result = {
        "n_runs_completed": 3, "n_runs_requested": 3, "base_seed": 42,
        "backend": "rules", "llm_model": None,
        "aggregate_metrics": {
            "clean_fraud_detection_rate": mean_std_ci95([0.95, 0.96, 0.94]),
            "adversarial_detection_rate": mean_std_ci95([0.4, 0.45, 0.42]),
            "detection_drop_off": mean_std_ci95([0.55, 0.51, 0.52]),
            "evasion_rate": mean_std_ci95([0.6, 0.55, 0.58]),
            "baseline_f1": mean_std_ci95([0.97, 0.96, 0.98]),
        },
        "paired_significance_clean_vs_adversarial": paired_test(
            [0.95, 0.96, 0.94], [0.4, 0.45, 0.42]
        ),
        "per_strategy_wilson_ci": {
            "amount_split": wilson_ci(18, 20),
            "temporal_dispersion": wilson_ci(0, 20),
        },
    }
    text = format_aggregate_report(result)
    assert "AGGREGATE REPORT" in text
    assert "Clean detection rate" in text
    assert "Paired significance" in text
    assert "amount_split" in text
    assert "WIDE" in text  # both strategies pool to n=20 < threshold
