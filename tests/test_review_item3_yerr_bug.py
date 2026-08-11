"""Discovered while validating REVIEW.md item 3's fix against the real PaySim
file (Task 6 unblocked): a strategy with 0/n evasions produces a Wilson
ci95_low of ~2.8e-17 instead of exactly 0.0 (floating-point noise from
`center - half` in stats.wilson_ci, which is mathematically correct but not
exactly zero). Every plotting function that derives yerr as
`proportion - ci95_low` inherited a tiny negative from that noise, which
matplotlib's bar(..., yerr=...) rejects outright — non-fatal (generate_all_
figures catches it) but silently drops the chart. Confirmed reproducible
directly against stats.wilson_ci(0, 6) without any mocking.
"""

from __future__ import annotations

from fraudprobe.figures import (
    plot_aggregate_dropoff,
    plot_aggregate_evasion_bar,
    plot_evasion_bar_chart,
)
from fraudprobe.stats import wilson_ci


def test_wilson_ci_zero_evasions_reproduces_the_floating_point_noise():
    # This is the exact real-data cell that triggered the bug: temporal_dispersion
    # with 0/6 evasions. Documents the root cause; the fix lives in figures.py,
    # not stats.py, since wilson_ci's interval is mathematically correct.
    w = wilson_ci(0, 6)
    assert w["proportion"] == 0.0
    assert w["ci95_low"] >= 0.0  # true today; the bug was assuming this is == 0.0


def test_plot_evasion_bar_chart_survives_zero_evasion_strategy(tmp_path):
    stress = {
        "per_strategy": {
            "amount_split": {"n": 30, "detection_rate": 0.3333, "evasion_rate": 0.6667},
            "temporal_dispersion": {"n": 6, "detection_rate": 1.0, "evasion_rate": 0.0},
        }
    }
    plot_evasion_bar_chart(stress, tmp_path)  # must not raise
    assert (tmp_path / "per_strategy_evasion.png").exists()


def test_plot_aggregate_evasion_bar_survives_zero_evasion_strategy(tmp_path):
    per_strategy_wilson = {
        "temporal_dispersion": wilson_ci(0, 6),
        "amount_split": wilson_ci(20, 30),
    }
    plot_aggregate_evasion_bar(per_strategy_wilson, tmp_path)  # must not raise
    assert (tmp_path / "aggregate_per_strategy_evasion.png").exists()


def test_plot_aggregate_dropoff_survives_identical_clean_and_adversarial_rate(tmp_path):
    # clean_fraud_detection_rate == adversarial_detection_rate (mean and CI
    # identical) is the analogous zero-width-difference edge case for this chart.
    from fraudprobe.stats import mean_std_ci95

    same = mean_std_ci95([1.0, 1.0, 1.0], bounds=(0.0, 1.0))
    aggregate_metrics = {
        "clean_fraud_detection_rate": same,
        "adversarial_detection_rate": same,
        "detection_drop_off": mean_std_ci95([0.0, 0.0, 0.0], bounds=(-1.0, 1.0)),
    }
    plot_aggregate_dropoff(aggregate_metrics, tmp_path)  # must not raise
    assert (tmp_path / "aggregate_detection_dropoff.png").exists()
