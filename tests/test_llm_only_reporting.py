"""Dual per-strategy evasion reporting: once over all rows, once restricted to
source == "llm" only. Added alongside the schema-error retry as the safer
substitute for --require-llm at scale — this is the pure-LLM number
--require-llm was meant to guarantee, computed after the fact instead of by
aborting the entire run on the first fallback.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraudprobe.manifest import write_results_tables
from fraudprobe.pipeline import ProbeConfig, _pool_per_strategy_wilson, run_repeated
from fraudprobe.stress import format_report, stress_test


def _fast_cfg(tmp_path, **overrides):
    defaults = dict(
        demo=True, demo_rows=6_000, model_type="rf", tune=False,
        strategies=("amount_split", "balance_camouflage"), max_seeds=15, explain=False,
        out=str(tmp_path), seed=300,
    )
    defaults.update(overrides)
    return ProbeConfig(**defaults)


def test_stress_test_already_exposes_per_strategy_llm_only(seed_frauds):
    from fraudprobe.adversary import generate_adversarial_corpus
    from fraudprobe.data import make_demo_data
    from fraudprobe.models import train_baseline

    df = make_demo_data(n_rows=4000, seed=1)
    model, clean_test, _ = train_baseline(df, model_type="rf", seed=1, tune=False)
    clean_fraud = clean_test[clean_test["isFraud"] == 1].reset_index(drop=True)
    corpus, _ = generate_adversarial_corpus(
        seed_frauds, strategies=("amount_split",), backend="rules", seed=1,
    )
    stress = stress_test(model, clean_test, corpus)
    assert "per_strategy_llm_only" in stress
    assert stress["per_strategy_llm_only"] == {}  # rules backend: no row is ever source=="llm"


def test_pool_per_strategy_wilson_handles_missing_key_gracefully():
    per_run = [{"stress": {"per_strategy": {"amount_split": {"n": 10, "evasion_rate": 0.5}}}}]
    assert _pool_per_strategy_wilson(per_run, "per_strategy_llm_only") == {}


def test_run_repeated_rules_backend_has_empty_llm_only_wilson_ci(tmp_path):
    cfg = _fast_cfg(tmp_path, backend="rules")
    result = run_repeated(cfg, n_runs=2)
    assert result["per_strategy_wilson_ci_llm_only"] == {}
    assert result["per_strategy_wilson_ci"] != {}  # sanity: all-rows version is populated


def test_write_results_tables_single_mode_exports_llm_only_table(tmp_path):
    result = {
        "mode": "single", "backend": "llm", "llm_model": "llama3",
        "stress": {
            "per_strategy": {"amount_split": {"n": 10, "detection_rate": 0.5, "evasion_rate": 0.5}},
            "per_strategy_llm_only": {"amount_split": {"n": 6, "detection_rate": 0.3, "evasion_rate": 0.7}},
        },
        "adversary": {}, "explanation": None,
    }
    written = write_results_tables(result, tmp_path)
    assert "per_strategy_evasion" in written
    assert "per_strategy_evasion_llm_only" in written
    df = pd.read_csv(tmp_path / "per_strategy_evasion_llm_only.csv")
    assert df.loc[0, "n"] == 6
    assert df.loc[0, "evasion_rate"] == pytest.approx(0.7)


def test_write_results_tables_repeated_mode_exports_llm_only_wilson_table(tmp_path):
    result = {
        "mode": "repeated",
        "aggregate_metrics": {"clean_fraud_detection_rate": {"n": 2, "mean": 0.9, "std": 0.0,
                                                              "ci95_low": 0.9, "ci95_high": 0.9}},
        "per_strategy_wilson_ci": {"amount_split": {"n": 20, "proportion": 0.5,
                                                     "ci95_low": 0.3, "ci95_high": 0.7, "wide": True}},
        "per_strategy_wilson_ci_llm_only": {"amount_split": {"n": 12, "proportion": 0.6,
                                                              "ci95_low": 0.3, "ci95_high": 0.8, "wide": True}},
        "per_run_summary": [],
    }
    written = write_results_tables(result, tmp_path)
    assert "per_strategy_wilson_ci_llm_only" in written
    df = pd.read_csv(tmp_path / "per_strategy_wilson_ci_llm_only.csv")
    assert df.loc[0, "n"] == 12


def test_format_report_includes_llm_only_section_when_present():
    baseline = {"model_type": "xgboost", "f1": 0.9, "pr_auc": 0.9, "support_fraud": 10}
    stress = {
        "clean_fraud_detection_rate": 0.9, "adversarial_detection_rate": 0.4,
        "detection_drop_off": 0.5, "evasion_rate": 0.6,
        "per_strategy": {"amount_split": {"n": 10, "detection_rate": 0.4, "evasion_rate": 0.6}},
        "per_strategy_llm_only": {"amount_split": {"n": 6, "detection_rate": 0.2, "evasion_rate": 0.8}},
        "threshold_sweep": {"grid": [], "fixed_fpr": []},
    }

    class _FakeReport:
        backend, n_generated, n_rejected_invalid, validation_mode = "llm", 10, 0, "strict"
        rejection_reasons, n_rejected_low_value, value_retention = {}, 0, {}
        n_llm_attempted, n_llm_success, n_llm_fallback = 0, 0, 0
        llm_success_rate, fallback_reasons = None, {}
        llm_mean_call_seconds = None
        n_llm_retried = 0

    report_text = format_report(baseline, stress, _FakeReport())
    assert "source == 'llm' rows only" in report_text
    assert "80.0% evaded" in report_text


def test_format_report_omits_llm_only_section_when_absent():
    baseline = {"model_type": "xgboost", "f1": 0.9, "pr_auc": 0.9, "support_fraud": 10}
    stress = {
        "clean_fraud_detection_rate": 0.9, "adversarial_detection_rate": 0.4,
        "detection_drop_off": 0.5, "evasion_rate": 0.6,
        "per_strategy": {"amount_split": {"n": 10, "detection_rate": 0.4, "evasion_rate": 0.6}},
        "per_strategy_llm_only": {},
        "threshold_sweep": {"grid": [], "fixed_fpr": []},
    }

    class _FakeReport:
        backend, n_generated, n_rejected_invalid, validation_mode = "rules", 10, 0, "strict"
        rejection_reasons, n_rejected_low_value, value_retention = {}, 0, {}
        n_llm_attempted, n_llm_success, n_llm_fallback = 0, 0, 0
        llm_success_rate, fallback_reasons = None, {}
        llm_mean_call_seconds = None
        n_llm_retried = 0

    report_text = format_report(baseline, stress, _FakeReport())
    assert "source == 'llm' rows only" not in report_text
