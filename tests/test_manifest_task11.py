"""Task 11: MANIFEST.json and results_tables/*.csv."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fraudprobe.manifest import write_manifest, write_results_tables


def _single_mode_result():
    return {
        "version": "0.1.0",
        "mode": "single",
        "seed": 42,
        "data_source": "synthetic:8000rows",
        "backend": "rules",
        "llm_model": None,
        "stress": {
            "per_strategy": {
                "amount_split": {"n": 30, "detection_rate": 0.2, "evasion_rate": 0.8},
                "balance_camouflage": {"n": 10, "detection_rate": 0.9, "evasion_rate": 0.1},
            },
            "threshold_sweep": {
                "grid": [{"threshold": 0.5, "clean_detection_rate": 0.9,
                          "adversarial_detection_rate": 0.4, "drop_off": 0.5}],
                "fixed_fpr": [{"target_fpr": 0.01, "threshold": 0.3, "actual_fpr": 0.01,
                              "clean_detection_rate": 0.9, "adversarial_detection_rate": 0.5,
                              "drop_off": 0.4}],
            },
        },
        "adversary": {
            "value_retention": {"amount_split": {"n": 10, "mean": 1.0, "median": 1.0,
                                                  "min": 1.0, "prop_within_1pct": 1.0}},
            "rejection_reasons": {"insufficient_funds": 2},
            "fallback_reasons": {},
        },
        "explanation": {
            "available": True,
            "evasion_levers": [{"feature": "amount_to_balance_ratio", "clean_mean_shap": 1.2,
                               "adversarial_mean_shap": 0.1, "fraud_signal_drop": 1.1}],
        },
    }


def test_write_manifest_structure(tmp_path):
    manifest = write_manifest(_single_mode_result(), tmp_path, wall_clock_seconds=12.3)
    assert (tmp_path / "MANIFEST.json").exists()
    on_disk = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert on_disk == manifest
    assert manifest["python_version"]
    assert manifest["wall_clock_seconds"] == 12.3
    assert "numpy" in manifest["package_versions"]
    assert manifest["package_versions"]["numpy"] is not None
    assert manifest["seed"] == 42
    assert manifest["data_source"] == "synthetic:8000rows"


def test_write_manifest_records_cli_invocation(tmp_path):
    manifest = write_manifest(_single_mode_result(), tmp_path, 1.0, cli_argv=["fraudprobe", "run", "--demo"])
    assert manifest["cli_invocation"] == ["fraudprobe", "run", "--demo"]


def test_write_manifest_ollama_none_for_rules_backend(tmp_path):
    manifest = write_manifest(_single_mode_result(), tmp_path, 1.0)
    assert manifest["ollama_model"] is None


def test_write_results_tables_single_mode(tmp_path):
    written = write_results_tables(_single_mode_result(), tmp_path)
    assert "per_strategy_evasion" in written
    assert "threshold_sweep" in written
    assert "threshold_fixed_fpr" in written
    assert "value_retention" in written
    assert "rejection_reasons" in written
    assert "shap_evasion_levers" in written
    assert "fallback_reasons" not in written  # empty dict -> not written

    df = pd.read_csv(tmp_path / "per_strategy_evasion.csv")
    assert set(df["strategy"]) == {"amount_split", "balance_camouflage"}


def test_write_results_tables_compare_mode(tmp_path):
    single = _single_mode_result()
    compare_result = {
        "mode": "compare",
        "variants": [
            {**single, "backend": "rules"},
            {**single, "backend": "llm"},
        ],
        "paired_backend_comparison": {
            "backend_a": "rules", "backend_b": "llm", "n_paired_groups": 5,
            "mean_evasion_a": 0.5, "mean_evasion_b": 0.6,
            "mean_difference_b_minus_a": 0.1,
            "difference_ci95": {"low": -0.1, "high": 0.3},
            "paired_test": {"n_pairs": 5, "mean_diff": -0.1},
        },
    }
    written = write_results_tables(compare_result, tmp_path)
    assert "per_strategy_evasion_rules" in written
    assert "per_strategy_evasion_llm" in written
    assert "paired_backend_comparison" in written
    pc_df = pd.read_csv(tmp_path / "paired_backend_comparison.csv")
    assert pc_df.loc[0, "n_paired_groups"] == 5
    assert pc_df.loc[0, "difference_ci95_low"] == -0.1


def test_write_results_tables_repeated_mode(tmp_path):
    result = {
        "mode": "repeated",
        "aggregate_metrics": {
            "clean_fraud_detection_rate": {"n": 3, "mean": 0.95, "std": 0.01,
                                           "ci95_low": 0.9, "ci95_high": 1.0},
        },
        "per_strategy_wilson_ci": {
            "amount_split": {"n": 60, "proportion": 0.8, "ci95_low": 0.7, "ci95_high": 0.9, "wide": False},
        },
        "per_run_summary": [
            {"seed": 1, "clean_detection_rate": 0.95, "adversarial_detection_rate": 0.5, "evasion_rate": 0.5},
        ],
    }
    written = write_results_tables(result, tmp_path)
    assert "aggregate_metrics" in written
    assert "per_strategy_wilson_ci" in written
    assert "per_run_summary" in written


def test_write_results_tables_handles_missing_sections_gracefully(tmp_path):
    written = write_results_tables({"mode": "single", "stress": {}, "adversary": {}}, tmp_path)
    assert written == []
