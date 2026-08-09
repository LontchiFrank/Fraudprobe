"""Task 6: the real PaySim path must load memory-safely and report the fraud
prevalence it actually trained on. We don't have the real 6,362,620-row CSV in
this environment (needs Kaggle auth) — these tests exercise the same code path
against a small, correctly-shaped stand-in file, and a separate manual smoke test
(see FINDINGS.md / task notes) exercises it at a more realistic scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudprobe.data import RAW_COLUMNS, load_paysim, make_demo_data
from fraudprobe.pipeline import _data_stats


def _write_paysim_csv(tmp_path, n_rows=2000, fraud_rate=0.02, seed=1):
    df = make_demo_data(n_rows=n_rows, fraud_rate=fraud_rate, seed=seed)
    path = tmp_path / "paysim_standin.csv"
    df.to_csv(path, index=False)
    return path, df


def test_load_paysim_downcasts_numeric_dtypes(tmp_path):
    path, _ = _write_paysim_csv(tmp_path)
    df = load_paysim(str(path))
    assert df["step"].dtype == np.int32
    assert df["amount"].dtype == np.float32
    assert df["oldbalanceOrg"].dtype == np.float32
    assert df["newbalanceOrig"].dtype == np.float32
    assert df["oldbalanceDest"].dtype == np.float32
    assert df["newbalanceDest"].dtype == np.float32
    assert df["isFraud"].dtype == np.int8


def test_load_paysim_downcast_uses_less_memory_than_default(tmp_path):
    path, _ = _write_paysim_csv(tmp_path, n_rows=5000)
    downcast = load_paysim(str(path))
    default = pd.read_csv(path)
    assert downcast.memory_usage(deep=True).sum() < default.memory_usage(deep=True).sum()


def test_load_paysim_missing_columns_still_raises_friendly_error(tmp_path):
    path = tmp_path / "not_paysim.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Not a PaySim-shaped CSV"):
        load_paysim(str(path))


def test_load_paysim_column_order_preserved(tmp_path):
    path, _ = _write_paysim_csv(tmp_path)
    df = load_paysim(str(path))
    assert list(df.columns[: len(RAW_COLUMNS)]) == RAW_COLUMNS


def test_sample_legit_keeps_all_fraud(tmp_path):
    path, original = _write_paysim_csv(tmp_path, n_rows=5000, fraud_rate=0.02)
    n_fraud = int(original["isFraud"].sum())
    df = load_paysim(str(path), sample_legit=100, seed=1)
    assert int(df["isFraud"].sum()) == n_fraud
    assert len(df) == n_fraud + 100


def test_sample_legit_larger_than_available_keeps_all_legit(tmp_path):
    path, original = _write_paysim_csv(tmp_path, n_rows=1000, fraud_rate=0.02)
    n_legit = int((original["isFraud"] == 0).sum())
    df = load_paysim(str(path), sample_legit=10_000_000, seed=1)
    assert len(df) == len(original)
    assert int((df["isFraud"] == 0).sum()) == n_legit


def test_sample_legit_is_deterministic_given_seed(tmp_path):
    path, _ = _write_paysim_csv(tmp_path, n_rows=5000, fraud_rate=0.02)
    a = load_paysim(str(path), sample_legit=200, seed=7)
    b = load_paysim(str(path), sample_legit=200, seed=7)
    pd.testing.assert_frame_equal(a.sort_values("nameOrig").reset_index(drop=True),
                                  b.sort_values("nameOrig").reset_index(drop=True))


def test_sample_legit_changes_observed_prevalence():
    # Building the point of --sample-legit directly: a run on all-legit-kept data
    # has a much lower observed rate than one where legit rows are downsampled.
    df_full = make_demo_data(n_rows=20_000, fraud_rate=0.001, seed=1)
    full_stats = _data_stats(df_full)
    n_fraud = int(df_full["isFraud"].sum())
    df_sampled = pd.concat(
        [df_full[df_full["isFraud"] == 1], df_full[df_full["isFraud"] == 0].sample(n=500, random_state=1)]
    )
    sampled_stats = _data_stats(df_sampled)
    assert sampled_stats["observed_fraud_rate"] > full_stats["observed_fraud_rate"]
    assert sampled_stats["n_fraud"] == full_stats["n_fraud"] == n_fraud


# --------------------------------------------------------------------------- #
# _data_stats: n_rows/observed_fraud_rate must be reported for every run
# --------------------------------------------------------------------------- #
def test_data_stats_matches_synthetic_config():
    df = make_demo_data(n_rows=10_000, fraud_rate=0.013, seed=1)
    stats = _data_stats(df)
    assert stats["n_rows"] == 10_000
    assert stats["observed_fraud_rate"] == pytest.approx(0.013, abs=0.002)
    assert stats["n_fraud"] == int(df["isFraud"].sum())


def test_run_probe_reports_n_rows_and_observed_fraud_rate(tmp_path):
    from fraudprobe.pipeline import ProbeConfig, run_probe

    cfg = ProbeConfig(
        demo=True, demo_rows=8_000, model_type="rf", backend="rules",
        strategies=("amount_split",), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1,
    )
    result = run_probe(cfg)
    assert result["n_rows"] == 8_000
    assert "observed_fraud_rate" in result
    assert "n_fraud" in result
    assert result["n_fraud"] > 0
