"""Task 5: results run against synthetic data with temporal_dispersion must carry
a fixture-dependency warning through results.json and the activity log — the
weakness ranking those runs produce for that strategy is not a measured property
of real fraud.

(Task 12 added a second, unconditional synthetic-data warning for
txn_velocity_orig/dest_txn_count — see test_data_task12.py for the feature
itself, and test_fixture_warning_* below for how it composes with the
existing temporal_dispersion warning.)
"""

from __future__ import annotations

from fraudprobe.pipeline import ProbeConfig, _fixture_warnings, run_probe


def test_fixture_warning_present_for_synthetic_plus_temporal():
    warnings = _fixture_warnings(("temporal_dispersion", "amount_split"), "synthetic:20000rows")
    assert len(warnings) == 2
    temporal = next(w for w in warnings if "temporal_dispersion" in w)
    assert "synthetic" in temporal.lower() or "fixture" in temporal.lower()


def test_synthetic_data_always_carries_velocity_warning_regardless_of_strategy():
    warnings = _fixture_warnings(("amount_split", "balance_camouflage"), "synthetic:20000rows")
    assert len(warnings) == 1
    assert "txn_velocity_orig" in warnings[0]
    assert "dest_txn_count" in warnings[0]


def test_no_fixture_warnings_for_real_paysim_data():
    assert _fixture_warnings(("temporal_dispersion",), "paysim:PS_20174392719.csv") == []
    assert _fixture_warnings(("amount_split",), "paysim:PS_20174392719.csv") == []


def test_run_probe_surfaces_warning_in_result_and_log(tmp_path):
    logs = []
    cfg = ProbeConfig(
        demo=True, demo_rows=8_000, model_type="rf", tune=False, backend="rules",
        strategies=("temporal_dispersion",), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1, log=logs.append,
    )
    result = run_probe(cfg)
    assert len(result["warnings"]) == 2
    assert any("temporal_dispersion" in w for w in result["warnings"])
    assert any("txn_velocity_orig" in w for w in result["warnings"])
    assert any("WARNING" in line for line in logs)


def test_run_probe_only_velocity_warning_for_non_temporal_strategy(tmp_path):
    cfg = ProbeConfig(
        demo=True, demo_rows=8_000, model_type="rf", tune=False, backend="rules",
        strategies=("amount_split",), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1,
    )
    result = run_probe(cfg)
    assert len(result["warnings"]) == 1
    assert "txn_velocity_orig" in result["warnings"][0]
