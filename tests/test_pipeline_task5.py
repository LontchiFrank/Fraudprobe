"""Task 5: results run against synthetic data with temporal_dispersion must carry
a fixture-dependency warning through results.json and the activity log — the
weakness ranking those runs produce for that strategy is not a measured property
of real fraud.
"""

from __future__ import annotations

from fraudprobe.pipeline import ProbeConfig, _fixture_warnings, run_probe


def test_fixture_warning_present_for_synthetic_plus_temporal():
    warnings = _fixture_warnings(("temporal_dispersion", "amount_split"), "synthetic:20000rows")
    assert len(warnings) == 1
    assert "temporal_dispersion" in warnings[0]
    assert "synthetic" in warnings[0].lower() or "fixture" in warnings[0].lower()


def test_no_fixture_warning_without_temporal_strategy():
    assert _fixture_warnings(("amount_split", "balance_camouflage"), "synthetic:20000rows") == []


def test_no_fixture_warning_for_real_paysim_data():
    assert _fixture_warnings(("temporal_dispersion",), "paysim:PS_20174392719.csv") == []


def test_run_probe_surfaces_warning_in_result_and_log(tmp_path):
    logs = []
    cfg = ProbeConfig(
        demo=True, demo_rows=8_000, model_type="rf", tune=False, backend="rules",
        strategies=("temporal_dispersion",), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1, log=logs.append,
    )
    result = run_probe(cfg)
    assert len(result["warnings"]) == 1
    assert "temporal_dispersion" in result["warnings"][0]
    assert any("WARNING" in line for line in logs)


def test_run_probe_no_warning_for_non_temporal_strategy(tmp_path):
    cfg = ProbeConfig(
        demo=True, demo_rows=8_000, model_type="rf", tune=False, backend="rules",
        strategies=("amount_split",), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1,
    )
    result = run_probe(cfg)
    assert result["warnings"] == []
