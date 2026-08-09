"""Task 9 integration: run_repeated() actually re-runs the pipeline across
distinct seeds and produces a well-formed aggregate."""

from __future__ import annotations

import json

from fraudprobe.pipeline import ProbeConfig, run_repeated


def _fast_cfg(tmp_path, **overrides):
    defaults = dict(
        demo=True, demo_rows=6_000, model_type="rf", tune=False, backend="rules",
        strategies=("amount_split", "balance_camouflage"), max_seeds=15, explain=False,
        out=str(tmp_path), seed=100,
    )
    defaults.update(overrides)
    return ProbeConfig(**defaults)


def test_run_repeated_completes_all_runs_with_distinct_seeds(tmp_path):
    cfg = _fast_cfg(tmp_path)
    result = run_repeated(cfg, n_runs=3)
    assert result["n_runs_completed"] == 3
    assert result["n_runs_requested"] == 3
    assert result["seeds"] == [100, 101, 102]
    assert len(set(result["seeds"])) == 3


def test_run_repeated_aggregate_metrics_structure(tmp_path):
    cfg = _fast_cfg(tmp_path)
    result = run_repeated(cfg, n_runs=3)
    for key in ["clean_fraud_detection_rate", "adversarial_detection_rate",
                "detection_drop_off", "evasion_rate", "baseline_f1"]:
        assert key in result["aggregate_metrics"]
        assert result["aggregate_metrics"][key]["n"] == 3


def test_run_repeated_writes_aggregate_json(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=3)
    out_file = tmp_path / "results_aggregate.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["mode"] == "repeated"
    assert data["n_runs_completed"] == 3


def test_run_repeated_does_not_write_per_run_artefacts(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=3)
    # Only the aggregate should land in outdir — no per-run adversarial_corpus.csv.
    assert not (tmp_path / "adversarial_corpus.csv").exists()
    assert not (tmp_path / "report.txt").exists()


def test_run_repeated_includes_paired_significance_and_wilson_ci(tmp_path):
    cfg = _fast_cfg(tmp_path)
    result = run_repeated(cfg, n_runs=3)
    assert "paired_significance_clean_vs_adversarial" in result
    assert "per_strategy_wilson_ci" in result
    assert "amount_split" in result["per_strategy_wilson_ci"]


def test_run_repeated_single_run_still_works(tmp_path):
    cfg = _fast_cfg(tmp_path)
    result = run_repeated(cfg, n_runs=1)
    assert result["n_runs_completed"] == 1
    sig = result["paired_significance_clean_vs_adversarial"]
    assert sig["ttest"] is None  # can't test significance from one run
    assert "note" in sig
