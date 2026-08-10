"""REVIEW.md item 3: --figures / MANIFEST.json / results_tables were a no-op in
--n-runs (repeated) mode — the exact run shape the dissertation cites for
reportable results. This exercises the fix end to end."""

from __future__ import annotations

import json

from fraudprobe.pipeline import ProbeConfig, run_repeated


def _fast_cfg(tmp_path, **overrides):
    defaults = dict(
        demo=True, demo_rows=6_000, model_type="rf", tune=False, backend="rules",
        strategies=("amount_split", "balance_camouflage"), max_seeds=15, explain=False,
        out=str(tmp_path), seed=200, figures=True,
    )
    defaults.update(overrides)
    return ProbeConfig(**defaults)


def test_repeated_mode_writes_aggregate_figures(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=3)
    figdir = tmp_path / "figures"
    assert (figdir / "aggregate_per_strategy_evasion.png").exists()
    assert (figdir / "aggregate_per_strategy_evasion.pdf").exists()
    assert (figdir / "aggregate_detection_dropoff.png").exists()
    assert (figdir / "aggregate_detection_dropoff.pdf").exists()


def test_repeated_mode_writes_results_tables(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=3)
    tabledir = tmp_path / "results_tables"
    assert (tabledir / "aggregate_metrics.csv").exists()
    assert (tabledir / "per_strategy_wilson_ci.csv").exists()
    assert (tabledir / "per_run_summary.csv").exists()


def test_repeated_mode_writes_manifest_with_seed_list_and_total_wallclock(tmp_path):
    cfg = _fast_cfg(tmp_path)
    result = run_repeated(cfg, n_runs=3)
    manifest_path = tmp_path / "MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["mode"] == "repeated"
    assert manifest["seeds"] == result["seeds"]
    assert manifest["wall_clock_seconds"] > 0


def test_repeated_mode_without_figures_flag_writes_no_figures(tmp_path):
    cfg = _fast_cfg(tmp_path, figures=False)
    run_repeated(cfg, n_runs=2)
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "MANIFEST.json").exists()
    assert not (tmp_path / "results_tables").exists()


def test_repeated_mode_per_run_artefacts_include_figures_when_enabled(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=2)
    for seed in (200, 201):
        run_figdir = tmp_path / "runs" / f"seed_{seed}" / "figures"
        assert run_figdir.exists()
        assert any(run_figdir.glob("*.png"))
        assert (tmp_path / "runs" / f"seed_{seed}" / "MANIFEST.json").exists()
