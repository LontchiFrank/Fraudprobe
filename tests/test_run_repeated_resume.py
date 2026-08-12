"""run_repeated resumability: added after the mid-session decision to run
reduced-scope --n-runs invocations overnight via nohup — a crash or kill
partway through must not lose already-completed seeds' work, since each
individual seed's artefacts (and, for the llm backend, real API calls) are
expensive to reproduce.
"""

from __future__ import annotations

import fraudprobe.pipeline as pipeline_module
from fraudprobe.pipeline import ProbeConfig, run_repeated


def _fast_cfg(tmp_path, **overrides):
    defaults = dict(
        demo=True, demo_rows=6_000, model_type="rf", tune=False, backend="rules",
        strategies=("amount_split", "balance_camouflage"), max_seeds=15, explain=False,
        out=str(tmp_path), seed=500,
    )
    defaults.update(overrides)
    return ProbeConfig(**defaults)


def test_second_invocation_reuses_all_cached_seeds_without_recomputing(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=2)  # first pass: writes runs/seed_500, runs/seed_501

    calls = {"n": 0}
    real_run_probe = pipeline_module.run_probe

    def counting_run_probe(run_cfg):
        calls["n"] += 1
        return real_run_probe(run_cfg)

    import fraudprobe.pipeline as pm
    original = pm.run_probe
    pm.run_probe = counting_run_probe
    try:
        result = run_repeated(cfg, n_runs=2)
    finally:
        pm.run_probe = original

    assert calls["n"] == 0  # both seeds fully cached — run_probe never called again
    assert result["n_runs_completed"] == 2
    assert result["seeds"] == [500, 501]


def test_extending_n_runs_only_computes_the_new_seed(tmp_path):
    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=2)  # writes runs/seed_500, runs/seed_501

    computed_seeds = []
    real_run_probe = pipeline_module.run_probe

    def tracking_run_probe(run_cfg):
        computed_seeds.append(run_cfg.seed)
        return real_run_probe(run_cfg)

    import fraudprobe.pipeline as pm
    original = pm.run_probe
    pm.run_probe = tracking_run_probe
    try:
        result = run_repeated(cfg, n_runs=3)  # one new seed: 502
    finally:
        pm.run_probe = original

    assert computed_seeds == [502]  # only the new seed actually ran
    assert result["n_runs_completed"] == 3
    assert result["seeds"] == [500, 501, 502]
    assert (tmp_path / "runs" / "seed_502" / "results.json").exists()


def test_resume_message_logged_for_cached_seeds(tmp_path):
    logs = []
    cfg = _fast_cfg(tmp_path, log=logs.append)
    run_repeated(cfg, n_runs=1)
    logs.clear()
    run_repeated(cfg, n_runs=1)
    assert any("resume" in line.lower() for line in logs)


def test_deleting_a_seed_dir_forces_it_to_rerun(tmp_path):
    import shutil

    cfg = _fast_cfg(tmp_path)
    run_repeated(cfg, n_runs=2)
    shutil.rmtree(tmp_path / "runs" / "seed_501")

    computed_seeds = []
    real_run_probe = pipeline_module.run_probe

    def tracking_run_probe(run_cfg):
        computed_seeds.append(run_cfg.seed)
        return real_run_probe(run_cfg)

    import fraudprobe.pipeline as pm
    original = pm.run_probe
    pm.run_probe = tracking_run_probe
    try:
        run_repeated(cfg, n_runs=2)
    finally:
        pm.run_probe = original

    assert computed_seeds == [501]  # only the deleted seed reruns
