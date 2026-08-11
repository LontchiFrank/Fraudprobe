"""REVIEW.md item 7: make_demo_data's fraud_rate default (0.013) is ~10x real
PaySim's ~0.129%, which invites the same demo-vs-real confusion that produced
the original validity gap TASKS.md was written to fix. Added --demo-fraud-rate,
defaulting the CLI (not the library function) to 0.00129."""

from __future__ import annotations

from fraudprobe.cli import build_parser
from fraudprobe.pipeline import ProbeConfig, run_probe


def test_cli_demo_fraud_rate_defaults_to_real_paysim_rate():
    args = build_parser().parse_args(["run", "--demo"])
    assert args.demo_fraud_rate == 0.00129


def test_cli_demo_fraud_rate_is_overridable():
    args = build_parser().parse_args(["run", "--demo", "--demo-fraud-rate", "0.05"])
    assert args.demo_fraud_rate == 0.05


def test_probeconfig_default_unchanged_for_backward_compatibility():
    # Direct API/library callers (and the existing test suite) construct
    # ProbeConfig without this field — must not silently change their fraud
    # counts. Only the CLI's own default changed.
    assert ProbeConfig().demo_fraud_rate == 0.013


def test_run_probe_honours_demo_fraud_rate(tmp_path):
    cfg = ProbeConfig(
        demo=True, demo_rows=20_000, demo_fraud_rate=0.00129, model_type="rf",
        tune=False, backend="rules", max_seeds=5, explain=False,
        out=str(tmp_path), seed=1,
    )
    result = run_probe(cfg)
    # observed_fraud_rate should land close to the requested 0.129%, not 1.3%.
    assert result["observed_fraud_rate"] < 0.005
    assert result["observed_fraud_rate"] > 0.0
