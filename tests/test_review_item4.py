"""REVIEW.md item 4: the superseded, near-vacuous value_preserved check
(total_value(corpus) <= total_value(seed) * len(strategies) + 1.0) shadowed
the real per-strategy value_retention numbers added in Task 4, with both
printed in the same report and no way to tell which was meaningful."""

from __future__ import annotations

from dataclasses import fields

from fraudprobe.adversary import AdversaryReport, generate_adversarial_corpus


def test_adversary_report_has_no_value_preserved_field():
    assert "value_preserved" not in {f.name for f in fields(AdversaryReport)}


def test_format_report_omits_economic_value_check_line(seed_frauds):
    from fraudprobe.stress import format_report

    corpus, adv_report = generate_adversarial_corpus(
        seed_frauds, strategies=("amount_split",), backend="rules", seed=1,
    )
    baseline = {"model_type": "xgboost", "f1": 0.9, "pr_auc": 0.9, "support_fraud": 10}
    stress = {
        "clean_fraud_detection_rate": 0.9, "adversarial_detection_rate": 0.5,
        "detection_drop_off": 0.4, "evasion_rate": 0.5, "per_strategy": {},
        "threshold_sweep": {"grid": [], "fixed_fpr": []},
    }
    report_text = format_report(baseline, stress, adv_report)
    assert "Economic value check" not in report_text
    assert "value retention" in report_text.lower() or "Value retention" in report_text
