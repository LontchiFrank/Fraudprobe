"""Task 4: measure value preservation properly.

Before this task the only check was `total_value(corpus) <= total_value(seed) *
len(strategies) + 1.0`, which only rules out the corpus moving more than 3x the
seed total — a mutation retaining 1% of the original value would pass. This
verifies the real per-strategy retention distribution, and that mutations which
abandon most of the money are excluded from the corpus rather than counted as an
evasion.
"""

from __future__ import annotations

import pytest

from fraudprobe.adversary import AMOUNT_SPLIT_PARTS, generate_adversarial_corpus


def test_rules_amount_split_retains_full_value(seed_frauds):
    _, report = generate_adversarial_corpus(seed_frauds, strategies=("amount_split",), backend="rules", seed=1)
    vr = report.value_retention["amount_split"]
    assert vr["n"] == len(seed_frauds)
    assert vr["mean"] == pytest.approx(1.0, abs=0.001)
    assert vr["min"] == pytest.approx(1.0, abs=0.001)
    assert vr["prop_within_1pct"] == 1.0


def test_rules_temporal_dispersion_retains_full_value(seed_frauds):
    _, report = generate_adversarial_corpus(
        seed_frauds, strategies=("temporal_dispersion",), backend="rules", seed=1
    )
    vr = report.value_retention["temporal_dispersion"]
    assert vr["mean"] == pytest.approx(1.0, abs=1e-9)


def test_rules_balance_camouflage_retains_honest_partial_value(seed_frauds):
    # The residual is deliberate — should land close to but below 1.0, and must be
    # reported as such rather than forced to 1.0.
    _, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="rules", seed=1
    )
    vr = report.value_retention["balance_camouflage"]
    assert 0.90 < vr["mean"] < 1.0
    assert vr["mean"] != 1.0


def test_low_retention_group_is_dropped_from_corpus_but_still_recorded(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    # Retains only 5% of the original value — economically valid (balances left
    # self-consistent) but abandons almost all the money.
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": '{"amount": 500.0, "oldbalanceOrg": 10000.0, "newbalanceOrig": 9500.0,'
                                   ' "oldbalanceDest": 0.0, "newbalanceDest": 500.0}'},
    )
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1,
    )
    assert report.n_rejected_low_value > 0
    assert "balance_camouflage" in report.value_retention  # still recorded...
    assert report.value_retention["balance_camouflage"]["min"] < 0.90
    assert corpus.empty  # ...but dropped from the corpus (only strategy under test)


def test_min_value_retention_threshold_is_configurable(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": '{"amount": 500.0, "oldbalanceOrg": 10000.0, "newbalanceOrig": 9500.0,'
                                   ' "oldbalanceDest": 0.0, "newbalanceDest": 500.0}'},
    )
    # 0.05 retention passes a permissive 0.01 threshold.
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1, min_value_retention=0.01,
    )
    assert report.n_rejected_low_value == 0
    assert not corpus.empty


def test_value_retention_reported_even_when_group_is_dropped(monkeypatch, seed_frauds):
    """The distribution must be honest, not survivor-biased by the drop filter."""
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": '{"amount": 100.0, "oldbalanceOrg": 10000.0, "newbalanceOrig": 9900.0,'
                                   ' "oldbalanceDest": 0.0, "newbalanceDest": 100.0}'},
    )
    _, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1,
    )
    vr = report.value_retention["balance_camouflage"]
    assert vr["n"] == len(seed_frauds)  # both seeds' groups counted, despite being dropped
    assert vr["mean"] < 0.02


def test_default_min_value_retention_is_090(seed_frauds):
    _, report = generate_adversarial_corpus(seed_frauds, backend="rules", seed=1)
    assert report.min_value_retention == pytest.approx(0.90)


def test_amount_split_row_count_unaffected_by_value_filter(seed_frauds):
    # Sanity: the value filter operates per (seed, strategy) group, not per row —
    # amount_split's 3 rows either all survive together or are dropped together.
    corpus, report = generate_adversarial_corpus(seed_frauds, strategies=("amount_split",), backend="rules", seed=1)
    assert len(corpus) == len(seed_frauds) * AMOUNT_SPLIT_PARTS
