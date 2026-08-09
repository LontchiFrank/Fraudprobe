"""Task 3: the economic validator must enforce that balances actually reconcile.

Before this task, is_economically_valid checked signs and that the origin couldn't
overspend, but never that newbalanceOrig/newbalanceDest were consistent with the
row's own amount — so a mutation could be "valid" while contradicting itself, which
is arithmetic a real bank would reject before any model ever ran.
"""

from __future__ import annotations

import pytest

from fraudprobe.adversary import (
    REJECTION_REASONS,
    generate_adversarial_corpus,
    is_economically_valid,
)


def _row(seed_frauds):
    return seed_frauds.iloc[0].copy()


def test_default_mode_is_strict(seed_frauds):
    row = _row(seed_frauds)
    row["newbalanceOrig"] = row["oldbalanceOrg"]  # amount "moved" but balance didn't change
    valid, reason = is_economically_valid(row)
    assert not valid
    assert reason == "orig_arithmetic"


def test_lenient_mode_ignores_arithmetic_mismatch(seed_frauds):
    row = _row(seed_frauds)
    row["newbalanceOrig"] = row["oldbalanceOrg"]
    valid, reason = is_economically_valid(row, mode="lenient")
    assert valid
    assert reason is None


def test_strict_mode_rejects_dest_arithmetic_mismatch(seed_frauds):
    row = _row(seed_frauds)
    row["newbalanceOrig"] = row["oldbalanceOrg"] - row["amount"]  # orig reconciles...
    row["newbalanceDest"] = row["oldbalanceDest"]  # ...but dest doesn't move
    valid, reason = is_economically_valid(row, mode="strict")
    assert not valid
    assert reason == "dest_arithmetic"


def test_strict_mode_accepts_reconciling_row(seed_frauds):
    row = _row(seed_frauds)
    row["newbalanceOrig"] = row["oldbalanceOrg"] - row["amount"]
    row["newbalanceDest"] = row["oldbalanceDest"] + row["amount"]
    valid, reason = is_economically_valid(row, mode="strict")
    assert valid
    assert reason is None


def test_reconciliation_within_tolerance_is_accepted(seed_frauds):
    row = _row(seed_frauds)
    row["newbalanceOrig"] = row["oldbalanceOrg"] - row["amount"] + 0.5  # within tol=1.0
    row["newbalanceDest"] = row["oldbalanceDest"] + row["amount"] - 0.5
    valid, reason = is_economically_valid(row, mode="strict")
    assert valid


def test_non_positive_amount(seed_frauds):
    row = _row(seed_frauds)
    row["amount"] = 0.0
    valid, reason = is_economically_valid(row)
    assert not valid and reason == "non_positive_amount"


@pytest.mark.parametrize("field", ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
def test_negative_balance_any_field(seed_frauds, field):
    row = _row(seed_frauds)
    row[field] = -5.0
    valid, reason = is_economically_valid(row)
    assert not valid and reason == "negative_balance"


def test_insufficient_funds(seed_frauds):
    row = _row(seed_frauds)
    row["amount"] = row["oldbalanceOrg"] * 10
    row["newbalanceOrig"] = 0.0  # keep orig_arithmetic from firing first
    valid, reason = is_economically_valid(row)
    assert not valid and reason == "insufficient_funds"


def test_unknown_mode_raises(seed_frauds):
    with pytest.raises(ValueError):
        is_economically_valid(_row(seed_frauds), mode="yolo")


def test_all_rejection_reasons_are_declared():
    # Every reason string used above must be one fraudprobe advertises via
    # REJECTION_REASONS, so results.json consumers can enumerate them upfront.
    exercised = {
        "non_positive_amount", "negative_balance", "insufficient_funds",
        "orig_arithmetic", "dest_arithmetic",
    }
    assert exercised == set(REJECTION_REASONS)


# --------------------------------------------------------------------------- #
# generate_adversarial_corpus: validation mode threading + rejection counting
# --------------------------------------------------------------------------- #
def test_rules_backend_rejects_nothing_under_strict(seed_frauds):
    # The rules mutations construct newbalance*/oldbalance* to reconcile by
    # construction, so switching the default to strict should reject zero rows.
    corpus, report = generate_adversarial_corpus(seed_frauds, backend="rules", seed=1, validation="strict")
    assert report.n_rejected_invalid == 0
    assert report.rejection_reasons == {}
    assert report.validation_mode == "strict"


def test_llm_backend_records_rejection_reasons_under_strict(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    # A response that changes 'amount' but leaves the balances untouched — passes
    # the old lenient checks (signs ok, doesn't overspend) but fails to reconcile.
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": '{"amount": 500.0}'},  # oldbalance/newbalance left as the seed's
    )
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1, validation="strict",
    )
    assert report.n_rejected_invalid > 0
    assert "orig_arithmetic" in report.rejection_reasons or "dest_arithmetic" in report.rejection_reasons

    # The exact same corpus generation under lenient validation keeps those rows.
    corpus_lenient, report_lenient = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1, validation="lenient",
    )
    assert report_lenient.n_rejected_invalid < report.n_rejected_invalid
    assert report_lenient.n_generated > report.n_generated
