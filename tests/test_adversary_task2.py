"""Task 2: the rules and LLM backends must implement the same intervention.

Before this task, rules' amount_split emitted 3 rows per seed (a real structuring
attack) while the LLM path emitted 1 row for every strategy — so "amount_split"
meant two different things depending on backend, and per-strategy evasion rates
were not comparable across them. The fix: both backends map one seed row to a
DataFrame of one-or-more rows, with the LLM asked for split *proportions* (not
finished rows) so the actual balance arithmetic is always done in Python.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudprobe.adversary import (
    AMOUNT_SPLIT_PARTS,
    STRATEGIES,
    _mutate_llm,
    _split_amount,
    generate_adversarial_corpus,
    is_economically_valid,
    total_value,
)


def _rng():
    return np.random.default_rng(0)


def _row(seed_frauds):
    return seed_frauds.iloc[0]


def _mock_proportions_response(monkeypatch, proportions):
    ollama = pytest.importorskip("ollama")
    payload = "{" + f'"proportions": {list(proportions)}' + "}"
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": payload})
    return ollama


# --------------------------------------------------------------------------- #
# _split_amount: the shared row-construction both backends funnel through
# --------------------------------------------------------------------------- #
def test_split_amount_preserves_total_value(seed_frauds):
    row = _row(seed_frauds)
    frame = _split_amount(row, weights=[0.5, 0.3, 0.2])
    assert len(frame) == 3
    assert total_value(frame) == pytest.approx(row["amount"], abs=0.01)


def test_split_amount_normalises_arbitrary_scale_weights(seed_frauds):
    row = _row(seed_frauds)
    # Weights need not sum to 1 — arbitrary positive numbers, any scale.
    frame = _split_amount(row, weights=[5, 3, 2])
    assert len(frame) == 3
    assert total_value(frame) == pytest.approx(row["amount"], abs=0.01)


def test_split_amount_all_rows_economically_valid(seed_frauds):
    row = _row(seed_frauds)
    frame = _split_amount(row, weights=[0.4, 0.35, 0.25])
    results = [is_economically_valid(r) for _, r in frame.iterrows()]
    assert all(valid for valid, _reason in results), results


# --------------------------------------------------------------------------- #
# _mutate_llm: amount_split's proportions contract
# --------------------------------------------------------------------------- #
def test_llm_amount_split_success_returns_multiple_rows(monkeypatch, seed_frauds):
    _mock_proportions_response(monkeypatch, [0.5, 0.3, 0.2])
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is True
    assert reason is None
    assert len(frame) == 3
    assert total_value(frame) == pytest.approx(_row(seed_frauds)["amount"], abs=0.01)


def test_llm_amount_split_missing_proportions_is_schema_error(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": 999.0}'})
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_amount_split_too_few_proportions_is_schema_error(monkeypatch, seed_frauds):
    _mock_proportions_response(monkeypatch, [1.0])
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_amount_split_non_numeric_proportions_is_schema_error(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"proportions": ["a", "b"]}'})
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_amount_split_negative_proportion_is_schema_error(monkeypatch, seed_frauds):
    _mock_proportions_response(monkeypatch, [0.5, -0.3, 0.2])
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_amount_split_fallback_still_yields_rules_row_count(monkeypatch, seed_frauds):
    """Even when the LLM fails, the fallback keeps amount_split's multi-row shape."""
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": 999.0}'})
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert len(frame) == AMOUNT_SPLIT_PARTS


# --------------------------------------------------------------------------- #
# Cross-backend contract: same strategy + seed -> same row count
# --------------------------------------------------------------------------- #
def test_backends_return_same_row_count_for_amount_split(monkeypatch, seed_frauds):
    _mock_proportions_response(monkeypatch, [0.5, 0.3, 0.2])
    one_seed = seed_frauds.iloc[[0]]
    rules_corpus, _ = generate_adversarial_corpus(one_seed, strategies=("amount_split",), backend="rules", seed=1)
    llm_corpus, _ = generate_adversarial_corpus(one_seed, strategies=("amount_split",), backend="llm", seed=1)
    assert len(rules_corpus) == len(llm_corpus) == AMOUNT_SPLIT_PARTS


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_backends_return_same_row_count_for_every_strategy(monkeypatch, seed_frauds, strategy):
    if strategy == "amount_split":
        _mock_proportions_response(monkeypatch, [0.5, 0.3, 0.2])
    else:
        ollama = pytest.importorskip("ollama")
        monkeypatch.setattr(
            ollama, "generate",
            lambda **kw: {"response": '{"amount": 9000.0, "oldbalanceOrg": 10000.0, '
                                       '"newbalanceOrig": 1000.0, "oldbalanceDest": 0.0, "newbalanceDest": 9000.0}'},
        )
    one_seed = seed_frauds.iloc[[0]]
    rules_corpus, _ = generate_adversarial_corpus(one_seed, strategies=(strategy,), backend="rules", seed=1)
    llm_corpus, _ = generate_adversarial_corpus(one_seed, strategies=(strategy,), backend="llm", seed=1)
    assert len(rules_corpus) == len(llm_corpus), (
        f"backends disagree on row count for {strategy}: "
        f"rules={len(rules_corpus)} llm={len(llm_corpus)}"
    )


def test_full_corpus_amount_split_llm_produces_amount_split_parts_rows_per_seed(monkeypatch, seed_frauds):
    _mock_proportions_response(monkeypatch, [0.5, 0.3, 0.2])
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=("amount_split",), backend="llm", seed=1
    )
    assert len(corpus) == len(seed_frauds) * AMOUNT_SPLIT_PARTS
    assert set(corpus["source"].unique()) == {"llm"}
