"""Task 1: LLM provenance tracking and fallback categorisation.

(Task 2's cross-backend row-count contract lives in test_adversary_task2.py,
added once that task's implementation lands.)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from fraudprobe.adversary import (
    STRATEGIES,
    LLMRequiredError,
    _mutate_llm,
    generate_adversarial_corpus,
    is_economically_valid,
)


def _rng():
    return np.random.default_rng(0)


def _row(seed_frauds):
    return seed_frauds.iloc[0]


# --------------------------------------------------------------------------- #
# _mutate_llm: every failure mode is categorised, never swallowed silently
# --------------------------------------------------------------------------- #
def test_llm_import_error_is_categorised(monkeypatch, seed_frauds):
    monkeypatch.setitem(sys.modules, "ollama", None)
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "import_error"
    assert not frame.empty  # still falls back to a usable row


def test_llm_connection_error_is_categorised(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")

    def boom(**kwargs):
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(ollama, "generate", boom)
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "connection_error"
    assert not frame.empty


def test_llm_timeout_is_categorised(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    httpx = pytest.importorskip("httpx")

    def boom(**kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ollama, "generate", boom)
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "timeout"


def test_llm_json_parse_error_is_categorised(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": "not json at all"})
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "json_parse_error"


def test_llm_schema_error_missing_required_field(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"step": 12}'})
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_schema_error_non_numeric_field(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": "a lot of money"}'})
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"


def test_llm_success(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": 'Sure! {"amount": 4000.0, "oldbalanceOrg": 10000.0, '
                                   '"newbalanceOrig": 6000.0, "oldbalanceDest": 0.0, "newbalanceDest": 4000.0}'},
    )
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is True
    assert reason is None
    assert frame.iloc[0]["amount"] == 4000.0


def test_llm_unexpected_exception_is_other(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")

    def boom(**kwargs):
        raise RuntimeError("something unrelated broke")

    monkeypatch.setattr(ollama, "generate", boom)
    frame, used_llm, reason = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "other"


# --------------------------------------------------------------------------- #
# generate_adversarial_corpus: aggregate LLM provenance stats + source tagging
# --------------------------------------------------------------------------- #
def test_rules_backend_never_attempts_llm(seed_frauds):
    corpus, report = generate_adversarial_corpus(seed_frauds, backend="rules", seed=1)
    assert report.n_llm_attempted == 0
    assert report.n_llm_success == 0
    assert report.llm_success_rate is None
    assert report.fallback_reasons == {}
    assert set(corpus["source"].unique()) == {"rules"}


def test_llm_backend_reports_success_rate_and_fallback_reasons(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ConnectionError("down")
        return {"response": '{"amount": 999.0}'}

    monkeypatch.setattr(ollama, "generate", flaky)
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=STRATEGIES, backend="llm", seed=1
    )
    n_expected = len(seed_frauds) * len(STRATEGIES)
    assert report.n_llm_attempted == n_expected
    assert report.n_llm_success == n_expected // 2
    assert report.n_llm_fallback == n_expected - n_expected // 2
    assert report.llm_success_rate == pytest.approx(report.n_llm_success / n_expected)
    assert report.fallback_reasons == {"connection_error": n_expected - n_expected // 2}
    assert set(corpus["source"].unique()) <= {"llm", "rules_fallback"}
    assert "llm" in corpus["source"].unique()
    assert "rules_fallback" in corpus["source"].unique()


def test_require_llm_raises_on_first_fallback(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(LLMRequiredError):
        generate_adversarial_corpus(
            seed_frauds, strategies=STRATEGIES, backend="llm", seed=1, require_llm=True
        )


def test_require_llm_passes_when_llm_always_succeeds(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": 999.0}'})
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=STRATEGIES, backend="llm", seed=1, require_llm=True
    )
    assert report.n_llm_fallback == 0
    assert set(corpus["source"].unique()) == {"llm"}


def test_is_economically_valid_rejects_negative_amount(seed_frauds):
    row = _row(seed_frauds).copy()
    row["amount"] = -1.0
    assert not is_economically_valid(row)
