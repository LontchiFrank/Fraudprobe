"""Schema-error retry (added mid-session after validating --require-llm at real
PaySim scale): a real llm+xgboost run against data/paysim.csv measured llama3
hitting schema_error on ~1 in 13 calls — at hundreds of calls per reportable
run, --require-llm was virtually guaranteed to abort with "nothing to
aggregate". Rather than dropping --require-llm's guarantee entirely, one
same-prompt retry is attempted on schema_error only (not other failure modes,
since e.g. retrying immediately after connection_error is pointless), and the
single-shot vs. with-retry success rates are both recorded so neither number
is lost.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudprobe.adversary import STRATEGIES, generate_adversarial_corpus, _mutate_llm


def _rng():
    return np.random.default_rng(0)


def _row(seed_frauds):
    return seed_frauds.iloc[0]


def test_first_attempt_success_is_not_marked_as_retried(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    calls = {"n": 0}

    def ok(**kwargs):
        calls["n"] += 1
        return {"response": '{"proportions": [0.5, 0.3, 0.2]}'}

    monkeypatch.setattr(ollama, "generate", ok)
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is True
    assert reason is None
    assert retried is False
    assert calls["n"] == 1  # no retry attempted


def test_schema_error_then_success_counts_as_retried(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    calls = {"n": 0}

    def flaky_then_ok(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"response": '{"amount": 999.0}'}  # missing "proportions" -> schema_error
        return {"response": '{"proportions": [0.5, 0.3, 0.2]}'}

    monkeypatch.setattr(ollama, "generate", flaky_then_ok)
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is True
    assert reason is None
    assert retried is True
    assert calls["n"] == 2  # exactly one retry, not more


def test_schema_error_on_both_attempts_falls_back_after_exactly_one_retry(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    calls = {"n": 0}

    def always_broken(**kwargs):
        calls["n"] += 1
        return {"response": '{"amount": 999.0}'}  # always missing "proportions"

    monkeypatch.setattr(ollama, "generate", always_broken)
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "schema_error"
    assert retried is True
    assert calls["n"] == 2  # exactly one retry — not an unbounded retry loop
    assert not frame.empty  # still falls back to a usable row


def test_non_schema_error_is_never_retried(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    calls = {"n": 0}

    def boom(**kwargs):
        calls["n"] += 1
        raise ConnectionError("down")

    monkeypatch.setattr(ollama, "generate", boom)
    frame, used_llm, reason, retried = _mutate_llm(_row(seed_frauds), _rng(), "amount_split", "llama3")
    assert used_llm is False
    assert reason == "connection_error"
    assert retried is False
    assert calls["n"] == 1  # no retry for a transport failure


def test_generate_adversarial_corpus_aggregates_retry_stats(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")

    # Explicit, ordered script of raw ollama.generate responses. seed_frauds has
    # 2 rows x STRATEGIES has 3 entries (amount_split, temporal_dispersion,
    # balance_camouflage) = 6 logical mutation requests, processed in that order.
    # "amount" is temporal_dispersion/balance_camouflage's only required field
    # (_LLM_REQUIRED_FIELDS) — a payload missing it is a genuine schema_error for
    # those strategies; amount_split instead requires "proportions".
    script = [
        {"response": '{"proportions": [0.5, 0.3, 0.2]}'},   # seed0/amount_split: success
        {"response": '{"step": 5}'},                         # seed0/temporal: schema_error #1 (no amount)
        {"response": '{"step": 5, "amount": 500.0}'},        # retry succeeds
        {"response": 'not json'},                            # seed0/balance: json_parse_error (no retry)
        {"response": '{"proportions": [0.5, 0.3, 0.2]}'},    # seed1/amount_split: success
        {"response": '{"step": 5}'},                         # seed1/temporal: schema_error #1 (no amount)
        {"response": '{"step": 5}'},                         # retry also schema_error (no amount)
        {"response": '{"step": 5, "amount": 500.0}'},        # seed1/balance: success
    ]
    idx = {"i": 0}

    def scripted(**kwargs):
        r = script[idx["i"]]
        idx["i"] += 1
        return r

    monkeypatch.setattr(ollama, "generate", scripted)
    corpus, report = generate_adversarial_corpus(
        seed_frauds, strategies=STRATEGIES, backend="llm", seed=1
    )
    n_expected = len(seed_frauds) * len(STRATEGIES)
    assert report.n_llm_attempted == n_expected  # logical requests, not raw HTTP calls
    assert idx["i"] == len(script)  # every scripted response was consumed exactly once

    # 4 logical requests succeeded (2 first-try, 1 retry-rescued... wait: seed0
    # amount_split first-try, seed0 temporal retry-rescued, seed1 amount_split
    # first-try, seed1 balance first-try) = 4 successes; seed0 balance
    # (json_parse_error) and seed1 temporal (schema_error x2) fail = 2 fallbacks.
    assert report.n_llm_success == 4
    assert report.n_llm_fallback == 2
    assert report.n_llm_first_attempt_success == 3  # seed0/amount_split, seed1/amount_split, seed1/balance
    assert report.n_llm_retried == 2  # seed0/temporal + seed1/temporal both hit schema_error first
    assert report.n_llm_retry_success == 1  # only seed0/temporal recovered
    assert report.llm_single_shot_success_rate == pytest.approx(3 / n_expected)
    assert report.llm_success_rate == pytest.approx(4 / n_expected)
    assert report.fallback_reasons == {"json_parse_error": 1, "schema_error": 1}
