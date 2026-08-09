"""Task 10: run the rules-vs-LLM comparison at scale.

run_comparison already shared the seed set across backends (Task 10 point 2 was
already satisfied) — what was missing: a statistical test of the *difference* in
evasion rate between backends (with a CI on the difference), and visibility into
LLM wall-clock cost. Both are exercised here against a mocked Ollama so the tests
stay fast; a live small-scale run is verified separately against the real server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudprobe.adversary import generate_adversarial_corpus
from fraudprobe.pipeline import ProbeConfig, _paired_backend_comparison, run_comparison


def test_generated_corpus_carries_seed_idx(seed_frauds):
    corpus, _ = generate_adversarial_corpus(seed_frauds, strategies=("amount_split",), backend="rules", seed=1)
    assert "seed_idx" in corpus.columns
    assert set(corpus["seed_idx"].unique()) == set(range(len(seed_frauds)))


def test_llm_timing_fields_populated(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": 999.0}'})
    _, report = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1
    )
    assert report.llm_wall_clock_seconds >= 0.0
    assert report.llm_mean_call_seconds is not None
    assert report.llm_min_call_seconds is not None
    assert report.llm_max_call_seconds is not None
    assert report.llm_min_call_seconds <= report.llm_mean_call_seconds <= report.llm_max_call_seconds


def test_llm_timing_fields_none_for_rules_backend(seed_frauds):
    _, report = generate_adversarial_corpus(seed_frauds, backend="rules", seed=1)
    assert report.llm_wall_clock_seconds == 0.0
    assert report.llm_mean_call_seconds is None


def test_progress_log_called_for_llm_backend(monkeypatch, seed_frauds):
    ollama = pytest.importorskip("ollama")
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": '{"amount": 999.0}'})
    logs = []
    generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="llm", seed=1, log=logs.append,
    )
    # 2 seeds x 1 strategy = 2 calls; the final call always logs (progress or completion).
    assert any("LLM call" in m or "LLM calls complete" in m for m in logs)


# --------------------------------------------------------------------------- #
# _paired_backend_comparison
# --------------------------------------------------------------------------- #
class _FakeModel:
    threshold = 0.5

    def score_rows(self, df: pd.DataFrame) -> np.ndarray:
        return df["score"].to_numpy(dtype=float)


def _corpus(seed_idxs, mutations, scores):
    return pd.DataFrame({"seed_idx": seed_idxs, "mutation": mutations, "score": scores})


def test_paired_comparison_finds_a_real_difference():
    model = _FakeModel()
    # 3 seeds x 1 strategy each. backend a evades all (low score); backend b is caught (high score).
    a = _corpus([0, 1, 2], ["amount_split"] * 3, [0.1, 0.1, 0.1])
    b = _corpus([0, 1, 2], ["amount_split"] * 3, [0.9, 0.9, 0.9])
    result = _paired_backend_comparison(model, {"rules": a, "llm": b})
    assert result["n_paired_groups"] == 3
    assert result["mean_evasion_a"] == 1.0
    assert result["mean_evasion_b"] == 0.0
    assert result["mean_difference_b_minus_a"] == pytest.approx(-1.0)


def test_paired_comparison_only_pairs_shared_seed_strategy_groups():
    model = _FakeModel()
    a = _corpus([0, 1, 2], ["amount_split"] * 3, [0.1, 0.1, 0.1])
    b = _corpus([0, 1], ["amount_split"] * 2, [0.9, 0.9])  # seed 2 missing from b
    result = _paired_backend_comparison(model, {"rules": a, "llm": b})
    assert result["n_paired_groups"] == 2


def test_paired_comparison_none_when_not_exactly_two_backends():
    model = _FakeModel()
    a = _corpus([0], ["amount_split"], [0.1])
    assert _paired_backend_comparison(model, {"rules": a}) is None
    assert _paired_backend_comparison(model, {}) is None


def test_paired_comparison_none_when_no_overlap():
    model = _FakeModel()
    a = _corpus([0, 1], ["amount_split", "amount_split"], [0.1, 0.1])
    b = _corpus([0, 1], ["balance_camouflage", "balance_camouflage"], [0.9, 0.9])
    assert _paired_backend_comparison(model, {"rules": a, "llm": b}) is None


def test_paired_comparison_none_when_corpus_empty():
    model = _FakeModel()
    a = _corpus([0], ["amount_split"], [0.1])
    b = pd.DataFrame(columns=["seed_idx", "mutation", "score"])
    assert _paired_backend_comparison(model, {"rules": a, "llm": b}) is None


# --------------------------------------------------------------------------- #
# run_comparison integration (mocked LLM for speed)
# --------------------------------------------------------------------------- #
def test_run_comparison_includes_paired_backend_comparison(tmp_path, monkeypatch):
    ollama = pytest.importorskip("ollama")

    def fake_generate(**kw):
        if "proportions" in kw.get("prompt", ""):
            return {"response": '{"proportions": [0.5, 0.3, 0.2]}'}
        return {"response": '{"amount": 9000.0, "oldbalanceOrg": 10000.0, "newbalanceOrig": 1000.0,'
                             ' "oldbalanceDest": 0.0, "newbalanceDest": 9000.0}'}

    monkeypatch.setattr(ollama, "generate", fake_generate)
    cfg = ProbeConfig(
        demo=True, demo_rows=6_000, model_type="rf", tune=False,
        strategies=("amount_split", "balance_camouflage"), max_seeds=10, explain=False,
        out=str(tmp_path), seed=1,
    )
    result = run_comparison(cfg)
    assert len(result["variants"]) == 2
    pc = result["paired_backend_comparison"]
    assert pc is not None
    assert {pc["backend_a"], pc["backend_b"]} == {"rules", "llm"}
    assert pc["n_paired_groups"] > 0
    assert "paired_test" in pc
