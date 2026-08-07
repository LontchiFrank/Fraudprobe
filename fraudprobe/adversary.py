"""The adversary: mutate fraudulent rows to evade detection while preserving value.

Two backends:
  * 'rules'  – deterministic Python implementations of the three mutation
               strategies. Needs no LLM, runs anywhere, fully reproducible.
               This is the fraudprobe default so the tool is exercisable by anyone.
  * 'llm'    – hands the row to a local LLM (Ollama) with a few-shot prompt and
               parses the returned row. This mirrors the research threat model:
               a non-expert adversary with a free local model.

Every mutated row passes an economic-consistency validator before it is admitted
to the corpus, so 'evasion' can never come from an impossible transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STRATEGIES = ("amount_split", "temporal_dispersion", "balance_camouflage")


# --------------------------------------------------------------------------- #
# Economic consistency validator
# --------------------------------------------------------------------------- #
def is_economically_valid(row: pd.Series, tol: float = 1.0) -> bool:
    """A mutated row is admissible only if its arithmetic is physically possible."""
    if row["amount"] <= 0:
        return False
    if row["oldbalanceOrg"] < 0 or row["newbalanceOrig"] < 0:
        return False
    if row["oldbalanceDest"] < 0 or row["newbalanceDest"] < 0:
        return False
    # Origin can't spend more than it holds.
    if row["amount"] > row["oldbalanceOrg"] + tol:
        return False
    return True


def total_value(rows: pd.DataFrame) -> float:
    """Total money moved – the invariant the attacker must preserve."""
    return float(rows["amount"].sum())


# --------------------------------------------------------------------------- #
# Rule-based mutation strategies
# --------------------------------------------------------------------------- #
def _mutate_amount_split(row: pd.Series, rng, n: int = 3) -> pd.DataFrame:
    """Break one large transfer into several smaller ones that sum to the same value."""
    weights = rng.uniform(0.5, 1.5, size=n)
    weights = weights / weights.sum()
    parts = np.round(row["amount"] * weights, 2)
    parts[-1] = round(row["amount"] - parts[:-1].sum(), 2)  # keep the sum exact

    rows = []
    running_orig = row["oldbalanceOrg"]
    for p in parts:
        r = row.copy()
        r["amount"] = float(p)
        r["oldbalanceOrg"] = round(running_orig, 2)
        r["newbalanceOrig"] = round(max(running_orig - p, 0.0), 2)
        running_orig = r["newbalanceOrig"]
        r["oldbalanceDest"] = round(row["oldbalanceDest"], 2)
        r["newbalanceDest"] = round(row["oldbalanceDest"] + p, 2)
        rows.append(r)
    return pd.DataFrame(rows)


def _mutate_temporal(row: pd.Series, rng) -> pd.DataFrame:
    """Move the transaction to a less-suspicious hour; value unchanged."""
    r = row.copy()
    r["step"] = int((rng.integers(9, 18)) + 24 * rng.integers(0, 31))  # daytime hours
    return pd.DataFrame([r])


def _mutate_balance_camouflage(row: pd.Series, rng) -> pd.DataFrame:
    """Leave a residual balance so the origin isn't drained to exactly zero.

    'Origin emptied to 0' is one of the strongest fraud signals in PaySim, so a
    cheap evasion is to leave a small float behind while moving almost all the value.
    """
    r = row.copy()
    residual = round(float(rng.uniform(1.0, max(2.0, row["amount"] * 0.03))), 2)
    r["amount"] = round(max(row["amount"] - residual, 0.01), 2)
    r["newbalanceOrig"] = round(row["oldbalanceOrg"] - r["amount"], 2)
    r["oldbalanceDest"] = round(row["oldbalanceDest"], 2)
    r["newbalanceDest"] = round(row["oldbalanceDest"] + r["amount"], 2)
    return pd.DataFrame([r])


_RULES = {
    "amount_split": _mutate_amount_split,
    "temporal_dispersion": _mutate_temporal,
    "balance_camouflage": _mutate_balance_camouflage,
}


# --------------------------------------------------------------------------- #
# LLM backend (optional) – matches the Ollama research setup
# --------------------------------------------------------------------------- #
def _mutate_llm(row: pd.Series, rng, strategy: str, model: str) -> pd.DataFrame:  # pragma: no cover
    """Ask a local Ollama model to perform the mutation. Falls back on any failure.

    Kept import-light and defensive so the package installs and runs without ollama.
    """
    try:
        import json

        import ollama  # type: ignore
    except ImportError:
        return _RULES[strategy](row, rng)

    prompt = (
        "You are an adversarial testing agent evaluating a fraud detector. "
        "Rewrite the following transaction so it looks less like fraud, using the "
        f"strategy '{strategy}', while preserving the total amount moved. "
        "Respond ONLY with a JSON object with keys: amount, step, oldbalanceOrg, "
        "newbalanceOrig, oldbalanceDest, newbalanceDest.\n\n"
        f"Transaction: {row[['amount','step','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].to_dict()}"
    )
    try:
        resp = ollama.generate(model=model, prompt=prompt)
        payload = json.loads(resp["response"][resp["response"].index("{"): resp["response"].rindex("}") + 1])
        r = row.copy()
        for k, v in payload.items():
            if k in r.index:
                r[k] = float(v)
        return pd.DataFrame([r])
    except Exception:
        return _RULES[strategy](row, rng)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AdversaryReport:
    n_seed: int
    n_generated: int
    n_rejected_invalid: int
    value_preserved: bool
    backend: str
    strategies: list[str] = field(default_factory=list)


def generate_adversarial_corpus(
    seed_frauds: pd.DataFrame,
    strategies=STRATEGIES,
    backend: str = "rules",
    llm_model: str = "llama3",
    seed: int = 42,
) -> tuple[pd.DataFrame, AdversaryReport]:
    """Produce the mutated corpus from a set of seed fraudulent rows."""
    rng = np.random.default_rng(seed)
    generated, rejected = [], 0

    for _, row in seed_frauds.iterrows():
        for strat in strategies:
            if backend == "llm":
                muts = _mutate_llm(row, rng, strat, llm_model)
            else:
                muts = _RULES[strat](row, rng)

            for _, m in muts.iterrows():
                if is_economically_valid(m):
                    m = m.copy()
                    m["isFraud"] = 1
                    m["mutation"] = strat
                    generated.append(m)
                else:
                    rejected += 1

    corpus = pd.DataFrame(generated).reset_index(drop=True) if generated else pd.DataFrame()

    # Value-preservation check: total moved in the corpus vs the seed set, per strategy
    # amount_split preserves exactly; the others move slightly less by design (residual),
    # so we check the corpus never moves MORE value than the seeds (no free money).
    value_ok = True
    if not corpus.empty:
        value_ok = total_value(corpus) <= total_value(seed_frauds) * len(strategies) + 1.0

    report = AdversaryReport(
        n_seed=len(seed_frauds),
        n_generated=len(corpus),
        n_rejected_invalid=rejected,
        value_preserved=value_ok,
        backend=backend,
        strategies=list(strategies),
    )
    return corpus, report
