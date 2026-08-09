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

# Two response contracts, one per strategy shape:
#  - amount_split asks for {"proportions": [...]} — a list of >=2 positive numbers
#    (see _split_amount / _llm_prompt); it never asks the model to do the balance
#    arithmetic itself.
#  - every other strategy asks for a single finished row. "amount" is the only
#    field we treat as mandatory for schema-validity; the rest fall back to the
#    seed row's own values if the model omits them.
_LLM_RESPONSE_FIELDS = (
    "amount", "step", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
)
_LLM_REQUIRED_FIELDS = ("amount",)

# Reasons an LLM mutation can fall back to the rules implementation. Every fallback
# is labelled with exactly one of these — never swallowed silently.
FALLBACK_REASONS = (
    "import_error", "connection_error", "timeout", "json_parse_error", "schema_error", "other",
)


class LLMRequiredError(Exception):
    """Raised when --require-llm is set and a mutation fell back to the rules backend.

    Deliberately NOT a RuntimeError subclass — pipeline.py catches RuntimeError to
    mean "this backend produced an empty corpus, log it and move on" (see
    run_comparison). A --require-llm violation is a harder failure than that and
    must abort the run loudly instead of being logged and skipped.
    """


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
# amount_split's row-construction contract, shared by both backends (Task 2): a
# strategy maps one seed row to a DataFrame of one or more rows. The rules backend
# picks its own random weights; the LLM backend picks the *proportions* (how many
# parts, and their relative sizes) but the actual balance arithmetic is always done
# here in Python, never trusted to the model's own arithmetic.
def _split_amount(row: pd.Series, weights) -> pd.DataFrame:
    """Break one large transfer into len(weights) smaller ones that sum to the same value."""
    weights = np.clip(np.asarray(weights, dtype=float), 1e-6, None)
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


# Number of parts the rules backend splits into, and what the LLM is asked to
# match — keeping both at the same n is what makes per-strategy evasion rates
# comparable across backends (Task 2's core requirement).
AMOUNT_SPLIT_PARTS = 3


def _mutate_amount_split(row: pd.Series, rng, n: int = AMOUNT_SPLIT_PARTS) -> pd.DataFrame:
    """Break one large transfer into several smaller ones that sum to the same value."""
    weights = rng.uniform(0.5, 1.5, size=n)
    return _split_amount(row, weights)


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
def _llm_prompt(row: pd.Series, strategy: str) -> str:
    txn = row[["amount", "step", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].to_dict()
    if strategy == "amount_split":
        # We ask for PROPORTIONS, not finished rows: the model decides the shape of
        # the split (how many parts, how uneven), but the actual balance arithmetic
        # for each resulting row is always done in Python via _split_amount — never
        # trusted to the model, which is what keeps every row economically valid.
        return (
            "You are an adversarial testing agent evaluating a fraud detector. "
            f"Split this transaction's amount into exactly {AMOUNT_SPLIT_PARTS} parts that sum to "
            "the original amount, structured as a real 'structuring' attack (parts of uneven, "
            "plausible size — not a mechanical equal split). "
            "Respond ONLY with a JSON object: {\"proportions\": [p1, p2, ...]} where the p_i are "
            f"positive numbers (any scale — they will be normalised) and there are exactly "
            f"{AMOUNT_SPLIT_PARTS} of them.\n\n"
            f"Transaction: {txn}"
        )
    return (
        "You are an adversarial testing agent evaluating a fraud detector. "
        "Rewrite the following transaction so it looks less like fraud, using the "
        f"strategy '{strategy}', while preserving the total amount moved. "
        "Respond ONLY with a JSON object with keys: amount, step, oldbalanceOrg, "
        "newbalanceOrig, oldbalanceDest, newbalanceDest.\n\n"
        f"Transaction: {txn}"
    )


def _mutate_llm(row: pd.Series, rng, strategy: str, model: str) -> tuple[pd.DataFrame, bool, str | None]:
    """Ask a local Ollama model to perform the mutation.

    Returns ``(frame, used_llm, reason)``. ``used_llm`` is True only if the model's
    own output was used; on any failure the rules implementation is substituted so
    the pipeline always gets a usable row, but the failure is never swallowed —
    ``reason`` is always one of ``FALLBACK_REASONS`` when ``used_llm`` is False, and
    None when it succeeded. Kept import-light so the package installs and runs
    without ollama; that specific case is reported as ``import_error``.

    ``amount_split`` returns multiple rows (one per proportion the model chose),
    matching the rules backend's contract of one seed row -> one-or-more rows —
    see _llm_prompt for why this is proportions, not finished rows (Task 2).
    """
    try:
        import json

        import httpx  # transitive dependency of ollama; import alongside it
        import ollama  # type: ignore
    except ImportError:
        return _RULES[strategy](row, rng), False, "import_error"

    prompt = _llm_prompt(row, strategy)

    try:
        resp = ollama.generate(model=model, prompt=prompt)
    except ConnectionError:
        # The ollama client wraps httpx.ConnectError (server not running) as this.
        return _RULES[strategy](row, rng), False, "connection_error"
    except httpx.TimeoutException:
        return _RULES[strategy](row, rng), False, "timeout"
    except Exception:
        return _RULES[strategy](row, rng), False, "other"

    try:
        text = resp["response"]
        payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (KeyError, ValueError):
        # No '{'/'}' found (.index/.rindex raise ValueError), or invalid JSON.
        return _RULES[strategy](row, rng), False, "json_parse_error"

    if not isinstance(payload, dict):
        return _RULES[strategy](row, rng), False, "schema_error"

    if strategy == "amount_split":
        proportions = payload.get("proportions")
        if not isinstance(proportions, list) or len(proportions) < 2:
            return _RULES[strategy](row, rng), False, "schema_error"
        try:
            weights = [float(p) for p in proportions]
        except (TypeError, ValueError):
            return _RULES[strategy](row, rng), False, "schema_error"
        if any(w <= 0 for w in weights):
            return _RULES[strategy](row, rng), False, "schema_error"
        return _split_amount(row, weights), True, None

    if not all(f in payload for f in _LLM_REQUIRED_FIELDS):
        return _RULES[strategy](row, rng), False, "schema_error"

    try:
        r = row.copy()
        for k in _LLM_RESPONSE_FIELDS:
            if k in payload:
                r[k] = float(payload[k])
        return pd.DataFrame([r]), True, None
    except (TypeError, ValueError):
        # A field was present but not coercible to float (e.g. a string label).
        return _RULES[strategy](row, rng), False, "schema_error"


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
    # LLM provenance (Task 1). Zero/None throughout for the 'rules' backend, since
    # the LLM is never attempted — that is distinct from "attempted and failed".
    n_llm_attempted: int = 0
    n_llm_success: int = 0
    n_llm_fallback: int = 0
    llm_success_rate: float | None = None
    fallback_reasons: dict[str, int] = field(default_factory=dict)


def generate_adversarial_corpus(
    seed_frauds: pd.DataFrame,
    strategies=STRATEGIES,
    backend: str = "rules",
    llm_model: str = "llama3",
    seed: int = 42,
    require_llm: bool = False,
) -> tuple[pd.DataFrame, AdversaryReport]:
    """Produce the mutated corpus from a set of seed fraudulent rows.

    Every generated row is tagged with a ``source`` column: ``"llm"`` for rows the
    model itself produced, ``"rules_fallback"`` for LLM-backend rows that fell back
    after a failure, and ``"rules"`` for a genuine rules-backend run (never
    attempted the LLM at all — not the same thing as a fallback). This is what lets
    per-strategy evasion be recomputed for LLM-sourced rows only.

    If ``require_llm`` is set and any mutation falls back to rules, raises
    ``LLMRequiredError`` immediately rather than degrading the corpus — for
    producing a headline result that is guaranteed pure-LLM.
    """
    rng = np.random.default_rng(seed)
    generated, rejected = [], 0
    n_llm_attempted = n_llm_success = 0
    fallback_reasons: dict[str, int] = {}

    for _, row in seed_frauds.iterrows():
        for strat in strategies:
            if backend == "llm":
                n_llm_attempted += 1
                muts, used_llm, reason = _mutate_llm(row, rng, strat, llm_model)
                if used_llm:
                    n_llm_success += 1
                    source = "llm"
                else:
                    fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                    source = "rules_fallback"
                    if require_llm:
                        raise LLMRequiredError(
                            f"--require-llm set but a mutation fell back to rules "
                            f"(reason={reason}) after {n_llm_success}/{n_llm_attempted} "
                            f"LLM calls succeeded."
                        )
            else:
                muts = _RULES[strat](row, rng)
                source = "rules"

            for _, m in muts.iterrows():
                if is_economically_valid(m):
                    m = m.copy()
                    m["isFraud"] = 1
                    m["mutation"] = strat
                    m["source"] = source
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

    n_llm_fallback = sum(fallback_reasons.values())
    report = AdversaryReport(
        n_seed=len(seed_frauds),
        n_generated=len(corpus),
        n_rejected_invalid=rejected,
        value_preserved=value_ok,
        backend=backend,
        strategies=list(strategies),
        n_llm_attempted=n_llm_attempted,
        n_llm_success=n_llm_success,
        n_llm_fallback=n_llm_fallback,
        llm_success_rate=(n_llm_success / n_llm_attempted) if n_llm_attempted else None,
        fallback_reasons=fallback_reasons,
    )
    return corpus, report
