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

import time
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
# Reasons a mutated row can be rejected, in the order they are checked. Recorded
# per-reason on AdversaryReport.rejection_reasons (Task 3) so "how much evasion
# survives once impossible transactions are excluded" is an answerable question.
REJECTION_REASONS = (
    "non_positive_amount", "negative_balance", "insufficient_funds",
    "orig_arithmetic", "dest_arithmetic",
)
VALIDATION_MODES = ("lenient", "strict")


def is_economically_valid(row: pd.Series, tol: float = 1.0, mode: str = "strict") -> tuple[bool, str | None]:
    """A mutated row is admissible only if its arithmetic is physically possible.

    Returns ``(valid, reason)`` — ``reason`` is None when valid, else one of
    ``REJECTION_REASONS``. Two validation modes:

    - ``lenient`` (fraudprobe's original behaviour): checks signs and that the
      origin can't spend more than it holds, but never checks that the resulting
      balances actually reconcile.
    - ``strict`` (default): additionally requires ``newbalanceOrig ≈ oldbalanceOrg
      - amount`` and ``newbalanceDest ≈ oldbalanceDest + amount`` within ``tol``.
      Without this, a mutation can leave the ledger self-contradictory — arithmetic
      a real bank would reject at input validation, before any model ever runs —
      and that non-reconciliation can itself push a classifier away from "fraud"
      for reasons having nothing to do with the mutation strategy being tested.
    """
    if mode not in VALIDATION_MODES:
        raise ValueError(f"Unknown validation mode: {mode!r} (expected one of {VALIDATION_MODES})")

    if row["amount"] <= 0:
        return False, "non_positive_amount"
    if row["oldbalanceOrg"] < 0 or row["newbalanceOrig"] < 0 or row["oldbalanceDest"] < 0 or row["newbalanceDest"] < 0:
        return False, "negative_balance"
    # Origin can't spend more than it holds.
    if row["amount"] > row["oldbalanceOrg"] + tol:
        return False, "insufficient_funds"

    if mode == "strict":
        if abs(row["newbalanceOrig"] - (row["oldbalanceOrg"] - row["amount"])) > tol:
            return False, "orig_arithmetic"
        if abs(row["newbalanceDest"] - (row["oldbalanceDest"] + row["amount"])) > tol:
            return False, "dest_arithmetic"

    return True, None


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


def _call_ollama_and_parse(
    row: pd.Series, strategy: str, model: str, prompt: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """One Ollama call + response parse/validation. Returns ``(frame, None)`` on
    success or ``(None, reason)`` on failure, ``reason`` always one of
    ``FALLBACK_REASONS``. Never touches the rules fallback itself — that's
    `_mutate_llm`'s job, so it can decide whether to retry first.
    """
    import json

    import httpx  # transitive dependency of ollama; import alongside it
    import ollama  # type: ignore

    try:
        resp = ollama.generate(model=model, prompt=prompt)
    except httpx.TimeoutException:
        return None, "timeout"
    except (httpx.ConnectError, ConnectionError):
        # The ollama client raises httpx.ConnectError when the server isn't
        # running — NOT Python's builtin ConnectionError (httpx.ConnectError
        # is not a subclass of it). Catch both explicitly: this is the single
        # most likely failure mode in a long overnight run, and getting it
        # mislabelled as "other" would corrupt the fallback-reason breakdown
        # that's reported as a result (REVIEW.md item 5). httpx.TimeoutException
        # is caught first since it and ConnectError are unrelated siblings
        # under httpx.TransportError.
        return None, "connection_error"
    except Exception:
        return None, "other"

    try:
        text = resp["response"]
        payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (KeyError, ValueError):
        # No '{'/'}' found (.index/.rindex raise ValueError), or invalid JSON.
        return None, "json_parse_error"

    if not isinstance(payload, dict):
        return None, "schema_error"

    if strategy == "amount_split":
        proportions = payload.get("proportions")
        if not isinstance(proportions, list) or len(proportions) < 2:
            return None, "schema_error"
        try:
            weights = [float(p) for p in proportions]
        except (TypeError, ValueError):
            return None, "schema_error"
        if any(w <= 0 for w in weights):
            return None, "schema_error"
        return _split_amount(row, weights), None

    if not all(f in payload for f in _LLM_REQUIRED_FIELDS):
        return None, "schema_error"

    try:
        r = row.copy()
        for k in _LLM_RESPONSE_FIELDS:
            if k in payload:
                r[k] = float(payload[k])
        return pd.DataFrame([r]), None
    except (TypeError, ValueError):
        # A field was present but not coercible to float (e.g. a string label).
        return None, "schema_error"


def _mutate_llm(
    row: pd.Series, rng, strategy: str, model: str,
) -> tuple[pd.DataFrame, bool, str | None, bool]:
    """Ask a local Ollama model to perform the mutation.

    Returns ``(frame, used_llm, reason, retried)``. ``used_llm`` is True only if
    the model's own output was used; on any failure the rules implementation is
    substituted so the pipeline always gets a usable row, but the failure is
    never swallowed — ``reason`` is always one of ``FALLBACK_REASONS`` when
    ``used_llm`` is False, and None when it succeeded. Kept import-light so the
    package installs and runs without ollama; that specific case is reported as
    ``import_error``.

    ``retried`` is True iff a ``schema_error`` on the first attempt triggered
    exactly one same-prompt retry — regardless of whether that retry then
    succeeded. Scoped to ``schema_error`` only (not ``json_parse_error`` or the
    transport failures): a validated real-PaySim run measured llama3 hitting
    schema_error on roughly 1 in 13 calls, at a scale (hundreds of calls per
    run) where that's a near-certainty to occur at least once, but a
    same-prompt retry is pointless for e.g. connection_error (the server being
    down doesn't fix itself on the next call). This lets a caller compute both
    a single-shot success rate (would the model get it right first try) and a
    with-retry success rate (does one free retry rescue most of the misses)
    without conflating the two.

    ``amount_split`` returns multiple rows (one per proportion the model chose),
    matching the rules backend's contract of one seed row -> one-or-more rows —
    see _llm_prompt for why this is proportions, not finished rows (Task 2).
    """
    try:
        import ollama  # noqa: F401  — import-only probe; _call_ollama_and_parse does the real import
    except ImportError:
        return _RULES[strategy](row, rng), False, "import_error", False

    prompt = _llm_prompt(row, strategy)
    frame, reason = _call_ollama_and_parse(row, strategy, model, prompt)
    if frame is not None:
        return frame, True, None, False
    if reason != "schema_error":
        return _RULES[strategy](row, rng), False, reason, False

    frame, reason = _call_ollama_and_parse(row, strategy, model, prompt)
    if frame is not None:
        return frame, True, None, True
    return _RULES[strategy](row, rng), False, reason, True


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AdversaryReport:
    n_seed: int
    n_generated: int
    n_rejected_invalid: int
    backend: str
    strategies: list[str] = field(default_factory=list)
    # LLM provenance (Task 1). Zero/None throughout for the 'rules' backend, since
    # the LLM is never attempted — that is distinct from "attempted and failed".
    n_llm_attempted: int = 0
    n_llm_success: int = 0
    n_llm_fallback: int = 0
    llm_success_rate: float | None = None
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    # Schema-error retry (same prompt, one retry, schema_error only — see
    # _mutate_llm's docstring for why). n_llm_success above already counts a
    # recovered retry as a success ("with-retry" rate); these let a caller
    # recover the "single-shot" rate too: n_llm_first_attempt_success /
    # n_llm_attempted, vs. n_llm_success / n_llm_attempted for with-retry.
    n_llm_first_attempt_success: int = 0
    n_llm_retried: int = 0
    n_llm_retry_success: int = 0
    llm_single_shot_success_rate: float | None = None
    # Economic validation (Task 3).
    validation_mode: str = "strict"
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    # Value preservation (Task 4). value_retention[strategy] = {n, mean, median,
    # min, prop_within_1pct} computed over one retention ratio per (seed, strategy)
    # group that survived economic validation — i.e. this measures how much of the
    # money the *intervention* preserved, not an arbitrary per-row artefact.
    min_value_retention: float = 0.90
    n_rejected_low_value: int = 0
    value_retention: dict[str, dict[str, float]] = field(default_factory=dict)
    # LLM call timing (Task 10) — wall-clock cost is real at scale (~8-12s/call),
    # so a large run needs this logged and recorded, not just a success rate.
    llm_wall_clock_seconds: float = 0.0
    llm_mean_call_seconds: float | None = None
    llm_min_call_seconds: float | None = None
    llm_max_call_seconds: float | None = None


def generate_adversarial_corpus(
    seed_frauds: pd.DataFrame,
    strategies=STRATEGIES,
    backend: str = "rules",
    llm_model: str = "llama3",
    seed: int = 42,
    require_llm: bool = False,
    validation: str = "strict",
    min_value_retention: float = 0.90,
    log=lambda msg: None,
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

    ``validation`` is ``"strict"`` (default) or ``"lenient"`` — see
    ``is_economically_valid``. Rejections are counted per reason on the returned
    report so "how much evasion survives once impossible transactions are
    excluded" is answerable from the artefacts alone.

    ``min_value_retention`` (Task 4): for each (seed, strategy) pair, the fraction
    of the seed's original amount retained across whichever of its mutated rows
    passed economic validation. Groups below this threshold are dropped from the
    corpus entirely — an "evasion" that abandons most of the money isn't one — but
    every group's retention ratio is still recorded in ``value_retention`` (mean,
    median, min, proportion within ±1% of 1.0, per strategy) regardless of whether
    it was kept, so the reported distribution is honest rather than survivor-biased.

    Every generated row also carries a ``seed_idx`` column — the seed fraud's
    positional index in ``seed_frauds`` — so a rules-vs-LLM comparison sharing the
    same seed set (Task 10) can pair rows back to the specific underlying fraud
    each backend mutated, not just compare aggregate rates.

    ``log`` receives periodic progress lines during an LLM-backend run (every 10
    calls) with the running mean call latency and an ETA — real wall-clock cost at
    ~8-12s/call, so a large run needs visible progress, not a single number at the
    end. Full timing summary statistics land on the returned report regardless.
    """
    rng = np.random.default_rng(seed)
    generated, rejected = [], 0
    n_llm_attempted = n_llm_success = 0
    n_llm_first_attempt_success = n_llm_retried = n_llm_retry_success = 0
    fallback_reasons: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    n_rejected_low_value = 0
    retention_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}
    llm_call_latencies: list[float] = []
    total_llm_calls_planned = len(seed_frauds) * len(strategies) if backend == "llm" else 0

    for seed_idx, (_, row) in enumerate(seed_frauds.iterrows()):
        for strat in strategies:
            if backend == "llm":
                n_llm_attempted += 1
                t0 = time.perf_counter()
                muts, used_llm, reason, retried = _mutate_llm(row, rng, strat, llm_model)
                llm_call_latencies.append(time.perf_counter() - t0)
                if n_llm_attempted % 10 == 0 or n_llm_attempted == total_llm_calls_planned:
                    mean_lat = sum(llm_call_latencies) / len(llm_call_latencies)
                    remaining = total_llm_calls_planned - n_llm_attempted
                    eta = remaining * mean_lat
                    log(f"LLM call {n_llm_attempted}/{total_llm_calls_planned} "
                        f"(mean {mean_lat:.1f}s/call, ETA {eta/60:.1f} min)")
                if retried:
                    n_llm_retried += 1
                if used_llm:
                    n_llm_success += 1
                    if retried:
                        n_llm_retry_success += 1
                    else:
                        n_llm_first_attempt_success += 1
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

            valid_rows = []
            for _, m in muts.iterrows():
                valid, why_not = is_economically_valid(m, mode=validation)
                if valid:
                    m = m.copy()
                    m["isFraud"] = 1
                    m["mutation"] = strat
                    m["source"] = source
                    m["seed_idx"] = seed_idx
                    valid_rows.append(m)
                else:
                    rejected += 1
                    rejection_reasons[why_not] = rejection_reasons.get(why_not, 0) + 1

            if not valid_rows:
                continue  # nothing survived economic validation for this seed+strategy

            retained_fraction = sum(r["amount"] for r in valid_rows) / row["amount"] if row["amount"] else 0.0
            retention_by_strategy[strat].append(retained_fraction)

            if retained_fraction < min_value_retention:
                n_rejected_low_value += 1
                continue  # abandons too much of the money to count as an evasion

            generated.extend(valid_rows)

    corpus = pd.DataFrame(generated).reset_index(drop=True) if generated else pd.DataFrame()

    value_retention: dict[str, dict[str, float]] = {}
    for strat, values in retention_by_strategy.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        value_retention[strat] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "prop_within_1pct": float(np.mean(np.abs(arr - 1.0) <= 0.01)),
        }

    n_llm_fallback = sum(fallback_reasons.values())
    if llm_call_latencies:
        total_seconds = sum(llm_call_latencies)
        log(f"LLM calls complete: {len(llm_call_latencies)} calls, "
            f"{total_seconds:.1f}s total ({total_seconds/60:.1f} min), "
            f"mean {total_seconds/len(llm_call_latencies):.1f}s/call.")

    report = AdversaryReport(
        n_seed=len(seed_frauds),
        n_generated=len(corpus),
        n_rejected_invalid=rejected,
        backend=backend,
        strategies=list(strategies),
        n_llm_attempted=n_llm_attempted,
        n_llm_success=n_llm_success,
        n_llm_fallback=n_llm_fallback,
        llm_success_rate=(n_llm_success / n_llm_attempted) if n_llm_attempted else None,
        n_llm_first_attempt_success=n_llm_first_attempt_success,
        n_llm_retried=n_llm_retried,
        n_llm_retry_success=n_llm_retry_success,
        llm_single_shot_success_rate=(
            (n_llm_first_attempt_success / n_llm_attempted) if n_llm_attempted else None
        ),
        fallback_reasons=fallback_reasons,
        validation_mode=validation,
        rejection_reasons=rejection_reasons,
        min_value_retention=min_value_retention,
        n_rejected_low_value=n_rejected_low_value,
        value_retention=value_retention,
        llm_wall_clock_seconds=sum(llm_call_latencies),
        llm_mean_call_seconds=(sum(llm_call_latencies) / len(llm_call_latencies)) if llm_call_latencies else None,
        llm_min_call_seconds=min(llm_call_latencies) if llm_call_latencies else None,
        llm_max_call_seconds=max(llm_call_latencies) if llm_call_latencies else None,
    )
    return corpus, report
