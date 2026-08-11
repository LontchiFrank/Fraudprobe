"""Data loading and feature engineering for PaySim-shaped transaction data."""

from __future__ import annotations

import numpy as np
import pandas as pd

# The PaySim schema fraudprobe expects.
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Hour-of-day activity profiles for the synthetic fixture (Task 5). PaySim's `step`
# is one simulated hour, so `step % 24` is hour-of-day. Documented assumption, not
# a measured PaySim statistic: legitimate traffic follows realistic daytime/
# business-hours activity, while fraud is weighted toward the early-morning hours
# (00:00-05:00) when a stolen account is least likely to be watched. Without this,
# hour_of_day carries no fraud signal at all, which would make any temporal-based
# mutation strategy trivially evade for reasons that have nothing to do with the
# mutation itself — a property of the fixture, not a finding about fraud detection.
_LEGIT_HOURLY_WEIGHTS = np.array([
    0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.8, 1.5, 2.2, 2.6, 2.8, 2.9,
    2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.0, 1.6, 1.2, 0.9, 0.6, 0.4,
])
_FRAUD_HOURLY_WEIGHTS = np.array([
    3.0, 3.2, 3.4, 3.5, 3.3, 2.8, 1.8, 1.0, 0.6, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 2.0, 2.6,
])
_SIM_DAYS = 31  # matches PaySim's ~744-hour (31-day) simulation window


def _sample_steps(rng: np.random.Generator, n: int, hourly_weights: np.ndarray) -> np.ndarray:
    """Sample `step` values (1..744) whose hour-of-day follows `hourly_weights`."""
    days = rng.integers(0, _SIM_DAYS, size=n)
    hours = rng.choice(24, size=n, p=hourly_weights / hourly_weights.sum())
    return days * 24 + hours + 1


# Explicit dtypes for the numeric PaySim columns (Task 6). float64->float32 and
# int64->int32/int8 roughly halves those columns' memory; the two identifier
# columns dominate memory regardless of dtype choice (high-cardinality strings)
# and are left as pandas' default. Together this keeps a full 6,362,620-row
# PaySim load in the low gigabytes without needing chunked reading. pandas'
# dtype= silently ignores any key here that isn't an actual column in the file,
# so this is safe to pass even against a malformed CSV — the RAW_COLUMNS check
# below still produces the friendly "Not a PaySim-shaped CSV" error in that case.
_PAYSIM_DTYPES = {
    "step": "int32",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
}


def load_paysim(path: str, sample_legit: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load the real PaySim CSV (Lopez-Rojas & Axelsson, 2016) from disk.

    Downcasts the numeric columns on read so the full file loads comfortably in
    memory (see _PAYSIM_DTYPES) rather than requiring chunked reading.

    If ``sample_legit`` is given, keeps every fraud row and a stratified random
    sample of ``sample_legit`` legitimate rows — for iterating on the full file's
    realistic (~0.129%) fraud prevalence without training on all 6M+ legitimate
    rows every run. Callers must report the *observed* fraud rate of whatever was
    actually trained on (see pipeline.py's ``observed_fraud_rate``), never assume
    it still matches PaySim's population rate after sampling.
    """
    df = pd.read_csv(path, dtype=_PAYSIM_DTYPES)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Not a PaySim-shaped CSV. Missing columns: {missing}")
    df = df[RAW_COLUMNS + [c for c in df.columns if c not in RAW_COLUMNS]]

    if sample_legit is not None:
        rng = np.random.default_rng(seed)
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]
        if sample_legit < len(legit):
            keep_idx = rng.choice(legit.index.to_numpy(), size=sample_legit, replace=False)
            legit = legit.loc[keep_idx]
        df = pd.concat([fraud, legit], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def make_demo_data(n_rows: int = 60_000, fraud_rate: float = 0.013, seed: int = 42) -> pd.DataFrame:
    """Generate a small PaySim-shaped dataset so the tool runs with zero setup.

    This is NOT a substitute for PaySim in research: it reproduces the schema and
    the headline structural regularities (fraud concentrated in TRANSFER/CASH_OUT,
    origin account drained to ~0, fraud clustered in the 00:00-05:00 hours per the
    documented assumption on _FRAUD_HOURLY_WEIGHTS above) so the pipeline is
    exercisable end to end. Use --data with the real PaySim CSV for any result you
    intend to report.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # --- legitimate traffic -------------------------------------------------
    legit_type = rng.choice(TXN_TYPES, size=n_legit, p=[0.22, 0.35, 0.02, 0.32, 0.09])
    legit_amount = np.round(rng.lognormal(mean=8.0, sigma=1.3, size=n_legit), 2)
    legit_old_org = np.round(legit_amount * rng.uniform(0.5, 25.0, size=n_legit), 2)
    legit_new_org = np.round(np.maximum(legit_old_org - legit_amount, 0.0), 2)
    legit_old_dest = np.round(rng.lognormal(mean=9.0, sigma=1.6, size=n_legit), 2)
    legit_new_dest = np.round(legit_old_dest + legit_amount, 2)

    legit = pd.DataFrame(
        {
            "step": _sample_steps(rng, n_legit, _LEGIT_HOURLY_WEIGHTS),
            "type": legit_type,
            "amount": legit_amount,
            "nameOrig": [f"C{i:09d}" for i in rng.integers(1e8, 9.9e8, size=n_legit)],
            "oldbalanceOrg": legit_old_org,
            "newbalanceOrig": legit_new_org,
            "nameDest": [f"C{i:09d}" for i in rng.integers(1e8, 9.9e8, size=n_legit)],
            "oldbalanceDest": legit_old_dest,
            "newbalanceDest": legit_new_dest,
            "isFraud": 0,
        }
    )

    # --- fraudulent traffic: drain the origin account, TRANSFER/CASH_OUT only
    fraud_old_org = np.round(rng.lognormal(mean=10.2, sigma=1.0, size=n_fraud), 2)
    fraud_amount = np.round(fraud_old_org * rng.uniform(0.92, 1.0, size=n_fraud), 2)
    fraud_old_dest = np.round(rng.lognormal(mean=6.0, sigma=2.0, size=n_fraud), 2)

    fraud = pd.DataFrame(
        {
            "step": _sample_steps(rng, n_fraud, _FRAUD_HOURLY_WEIGHTS),
            "type": rng.choice(["TRANSFER", "CASH_OUT"], size=n_fraud),
            "amount": fraud_amount,
            "nameOrig": [f"C{i:09d}" for i in rng.integers(1e8, 9.9e8, size=n_fraud)],
            "oldbalanceOrg": fraud_old_org,
            "newbalanceOrig": np.round(np.maximum(fraud_old_org - fraud_amount, 0.0), 2),
            "nameDest": [f"C{i:09d}" for i in rng.integers(1e8, 9.9e8, size=n_fraud)],
            "oldbalanceDest": fraud_old_dest,
            "newbalanceDest": np.round(fraud_old_dest + fraud_amount, 2),
            "isFraud": 1,
        }
    )

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# PaySim's `step` = 1 simulated hour. txn_velocity_orig counts same-nameOrig
# transactions in the preceding VELOCITY_WINDOW_STEPS hours, backward-looking
# only (never counting a later row) since a real-time fraud system can only
# see past transactions when scoring a new one.
VELOCITY_WINDOW_STEPS = 24


def _rolling_txn_count(df: pd.DataFrame, account_col: str, window: int) -> np.ndarray:
    """For each row, count rows sharing `account_col` whose `step` falls in
    [step - window, step] — i.e. a backward-looking rolling velocity, computed
    strictly from the rows present in `df` (see engineer_features docstring for
    what that means when `df` is a small adversarial corpus rather than the
    full dataset). Vectorized per account group via sorted-step searchsorted;
    a Python loop only over the number of *distinct* accounts, not rows.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    steps = df["step"].to_numpy()
    accounts = df[account_col].to_numpy()
    result = np.empty(n, dtype=np.int32)
    order = np.argsort(accounts, kind="stable")
    sorted_accounts = accounts[order]
    sorted_steps = steps[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_accounts[1:] != sorted_accounts[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_steps = sorted_steps[start:end]
        step_order = np.argsort(group_steps, kind="stable")
        gs = group_steps[step_order]
        lo = np.searchsorted(gs, gs - window, side="left")
        hi = np.searchsorted(gs, gs, side="right")  # inclusive of self and same-step ties
        counts = (hi - lo).astype(np.int32)
        result[order[start:end][step_order]] = counts
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the behavioural features the baseline classifier trains on.

    Returns a new frame with FEATURE_COLUMNS present. Deterministic and a pure
    function of `df` alone — no external state or transaction-history lookup —
    so it can be reapplied to mutated rows unchanged. This matters for
    `txn_velocity_orig`/`dest_txn_count` (Task 12) specifically: they are
    counted *only among the rows present in `df` itself*, never joined against
    a separate history table. Two consequences of that design choice, recorded
    here rather than left implicit:

    1. Called on the full clean dataset (training, or `clean_test` at score
       time), both features reflect genuine population-level history, since
       every real transaction that account made within the loaded data is
       present in `df`.
    2. Called on a small adversarial corpus alone (stress_test's `model.
       score_rows(corpus)`, SHAP's per-row attribution, etc.) — which is every
       call site except the clean-side ones — the account's real prior history
       from `clean_test` is *not* in `df`, so these features can only see the
       corpus's own sibling rows. Concretely: `amount_split` turns one seed
       transaction into `AMOUNT_SPLIT_PARTS` rows sharing the same `nameOrig`,
       `nameDest`, and `step`, so those rows genuinely earn a velocity count of
       `AMOUNT_SPLIT_PARTS` — a real structuring signal, not an artefact.
       `temporal_dispersion` and `balance_camouflage` each produce exactly one
       row per seed, so they see a velocity of 1 regardless of window size.
       A history-joined implementation was considered and rejected: it would
       require `ScoredModel` to carry the training population everywhere it's
       scored, breaking the "reapply to mutated rows unchanged" contract this
       docstring already promises and that Task 2's cross-backend row-count
       comparison depends on.
    """
    out = df.copy()

    out["balance_delta_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balance_delta_dest"] = out["newbalanceDest"] - out["oldbalanceDest"]

    # Does the arithmetic of the transaction actually add up?
    out["error_balance_orig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["error_balance_dest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]

    eps = 1e-9
    out["amount_to_balance_ratio"] = out["amount"] / (out["oldbalanceOrg"] + eps)
    out["orig_emptied"] = (out["newbalanceOrig"] <= 0.01).astype(int)
    out["dest_was_empty"] = (out["oldbalanceDest"] <= 0.01).astype(int)
    out["hour_of_day"] = out["step"] % 24
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))

    # Task 12: transaction velocity / destination behavioural consistency.
    out["txn_velocity_orig"] = _rolling_txn_count(out, "nameOrig", VELOCITY_WINDOW_STEPS)
    out["dest_txn_count"] = out.groupby("nameDest")["nameDest"].transform("count").astype(np.int32)

    for t in TXN_TYPES:
        out[f"type_{t}"] = (out["type"] == t).astype(int)

    return out


FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_delta_orig",
    "balance_delta_dest",
    "error_balance_orig",
    "error_balance_dest",
    "amount_to_balance_ratio",
    "orig_emptied",
    "dest_was_empty",
    "hour_of_day",
    "txn_velocity_orig",
    "dest_txn_count",
] + [f"type_{t}" for t in TXN_TYPES]
