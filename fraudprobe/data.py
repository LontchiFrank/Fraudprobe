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


def load_paysim(path: str) -> pd.DataFrame:
    """Load the real PaySim CSV (Lopez-Rojas & Axelsson, 2016) from disk."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Not a PaySim-shaped CSV. Missing columns: {missing}")
    return df[RAW_COLUMNS + [c for c in df.columns if c not in RAW_COLUMNS]]


def make_demo_data(n_rows: int = 60_000, fraud_rate: float = 0.013, seed: int = 42) -> pd.DataFrame:
    """Generate a small PaySim-shaped dataset so the tool runs with zero setup.

    This is NOT a substitute for PaySim in research: it reproduces the schema and
    the headline structural regularities (fraud concentrated in TRANSFER/CASH_OUT,
    origin account drained to ~0) so the pipeline is exercisable end to end.
    Use --data with the real PaySim CSV for any result you intend to report.
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
            "step": rng.integers(1, 745, size=n_legit),
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
            "step": rng.integers(1, 745, size=n_fraud),
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the behavioural features the baseline classifier trains on.

    Returns a new frame with FEATURE_COLUMNS present. Deterministic and
    row-independent, so it can be reapplied to mutated rows unchanged.
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
] + [f"type_{t}" for t in TXN_TYPES]
