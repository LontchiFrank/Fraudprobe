"""Shared fixtures for the fraudprobe test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def seed_frauds() -> pd.DataFrame:
    """A tiny, deterministic PaySim-shaped set of "caught" frauds to mutate."""
    return pd.DataFrame(
        [
            {
                "step": 10, "type": "TRANSFER", "amount": 10000.0,
                "nameOrig": "C1", "oldbalanceOrg": 10000.0, "newbalanceOrig": 0.0,
                "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 10000.0,
                "isFraud": 1,
            },
            {
                "step": 400, "type": "CASH_OUT", "amount": 5000.0,
                "nameOrig": "C3", "oldbalanceOrg": 5000.0, "newbalanceOrig": 0.0,
                "nameDest": "C4", "oldbalanceDest": 200.0, "newbalanceDest": 5200.0,
                "isFraud": 1,
            },
        ]
    )
