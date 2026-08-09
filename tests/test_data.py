"""Task 5: the synthetic demo fixture must give `step` (hour-of-day) real temporal
structure, or temporal_dispersion evades for reasons that are a property of the
fixture, not a finding about fraud detection.
"""

from __future__ import annotations

import numpy as np

from fraudprobe.data import make_demo_data


def test_fraud_hour_of_day_concentrates_in_early_morning():
    df = make_demo_data(n_rows=20_000, seed=1)
    fraud_hours = df.loc[df["isFraud"] == 1, "step"] % 24
    # Documented assumption: fraud weighted toward 00:00-05:00.
    frac_night = fraud_hours.isin(range(0, 6)).mean()
    assert frac_night > 0.4, f"expected fraud concentrated 00:00-05:00, got {frac_night:.2%}"


def test_legit_hour_of_day_is_daytime_weighted():
    df = make_demo_data(n_rows=20_000, seed=1)
    legit_hours = df.loc[df["isFraud"] == 0, "step"] % 24
    frac_business_hours = legit_hours.isin(range(8, 20)).mean()
    assert frac_business_hours > 0.5, f"expected legit weighted to daytime, got {frac_business_hours:.2%}"


def test_fraud_and_legit_hour_distributions_differ():
    """The whole point: hour_of_day must carry a fraud signal, unlike before."""
    df = make_demo_data(n_rows=30_000, seed=1)
    fraud_hours = df.loc[df["isFraud"] == 1, "step"] % 24
    legit_hours = df.loc[df["isFraud"] == 0, "step"] % 24
    frac_night_fraud = fraud_hours.isin(range(0, 6)).mean()
    frac_night_legit = legit_hours.isin(range(0, 6)).mean()
    assert frac_night_fraud > frac_night_legit + 0.2


def test_step_stays_in_valid_paysim_range():
    df = make_demo_data(n_rows=5_000, seed=1)
    assert df["step"].min() >= 1
    assert df["step"].max() <= 744


def test_deterministic_given_seed():
    a = make_demo_data(n_rows=2_000, seed=7)
    b = make_demo_data(n_rows=2_000, seed=7)
    assert (a["step"] == b["step"]).all()
