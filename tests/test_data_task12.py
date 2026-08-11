"""Task 12 / REVIEW.md item 1: txn_velocity_orig and dest_txn_count.

FEATURE_COLUMNS previously contained neither, despite the dissertation
claiming both a transaction-velocity feature and a destination behavioural-
consistency feature exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudprobe.adversary import AMOUNT_SPLIT_PARTS, generate_adversarial_corpus
from fraudprobe.data import FEATURE_COLUMNS, VELOCITY_WINDOW_STEPS, engineer_features


def _row(step, orig, dest, amount=1000.0):
    return {
        "step": step, "type": "TRANSFER", "amount": amount,
        "nameOrig": orig, "oldbalanceOrg": amount, "newbalanceOrig": 0.0,
        "nameDest": dest, "oldbalanceDest": 0.0, "newbalanceDest": amount,
        "isFraud": 1,
    }


def test_feature_columns_include_both_new_features():
    assert "txn_velocity_orig" in FEATURE_COLUMNS
    assert "dest_txn_count" in FEATURE_COLUMNS


def test_engineer_features_produces_both_columns():
    df = pd.DataFrame([_row(1, "A", "X"), _row(2, "B", "Y")])
    out = engineer_features(df)
    assert "txn_velocity_orig" in out.columns
    assert "dest_txn_count" in out.columns


def test_txn_velocity_counts_only_within_window_backward_looking():
    # Account "A": transactions at step 1, 10, 30, 100.
    # Window = 24: for step=30, in-window predecessors are step 10 and 30 itself
    # (30-10=20<=24), but NOT step 1 (30-1=29>24) or step 100 (future, excluded).
    df = pd.DataFrame([
        _row(1, "A", "X"), _row(10, "A", "X"), _row(30, "A", "X"), _row(100, "A", "X"),
    ])
    out = engineer_features(df)
    velocities = dict(zip(df["step"], out["txn_velocity_orig"]))
    assert velocities[1] == 1        # nothing precedes it
    assert velocities[10] == 2       # step 1 and 10 (10-1=9<=24)
    assert velocities[30] == 2       # step 10 and 30 (20<=24); step 1 excluded (29>24)
    assert velocities[100] == 1      # nothing else within 24 steps behind it


def test_txn_velocity_never_counts_future_transactions():
    # A transaction at step 1 must not see a same-account transaction at step 2.
    df = pd.DataFrame([_row(1, "A", "X"), _row(2, "A", "X")])
    out = engineer_features(df)
    assert out.loc[df["step"] == 1, "txn_velocity_orig"].iloc[0] == 1
    assert out.loc[df["step"] == 2, "txn_velocity_orig"].iloc[0] == 2


def test_txn_velocity_is_per_account_not_global():
    df = pd.DataFrame([_row(5, "A", "X"), _row(5, "B", "Y"), _row(5, "A", "X")])
    out = engineer_features(df)
    assert list(out["txn_velocity_orig"]) == [2, 1, 2]


def test_dest_txn_count_is_total_within_frame_not_windowed():
    # dest_txn_count is a plain count, unlike the windowed orig velocity —
    # two receipts far apart in step still both count.
    df = pd.DataFrame([_row(1, "A", "Z"), _row(1000, "B", "Z"), _row(5, "C", "W")])
    out = engineer_features(df)
    counts = dict(zip(df["nameOrig"], out["dest_txn_count"]))
    assert counts["A"] == 2
    assert counts["B"] == 2
    assert counts["C"] == 1


def test_velocity_window_constant_is_positive():
    assert VELOCITY_WINDOW_STEPS > 0


# --------------------------------------------------------------------------- #
# Design-constraint consequence (documented in engineer_features' docstring):
# scoring a small adversarial corpus alone means these features only see the
# corpus's own sibling rows, not the account's real history from clean_test.
# --------------------------------------------------------------------------- #
def test_amount_split_corpus_rows_show_full_velocity_from_sibling_splits(seed_frauds):
    corpus, _ = generate_adversarial_corpus(
        seed_frauds, strategies=("amount_split",), backend="rules", seed=1,
    )
    feats = engineer_features(corpus)
    # Every seed produced AMOUNT_SPLIT_PARTS rows sharing nameOrig and step —
    # a real structuring signal, not a bug in the feature.
    for _, group in feats.groupby("nameOrig"):
        assert (group["txn_velocity_orig"] == AMOUNT_SPLIT_PARTS).all()


def test_single_row_strategies_show_velocity_of_one_in_isolation(seed_frauds):
    corpus, _ = generate_adversarial_corpus(
        seed_frauds, strategies=("balance_camouflage",), backend="rules", seed=1,
    )
    feats = engineer_features(corpus)
    assert (feats["txn_velocity_orig"] == 1).all()
