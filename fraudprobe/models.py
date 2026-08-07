"""Baseline classifier training and the 'bring your own classifier' adapter.

The adapter is the point of the tool: fraudprobe does not care how your fraud
model was built, only that it can score a PaySim-shaped row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import FEATURE_COLUMNS, engineer_features

try:  # optional: preferred when installed
    from xgboost import XGBClassifier

    HAVE_XGB = True
except ImportError:  # pragma: no cover
    HAVE_XGB = False

try:  # optional: proper SMOTE if available
    from imblearn.over_sampling import SMOTE

    HAVE_SMOTE = True
except ImportError:  # pragma: no cover
    HAVE_SMOTE = False


@dataclass
class ScoredModel:
    """Wraps any fitted estimator so fraudprobe can score raw transaction rows.

    `estimator` must expose predict_proba(X) -> (n, 2). Anything scikit-learn
    compatible works, as do XGBoost/LightGBM/CatBoost sklearn APIs. If you have
    a bespoke model, wrap it in a shim with a predict_proba method.
    """

    estimator: object
    feature_columns: list[str]
    threshold: float = 0.5

    def score_rows(self, raw: pd.DataFrame) -> np.ndarray:
        """Take raw PaySim-shaped rows, return P(fraud) for each."""
        feats = engineer_features(raw)
        missing = [c for c in self.feature_columns if c not in feats.columns]
        if missing:
            raise ValueError(f"Model expects features not produced by the pipeline: {missing}")
        X = feats[self.feature_columns].to_numpy(dtype=float)
        return self.estimator.predict_proba(X)[:, 1]

    def flags_fraud(self, raw: pd.DataFrame) -> np.ndarray:
        return self.score_rows(raw) >= self.threshold

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScoredModel":
        blob = joblib.load(path)
        if isinstance(blob, dict) and "estimator" in blob:
            return cls(
                estimator=blob["estimator"],
                feature_columns=blob.get("feature_columns", FEATURE_COLUMNS),
                threshold=blob.get("threshold", 0.5),
            )
        # A bare estimator was pickled: assume the standard feature contract.
        return cls(estimator=blob, feature_columns=FEATURE_COLUMNS)


def _build_estimator(kind: str, seed: int):
    if kind == "xgboost":
        if not HAVE_XGB:
            raise RuntimeError("xgboost is not installed. Use --model-type gbdt or rf.")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "gbdt":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model type: {kind}")


def _resample(X: np.ndarray, y: np.ndarray, seed: int):
    """Oversample the minority class on the TRAINING partition only."""
    if HAVE_SMOTE:
        return SMOTE(random_state=seed).fit_resample(X, y)
    # Fallback: random oversampling with replacement. Cruder than SMOTE but keeps
    # the tool installable with just scikit-learn.
    rng = np.random.default_rng(seed)
    minority = np.flatnonzero(y == 1)
    majority = np.flatnonzero(y == 0)
    if len(minority) == 0 or len(minority) >= len(majority):
        return X, y
    picks = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    idx = np.concatenate([majority, minority, picks])
    rng.shuffle(idx)
    return X[idx], y[idx]


def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "support_fraud": int(y_true.sum()),
        "n": int(len(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    return out


def train_baseline(
    df: pd.DataFrame,
    model_type: str = "auto",
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[ScoredModel, pd.DataFrame, dict]:
    """Train the stand-in bank classifier. Returns (model, held-out test set, metrics)."""
    if model_type == "auto":
        model_type = "xgboost" if HAVE_XGB else "rf"

    feats = engineer_features(df)
    y = feats["isFraud"].to_numpy(dtype=int)

    train_idx, test_idx = train_test_split(
        np.arange(len(feats)), test_size=test_size, stratify=y, random_state=seed
    )

    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_tr, y_tr = _resample(X[train_idx], y[train_idx], seed)

    est = _build_estimator(model_type, seed)
    est.fit(X_tr, y_tr)

    model = ScoredModel(estimator=est, feature_columns=list(FEATURE_COLUMNS))

    # Evaluate on the untouched, realistically-imbalanced held-out set.
    test_raw = df.iloc[test_idx].reset_index(drop=True)
    proba = model.score_rows(test_raw)
    m = metrics(y[test_idx], proba)
    m["model_type"] = model_type
    m["smote"] = HAVE_SMOTE
    return model, test_raw, m


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
