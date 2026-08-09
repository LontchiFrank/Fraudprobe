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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .data import FEATURE_COLUMNS, engineer_features

try:  # optional: preferred when installed
    from xgboost import XGBClassifier

    HAVE_XGB = True
except ImportError:  # pragma: no cover
    HAVE_XGB = False

try:  # optional: proper SMOTE if available. imblearn.pipeline.Pipeline (not
    # sklearn's) is required for grid search to fit SMOTE inside each CV fold
    # rather than once on the whole training set before CV — the latter lets
    # synthetic minority samples "see" points that end up in the validation fold,
    # leaking information across the split (Task 7).
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAVE_SMOTE = True
except ImportError:  # pragma: no cover
    HAVE_SMOTE = False

# Hyperparameter search spaces for GridSearchCV, one per model_type (Task 7).
# Deliberately modest (2-3 values per axis) so a tuned run stays usable without
# --no-tune; substitute a wider grid for a dedicated reportable run. xgboost's
# axes match the dissertation's specified max_depth/learning_rate/n_estimators/
# scale_pos_weight; rf and gbdt don't have a scale_pos_weight equivalent for
# XGBoost's exact knob, so their imbalance-handling axis is documented separately
# per model (rf: class_weight; gbdt: none — GradientBoostingClassifier has no
# built-in class weighting, hence SMOTE doing the balancing work for it).
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "xgboost": {
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.1, 0.2],
        "clf__n_estimators": [150, 300],
        "clf__scale_pos_weight": [1, 5],
    },
    "rf": {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [None, 12],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": ["balanced", "balanced_subsample"],
    },
    "gbdt": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.1, 0.2],
    },
}
CV_FOLDS = 5


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


def _build_estimator(kind: str, seed: int, n_jobs: int = -1):
    """n_jobs=1 is used inside GridSearchCV (which parallelises across the
    fold x combination grid itself); nesting n_jobs=-1 inside AND outside a grid
    search means many processes fighting over the same cores, which is dramatically
    slower than either alone rather than faster."""
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
            n_jobs=n_jobs,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
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


# Grid search runs on a bounded subsample of the training partition, capped here,
# regardless of the overall dataset size — CV cost is (combinations x folds) fits,
# so it must not scale with the full dataset or a 6M-row PaySim run would never
# finish tuning. This is exactly what "refit the best configuration on the full
# training partition" (Task 7) implies: the *search* finds hyperparameters on a
# representative sample, then exactly one further fit uses the full partition.
TUNING_SUBSAMPLE_CAP = 20_000


def _fit_tuned(model_type: str, X_tr: np.ndarray, y_tr: np.ndarray, seed: int, log) -> tuple[object, dict]:
    """GridSearchCV over SEARCH_SPACES[model_type], SMOTE fit inside each fold via
    an imblearn Pipeline (Task 7). Returns (fitted_estimator, tuning_info) where
    fitted_estimator has been refit on the FULL (X_tr, y_tr) with the winning
    hyperparameters — search itself runs on a bounded, stratified subsample (see
    TUNING_SUBSAMPLE_CAP) so its cost doesn't scale with the overall dataset size.
    """
    grid = SEARCH_SPACES.get(model_type)
    if grid is None:
        return None, {"tuned": False, "reason": f"no search space defined for model_type={model_type!r}"}

    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)

    if len(X_tr) > TUNING_SUBSAMPLE_CAP:
        sub_idx, _ = train_test_split(
            np.arange(len(X_tr)), train_size=TUNING_SUBSAMPLE_CAP, stratify=y_tr, random_state=seed,
        )
        X_search, y_search = X_tr[sub_idx], y_tr[sub_idx]
    else:
        X_search, y_search = X_tr, y_tr

    log(f"Tuning {model_type} via {CV_FOLDS}-fold grid search on {len(X_search)} rows "
        f"({n_combos} combinations x {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits)...")

    # n_jobs=1 on the inner estimator: GridSearchCV parallelises across the fold x
    # combination grid at the outer level (n_jobs=-1 below); nesting -1 inside AND
    # outside means many processes fighting over the same cores, which measured
    # roughly 10x slower than either alone rather than faster.
    pipe = ImbPipeline([("smote", SMOTE(random_state=seed)), ("clf", _build_estimator(model_type, seed, n_jobs=1))])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, refit=False)
    search.fit(X_search, y_search)

    best_idx = search.best_index_
    best_params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
    tuning_info = {
        "tuned": True,
        "search_space": grid,
        "cv_folds": CV_FOLDS,
        "n_combinations": n_combos,
        "tuning_subsample_size": len(X_search),
        "best_params": best_params,
        "cv_mean_f1": float(search.cv_results_["mean_test_score"][best_idx]),
        "cv_std_f1": float(search.cv_results_["std_test_score"][best_idx]),
    }
    log(f"Best params: {best_params} "
        f"(CV F1 = {tuning_info['cv_mean_f1']:.3f} +/- {tuning_info['cv_std_f1']:.3f})")

    # Exactly one further fit, with the winning hyperparameters, on the FULL
    # training partition — not the tuning subsample above.
    final_est = _build_estimator(model_type, seed, n_jobs=-1)
    final_est.set_params(**best_params)
    X_res, y_res = _resample(X_tr, y_tr, seed)
    final_est.fit(X_res, y_res)
    return final_est, tuning_info


def train_baseline(
    df: pd.DataFrame,
    model_type: str = "auto",
    seed: int = 42,
    test_size: float = 0.2,
    tune: bool = True,
    log=lambda msg: None,
) -> tuple[ScoredModel, pd.DataFrame, dict]:
    """Train the stand-in bank classifier. Returns (model, held-out test set, metrics).

    When ``tune`` (default) and imbalanced-learn is installed, hyperparameters are
    selected via stratified 5-fold grid search with SMOTE fitted inside each fold
    (see _fit_tuned / SEARCH_SPACES) — the methodology the dissertation specifies.
    ``tune=False`` (``--no-tune``) skips straight to the previous fixed-hyperparameter
    behaviour for fast iteration; used automatically as a fallback whenever
    imbalanced-learn isn't installed or a model_type has no declared search space.
    """
    if model_type == "auto":
        model_type = "xgboost" if HAVE_XGB else "rf"

    feats = engineer_features(df)
    y = feats["isFraud"].to_numpy(dtype=int)

    train_idx, test_idx = train_test_split(
        np.arange(len(feats)), test_size=test_size, stratify=y, random_state=seed
    )

    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_tr, y_tr = X[train_idx], y[train_idx]

    tuning_info = {"tuned": False, "reason": "tune=False"}
    est = None
    if tune and HAVE_SMOTE:
        est, tuning_info = _fit_tuned(model_type, X_tr, y_tr, seed, log)
    elif tune and not HAVE_SMOTE:
        tuning_info = {"tuned": False, "reason": "imbalanced-learn not installed"}

    if est is None:
        # Untuned path: fixed hyperparameters, single fit, SMOTE (or its fallback)
        # applied once on the training partition only — the pre-Task-7 behaviour.
        X_res, y_res = _resample(X_tr, y_tr, seed)
        est = _build_estimator(model_type, seed)
        est.fit(X_res, y_res)

    model = ScoredModel(estimator=est, feature_columns=list(FEATURE_COLUMNS))

    # Evaluate on the untouched, realistically-imbalanced held-out set.
    test_raw = df.iloc[test_idx].reset_index(drop=True)
    proba = model.score_rows(test_raw)
    m = metrics(y[test_idx], proba)
    m["model_type"] = model_type
    m["smote"] = HAVE_SMOTE
    m["tuning"] = tuning_info
    return model, test_raw, m


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
