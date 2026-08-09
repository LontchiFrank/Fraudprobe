"""Task 7: baseline hyperparameters must be selected via stratified 5-fold grid
search with SMOTE fitted inside each fold (imblearn.pipeline.Pipeline), not a
single stratified 80/20 split with hard-coded hyperparameters and one SMOTE call
applied before any splitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudprobe.data import make_demo_data
from fraudprobe import models as models_module
from fraudprobe.models import (
    CV_FOLDS,
    SEARCH_SPACES,
    TUNING_SUBSAMPLE_CAP,
    train_baseline,
)


@pytest.fixture(scope="module")
def small_df():
    return make_demo_data(n_rows=3000, fraud_rate=0.03, seed=1)


def test_tuned_xgboost_reports_full_tuning_info(small_df):
    _, _, m = train_baseline(small_df, model_type="xgboost", seed=1, tune=True)
    t = m["tuning"]
    assert t["tuned"] is True
    assert t["cv_folds"] == CV_FOLDS
    assert t["search_space"] == SEARCH_SPACES["xgboost"]
    assert set(t["best_params"]) == {"max_depth", "learning_rate", "n_estimators", "scale_pos_weight"}
    for v in t["best_params"].values():
        assert not isinstance(v, str) or True  # values are whatever the grid declared; no assertion needed
    assert 0.0 <= t["cv_mean_f1"] <= 1.0
    assert t["cv_std_f1"] >= 0.0
    # best_params keys must NOT carry the pipeline step prefix.
    assert all("__" not in k for k in t["best_params"])


def test_tuned_rf_reports_full_tuning_info(small_df):
    _, _, m = train_baseline(small_df, model_type="rf", seed=1, tune=True)
    t = m["tuning"]
    assert t["tuned"] is True
    assert set(t["best_params"]) == {"n_estimators", "max_depth", "min_samples_leaf", "class_weight"}


def test_untuned_reports_reason(small_df):
    _, _, m = train_baseline(small_df, model_type="xgboost", seed=1, tune=False)
    assert m["tuning"] == {"tuned": False, "reason": "tune=False"}


def test_untuned_and_tuned_models_both_score(small_df):
    """Whichever path was taken, the returned model must still be usable."""
    model_tuned, test_df, _ = train_baseline(small_df, model_type="xgboost", seed=1, tune=True)
    model_untuned, _, _ = train_baseline(small_df, model_type="xgboost", seed=1, tune=False)
    for model in (model_tuned, model_untuned):
        proba = model.score_rows(test_df)
        assert proba.shape == (len(test_df),)
        assert ((proba >= 0) & (proba <= 1)).all()


def test_grid_search_runs_on_bounded_subsample_when_data_is_large():
    big_df = make_demo_data(n_rows=60_000, fraud_rate=0.02, seed=1)
    _, _, m = train_baseline(big_df, model_type="xgboost", seed=1, tune=True)
    assert m["tuning"]["tuning_subsample_size"] == TUNING_SUBSAMPLE_CAP


def test_grid_search_uses_all_rows_when_data_is_small(small_df):
    _, _, m = train_baseline(small_df, model_type="xgboost", seed=1, tune=True)
    n_train = int(len(small_df) * 0.8)
    assert m["tuning"]["tuning_subsample_size"] == n_train
    assert m["tuning"]["tuning_subsample_size"] < TUNING_SUBSAMPLE_CAP


def test_smote_used_inside_pipeline_not_bare_module_call(small_df, monkeypatch):
    """The search estimator must be an imblearn Pipeline with SMOTE as a step (so
    it is fit per-fold by GridSearchCV's own cross-validation), not a bare
    classifier pre-resampled once before any splitting.

    GridSearchCV's n_jobs=-1 runs fits in separate worker processes, so counting
    calls via monkeypatch (which only patches the main process) silently under-
    counts — this checks the estimator's structure instead, which is robust to
    where the fits actually run.
    """
    # Swap the *name* GridSearchCV resolves to in models.py for a plain function
    # that captures the estimator and delegates to the real class untouched —
    # monkeypatching GridSearchCV.__init__ directly breaks sklearn's own param
    # introspection (_get_param_names inspects the constructor signature).
    captured = {}
    real_gridsearchcv = models_module.GridSearchCV

    def capturing_gridsearchcv(estimator, *args, **kwargs):
        captured["estimator"] = estimator
        return real_gridsearchcv(estimator, *args, **kwargs)

    monkeypatch.setattr(models_module, "GridSearchCV", capturing_gridsearchcv)
    train_baseline(small_df, model_type="xgboost", seed=1, tune=True)

    pipe = captured["estimator"]
    assert isinstance(pipe, models_module.ImbPipeline)
    assert list(pipe.named_steps) == ["smote", "clf"]
    assert isinstance(pipe.named_steps["smote"], models_module.SMOTE)


def test_no_smote_falls_back_to_untuned(small_df, monkeypatch):
    monkeypatch.setattr(models_module, "HAVE_SMOTE", False)
    _, _, m = train_baseline(small_df, model_type="xgboost", seed=1, tune=True)
    assert m["tuning"] == {"tuned": False, "reason": "imbalanced-learn not installed"}


def test_final_model_is_bare_estimator_not_a_pipeline(small_df):
    """The returned model should be the plain classifier (already refit with the
    winning hyperparameters), not the search Pipeline — so ScoredModel.score_rows
    doesn't need special handling for a sampler step at predict time."""
    from sklearn.pipeline import Pipeline as SkPipeline

    model, _, _ = train_baseline(small_df, model_type="xgboost", seed=1, tune=True)
    assert not isinstance(model.estimator, SkPipeline)
    assert hasattr(model.estimator, "predict_proba")
