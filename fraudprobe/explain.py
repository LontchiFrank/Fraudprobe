"""SHAP-based explanations: *why* did a mutated fraud slip past the detector?

The stress test tells you how many adversarial rows evaded and which strategy was
worst. This module answers the follow-up a fraud engineer immediately asks: which
features did the attacker move to push the model's score below its decision
threshold?

We use SHAP (SHapley Additive exPlanations) to attribute each row's fraud score to
its individual features, then contrast the attribution on the original caught fraud
against its mutated, evading counterpart. The features whose contribution collapsed
are the model's blind spots.

SHAP is optional — if it isn't installed the caller gets a graceful ``None`` and the
rest of the pipeline is unaffected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, engineer_features
from .models import ScoredModel

try:  # optional dependency
    import shap  # type: ignore

    HAVE_SHAP = True
except Exception:  # pragma: no cover - shap has heavy transitive deps
    HAVE_SHAP = False


@dataclass
class FeatureAttribution:
    feature: str
    mean_abs_shap: float
    mean_shap: float  # signed: >0 pushes toward "fraud", <0 toward "legit"


@dataclass
class ExplanationReport:
    available: bool
    backend: str  # "shap" | "unavailable"
    n_explained: int
    # Which features most drive the model TOWARD flagging fraud on clean frauds.
    clean_top_features: list[FeatureAttribution] = field(default_factory=list)
    # Which features most drive the model on the evading adversarial rows.
    adversarial_top_features: list[FeatureAttribution] = field(default_factory=list)
    # The biggest *drops* in fraud-pushing signal from clean -> adversarial.
    # These are the levers the attacker pulled to evade.
    evasion_levers: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _mean_attributions(shap_values: np.ndarray, columns: list[str]) -> list[FeatureAttribution]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        FeatureAttribution(
            feature=columns[i],
            mean_abs_shap=round(float(mean_abs[i]), 5),
            mean_shap=round(float(mean_signed[i]), 5),
        )
        for i in order
    ]


def _shap_matrix(explainer, X: np.ndarray, n_features: int) -> np.ndarray:
    """Return a (n_rows, n_features) array of SHAP values for the positive class.

    Handles the several shapes different SHAP explainer/model combinations return.
    """
    sv = explainer.shap_values(X)
    # Newer SHAP may return an Explanation object.
    if hasattr(sv, "values"):
        sv = sv.values
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_rows, n_features, n_classes) or (n_classes, n_rows, n_features)
        if sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        elif sv.shape[0] == 2:
            sv = sv[1]
        else:
            sv = sv[..., -1]
    if sv.shape != (X.shape[0], n_features):
        # Last-resort reshape guard.
        sv = sv.reshape(X.shape[0], n_features)
    return sv


def explain_evasion(
    model: ScoredModel,
    clean_fraud: pd.DataFrame,
    corpus: pd.DataFrame,
    max_rows: int = 200,
    background_size: int = 100,
) -> ExplanationReport:
    """Attribute the detection drop-off to individual transaction features via SHAP.

    Parameters
    ----------
    model : the ScoredModel under test.
    clean_fraud : raw rows of frauds the model catches cleanly (the seed population).
    corpus : the adversarial corpus of mutated, evading rows.
    """
    if not HAVE_SHAP:
        return ExplanationReport(
            available=False,
            backend="unavailable",
            n_explained=0,
            note="SHAP is not installed. `pip install shap` (needs Python <3.14) to enable.",
        )

    cols = model.feature_columns
    n_features = len(cols)

    clean_feats = engineer_features(clean_fraud)[cols].to_numpy(dtype=float)
    adv_feats = engineer_features(corpus)[cols].to_numpy(dtype=float)

    if len(clean_feats) == 0 or len(adv_feats) == 0:
        return ExplanationReport(
            available=False, backend="unavailable", n_explained=0,
            note="Not enough rows to explain (need both clean frauds and adversarial rows).",
        )

    clean_feats = clean_feats[:max_rows]
    adv_feats = adv_feats[:max_rows]

    est = model.estimator
    try:
        # TreeExplainer is exact and fast for the tree models fraudprobe trains.
        explainer = shap.TreeExplainer(est)
    except Exception:
        # Model-agnostic fallback (slower). Sample a background set from clean frauds.
        bg = clean_feats[:background_size]
        explainer = shap.Explainer(est.predict_proba, bg)

    clean_sv = _shap_matrix(explainer, clean_feats, n_features)
    adv_sv = _shap_matrix(explainer, adv_feats, n_features)

    clean_attr = _mean_attributions(clean_sv, cols)
    adv_attr = _mean_attributions(adv_sv, cols)

    # Evasion levers: features whose *fraud-pushing* signal dropped most when the
    # attacker mutated the row. Positive mean_shap pushes toward fraud, so a fall
    # in mean_shap (clean -> adversarial) means the attacker neutralised that signal.
    clean_by_feat = {a.feature: a.mean_shap for a in clean_attr}
    adv_by_feat = {a.feature: a.mean_shap for a in adv_attr}
    levers = []
    for feat in cols:
        drop = clean_by_feat[feat] - adv_by_feat[feat]
        levers.append(
            {
                "feature": feat,
                "clean_mean_shap": round(clean_by_feat[feat], 5),
                "adversarial_mean_shap": round(adv_by_feat[feat], 5),
                "fraud_signal_drop": round(float(drop), 5),
            }
        )
    levers.sort(key=lambda d: -d["fraud_signal_drop"])

    return ExplanationReport(
        available=True,
        backend="shap",
        n_explained=int(len(clean_feats) + len(adv_feats)),
        clean_top_features=clean_attr[:10],
        adversarial_top_features=adv_attr[:10],
        evasion_levers=levers[:10],
        note=(
            "Positive mean_shap pushes the model toward 'fraud'. `fraud_signal_drop` "
            "is how much each feature's fraud-pushing signal fell from the clean fraud "
            "to its mutated version — the largest drops are the levers the attacker pulled."
        ),
    )
