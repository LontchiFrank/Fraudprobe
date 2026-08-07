"""Phase 3: stress-test a model against the adversarial corpus and attribute failure.

Produces the resilience report a fraud engineer actually reads:
how much recall dropped, and which mutation strategy caused it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ScoredModel, metrics


def stress_test(model: ScoredModel, clean_test: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    """Compare detection on clean fraud vs the adversarial corpus."""
    clean_fraud = clean_test[clean_test["isFraud"] == 1].reset_index(drop=True)

    clean_proba = model.score_rows(clean_fraud)
    clean_caught = float((clean_proba >= model.threshold).mean()) if len(clean_fraud) else 0.0

    adv_proba = model.score_rows(corpus)
    adv_caught = float((adv_proba >= model.threshold).mean()) if len(corpus) else 0.0

    # Per-strategy evasion breakdown – the actionable part.
    per_strategy = {}
    if "mutation" in corpus.columns and len(corpus):
        corpus = corpus.copy()
        corpus["_proba"] = adv_proba
        for strat, grp in corpus.groupby("mutation"):
            caught = float((grp["_proba"] >= model.threshold).mean())
            per_strategy[strat] = {
                "n": int(len(grp)),
                "detection_rate": round(caught, 4),
                "evasion_rate": round(1.0 - caught, 4),
            }

    return {
        "clean_fraud_detection_rate": round(clean_caught, 4),
        "adversarial_detection_rate": round(adv_caught, 4),
        "detection_drop_off": round(clean_caught - adv_caught, 4),
        "evasion_rate": round(1.0 - adv_caught, 4),
        "n_clean_fraud": int(len(clean_fraud)),
        "n_adversarial": int(len(corpus)),
        "per_strategy": per_strategy,
    }


def rank_weaknesses(stress: dict) -> list[tuple[str, float]]:
    """Return strategies sorted by how well they evaded (worst weakness first)."""
    ps = stress.get("per_strategy", {})
    return sorted(((k, v["evasion_rate"]) for k, v in ps.items()), key=lambda x: -x[1])


def format_report(baseline: dict, stress: dict, adv_report) -> str:
    """Human-readable resilience report for the terminal / README."""
    lines = []
    add = lines.append
    add("=" * 62)
    add("  FRAUDPROBE — ADVERSARIAL RESILIENCE REPORT")
    add("=" * 62)
    add("")
    add(f"  Baseline model         : {baseline.get('model_type', '?')}")
    add(f"  Baseline F1 / PR-AUC   : {baseline.get('f1', 0):.3f} / {baseline.get('pr_auc', float('nan')):.3f}")
    add(f"  Held-out fraud support : {baseline.get('support_fraud', 0)}")
    add("")
    add(f"  Adversarial corpus     : {adv_report.n_generated} rows "
        f"({adv_report.backend} backend, {adv_report.n_rejected_invalid} rejected as invalid)")
    add(f"  Economic value check   : {'PASS' if adv_report.value_preserved else 'FAIL'}")
    add("")
    add("-" * 62)
    add(f"  Clean fraud detection rate       : {stress['clean_fraud_detection_rate']*100:5.1f}%")
    add(f"  Adversarial detection rate       : {stress['adversarial_detection_rate']*100:5.1f}%")
    add(f"  >> DETECTION DROP-OFF            : {stress['detection_drop_off']*100:5.1f} pts")
    add(f"  >> OVERALL EVASION RATE          : {stress['evasion_rate']*100:5.1f}%")
    add("-" * 62)
    add("")
    add("  Weakness ranking (which trick fooled the model most):")
    for strat, ev in rank_weaknesses(stress):
        bar = "#" * int(ev * 40)
        add(f"    {strat:22s} {ev*100:5.1f}% evaded  {bar}")
    add("")
    add("=" * 62)
    worst = rank_weaknesses(stress)
    if worst:
        add(f"  VERDICT: weakest against '{worst[0][0]}' "
            f"({worst[0][1]*100:.0f}% of those slipped through).")
    add("=" * 62)
    return "\n".join(lines)
