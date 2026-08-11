"""Report-ready figures (Task 11): 300 dpi PNG + vector PDF for every chart a
report needs, generated properly rather than screenshotted from the dashboard.

Optional dependency on matplotlib (and shap for the SHAP summary plots) — a
missing one degrades to skipping that specific figure with a log line, never a
hard failure of the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import wilson_ci

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: never try to open a GUI window
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _adversarial_eval_set(clean_test: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Same background legit population as the clean test, with adversarial fraud
    substituted in for the clean fraud — matches how stress_test computes
    adversarial_detection_rate, so these figures agree with the numbers in the text.
    """
    legit = clean_test[clean_test["isFraud"] == 0]
    return pd.concat([legit, corpus], ignore_index=True)


def plot_pr_roc_curves(model, clean_test: pd.DataFrame, corpus: pd.DataFrame, outdir: Path) -> None:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y_clean = clean_test["isFraud"].to_numpy()
    p_clean = model.score_rows(clean_test)
    adv_eval = _adversarial_eval_set(clean_test, corpus)
    y_adv = adv_eval["isFraud"].to_numpy()
    p_adv = model.score_rows(adv_eval)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for y, p, label in [(y_clean, p_clean, "Clean"), (y_adv, p_adv, "Adversarial")]:
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(rec, prec, label=f"{label} (AUC={auc(rec, prec):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall: clean vs. adversarial")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC: clean vs. adversarial")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "pr_roc_curves")


def plot_confusion_matrices(model, clean_test: pd.DataFrame, corpus: pd.DataFrame, outdir: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    y_clean = clean_test["isFraud"].to_numpy()
    pred_clean = (model.score_rows(clean_test) >= model.threshold).astype(int)
    adv_eval = _adversarial_eval_set(clean_test, corpus)
    y_adv = adv_eval["isFraud"].to_numpy()
    pred_adv = (model.score_rows(adv_eval) >= model.threshold).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_clean, pred_clean, labels=[0, 1]),
                           display_labels=["legit", "fraud"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Before attack (clean)")
    ConfusionMatrixDisplay(confusion_matrix(y_adv, pred_adv, labels=[0, 1]),
                           display_labels=["legit", "fraud"]).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("After attack (adversarial)")
    fig.tight_layout()
    _save(fig, outdir, "confusion_matrices")


def plot_evasion_bar_chart(stress: dict, outdir: Path) -> None:
    per_strategy = stress.get("per_strategy", {})
    if not per_strategy:
        return
    strategies, rates, err_low, err_high = [], [], [], []
    for s, vals in per_strategy.items():
        n = vals["n"]
        evaded = round(vals["evasion_rate"] * n)
        w = wilson_ci(evaded, n)
        p = w["proportion"] or 0.0
        strategies.append(s)
        rates.append(p)
        # max(0.0, ...): wilson_ci's own low/high are correct, but subtracting
        # two independently-rounded floats at a boundary (e.g. p=0.0 exactly,
        # ci95_low ~2.8e-17 instead of exactly 0.0) can yield a tiny negative
        # that matplotlib's yerr rejects outright — REVIEW.md item 3 smoke test
        # against real PaySim hit exactly this on a 0/6 temporal_dispersion cell.
        err_low.append(max(0.0, p - (w["ci95_low"] or p)))
        err_high.append(max(0.0, (w["ci95_high"] or p) - p))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(strategies, rates, yerr=[err_low, err_high], capsize=5, color="#c0392b")
    ax.set_ylabel("Evasion rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-strategy evasion rate (Wilson 95% CI)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, outdir, "per_strategy_evasion")


def plot_threshold_sweep(threshold_sweep: dict, outdir: Path) -> None:
    grid = threshold_sweep.get("grid", [])
    if not grid:
        return
    thresholds = [p["threshold"] for p in grid]
    clean = [p["clean_detection_rate"] for p in grid]
    adv = [p["adversarial_detection_rate"] for p in grid]
    drop = [p["drop_off"] for p in grid]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, clean, marker="o", label="Clean detection")
    ax.plot(thresholds, adv, marker="o", label="Adversarial detection")
    ax.plot(thresholds, drop, marker="x", linestyle="--", color="gray", label="Drop-off")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Detection drop-off vs. threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "threshold_sweep")


def plot_evasion_levers(explanation: dict | None, outdir: Path) -> None:
    if not explanation or not explanation.get("available") or not explanation.get("evasion_levers"):
        return
    levers = explanation["evasion_levers"][:10]
    features = [lv["feature"] for lv in levers][::-1]
    clean_vals = [lv["clean_mean_shap"] for lv in levers][::-1]
    adv_vals = [lv["adversarial_mean_shap"] for lv in levers][::-1]

    y = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y - 0.2, clean_vals, height=0.4, label="Clean fraud signal", color="#5b9dff")
    ax.barh(y + 0.2, adv_vals, height=0.4, label="After mutation", color="#f5a524")
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean SHAP value (pushes toward fraud →)")
    ax.set_title("Evasion levers: feature signal before vs. after mutation")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "evasion_levers")


def _extract_shap_matrix(explainer, X: np.ndarray, n_features: int) -> np.ndarray:
    sv = explainer.shap_values(X)
    if hasattr(sv, "values"):
        sv = sv.values
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        elif sv.shape[0] == 2:
            sv = sv[1]
        else:
            sv = sv[..., -1]
    if sv.shape != (X.shape[0], n_features):
        sv = sv.reshape(X.shape[0], n_features)
    return sv


def plot_shap_summary(model, clean_fraud: pd.DataFrame, corpus: pd.DataFrame, outdir: Path, max_rows=200) -> None:
    try:
        import shap
    except ImportError:
        return
    from .data import engineer_features

    cols = model.feature_columns
    clean_X = engineer_features(clean_fraud)[cols].to_numpy(dtype=float)[:max_rows]
    adv_X = engineer_features(corpus)[cols].to_numpy(dtype=float)[:max_rows]
    if len(clean_X) == 0 or len(adv_X) == 0:
        return

    try:
        explainer = shap.TreeExplainer(model.estimator)
    except Exception:
        return

    for name, X in [("clean", clean_X), ("adversarial", adv_X)]:
        try:
            sv = _extract_shap_matrix(explainer, X, len(cols))
        except Exception:
            continue

        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X, feature_names=cols, show=False)
        plt.title(f"SHAP summary ({name})")
        _save(plt.gcf(), outdir, f"shap_beeswarm_{name}")

        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X, feature_names=cols, plot_type="bar", show=False)
        plt.title(f"SHAP feature importance ({name})")
        _save(plt.gcf(), outdir, f"shap_bar_{name}")


def plot_aggregate_evasion_bar(per_strategy_wilson: dict, outdir: Path) -> None:
    """Task 9+11: per-strategy evasion pooled across repeated runs, using the
    Wilson CIs already computed by stats.wilson_ci rather than recomputing from
    a single run's (much smaller) per-strategy counts."""
    strategies, rates, err_low, err_high = [], [], [], []
    for s, w in per_strategy_wilson.items():
        p = w.get("proportion")
        if p is None:
            continue
        lo = w["ci95_low"] if w.get("ci95_low") is not None else p
        hi = w["ci95_high"] if w.get("ci95_high") is not None else p
        strategies.append(s)
        rates.append(p)
        err_low.append(max(0.0, p - lo))  # see plot_evasion_bar_chart for why
        err_high.append(max(0.0, hi - p))
    if not strategies:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(strategies, rates, yerr=[err_low, err_high], capsize=5, color="#c0392b")
    ax.set_ylabel("Evasion rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-strategy evasion rate, pooled across runs (Wilson 95% CI)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, outdir, "aggregate_per_strategy_evasion")


def plot_aggregate_dropoff(aggregate_metrics: dict, outdir: Path) -> None:
    """Task 9+11: clean vs. adversarial detection rate across repeated runs,
    with the 95% CIs already computed by stats.mean_std_ci95, plus the
    detection drop-off point estimate and CI in the title."""
    clean = aggregate_metrics.get("clean_fraud_detection_rate") or {}
    adv = aggregate_metrics.get("adversarial_detection_rate") or {}
    if clean.get("mean") is None or adv.get("mean") is None:
        return

    labels = ["Clean detection", "Adversarial detection"]
    means = [clean["mean"], adv["mean"]]
    lows = [clean.get("ci95_low", means[0]), adv.get("ci95_low", means[1])]
    highs = [clean.get("ci95_high", means[0]), adv.get("ci95_high", means[1])]
    err_low = [max(0.0, m - (lo if lo is not None else m)) for m, lo in zip(means, lows)]
    err_high = [max(0.0, (hi if hi is not None else m) - m) for m, hi in zip(means, highs)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=[err_low, err_high], capsize=5, color=["#2e7d32", "#c0392b"])
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1)
    drop = aggregate_metrics.get("detection_drop_off") or {}
    title = "Detection drop-off across runs (95% CI)"
    if drop.get("mean") is not None:
        title += (f"\ndrop-off = {drop['mean']:.3f} "
                  f"[{drop.get('ci95_low', float('nan')):.3f}, {drop.get('ci95_high', float('nan')):.3f}]")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outdir, "aggregate_detection_dropoff")


def generate_aggregate_figures(result: dict, outdir: Path, log=lambda msg: None) -> list[str]:
    """Task 11 fix: --figures for run_repeated (mode='repeated') was previously
    a no-op — the exact runs a dissertation cites (--n-runs 10) produced no
    figures at all. These two charts use the pooled statistics already computed
    by run_repeated (per_strategy_wilson_ci, aggregate_metrics), not
    recomputed from a single run.
    """
    if not HAVE_MPL:
        log("matplotlib not installed; skipping --figures.")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    plotters = [
        ("aggregate_per_strategy_evasion",
         lambda: plot_aggregate_evasion_bar(result.get("per_strategy_wilson_ci", {}), outdir)),
        ("aggregate_detection_dropoff",
         lambda: plot_aggregate_dropoff(result.get("aggregate_metrics", {}), outdir)),
    ]
    for name, fn in plotters:
        try:
            fn()
            written.append(name)
        except Exception as exc:  # a figure failing must never fail the whole run
            log(f"Figure '{name}' failed (non-fatal): {exc}")
    log(f"Aggregate figures written to {outdir}/ ({len(written)}/{len(plotters)} chart types, PNG+PDF each).")
    return written


def generate_all_figures(
    model, clean_test: pd.DataFrame, clean_fraud: pd.DataFrame, corpus: pd.DataFrame,
    stress: dict, explanation: dict | None, outdir: Path, log=lambda msg: None,
) -> list[str]:
    """Generate every Task 11 figure; returns the names actually written.

    Never raises — a single figure's failure (e.g. no SHAP installed, or a corpus
    too small for a stable curve) is logged and skipped, not fatal to the run.
    """
    if not HAVE_MPL:
        log("matplotlib not installed; skipping --figures.")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    plotters = [
        ("pr_roc_curves", lambda: plot_pr_roc_curves(model, clean_test, corpus, outdir)),
        ("confusion_matrices", lambda: plot_confusion_matrices(model, clean_test, corpus, outdir)),
        ("per_strategy_evasion", lambda: plot_evasion_bar_chart(stress, outdir)),
        ("threshold_sweep", lambda: plot_threshold_sweep(stress.get("threshold_sweep", {}), outdir)),
        ("evasion_levers", lambda: plot_evasion_levers(explanation, outdir)),
        ("shap_summary", lambda: plot_shap_summary(model, clean_fraud, corpus, outdir)),
    ]
    for name, fn in plotters:
        try:
            fn()
            written.append(name)
        except Exception as exc:  # a figure failing must never fail the whole run
            log(f"Figure '{name}' failed (non-fatal): {exc}")
    log(f"Figures written to {outdir}/ ({len(written)}/{len(plotters)} chart types, PNG+PDF each).")
    return written
