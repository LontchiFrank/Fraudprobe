"""Task 11: report-ready figures. Uses real (small) trained models/corpora since
the figures depend on genuine sklearn/SHAP objects, not just data shapes."""

from __future__ import annotations

import pytest

from fraudprobe.adversary import generate_adversarial_corpus
from fraudprobe.data import make_demo_data
from fraudprobe.explain import explain_evasion
from fraudprobe.figures import HAVE_MPL, generate_all_figures
from fraudprobe.models import train_baseline
from fraudprobe.stress import stress_test

pytestmark = pytest.mark.skipif(not HAVE_MPL, reason="matplotlib not installed")


@pytest.fixture(scope="module")
def scenario():
    df = make_demo_data(n_rows=4000, fraud_rate=0.03, seed=1)
    model, clean_test, baseline = train_baseline(df, model_type="xgboost", seed=1, tune=False)
    clean_fraud = clean_test[clean_test["isFraud"] == 1].reset_index(drop=True)
    proba = model.score_rows(clean_fraud)
    seed_frauds = clean_fraud[proba >= 0.95].reset_index(drop=True).head(15)
    corpus, adv_report = generate_adversarial_corpus(
        seed_frauds, strategies=("amount_split", "balance_camouflage"), backend="rules", seed=1
    )
    stress = stress_test(model, clean_test, corpus)
    explanation = explain_evasion(model, clean_fraud, corpus).to_dict()
    return {
        "model": model, "clean_test": clean_test, "clean_fraud": clean_fraud,
        "corpus": corpus, "stress": stress, "explanation": explanation,
    }


def test_generate_all_figures_writes_png_and_pdf(tmp_path, scenario):
    written = generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert len(written) > 0
    pngs = list(tmp_path.glob("*.png"))
    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pngs) > 0
    assert len(pdfs) > 0
    # Every PNG should have a matching PDF (same figures, both formats).
    png_stems = {p.stem for p in pngs}
    pdf_stems = {p.stem for p in pdfs}
    assert png_stems == pdf_stems


def test_pr_roc_curves_file_created(tmp_path, scenario):
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert (tmp_path / "pr_roc_curves.png").exists()
    assert (tmp_path / "pr_roc_curves.pdf").exists()


def test_confusion_matrices_file_created(tmp_path, scenario):
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert (tmp_path / "confusion_matrices.png").exists()


def test_per_strategy_evasion_chart_created(tmp_path, scenario):
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert (tmp_path / "per_strategy_evasion.png").exists()


def test_threshold_sweep_chart_created(tmp_path, scenario):
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert (tmp_path / "threshold_sweep.png").exists()


def test_evasion_levers_chart_created_when_explanation_available(tmp_path, scenario):
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    if scenario["explanation"].get("available"):
        assert (tmp_path / "evasion_levers.png").exists()


def test_shap_summary_charts_created(tmp_path, scenario):
    pytest.importorskip("shap")
    generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], scenario["explanation"], tmp_path,
    )
    assert (tmp_path / "shap_beeswarm_clean.png").exists()
    assert (tmp_path / "shap_bar_clean.png").exists()
    assert (tmp_path / "shap_beeswarm_adversarial.png").exists()


def test_generate_all_figures_never_raises_on_empty_corpus(tmp_path, scenario):
    import pandas as pd
    empty = pd.DataFrame(columns=scenario["corpus"].columns)
    empty_stress = {"per_strategy": {}, "threshold_sweep": {"grid": [], "fixed_fpr": []}}
    written = generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], empty,
        empty_stress, None, tmp_path,
    )
    # Should not raise; some figures legitimately produce nothing for empty input.
    assert isinstance(written, list)


def test_generate_all_figures_handles_missing_explanation(tmp_path, scenario):
    written = generate_all_figures(
        scenario["model"], scenario["clean_test"], scenario["clean_fraud"], scenario["corpus"],
        scenario["stress"], None, tmp_path,
    )
    assert not (tmp_path / "evasion_levers.png").exists()
    assert len(written) > 0
