"""One entry point that runs the full baseline -> attack -> stress -> explain flow.

Both the CLI ([cli.py](cli.py)) and the web dashboard ([webapp.py](webapp.py)) call
``run_probe`` so they can never drift apart. It returns a single JSON-serialisable
dict and (optionally) writes the reproducibility artefacts to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .adversary import STRATEGIES, generate_adversarial_corpus
from .data import load_paysim, make_demo_data
from .explain import explain_evasion
from .models import ScoredModel, train_baseline, write_json
from .stress import format_report, rank_weaknesses, stress_test


@dataclass
class ProbeConfig:
    data: str | None = None
    demo: bool = True
    demo_rows: int = 60_000
    sample_legit: int | None = None     # real PaySim only: keep all fraud + this many sampled legit rows
    model: str | None = None            # path to a user's fitted .joblib
    model_type: str = "auto"            # auto | xgboost | rf | gbdt
    tune: bool = True                   # grid-search CV hyperparameter tuning (--no-tune to skip)
    save_model: str | None = None
    backend: str = "rules"              # rules | llm
    llm_model: str = "llama3"
    require_llm: bool = False           # raise instead of degrading on any rules fallback
    validation: str = "strict"          # strict | lenient — see adversary.is_economically_valid
    min_value_retention: float = 0.90   # reject seed+strategy groups that abandon more value than this
    strategies: tuple[str, ...] = STRATEGIES
    max_seeds: int = 500
    out: str | None = "fraudprobe_out"
    seed: int = 42
    explain: bool = True                # run SHAP attribution
    log: object = print                 # progress callback

    def _emit(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:
            pass


def _sample_evading_rows(model: ScoredModel, corpus: pd.DataFrame, per_strategy: int = 3) -> list[dict]:
    """A few concrete evading transactions per strategy, with the model's score.

    These make the abstract 'evasion rate' tangible in the UI: here is a fraud the
    model scored as safe.
    """
    if corpus.empty:
        return []
    proba = model.score_rows(corpus)
    df = corpus.copy()
    df["_fraud_score"] = proba
    df["_flagged"] = (proba >= model.threshold)
    cols = [
        "mutation", "source", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "_fraud_score", "_flagged",
    ]
    cols = [c for c in cols if c in df.columns]
    out = []
    grouper = df.groupby("mutation") if "mutation" in df.columns else [("all", df)]
    for strat, grp in grouper:
        # Prefer the rows that actually evaded (lowest score first).
        evaded = grp[~grp["_flagged"]].sort_values("_fraud_score").head(per_strategy)
        pick = evaded if len(evaded) else grp.sort_values("_fraud_score").head(per_strategy)
        for _, r in pick.iterrows():
            rec = {c.lstrip("_"): (float(r[c]) if isinstance(r[c], (int, float, np.floating, np.integer))
                                   else bool(r[c]) if isinstance(r[c], (bool, np.bool_))
                                   else str(r[c])) for c in cols}
            out.append(rec)
    return out


def run_probe(cfg: ProbeConfig) -> dict:
    """Execute the pipeline and return a structured, JSON-serialisable result."""
    outdir = Path(cfg.out) if cfg.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    df, data_source = _load_data(cfg)
    data_stats = _data_stats(df)
    cfg._emit(f"{data_stats['n_rows']} rows, observed fraud rate = "
              f"{data_stats['observed_fraud_rate']:.4%} ({data_stats['n_fraud']} fraud).")
    model, clean_test, baseline = _fit_baseline(cfg, df)
    clean_fraud, seed_frauds = _seed_frauds(cfg, model, clean_test)

    variant = _attack_and_evaluate(cfg, model, clean_test, clean_fraud, seed_frauds,
                                   cfg.backend, baseline)

    warnings = _fixture_warnings(_resolve_strategies(cfg), data_source)
    for w in warnings:
        cfg._emit(f"WARNING: {w}")

    result = {
        "version": __version__,
        "mode": "single",
        "seed": cfg.seed,
        "data_source": data_source,
        **data_stats,
        "backend": cfg.backend,
        "llm_model": cfg.llm_model if cfg.backend == "llm" else None,
        "baseline": baseline,
        "warnings": warnings,
        **variant,
    }

    if outdir:
        variant["_corpus"].to_csv(outdir / "adversarial_corpus.csv", index=False)
        (outdir / "report.txt").write_text(variant["report_text"])
        write_json({k: v for k, v in result.items() if not k.startswith("_")},
                   outdir / "results.json")
        cfg._emit(f"Artefacts written to {outdir}/")

    return {k: v for k, v in result.items() if not k.startswith("_")}


def run_comparison(cfg: ProbeConfig, backends=("rules", "llm")) -> dict:
    """Attack the SAME trained model + seed frauds with each backend, side by side.

    Training once and sharing the seed set is what makes the comparison fair: the
    only thing that differs between the columns is *how the transactions were
    mutated* (deterministic rules vs a local LLM).
    """
    outdir = Path(cfg.out) if cfg.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    df, data_source = _load_data(cfg)
    data_stats = _data_stats(df)
    cfg._emit(f"{data_stats['n_rows']} rows, observed fraud rate = "
              f"{data_stats['observed_fraud_rate']:.4%} ({data_stats['n_fraud']} fraud).")
    model, clean_test, baseline = _fit_baseline(cfg, df)
    clean_fraud, seed_frauds = _seed_frauds(cfg, model, clean_test)

    variants = []
    for backend in backends:
        cfg._emit(f"--- Attacking with backend='{backend}' ---")
        try:
            v = _attack_and_evaluate(cfg, model, clean_test, clean_fraud, seed_frauds,
                                     backend, baseline)
        except RuntimeError as exc:
            cfg._emit(f"backend='{backend}' produced no valid rows: {exc}")
            continue
        if outdir and "_corpus" in v:
            v["_corpus"].to_csv(outdir / f"adversarial_corpus_{backend}.csv", index=False)
        variants.append({
            "backend": backend,
            "llm_model": cfg.llm_model if backend == "llm" else None,
            "baseline": baseline,
            "seed": cfg.seed,
            "data_source": data_source,
            **{k: val for k, val in v.items() if not k.startswith("_")},
        })

    warnings = _fixture_warnings(_resolve_strategies(cfg), data_source)
    for w in warnings:
        cfg._emit(f"WARNING: {w}")

    result = {
        "version": __version__,
        "mode": "compare",
        "seed": cfg.seed,
        "data_source": data_source,
        **data_stats,
        "baseline": baseline,
        "warnings": warnings,
        "variants": variants,
    }
    if outdir:
        write_json(result, outdir / "results.json")
        cfg._emit(f"Comparison artefacts written to {outdir}/")
    return result


# --------------------------------------------------------------------------- #
# Reusable pipeline stages
# --------------------------------------------------------------------------- #
def _load_data(cfg: ProbeConfig):
    if cfg.data and not cfg.demo:
        cfg._emit(f"Loading dataset from {cfg.data}"
                  + (f" (sampling {cfg.sample_legit} legit rows)" if cfg.sample_legit else ""))
        df = load_paysim(cfg.data, sample_legit=cfg.sample_legit, seed=cfg.seed)
        return df, f"paysim:{Path(cfg.data).name}"
    cfg._emit("Using built-in synthetic (PaySim-shaped) demo data.")
    return make_demo_data(n_rows=cfg.demo_rows, seed=cfg.seed), f"synthetic:{cfg.demo_rows}rows"


def _data_stats(df: pd.DataFrame) -> dict:
    """n_rows and observed_fraud_rate (Task 6) — written for every run, synthetic
    or real, so the two are never confused: a demo run's ~1.3% fraud rate is an
    order of magnitude easier than PaySim's real ~0.129%, and both must be
    traceable from results.json alone."""
    n_rows = int(len(df))
    n_fraud = int(df["isFraud"].sum())
    return {
        "n_rows": n_rows,
        "n_fraud": n_fraud,
        "observed_fraud_rate": (n_fraud / n_rows) if n_rows else 0.0,
    }


def _fit_baseline(cfg: ProbeConfig, df):
    if cfg.model:
        cfg._emit(f"Loading your classifier: {cfg.model}")
        model = ScoredModel.load(cfg.model)
        # Untuned and throwaway: this is only to get a clean_test split and a
        # reference baseline metrics dict, not the model actually being probed.
        _, clean_test, baseline = train_baseline(df, model_type="rf", seed=cfg.seed, tune=False)
        baseline["model_type"] = f"user:{Path(cfg.model).name}"
    else:
        cfg._emit(f"Training baseline classifier ({cfg.model_type})...")
        model, clean_test, baseline = train_baseline(
            df, model_type=cfg.model_type, seed=cfg.seed, tune=cfg.tune, log=cfg._emit,
        )
        if cfg.save_model:
            model.save(cfg.save_model)
            cfg._emit(f"Saved baseline model to {cfg.save_model}")
    cfg._emit(f"Baseline F1={baseline['f1']:.3f} PR-AUC={baseline.get('pr_auc', float('nan')):.3f}")
    return model, clean_test, baseline


def _seed_frauds(cfg: ProbeConfig, model, clean_test):
    clean_fraud = clean_test[clean_test["isFraud"] == 1].reset_index(drop=True)
    proba = model.score_rows(clean_fraud)
    seed_frauds = clean_fraud[proba >= 0.95].reset_index(drop=True)
    if len(seed_frauds) == 0:
        seed_frauds = clean_fraud.head(cfg.max_seeds)
    seed_frauds = seed_frauds.head(cfg.max_seeds)
    cfg._emit(f"{len(seed_frauds)} high-confidence frauds seed the attack.")
    return clean_fraud, seed_frauds


def _resolve_strategies(cfg: ProbeConfig) -> tuple[str, ...]:
    return tuple(cfg.strategies) if cfg.strategies else STRATEGIES


def _fixture_warnings(strategies: tuple[str, ...], data_source: str) -> list[str]:
    """Task 5: flag results that are an artefact of the synthetic fixture, not a
    finding about fraud detection. Extend this if other strategies gain a similar
    dependency on a fixture property the real PaySim data wouldn't share."""
    warnings = []
    if "temporal_dispersion" in strategies and data_source.startswith("synthetic:"):
        warnings.append(
            "temporal_dispersion was run against synthetic demo data. Its evasion rate "
            "reflects the documented hour-of-day concentration assumption in "
            "data.make_demo_data (fraud weighted toward 00:00-05:00), not a measured "
            "property of real fraud. Re-run against real PaySim data before citing this "
            "number."
        )
    return warnings


def _attack_and_evaluate(cfg, model, clean_test, clean_fraud, seed_frauds, backend, baseline):
    """Generate a corpus with `backend`, stress-test it, and explain the evasion."""
    strategies = _resolve_strategies(cfg)
    cfg._emit(f"Generating adversarial corpus (backend={backend}, strategies={list(strategies)})...")
    corpus, adv_report = generate_adversarial_corpus(
        seed_frauds, strategies=strategies, backend=backend,
        llm_model=cfg.llm_model, seed=cfg.seed, require_llm=cfg.require_llm,
        validation=cfg.validation, min_value_retention=cfg.min_value_retention,
    )
    cfg._emit(f"{adv_report.n_generated} adversarial rows "
              f"({adv_report.n_rejected_invalid} rejected as economically invalid, "
              f"validation={adv_report.validation_mode}).")
    if adv_report.rejection_reasons:
        cfg._emit(f"Rejection reasons: {adv_report.rejection_reasons}")
    if adv_report.n_rejected_low_value:
        cfg._emit(f"{adv_report.n_rejected_low_value} seed+strategy group(s) dropped for retaining "
                  f"less than {adv_report.min_value_retention:.0%} of the original value.")
    if adv_report.n_llm_attempted:
        rate = adv_report.llm_success_rate or 0.0
        cfg._emit(f"LLM success rate: {rate*100:.1f}% "
                  f"({adv_report.n_llm_success}/{adv_report.n_llm_attempted} calls)"
                  + (f"; fallbacks: {adv_report.fallback_reasons}" if adv_report.fallback_reasons else ""))
    if corpus.empty:
        raise RuntimeError("No valid adversarial rows generated.")

    stress = stress_test(model, clean_test, corpus)
    report_text = format_report(baseline, stress, adv_report)

    explanation = None
    if cfg.explain:
        cfg._emit("Explaining evasion with SHAP...")
        try:
            explanation = explain_evasion(model, clean_fraud, corpus).to_dict()
            cfg._emit("SHAP attribution complete." if explanation.get("available")
                      else f"SHAP unavailable: {explanation.get('note', '')}")
        except Exception as exc:  # never let explanation failure kill a run
            explanation = {"available": False, "backend": "error", "note": str(exc)}
            cfg._emit(f"SHAP failed (non-fatal): {exc}")

    return {
        "stress": stress,
        "adversary": adv_report.__dict__,
        "weakness_ranking": [{"strategy": s, "evasion_rate": e} for s, e in rank_weaknesses(stress)],
        "explanation": explanation,
        "sample_evasions": _sample_evading_rows(model, corpus),
        "report_text": report_text,
        "_corpus": corpus,
    }
