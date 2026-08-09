"""One entry point that runs the full baseline -> attack -> stress -> explain flow.

Both the CLI ([cli.py](cli.py)) and the web dashboard ([webapp.py](webapp.py)) call
``run_probe`` so they can never drift apart. It returns a single JSON-serialisable
dict and (optionally) writes the reproducibility artefacts to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .adversary import STRATEGIES, LLMRequiredError, generate_adversarial_corpus
from .data import load_paysim, make_demo_data
from .explain import explain_evasion
from .models import ScoredModel, train_baseline, write_json
from .stats import mean_std_ci95, paired_test, wilson_ci
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


def _paired_backend_comparison(model: ScoredModel, corpora: dict[str, pd.DataFrame]) -> dict | None:
    """Task 10: pair each backend's evasion by (seed_idx, mutation) — the same
    underlying seed fraud, mutated by each backend — and test the difference.

    Aggregate rate comparison ("rules evaded 55%, llm evaded 62%") can't tell you
    whether that gap is real or noise from which seeds happened to be attacked.
    Pairing by the shared seed set (guaranteed identical across backends — see
    run_comparison) turns it into exactly the paired design paired_test expects.
    Returns None if fewer than two backends produced a usable corpus.
    """
    usable = {b: df for b, df in corpora.items() if not df.empty and "seed_idx" in df.columns}
    if len(usable) != 2:
        return None
    (a_name, a_df), (b_name, b_df) = usable.items()

    def group_evasion(df: pd.DataFrame) -> pd.Series:
        proba = model.score_rows(df)
        flagged = pd.Series(proba >= model.threshold, index=df.index)
        evaded = 1.0 - flagged.groupby([df["seed_idx"], df["mutation"]]).mean()
        return evaded

    a_evasion, b_evasion = group_evasion(a_df), group_evasion(b_df)
    common = a_evasion.index.intersection(b_evasion.index)
    if len(common) == 0:
        return None

    a_vals, b_vals = a_evasion.loc[common].to_numpy(), b_evasion.loc[common].to_numpy()
    diff_stats = mean_std_ci95(b_vals - a_vals, bounds=(-1.0, 1.0))

    return {
        "backend_a": a_name,
        "backend_b": b_name,
        "n_paired_groups": int(len(common)),
        "mean_evasion_a": float(np.mean(a_vals)),
        "mean_evasion_b": float(np.mean(b_vals)),
        "mean_difference_b_minus_a": diff_stats["mean"],
        "difference_ci95": {"low": diff_stats["ci95_low"], "high": diff_stats["ci95_high"]},
        "paired_test": paired_test(a_vals, b_vals),
    }


def run_comparison(cfg: ProbeConfig, backends=("rules", "llm")) -> dict:
    """Attack the SAME trained model + seed frauds with each backend, side by side.

    Training once and sharing the seed set is what makes the comparison fair: the
    only thing that differs between the columns is *how the transactions were
    mutated* (deterministic rules vs a local LLM). See _paired_backend_comparison
    for the statistical test this enables (Task 10).
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
    cfg._emit(f"{len(seed_frauds)} seed frauds shared identically across all backends.")

    variants = []
    corpora: dict[str, pd.DataFrame] = {}
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
        corpora[backend] = v["_corpus"]
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

    paired_comparison = _paired_backend_comparison(model, corpora)
    if paired_comparison:
        cfg._emit(f"Paired comparison ({paired_comparison['backend_a']} vs "
                  f"{paired_comparison['backend_b']}, n={paired_comparison['n_paired_groups']} "
                  f"shared seed+strategy groups): mean difference = "
                  f"{paired_comparison['mean_difference_b_minus_a']:+.3f} "
                  f"(95% CI [{paired_comparison['difference_ci95']['low']:+.3f}, "
                  f"{paired_comparison['difference_ci95']['high']:+.3f}])")

    result = {
        "version": __version__,
        "mode": "compare",
        "seed": cfg.seed,
        "data_source": data_source,
        **data_stats,
        "baseline": baseline,
        "warnings": warnings,
        "variants": variants,
        "paired_backend_comparison": paired_comparison,
    }
    if outdir:
        write_json(result, outdir / "results.json")
        cfg._emit(f"Comparison artefacts written to {outdir}/")
    return result


def run_repeated(cfg: ProbeConfig, n_runs: int = 10) -> dict:
    """Task 9: re-run the full single-backend pipeline across n_runs distinct
    seeds (cfg.seed, cfg.seed+1, ..., cfg.seed+n_runs-1) and aggregate.

    A single run at a fixed seed cannot support the paired significance test or
    95% confidence intervals the methodology promises — this is what makes both
    computable. Individual runs are not written to disk (their `out` is
    suppressed); only the aggregate goes to results_aggregate.json.
    """
    outdir = Path(cfg.out) if cfg.out else None

    per_run = []
    for i in range(n_runs):
        run_cfg = replace(cfg, seed=cfg.seed + i, out=None)
        cfg._emit(f"--- Run {i + 1}/{n_runs} (seed={run_cfg.seed}) ---")
        try:
            per_run.append(run_probe(run_cfg))
        except (RuntimeError, LLMRequiredError) as exc:
            cfg._emit(f"Run {i + 1} (seed={run_cfg.seed}) failed, excluded from aggregate: {exc}")

    if not per_run:
        raise RuntimeError(f"All {n_runs} repeated runs failed; nothing to aggregate.")
    if len(per_run) < n_runs:
        cfg._emit(f"WARNING: only {len(per_run)}/{n_runs} runs succeeded; "
                  f"statistics below are computed from the successful runs only.")

    clean_rates = [r["stress"]["clean_fraud_detection_rate"] for r in per_run]
    adv_rates = [r["stress"]["adversarial_detection_rate"] for r in per_run]

    aggregate_metrics = {
        "clean_fraud_detection_rate": mean_std_ci95(clean_rates, bounds=(0.0, 1.0)),
        "adversarial_detection_rate": mean_std_ci95(adv_rates, bounds=(0.0, 1.0)),
        "detection_drop_off": mean_std_ci95(
            (r["stress"]["detection_drop_off"] for r in per_run), bounds=(-1.0, 1.0)
        ),
        "evasion_rate": mean_std_ci95((r["stress"]["evasion_rate"] for r in per_run), bounds=(0.0, 1.0)),
        "baseline_f1": mean_std_ci95((r["baseline"]["f1"] for r in per_run), bounds=(0.0, 1.0)),
    }

    significance = paired_test(clean_rates, adv_rates)

    # Per-strategy Wilson CIs, pooled across all successful runs — a single run's
    # seed-limited corpus (often ~20 rows per strategy) makes for a wide interval
    # on its own; pooling narrows it but does not always escape "wide" (flagged
    # explicitly on each entry, never silently presented as precise).
    pools: dict[str, dict[str, int]] = {}
    for r in per_run:
        for strat, s in r["stress"]["per_strategy"].items():
            pool = pools.setdefault(strat, {"n": 0, "evaded": 0})
            pool["n"] += s["n"]
            pool["evaded"] += round(s["evasion_rate"] * s["n"])
    per_strategy_wilson = {strat: wilson_ci(pool["evaded"], pool["n"]) for strat, pool in pools.items()}
    for strat, w in per_strategy_wilson.items():
        if w["wide"]:
            cfg._emit(f"NOTE: per-strategy evasion CI for '{strat}' is wide (n={w['n']} pooled "
                      f"observations, 95% CI [{w['ci95_low']:.2f}, {w['ci95_high']:.2f}]).")

    result = {
        "version": __version__,
        "mode": "repeated",
        "n_runs_requested": n_runs,
        "n_runs_completed": len(per_run),
        "base_seed": cfg.seed,
        "seeds": [r["seed"] for r in per_run],
        "backend": cfg.backend,
        "llm_model": cfg.llm_model if cfg.backend == "llm" else None,
        "aggregate_metrics": aggregate_metrics,
        "paired_significance_clean_vs_adversarial": significance,
        "per_strategy_wilson_ci": per_strategy_wilson,
        "per_run_summary": [
            {
                "seed": r["seed"],
                "clean_detection_rate": r["stress"]["clean_fraud_detection_rate"],
                "adversarial_detection_rate": r["stress"]["adversarial_detection_rate"],
                "evasion_rate": r["stress"]["evasion_rate"],
            }
            for r in per_run
        ],
    }

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(result, outdir / "results_aggregate.json")
        cfg._emit(f"Aggregate results written to {outdir}/results_aggregate.json")

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
        log=cfg._emit,
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
