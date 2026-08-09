"""Statistics for repeated runs (Task 9): means/CIs across seeds, a paired
significance test of clean vs. adversarial detection, and Wilson confidence
intervals on per-strategy evasion rates.

A single run at a fixed seed cannot support any of this — the dissertation's
methodology promises paired significance tests and 95% confidence intervals,
neither of which is computable from one observation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

# A pooled per-strategy sample below this size is flagged as "wide" — the
# interval is still reported, never hidden, but a reader must not treat a narrow-
# looking point estimate from ~20 observations as precise.
WIDE_INTERVAL_N_THRESHOLD = 100


def mean_std_ci95(values, bounds: tuple[float, float] | None = None) -> dict:
    """Mean, sample std, and a 95% CI via the t-distribution (appropriate for the
    small number of runs a repeated fraudprobe experiment actually produces).

    The t-based interval is symmetric around the mean, so for a value near a hard
    bound (e.g. a detection rate near 100%) it can extend past that bound — pass
    ``bounds=(0.0, 1.0)`` for any metric that is itself a bounded rate/proportion
    to clip the reported interval to what's actually possible. Left unclipped by
    default since not every metric this is used for is bounded.
    """
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        half = float(se * scipy_stats.t.ppf(0.975, df=n - 1))
    else:
        half = 0.0
    low, high = mean - half, mean + half
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return {"n": n, "mean": mean, "std": std, "ci95_low": low, "ci95_high": high}


def paired_test(clean, adversarial) -> dict:
    """Paired test of clean vs. adversarial detection across repeated runs.

    Reports scipy.stats.ttest_rel (parametric) and wilcoxon (its non-parametric
    fallback — robust when the per-run differences aren't approximately normal,
    which is likely with few runs), plus Cohen's d_z as the paired effect size.
    """
    clean = np.asarray(list(clean), dtype=float)
    adv = np.asarray(list(adversarial), dtype=float)
    if len(clean) != len(adv):
        raise ValueError("clean and adversarial must have the same length (paired runs)")

    diff = clean - adv
    out = {"n_pairs": int(len(clean)), "mean_diff": float(np.mean(diff)) if len(diff) else None}

    if len(clean) < 2:
        out.update(ttest=None, wilcoxon=None, cohens_d_z=None,
                    note="Fewer than 2 paired runs — no significance test is computable.")
        return out
    if np.allclose(diff, diff[0]):
        out.update(ttest=None, wilcoxon=None, cohens_d_z=None,
                    note="All paired differences are identical (zero variance) — "
                         "t-test and Wilcoxon are undefined.")
        return out

    t_stat, t_p = scipy_stats.ttest_rel(clean, adv)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(clean, adv)
        wilcoxon = {"statistic": float(w_stat), "p_value": float(w_p)}
    except ValueError:
        # e.g. all differences are zero after ranking, or n too small
        wilcoxon = None

    std_diff = float(np.std(diff, ddof=1))
    cohens_d_z = float(np.mean(diff) / std_diff) if std_diff > 0 else None

    out.update(
        ttest={"statistic": float(t_stat), "p_value": float(t_p)},
        wilcoxon=wilcoxon,
        cohens_d_z=cohens_d_z,
    )
    return out


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> dict:
    """Wilson score interval for a binomial proportion.

    Better-behaved than the normal approximation at small n or proportions near
    0/1 — exactly the regime per-strategy evasion counts (often tens of
    observations, sometimes 0% or 100% evaded) tend to land in.
    """
    if n == 0:
        return {"n": 0, "proportion": None, "ci95_low": None, "ci95_high": None, "wide": True}
    z = float(scipy_stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    low, high = max(0.0, center - half), min(1.0, center + half)
    return {
        "n": n,
        "proportion": float(p),
        "ci95_low": float(low),
        "ci95_high": float(high),
        # Not a hard statistical threshold — a practical flag so a reader doesn't
        # mistake a point estimate from a small pooled sample for a precise one.
        "wide": n < WIDE_INTERVAL_N_THRESHOLD,
    }


def format_aggregate_report(result: dict) -> str:
    """Human-readable summary of a run_repeated() result for the terminal."""
    lines = []
    add = lines.append
    add("=" * 62)
    add("  FRAUDPROBE — AGGREGATE REPORT ACROSS REPEATED RUNS")
    add("=" * 62)
    add("")
    add(f"  Runs completed          : {result['n_runs_completed']}/{result['n_runs_requested']}"
        f" (seeds {result['base_seed']}..{result['base_seed'] + result['n_runs_requested'] - 1})")
    add(f"  Backend                 : {result['backend']}"
        + (f" ({result['llm_model']})" if result.get("llm_model") else ""))
    add("")
    m = result["aggregate_metrics"]
    add("-" * 62)
    for key, label in [
        ("clean_fraud_detection_rate", "Clean detection rate"),
        ("adversarial_detection_rate", "Adversarial detection rate"),
        ("detection_drop_off", "Detection drop-off"),
        ("evasion_rate", "Evasion rate"),
        ("baseline_f1", "Baseline F1"),
    ]:
        s = m[key]
        add(f"  {label:28s}: {s['mean']*100:5.1f}% +/- {s['std']*100:4.1f}%  "
            f"(95% CI [{s['ci95_low']*100:5.1f}%, {s['ci95_high']*100:5.1f}%], n={s['n']})")
    add("-" * 62)
    add("")

    sig = result["paired_significance_clean_vs_adversarial"]
    add("  Paired significance (clean vs. adversarial detection, across runs):")
    if sig.get("note"):
        add(f"    {sig['note']}")
    else:
        add(f"    t-test:   t={sig['ttest']['statistic']:+.3f}, p={sig['ttest']['p_value']:.4g}")
        if sig["wilcoxon"]:
            add(f"    Wilcoxon: W={sig['wilcoxon']['statistic']:.3f}, p={sig['wilcoxon']['p_value']:.4g}")
        add(f"    Effect size (Cohen's d_z): {sig['cohens_d_z']:+.3f}")
    add("")

    add("  Per-strategy evasion rate (Wilson 95% CI, pooled across runs):")
    for strat, w in result["per_strategy_wilson_ci"].items():
        if w["proportion"] is None:
            add(f"    {strat:22s} no observations")
            continue
        flag = "  [WIDE — small sample]" if w["wide"] else ""
        add(f"    {strat:22s} {w['proportion']*100:5.1f}% "
            f"[{w['ci95_low']*100:5.1f}%, {w['ci95_high']*100:5.1f}%] (n={w['n']}){flag}")
    add("")
    add("=" * 62)
    return "\n".join(lines)
