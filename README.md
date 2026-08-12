# fraudprobe

**Can your fraud detector be fooled by AI? Find out in 30 seconds.**

Bank fraud-detection models score 95%+ on normal fraud. But what happens when an
attacker uses a **free, locally-hosted LLM** to reshape a fraudulent transaction —
splitting the amount, shifting the timing, leaving a residual balance — so it still
steals the same money but no longer *looks* like fraud?

`fraudprobe` answers that question for any fraud classifier. Point it at a model and
it generates a corpus of AI-mutated frauds, replays them, and tells you **how many
slipped through and which trick was the weak spot.**

```
==============================================================
  Clean fraud detection rate       :  97.4%
  Adversarial detection rate       :  44.3%
  >> DETECTION DROP-OFF            :  53.1 pts
  >> OVERALL EVASION RATE          :  55.7%
--------------------------------------------------------------
  Weakness ranking (which trick fooled the model most):
    amount_split            88.3% evaded  ###################################
    balance_camouflage      13.4% evaded  #####
    temporal_dispersion      0.0% evaded
==============================================================
  VERDICT: weakest against 'amount_split' (88% of those slipped through).
```

> ⚠️ **Defensive tool.** fraudprobe exists so teams can find and fix these weaknesses
> *before* attackers exploit them. It runs entirely offline on synthetic data.

## Why this matters

Adversarial ML is well studied for images and text, but **tabular financial data is
different** — mutations have to preserve economic reality (you can't spend money you
don't have). fraudprobe's mutations pass an economic-consistency validator, so every
evasion is a *plausible* transaction, not an arithmetic impossibility. And because the
attacker is a free local model, the threat it measures is one a non-expert could
actually mount today.

## Install

```bash
pip install fraudprobe            # core: scikit-learn + the web dashboard (Flask)
pip install "fraudprobe[full]"    # + XGBoost, SMOTE, matplotlib
pip install "fraudprobe[explain]" # + SHAP evasion explanations (needs Python <3.14)
pip install "fraudprobe[llm]"     # + Ollama LLM backend
```

> **Note on SHAP / Python 3.14.** SHAP depends on `numba`, which lags the newest
> CPython. If you're on 3.14, install the `explain` extra under a 3.10–3.13
> interpreter; the rest of fraudprobe runs fine on 3.14 without it (the dashboard
> just omits the SHAP panel).

## Quickstart

```bash
# Zero setup — synthetic data, no downloads, no GPU:
fraudprobe run --demo

# Web dashboard (run probes and see the report in your browser):
fraudprobe serve            # then open http://127.0.0.1:5000

# Real run against the PaySim dataset:
fraudprobe run --data paysim.csv --model-type xgboost

# Attack with a local LLM (llama3 via Ollama) instead of deterministic rules:
fraudprobe run --demo --backend llm --llm-model llama3 --max-seeds 20
```

## Web dashboard

`fraudprobe serve` launches a Flask dashboard that runs the full pipeline on demand
and renders the resilience report: the baseline-vs-adversarial headline tiles, the
weakness ranking, concrete evading transactions, and — when SHAP is installed — the
**evasion levers**: which transaction features the attacker neutralised to push the
fraud score below the model's threshold.

## Test YOUR own classifier

This is the point of the tool. Wrap any fitted model that exposes `predict_proba`
and fraudprobe will attack it:

```python
from fraudprobe import ScoredModel, train_baseline, make_demo_data

# ... you have your own fitted fraud model `clf` ...
ScoredModel(estimator=clf, feature_columns=my_features).save("my_model.joblib")
```

```bash
fraudprobe run --data paysim.csv --model my_model.joblib
```

Out comes a resilience report telling you your model's blind spots.

## How it works

Three phases, mirroring the research design behind it:

1. **Baseline** — trains (or loads) a fraud classifier and measures clean performance.
2. **Adversary** — takes high-confidence caught frauds and mutates them with three
   strategies (`amount_split`, `temporal_dispersion`, `balance_camouflage`), using
   either deterministic rules (default) or a local LLM (`--backend llm`).
3. **Stress test** — replays the corpus, measures the detection drop-off, and ranks
   which strategy evaded most.
4. **Explain** — uses SHAP to attribute the drop-off to individual features, so you
   see *which* signals the attacker neutralised (skip with `--no-explain`).

Every run writes `adversarial_corpus.csv`, `results.json` (now including the SHAP
attribution and sample evasions), and `report.txt` with a fixed seed for full
reproducibility.

## Methodological rigor

fraudprobe is being hardened against a validity review (see `TASKS.md`) so every
number in a report is measured, not assumed. So far:

- **LLM provenance** — every mutation is tagged `source=llm`, `source=rules_fallback`,
  or `source=rules`, and the corpus/`results.json` report `llm_success_rate` and a
  breakdown of *why* any call fell back (`connection_error`, `timeout`,
  `json_parse_error`, `schema_error`, ...). `--require-llm` aborts instead of
  silently degrading to rules, for headline results that must be pure-LLM — but
  measured against a real llama3 server at PaySim scale (hundreds of calls per
  run), a ~1-in-13 `schema_error` rate makes surviving a whole run with zero
  failures a near-certainty *not* to happen, so `--require-llm` is not what the
  reportable run set actually uses (see below). On a `schema_error` specifically
  (not other failure modes — retrying a dead connection is pointless), one
  same-prompt retry is attempted; `n_llm_retried`/`n_llm_retry_success` and
  `llm_single_shot_success_rate` (before any retry) vs. `llm_success_rate`
  (with retry) are both recorded, so neither number is lost to the other.
  `results.json`'s `stress.per_strategy_llm_only` (and `--n-runs`'s
  `per_strategy_wilson_ci_llm_only`) recompute per-strategy evasion restricted
  to genuinely `source=="llm"` rows — the pure-LLM number `--require-llm` was
  meant to guarantee, obtained without gambling an entire run on it.
- **Matched interventions** — `amount_split` produces the same number of rows under
  both backends (the LLM proposes *proportions*, Python always does the balance
  arithmetic), so per-strategy evasion is comparable across rules vs. LLM.
- **Economic validation** — `--validation strict` (default) additionally requires
  that `newbalanceOrig`/`newbalanceDest` actually reconcile with `amount`, not just
  that signs and funds are plausible. `--validation lenient` restores the original,
  looser check, so both can be run and the difference reported.
- **Value retention** — `results.json` reports mean/median/min retained value per
  strategy (not forced to 1.0 — `balance_camouflage`'s deliberate residual shows up
  honestly as ~97%). `--min-value-retention` (default `0.90`) drops mutations that
  abandon most of the money before they can be counted as an "evasion".
- **Real data path** — `load_paysim` downcasts numeric dtypes so the full 6.36M-row
  PaySim CSV loads in ~1.3GB without chunking. `--sample-legit N` keeps every fraud
  row plus a stratified sample of legitimate ones; every run (synthetic or real)
  reports `n_rows`/`observed_fraud_rate` so the two are never confused. The CLI's
  `--demo` mode defaults `--demo-fraud-rate` to `0.00129` (real PaySim's observed
  rate) rather than `make_demo_data`'s own `0.013` library default — a demo run an
  order of magnitude easier than the real problem is what produced the original
  validity gap this brief exists to fix.
- **Tuned baseline** — hyperparameters are selected via stratified 5-fold grid
  search with SMOTE fit *inside* each fold (never on validation data), then refit
  once on the full training partition. `results.json`'s `baseline.tuning` records
  the search space, winning parameters, and cross-validated F1 (mean ± std).
  Tuning is on by default; pass `--no-tune` for the old fixed-hyperparameter, fast
  behaviour when iterating quickly (grid search adds ~15-20s for XGBoost/`auto` at
  demo scale, more for `rf`/`gbdt`, which are inherently slower per fit).
- **Threshold sweep** — every run reports clean-vs-adversarial detection across a
  full threshold grid, plus at thresholds pinned to realistic false-positive
  budgets (0.1%/0.5%/1%) — a fixed 0.5 cutoff is a modelling convenience, not how
  a real fraud team operates one, and evasion can look very different at a
  better-chosen operating point.
- **Statistics across seeds** — `--n-runs N` repeats the full pipeline across N
  distinct seeds and writes `results_aggregate.json`: means with 95% CIs, a paired
  test (t-test + Wilcoxon fallback, Cohen's d_z) of clean vs. adversarial
  detection, and Wilson confidence intervals per strategy (flagged when the pooled
  sample is small, never silently presented as precise). When `--out` is set, each
  individual run's artefacts (corpus, report, `results.json`, and — with
  `--figures` — its own figures/tables/manifest, including SHAP) land under
  `out/runs/seed_<n>/` so any one run can be inspected on its own.
- **Rules vs. LLM at scale** — `--compare` attacks the *same* trained model and
  *identical* seed frauds with both backends and reports a paired comparison:
  mean evasion per backend, the difference with a 95% CI, and a significance test
  — isolating what the LLM specifically contributes over a heuristic. LLM calls
  run ~8-25s each depending on load; progress and an ETA are logged periodically,
  and total/mean/min/max call latency land in `results.json`.
- **Transaction velocity / destination behaviour** — `FEATURE_COLUMNS` now
  includes `txn_velocity_orig` (backward-looking rolling count of same-`nameOrig`
  transactions within a 24-step window) and `dest_txn_count` (total transactions
  received by `nameDest`), computed strictly from whatever frame `engineer_features`
  is given — never joined against an external history table, so it stays a pure,
  reapply-to-mutated-rows-unchanged function (see its docstring for the full design
  rationale). Concrete consequence: scored against the adversarial corpus alone
  (not the full clean population), `amount_split`'s `AMOUNT_SPLIT_PARTS` sibling
  rows genuinely earn a velocity count > 1 — a real structuring signal — while
  single-row strategies see 1 regardless. On `--demo` synthetic data both features
  are degenerate (every row gets an independently random account ID, so the clean
  population never has velocity > 1 and the features carry no training signal) —
  flagged automatically as a fixture-dependency warning in every `--demo` run,
  same treatment as `temporal_dispersion`'s existing warning. Real PaySim data does
  have repeat account IDs; any evasion-lever ranking involving these two features
  must be re-measured there before citing it.
- **Report-ready artefacts** — `--figures` writes 300 dpi PNG + vector PDF to
  `out/figures/` (PR/ROC curves, confusion matrices, per-strategy evasion with
  Wilson CIs, the threshold sweep, SHAP summary, and evasion levers), CSV tables
  to `out/results_tables/`, and `out/MANIFEST.json` (package/Python/OS/CPU/RAM,
  Ollama model, git commit + dirty flag (with diff stat when dirty), seed(s), wall-clock, full CLI
  invocation) — for single runs, `--compare`, and `--n-runs` alike, so the exact
  runs cited in a report all produce the same artefacts, generated rather than
  screenshotted.

## Research context

fraudprobe is the reference implementation for the MSc dissertation *"Evaluating the
Resilience of Open Banking Anti-Fraud Classifiers Against LLM-Generated Adversarial
Transaction Sequences"* (Edge Hill University, 2026). If you use it in academic work,
a citation is appreciated — see [CITATION.cff](CITATION.cff).

## License

MIT. Uses the PaySim synthetic dataset (Lopez-Rojas & Axelsson, 2016) — no real
financial data is ever processed.
