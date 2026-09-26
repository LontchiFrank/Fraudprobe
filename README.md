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
> *before* attackers exploit them. It runs entirely offline, and its zero-setup demo
> mode never touches real financial data.

## Why this matters

Adversarial ML is well studied for images and text, but **tabular financial data is
different** — mutations have to preserve economic reality (you can't spend money you
don't have). fraudprobe's mutations pass an economic-consistency validator, so every
evasion is a *plausible* transaction, not an arithmetic impossibility. And because the
attacker is a free local model, the threat it measures is one a non-expert could
actually mount today.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Opening the project](#opening-the-project)
- [How the pipeline works](#how-the-pipeline-works)
- [Using the CLI](#using-the-cli)
- [Using the web dashboard](#using-the-web-dashboard)
- [What gets written to disk](#what-gets-written-to-disk)
- [Test YOUR own classifier](#test-your-own-classifier)
- [Using fraudprobe as a library](#using-fraudprobe-as-a-library)
- [Methodological rigor](#methodological-rigor)
- [Research context](#research-context) · [License](#license)

## Requirements

- **Python 3.10–3.13.** (3.14 works for everything *except* the SHAP explanations —
  see the note below.)
- **macOS/Linux/Windows**, CPU only — no GPU is used or required anywhere in the tool.
- **Optional, only if you want them:**
  - [Ollama](https://ollama.com) running locally, with a model pulled (e.g.
    `ollama pull llama3`) — only needed for `--backend llm`. Everything else (the
    default `rules` backend, the dashboard, real-data runs) works without it.
  - The real [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) CSV —
    only needed if you want to test against real transaction data instead of the
    built-in synthetic demo data.

> **Note on SHAP / Python 3.14.** SHAP (the "why did it evade" feature attribution)
> depends on `numba`, which lags the newest CPython. If you're on 3.14, install the
> `explain` extra under a 3.10–3.13 interpreter instead; the rest of fraudprobe runs
> fine on 3.14 without it (the dashboard just omits the SHAP panel, and `--no-explain`
> skips it on the CLI).

## Installation

fraudprobe isn't published on PyPI yet, so install it straight from a clone:

```bash
# 1. Get the code
git clone https://github.com/LontchiFrank/Fraudprobe.git
cd Fraudprobe

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install fraudprobe and its dependencies
pip install -e .                   # core only: scikit-learn + the web dashboard (Flask)
pip install -e ".[full]"           # + XGBoost, SMOTE (imbalanced-learn), matplotlib figures
pip install -e ".[explain]"        # + SHAP evasion explanations (needs Python <3.14)
pip install -e ".[llm]"            # + the Ollama client, for the LLM backend
```

For the full experience (every classifier type, every figure, SHAP, and the LLM
backend), install all four extras together:

```bash
pip install -e ".[full,explain,llm]"
```

`-e` (editable install) means the `fraudprobe` command and `import fraudprobe` both
run directly against your checked-out copy of the code — useful if you're going to
read or modify it, and harmless if you're not.

**XGBoost on macOS** needs the OpenMP runtime, which isn't bundled by default:

```bash
brew install libomp
```

(If you skip this, everything still works — `--model-type auto`/`rf`/`gbdt` don't
need it, and fraudprobe falls back to Random Forest if XGBoost fails to import.)

Confirm the install:

```bash
fraudprobe --version
```

## Opening the project

There are two ways to "open" fraudprobe: as a **command-line tool** (scriptable,
reproducible, what the reportable/research runs use) or as a **web dashboard**
(interactive, point-and-click, best for exploring a first result).

### Option A — the CLI, zero setup

```bash
fraudprobe run --demo
```

This trains a classifier on built-in synthetic data, attacks it, and prints the
report shown at the top of this file straight to your terminal — no data download,
no LLM, no GPU. It also writes its artefacts to `./fraudprobe_out/` (see
[What gets written to disk](#what-gets-written-to-disk)).

### Option B — the web dashboard

```bash
fraudprobe serve
```

Then open **http://127.0.0.1:5000** in your browser. You'll land on a form (see
[Using the web dashboard](#using-the-web-dashboard) below for what every field does)
— pick a model and backend, press **Run probe**, and watch the report render live.

> **Port already in use?** On macOS, port 5000 is often held by the AirPlay Receiver
> (System Settings → General → AirDrop & Handoff). Run `fraudprobe serve --port 5050`
> instead and open `http://127.0.0.1:5050`.

## How the pipeline works

Every run — CLI or dashboard — goes through the same four stages:

1. **Baseline** — trains (or loads) a fraud classifier (XGBoost, Random Forest, or
   Gradient Boosting) and measures clean-conditions performance: F1, PR-AUC, recall
   on held-out fraud.
2. **Seed selection** — pulls the frauds the model catches with ≥95% confidence from
   the held-out set. These are the "obvious" frauds — the ones you'd assume are
   safely caught.
3. **Adversary** — mutates each seed fraud so it moves the *same amount of money* but
   looks less suspicious, using one or more strategies:
   | Strategy | What it does |
   |---|---|
   | `amount_split` | Breaks one large transfer into several smaller ones that sum to the same total (classic "structuring"). |
   | `temporal_dispersion` | Shifts the transaction to a less suspicious hour. |
   | `balance_camouflage` | Leaves a small residual balance instead of draining the account to exactly zero — an account drained to `0` is one of the strongest fraud signals in this data. |

   Two **backends** generate these mutations: `rules` (deterministic Python, instant)
   or `llm` (a local model — llama3 by default, via Ollama — does the same job in
   natural language; slower, and every output still has to pass an
   economic-validity check before it's accepted, exactly like the rules backend's
   output does).
4. **Stress test** — rescores the mutated corpus with the *same* baseline model and
   compares detection: clean vs. adversarial recall, the point-drop, and which
   strategy evaded most.
5. **Explain (SHAP, optional)** — for each feature, how much it pushed the model
   toward "fraud" on the original catches vs. the mutated versions. The biggest drops
   are the levers the attacker pulled to look legitimate.

## Using the CLI

```
fraudprobe run [options]      # baseline -> attack -> stress-test -> explain
fraudprobe serve [options]    # launch the web dashboard
fraudprobe --version
```

### `run` — data & baseline

| Flag | Default | What it does |
|---|---|---|
| `--demo` | on unless `--data` is given | Use the built-in synthetic PaySim-shaped data. |
| `--demo-rows N` | `60000` | Row count for `--demo`. |
| `--demo-fraud-rate F` | `0.00129` | Fraud prevalence for `--demo`, matching real PaySim's observed rate. |
| `--data PATH` | — | Path to a real PaySim-shaped CSV (see [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)). |
| `--sample-legit N` | all rows | With `--data`: keep every fraud row plus a stratified sample of `N` legitimate rows, instead of all ~6.36M — much faster to iterate on. |
| `--model-type {auto,xgboost,rf,gbdt}` | `auto` | Classifier architecture to train. |
| `--no-tune` | tuning **on** | Skip the stratified 5-fold grid-search hyperparameter search (faster, less rigorous). |
| `--model PATH` | — | Test *your own* fitted classifier instead of training one — see [Test YOUR own classifier](#test-your-own-classifier). |
| `--save-model PATH` | — | Save the trained baseline model to this path. |
| `--seed N` | `42` | Random seed, for reproducibility. |

### `run` — the adversary

| Flag | Default | What it does |
|---|---|---|
| `--backend {rules,llm}` | `rules` | How mutations are generated. |
| `--llm-model NAME` | `llama3` | Which Ollama model to use with `--backend llm`. |
| `--require-llm` | off | Abort instead of silently falling back to rules on any LLM failure. |
| `--strategies [...]` | all three | Restrict to specific mutation strategies. |
| `--max-seeds N` | `500` | Cap on how many caught frauds get attacked (bounds LLM wall-clock time). |
| `--validation {strict,lenient}` | `strict` | `strict` also requires balances to arithmetically reconcile; `lenient` only checks signs and sufficient funds. |
| `--min-value-retention F` | `0.90` | Reject a mutation group that abandons more than `1-F` of the original money — an "evasion" that gives most of the money away isn't one. |

### `run` — comparing and scaling up

| Flag | Default | What it does |
|---|---|---|
| `--compare` | off | Attack the *same* trained model + seed frauds with **both** backends and report a statistical (paired) comparison. |
| `--n-runs N` | `1` | Repeat the whole pipeline across `N` distinct seeds and report means, 95% confidence intervals, and significance tests. Use `10` for citable, reportable results. |

### `run` — output

| Flag | Default | What it does |
|---|---|---|
| `--out DIR` | `fraudprobe_out` | Where artefacts are written. Pass an empty value or omit results-affecting flags to skip disk writes when used as a library. |
| `--no-explain` | explanation **on** | Skip the SHAP evasion attribution (useful if SHAP isn't installed, or for speed). |
| `--figures` | off | Also write 300dpi PNG+PDF charts, CSV tables, and a full reproducibility manifest — see [What gets written to disk](#what-gets-written-to-disk). |

### `serve`

| Flag | Default |
|---|---|
| `--host` | `127.0.0.1` |
| `--port` | `5000` |

### Example commands

```bash
# Zero-setup demo:
fraudprobe run --demo

# Real run against PaySim, XGBoost, all report-ready artefacts:
fraudprobe run --data paysim.csv --model-type xgboost --figures

# Attack with a local LLM instead of deterministic rules:
fraudprobe run --demo --backend llm --llm-model llama3 --max-seeds 20

# Rules vs. LLM, head to head, on the same model and seed frauds:
fraudprobe run --demo --compare --max-seeds 30

# 10 repeated runs with confidence intervals, for a citable result:
fraudprobe run --data paysim.csv --model-type xgboost --n-runs 10 --figures
```

## Using the web dashboard

`fraudprobe serve` renders one page with two halves: a **controls panel** on the
left and a **results panel** on the right. It also has its own built-in "How this
works ▾" panel (top right) with the same pipeline/strategy explanation as this
README, plus a guide to reading the charts — worth opening on your first visit.

**Controls panel** (left):

| Field | What it controls |
|---|---|
| Baseline model | Which classifier architecture to train (`auto`, `xgboost`, `rf`, `gbdt`). |
| Adversary backend | `rules` (instant) or `llm` (local Ollama model). |
| LLM model | Which Ollama model to use, when the backend is `llm`. |
| Compare rules vs LLM side-by-side | Trains one model, attacks it with *both* backends, and renders the two reports side by side plus a head-to-head verdict. |
| Mutation strategies | Which of the three strategies to include (all checked by default). |
| Synthetic rows | Size of the demo dataset generated for this run. |
| Max seed frauds | Caps how many caught frauds get attacked — keep this low (10–30) for the `llm` backend, since each mutation is a real model call that takes seconds. |
| Explain evasion with SHAP | Toggle the SHAP feature-attribution panel. |
| **Run probe** button | Starts the run; disabled while one is in progress. |

**Results panel** (right), populated once a run finishes:

- **Activity** — a live log streamed from the run (data loading, training, per-strategy
  generation progress, LLM call latency if applicable).
- **Headline tiles** — Baseline F1, Clean detection, Adversarial detection, and
  Detection drop-off (the percentage-point gap — the number that matters most).
- **Weakness ranking** — a bar per strategy, worst (most-evaded) first.
- **SHAP evasion levers** — for each feature, a blue bar (its fraud-pushing signal on
  the original catch) next to an orange bar (the same feature after mutation). A
  feature whose orange bar collapses toward zero is a signal the attacker
  successfully neutralised.
- **Sample evading transactions** — concrete rows with the model's real fraud score,
  so "44% evasion" becomes an actual transaction the model was confident was safe.
- **Head-to-head strip** (compare mode only) — states which backend evaded more and
  by how much. With very few seed frauds this is noisy — use 20–30+ before trusting
  the verdict.

The dashboard always uses synthetic demo data and skips hyperparameter tuning (it's
built for fast interactive exploration, not citable numbers) — for a real-data,
tuned, statistically-repeated run, use the CLI's `--data`/`--n-runs`/`--figures`.

## What gets written to disk

Every CLI run with `--out` set (the default, `fraudprobe_out/`) writes:

| File | Contents |
|---|---|
| `adversarial_corpus.csv` | Every mutated transaction generated, tagged with its strategy and `source` (`rules`, `llm`, or `rules_fallback`). |
| `report.txt` | The same human-readable report printed to the terminal. |
| `results.json` | Everything, structured: baseline metrics, stress-test results, per-strategy evasion, value retention, LLM success/fallback rates, SHAP attribution, warnings. |

With `--n-runs N > 1`, you additionally get `results_aggregate.json` (means, 95% CIs,
significance tests, per-strategy Wilson intervals across all `N` runs) at the top
level, and each individual run's own three files above under `runs/seed_<n>/`.

With `--figures`, you additionally get:

| Path | Contents |
|---|---|
| `figures/*.png` + `*.pdf` | PR/ROC curves, confusion matrices, per-strategy evasion with confidence intervals, the threshold sweep, and (if SHAP is installed) the evasion-levers and SHAP summary charts — 300dpi PNG and vector PDF for every chart. |
| `results_tables/*.csv` | Every reportable table as its own CSV. |
| `MANIFEST.json` | Package/Python/OS/CPU/RAM, the Ollama model in use, git commit (+ whether the tree was dirty), seed(s), wall-clock timing, and the full CLI invocation — everything needed to say what exactly produced a given `results.json`. |

`--compare` writes `adversarial_corpus_<backend>.csv` per backend plus
`results.json` with both variants and the paired statistical comparison.

## Test YOUR own classifier

This is the point of the tool. Wrap any fitted model that exposes `predict_proba`
and fraudprobe will attack it:

```python
from fraudprobe import ScoredModel

# ... you have your own fitted fraud model `clf` and know which columns it expects ...
ScoredModel(estimator=clf, feature_columns=my_features).save("my_model.joblib")
```

```bash
fraudprobe run --data paysim.csv --model my_model.joblib
```

Out comes a resilience report telling you your model's blind spots.

## Using fraudprobe as a library

Every CLI/dashboard feature is a plain Python function underneath — see
[examples/quickstart.py](examples/quickstart.py) for a minimal end-to-end script:

```bash
python examples/quickstart.py
```

```python
from fraudprobe import generate_adversarial_corpus, make_demo_data, stress_test, train_baseline

df = make_demo_data(n_rows=40_000, seed=7)
model, clean_test, baseline = train_baseline(df, model_type="auto", seed=7)
clean_fraud = clean_test[clean_test["isFraud"] == 1]
caught = clean_fraud[model.score_rows(clean_fraud) >= 0.95].head(300)
corpus, adv_report = generate_adversarial_corpus(caught, seed=7)
stress = stress_test(model, clean_test, corpus)
```

## Methodological rigor

fraudprobe was hardened against a validity review (see `TASKS.md` and `Review.md`)
so every number in a report is measured, not assumed:

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
  `out/runs/seed_<n>/` so any one run can be inspected on its own, and a crashed
  overnight run only loses its in-progress seed (each seed's result is cached and
  reused on restart).
- **Rules vs. LLM at scale** — `--compare` attacks the *same* trained model and
  *identical* seed frauds with both backends and reports a paired comparison:
  mean evasion per backend, the difference with a 95% CI, and a significance test
  — isolating what the LLM specifically contributes over a heuristic. LLM calls
  run ~8-25s each depending on load; progress and an ETA are logged periodically,
  and total/mean/min/max call latency land in `results.json`.
- **Transaction velocity / destination behaviour** — `FEATURE_COLUMNS` includes
  `txn_velocity_orig` (backward-looking rolling count of same-`nameOrig`
  transactions within a 24-step window) and `dest_txn_count` (total transactions
  received by `nameDest`), computed strictly from whatever frame `engineer_features`
  is given — never joined against an external history table, so it stays a pure,
  reapply-to-mutated-rows-unchanged function (see its docstring for the full design
  rationale). Concrete consequence: scored against the adversarial corpus alone
  (not the full clean population), `amount_split`'s sibling rows genuinely earn a
  velocity count > 1 — a real structuring signal — while single-row strategies see
  1 regardless. On `--demo` synthetic data both features are degenerate (every row
  gets an independently random account ID, so the clean population never has
  velocity > 1 and the features carry no training signal) — flagged automatically
  as a fixture-dependency warning in every `--demo` run. Real PaySim data does have
  repeat account IDs; any evasion-lever ranking involving these two features must
  be re-measured there before citing it.
- **Report-ready artefacts** — `--figures` writes 300 dpi PNG + vector PDF to
  `out/figures/`, CSV tables to `out/results_tables/`, and `out/MANIFEST.json` — for
  single runs, `--compare`, and `--n-runs` alike, so the exact runs cited in a
  report all produce the same artefacts, generated rather than screenshotted.

## Research context

fraudprobe is the reference implementation for the MSc dissertation *"Evaluating the
Resilience of Open Banking Anti-Fraud Classifiers Against LLM-Generated Adversarial
Transaction Sequences"* (Edge Hill University, 2026). If you use it in academic work,
a citation is appreciated — see [CITATION.cff](CITATION.cff).

## License

MIT. Uses the PaySim synthetic dataset (Lopez-Rojas & Axelsson, 2016) — no real
financial data is ever processed.
