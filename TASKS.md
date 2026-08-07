<!-- @format -->

# fraudprobe — remediation brief

Work through this file in order. It fixes validity problems found during an
academic review of the codebase against the project's stated methodology
(CIS4517 MSc dissertation, Edge Hill University).

## Context you need

`fraudprobe` measures whether a locally-hosted LLM can mutate fraudulent
transactions so they evade an ML fraud classifier. The dissertation claims:

- the adversary is an **LLM** (Llama-3 via Ollama), not a heuristic
- mutations **preserve economic value** and remain **economically plausible**
- the baseline is tuned via **stratified 5-fold CV with grid search**
- results generalise across **two classifier architectures**
- the data is the **real PaySim dataset** (6,362,620 rows, 0.129% fraud)

The current code does not support several of these claims. Each task below
closes one gap. **Do not fabricate or hard-code any result.** Where a claim
cannot be supported, the correct outcome is to make the code measure the truth
so the dissertation can be rewritten to match it.

## Ground rules

- Preserve the public API (`run_probe`, `run_comparison`, CLI flags, `/api/probe`).
- Everything new must land in `results.json` so it can be cited in the report.
- Keep it CPU-only, offline, no GPU dependency.
- Add or extend tests for each change.
- Keep the defensive framing in docstrings and README.
- Commit after each task with a message naming the task number.

---

## Task 1 — Record whether the LLM actually ran ★ CRITICAL

**Problem.** `adversary._mutate_llm()` catches `ImportError` and bare
`Exception` and silently returns `_RULES[strategy](row, rng)`. Nothing counts
these. `AdversaryReport.backend` records what was _requested_, not what
_happened_. The dissertation's central claim — that an LLM produced the
evasions — is therefore unverifiable from the project's own outputs.

**Do this.**

1. Change `_mutate_llm` to return `(frame, used_llm: bool, reason: str | None)`.
2. Catch specific exceptions separately and label them: `import_error`,
   `connection_error`, `timeout`, `json_parse_error`, `schema_error`,
   `other`. Never swallow silently.
3. Extend `AdversaryReport` with:
   - `n_llm_attempted: int`
   - `n_llm_success: int`
   - `n_llm_fallback: int`
   - `llm_success_rate: float`
   - `fallback_reasons: dict[str, int]`
4. Tag every generated row with a `source` column (`llm` or `rules_fallback`)
   and persist it in the corpus CSV.
5. Surface `llm_success_rate` and the reason breakdown in `results.json`,
   `format_report()` and the dashboard activity log.
6. Add `--require-llm` to the CLI. When set, a fallback raises rather than
   degrades — so a "pure LLM" run can be guaranteed for the headline result.

**Acceptance.** A run reports, e.g., `llm_success_rate: 0.87` with
`{"json_parse_error": 6, "connection_error": 2}`, and per-strategy evasion can
be recomputed for LLM-sourced rows only.

---

## Task 2 — Make the two backends perform the same intervention ★

**Problem.** `_mutate_amount_split` (rules) emits **3 rows** per seed — a real
structuring attack. The LLM path emits **1 row** for every strategy, so under
the LLM backend "amount_split" splits nothing. The backends implement
different interventions under the same label, which invalidates any
rules-vs-LLM comparison and makes the per-strategy evasion rates
non-comparable.

**Do this.**

1. Define a single mutation contract: a strategy maps one seed row to a
   `DataFrame` of _one or more_ rows.
2. For `amount_split` under the LLM backend, prompt the model for a JSON
   **array** of parts (or for split proportions, then construct the rows in
   Python from those proportions — more robust, and keeps balance arithmetic
   under your control).
3. Assert in tests that both backends return the same row count for the same
   strategy and seed.
4. If the LLM cannot produce a valid multi-row split, that is a
   `schema_error` fallback under Task 1 — count it, don't paper over it.

---

## Task 3 — Enforce arithmetic in the economic validator ★

**Problem.** `is_economically_valid()` checks signs and
`amount <= oldbalanceOrg`, but never that the balances reconcile. SHAP output
shows `error_balance_orig` absent from clean top features but present in
adversarial ones with negative mean SHAP — the signature of mutated rows whose
arithmetic does not add up, and whose non-reconciliation is pushing the
classifier away from "fraud". Some measured evasion may come from transactions
a real bank would reject at input validation, before any model runs.

**Do this.**

1. Add to the validator, within tolerance `tol`:
   - `newbalanceOrig ≈ oldbalanceOrg - amount`
   - `newbalanceDest ≈ oldbalanceDest + amount`
2. Add `--validation {lenient,strict}` (default `strict`), where `lenient` is
   the current behaviour.
3. Record the rejection count **per reason** in `AdversaryReport`
   (`negative_balance`, `insufficient_funds`, `orig_arithmetic`,
   `dest_arithmetic`, `non_positive_amount`).
4. Run the full pipeline under both settings and keep both result files. The
   difference between them is a reportable finding: how much of the measured
   evasion survives when impossible transactions are excluded.

---

## Task 4 — Measure value preservation properly

**Problem.** The current check is
`total_value(corpus) <= total_value(seed_frauds) * len(strategies) + 1.0`,
which only rules out the corpus moving more than 3× the seed total. A mutation
retaining 1% of the original value passes. The dissertation claims mutations
"preserve the total economic value of the fraudulent transfer"; this does not
establish that.

**Do this.**

1. For each seed row, compute
   `value_retention = mutated_total_amount / seed_amount`, grouped by strategy.
2. Report `mean`, `median`, `min` and the proportion within ±1% of 1.0, per
   strategy, in `results.json`.
3. `balance_camouflage` deliberately leaves a residual, so ~0.97 is the
   expected honest answer — report it, don't force it to 1.0.
4. Add `--min-value-retention` (default `0.90`) and reject mutations below it,
   counting rejections. An "evasion" that abandons most of the money is not an
   evasion.

---

## Task 5 — Give the demo data temporal structure, or disable the temporal strategy on it

**Problem.** `make_demo_data()` assigns `step` uniformly at random (1–745) for
fraudulent **and** legitimate rows, so `hour_of_day` carries no fraud signal.
`_mutate_temporal` shifts timing into a distribution identical to the one it
left. This fully explains `temporal_dispersion` scoring 21.1% against
`amount_split`'s 88%. That is a property of the fixture, not a finding about
fraud detection.

**Do this.**

1. Give demo frauds a realistic temporal concentration (e.g. weighted toward
   00:00–05:00) while legitimate traffic stays broadly diurnal. Document the
   assumption in the docstring.
2. Emit a warning in `results.json` and the activity log when
   `temporal_dispersion` is run against `data_source` starting `synthetic:`,
   flagging that the result is fixture-dependent.
3. Re-run and record whether temporal evasion changes once there is a signal
   to evade.

---

## Task 6 — Verify the real PaySim path end to end ★

**Problem.** Every saved result uses `data_source: synthetic:*`. The demo
generator defaults to `fraud_rate=0.013` (1.3%) — roughly **ten times** PaySim's
0.129%. The classification problem is therefore an order of magnitude easier
than the one the dissertation describes, and `make_demo_data`'s own docstring
warns against reporting from it.

**Do this.**

1. Confirm `load_paysim()` handles the full 6,362,620-row CSV without
   exhausting memory; chunk or downcast dtypes if needed.
2. Add `--sample-legit N` so the run keeps **all** fraud rows and a stratified
   random sample of legitimate rows, and record the resulting fraud prevalence
   in `results.json` as `observed_fraud_rate`.
3. Always write `observed_fraud_rate` and `n_rows` for every run, synthetic or
   real, so the two are never confused in the write-up.
4. Verify the real-data run completes for `--model-type xgboost` and `rf`.

---

## Task 7 — Implement the validation procedure the methodology describes

**Problem.** The dissertation specifies stratified 5-fold CV, grid search over
`max_depth` / `learning_rate` / `n_estimators` / `scale_pos_weight`, and SMOTE
applied independently inside each training fold. `models.py` does a single
stratified 80/20 split with hard-coded hyperparameters and one SMOTE call.

**Do this.**

1. Use `imblearn.pipeline.Pipeline` so SMOTE is fitted **inside** each CV fold,
   never on validation data.
2. Wrap in `GridSearchCV(cv=StratifiedKFold(n_splits=5, shuffle=True,
random_state=seed), scoring="f1")` over a documented search space.
3. Refit the best configuration on the full training partition; evaluate once
   on the untouched held-out test set.
4. Write the search space, the selected hyperparameters and the mean
   cross-validated F1 (with standard deviation) into `results.json`.
5. Add `--no-tune` to skip grid search for fast iteration, defaulting to
   tuning on for reportable runs.

---

## Task 8 — Sweep the decision threshold

**Problem.** `ScoredModel.threshold` is fixed at 0.5 throughout. Operational
fraud systems tune the threshold to a false-positive budget, so a reviewer will
reasonably ask whether the evasions survive a better-chosen operating point.

**Do this.**

1. Compute clean and adversarial detection rates across a threshold grid
   (e.g. 0.05 → 0.95 in steps of 0.05).
2. Also report at thresholds fixed to realistic false-positive rates on the
   clean test set (0.1%, 0.5%, 1%).
3. Store the full sweep in `results.json` as `threshold_sweep` so it can be
   plotted as drop-off versus threshold.

---

## Task 9 — Repeat across seeds and compute statistics ★

**Problem.** Every saved result is a single run at `seed: 42`. The methodology
promises paired significance tests and 95% confidence intervals; neither is
computable from one run.

**Do this.**

1. Add `--n-runs N` (default 1, use 10 for reportable runs) iterating over
   distinct seeds, re-running the full pipeline each time.
2. Aggregate per-metric mean, standard deviation and 95% CI.
3. Paired test of clean versus adversarial detection across runs: `scipy.stats.
ttest_rel`, plus `wilcoxon` as the non-parametric fallback, reporting the
   statistic, p-value and effect size.
4. Wilson confidence intervals on every per-strategy evasion rate — with ~20
   observations per strategy these are wide, and the report must say so.
5. Write it all to `results_aggregate.json`.

---

## Task 10 — Run the rules-vs-LLM comparison at scale ★

**Problem.** `run_comparison()` exists and works, but the saved corpora hold
3 LLM rows and 5 rules rows. This comparison is the study's core control:
without it, the finding is "mutations evade a classifier", which a heuristic
could achieve. With it, the finding isolates what the LLM contributes.

**Do this.**

1. Ensure `run_comparison` inherits everything above (Tasks 1–4, 7–9).
2. Make the seed set **identical** across backends so the comparison is paired.
3. Add a statistical test of the difference in evasion rate between backends,
   with a confidence interval on the difference.
4. Target ≥200 seed frauds. Given ~8–12s per LLM call, batch requests or run
   overnight; log wall-clock time per call and total.

---

## Task 11 — Export report-ready artefacts

The dissertation needs figures and tables that are excluded from the word
count, so generate them properly rather than screenshotting.

1. Add `--figures` writing 300 dpi PNG **and** vector PDF to `outdir/figures/`:
   - precision-recall and ROC curves, clean versus adversarial
   - confusion matrices before and after attack
   - per-strategy evasion bar chart with confidence intervals
   - SHAP summary (beeswarm and bar) for clean and adversarial sets
   - the evasion-levers chart already in the dashboard
   - drop-off versus threshold (from Task 8)
2. Write `results_tables/*.csv` for every table destined for the report.
3. Write `MANIFEST.json` per run: package versions, Python version, OS, CPU,
   RAM, Ollama and model version, seeds, wall-clock timings, git commit hash,
   full CLI invocation.

---

## Task 12 — Reconcile the feature set

**Problem.** The dissertation promises `transaction velocity` and
`destination account behavioural consistency` features. `FEATURE_COLUMNS`
contains neither.

**Do this.**

1. Implement `txn_velocity_orig` (count of transactions by the same
   `nameOrig` within a rolling window of `step`) and `dest_txn_count`
   (transactions received by `nameDest`).
2. Keep `engineer_features` row-independent for mutated rows, or document
   clearly how the mutated corpus is joined to history for these features —
   this is a real design constraint, so record the decision either way.
3. If either feature proves impractical, **remove it from the report's claims**
   and note the removal in `MANIFEST.json`. Do not leave the mismatch.

---

## Final deliverables

After the tasks are complete, produce a **reportable run set**:

| Run | Data   | Model         | Backend | Config                                           |
| --- | ------ | ------------- | ------- | ------------------------------------------------ |
| A   | PaySim | XGBoost       | llm     | `--require-llm --validation strict --n-runs 10`  |
| B   | PaySim | Random Forest | llm     | as above                                         |
| C   | PaySim | XGBoost       | rules   | as above (paired control)                        |
| D   | PaySim | Random Forest | rules   | as above (paired control)                        |
| E   | PaySim | XGBoost       | llm     | `--validation lenient` (for the Task 3 contrast) |

Then write `FINDINGS.md` summarising, in plain prose with no spin:

- baseline performance per architecture, with the selected hyperparameters
- LLM success rate and fallback reasons
- detection drop-off per architecture, with 95% CIs and p-values
- per-strategy evasion with Wilson intervals
- the rules-vs-LLM difference, with a CI on the difference
- value retention per strategy
- strict-versus-lenient validation contrast
- anything that contradicts the project's original hypothesis, stated plainly

That last point matters most. A negative or partial result, reported honestly,
is worth more than an inflated one — and the dissertation will be marked on the
quality of the reasoning, not on whether the attack succeeded.
