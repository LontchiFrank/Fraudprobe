<!-- @format -->

# Code review — post-remediation findings

Review of the repository after Tasks 1–11 of `TASKS.md` were implemented.
Tasks 1–11 were verified working by running the CLI end to end in several
configurations. The items below are the remaining gaps.

**Fix items 3–6 first** (they affect the runs destined for the dissertation),
then items 1–2, then 7–8.

---

## 1. Task 12 was never done

No `txn_velocity_orig`, no `dest_txn_count`; `FEATURE_COLUMNS` in `data.py` is
unchanged. The dissertation claims both features exist.

**Do:** implement them per Task 12 in `TASKS.md`, or remove them from the
project's claims and record the removal in `MANIFEST.json`. Do not leave the
mismatch between what the report says and what the code computes.

## 2. Task 11 is uncommitted

`git status` shows `fraudprobe/figures.py`, `fraudprobe/manifest.py`,
`tests/test_figures_task11.py` and `tests/test_manifest_task11.py` untracked,
with `cli.py` and `pipeline.py` modified but unstaged.

**Do:** commit as `Task 11: export report-ready artefacts`. Until then the work
is absent from history and from any GitHub publication.

---

## 3. `--figures` and `MANIFEST.json` do not run in `--n-runs` mode ★

Most important item here. Task 9 specifies `--n-runs 10` for reportable runs,
but that path writes only `results_aggregate.json` — no figures, no manifest,
no per-run artefacts. **The exact runs destined for the dissertation currently
produce no figures at all.**

Verified:

```
python -m fraudprobe.cli run --demo --demo-rows 8000 --model-type rf \
  --backend rules --max-seeds 10 --n-runs 3 --out /tmp/smoke2 --no-explain --figures
# -> /tmp/smoke2/results_aggregate.json          (only file written)

# same command without --n-runs writes MANIFEST.json, figures/ (4 PNG + 4 PDF)
# and results_tables/ (4 CSV) correctly
```

**Do:**

- Write `MANIFEST.json` for the aggregate run (mode `multi`, plus the seed list
  and total wall-clock across runs).
- Honour `--figures` in multi-run mode, generating aggregate figures: per-strategy
  evasion with the Wilson intervals already computed in `stats.py`, and drop-off
  with its 95% CI across runs.
- Persist each individual run's artefacts under `outdir/runs/seed_<n>/` so a
  single run can be inspected, and so the SHAP output isn't lost.
- Export the aggregate tables to `results_tables/` as CSV like the single-run
  path does.

## 4. The superseded value check is still printed

`generate_adversarial_corpus` still computes `value_ok` as
`total_value(corpus) <= total_value(seed_frauds) * len(strategies) + 1.0`, and
`format_report` prints `Economic value check: PASS` directly above the real
per-strategy value-retention table added in Task 4.

Two value checks appear in the same report, one meaningful and one close to
vacuous. A reader cannot tell which is which.

**Do:** remove `value_preserved` from `AdversaryReport` and drop the line from
`format_report`. `value_retention` supersedes it entirely.

## 5. `except ConnectionError` in `_mutate_llm` will never fire

The ollama client raises `httpx.ConnectError`, which is **not** a subclass of
Python's builtin `ConnectionError`. Genuine "Ollama isn't running" failures
therefore fall through to `except Exception` and are labelled `other` rather
than `connection_error`.

This matters because that is the single most likely failure mode in a long
overnight run, and the fallback taxonomy is going to be reported as a result.

**Do:** catch `httpx.ConnectError` (or `httpx.TransportError`) explicitly before
the generic handler, and add a test that asserts the reason label is
`connection_error` when the client raises it.

## 6. `MANIFEST.json` records a commit hash but no dirty flag

The manifest currently reports `git_commit: 951ad20…` while the working tree
contains the uncommitted Task 11 code — so the manifest points at a commit that
does not contain the code that actually ran. That defeats the purpose of
recording the hash.

**Do:** add `git_dirty: true|false` via `git status --porcelain`, and consider
recording the diff stat when dirty.

---

## 7. Demo fraud rate still defaults to 0.013

`make_demo_data(fraud_rate=0.013)` is roughly ten times PaySim's 0.129%.
Harmless now that `observed_fraud_rate` is recorded on every run, but the
default invites the same confusion that produced the original problem.

**Do:** add `--demo-fraud-rate`, defaulting to `0.00129`.

## 8. `fraudprobe_out/` and `fraudprobe_out_llm/` hold pre-remediation results

Confirmed stale: `fraudprobe_out/results.json` has no `validation_mode`, no
`llm_success_rate`, no `warnings`, no `threshold_sweep`. These predate every fix
and will be mistaken for current findings.

**Do:** delete both directories, add them to `.gitignore`, and regenerate only
from the reportable run set defined under "Final deliverables" in `TASKS.md`.

---

## Note on expected results

Smoke runs under strict validation showed `balance_camouflage` and
`temporal_dispersion` evasion collapsing to roughly zero, while `amount_split`
held at 85–94%. Samples were small and no LLM backend was available, so treat
this as a direction rather than a measurement — but do not treat a large drop
from the earlier figures as a regression. It is the expected consequence of
excluding arithmetically impossible transactions, and it is the correct result.
