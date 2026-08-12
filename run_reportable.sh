#!/usr/bin/env bash
# Reportable run set for the fraudprobe dissertation (TASKS.md "Final deliverables").
#
# Usage (overnight, survives this shell/session ending):
#   nohup ./run_reportable.sh > run_reportable.log 2>&1 &
#   disown
#   tail -f run_reportable.log
#
# Prerequisites:
#   - Ollama running locally with llama3 pulled (`ollama serve`, `ollama pull llama3`).
#   - data/paysim.csv present: the REAL PaySim CSV, not the synthetic --demo
#     fixture. Verified this session via load_paysim: 6,362,620 rows,
#     observed_fraud_rate=0.129082% (8,213 fraud) — matches TASKS.md exactly.
#   - .venv312 with fraudprobe installed (this script calls it directly, no
#     activation needed).
#
# RESUMABLE. Safe to Ctrl-C or let it get killed and re-invoke identically:
#   - Each of the 6 named runs below is skipped entirely if its final output
#     file already exists (results_aggregate.json for A-E, results.json for F).
#   - Within a still-incomplete --n-runs run, run_repeated() itself now caches
#     each seed's results.json under out/runs/seed_<n>/ and reuses it on the
#     next invocation (see pipeline.py) — a crash during, say, run 4/5 does not
#     repeat runs 1-3.
#   - Run F (--compare) is NOT seed-level resumable (see the comment above it)
#     — a crash there re-does the whole comparison, but it's the smallest,
#     fastest piece of this script.
#
# ---------------------------------------------------------------------------
# SCOPE DEVIATION FROM TASKS.md's LITERAL TABLE — decided and measured this
# session, not silently assumed:
#
# 1. --require-llm is DROPPED from A, B, E. A real small-scale validation
#    against this exact PaySim file (--max-seeds 10 --n-runs 2 --require-llm)
#    failed outright TWICE ("nothing to aggregate") — llama3 hit schema_error
#    on 2/26 real calls (~7.7%). --require-llm aborts the entire run on the
#    FIRST fallback of any kind, and each full-scope run makes ~600+ calls, so
#    the odds of finishing with zero fallbacks are negligible. Since then:
#      - One same-prompt retry is attempted on schema_error specifically
#        (not other failure modes). n_llm_retried / n_llm_retry_success and
#        llm_single_shot_success_rate (before any retry) vs. llm_success_rate
#        (with retry) are both recorded in results.json — neither number is
#        lost to the other.
#      - results.json's stress.per_strategy_llm_only, and results_aggregate.
#        json's per_strategy_wilson_ci_llm_only, recompute per-strategy
#        evasion restricted to genuinely source=="llm" rows (excluding rules
#        and rules_fallback rows entirely) — the pure-LLM number --require-llm
#        was meant to guarantee by construction, obtained here by measurement
#        instead, without gambling an entire run on it.
#    FINDINGS.md should report llm_success_rate, llm_single_shot_success_rate,
#    fallback_reasons, and the per_strategy_llm_only breakdown for A/B/E
#    plainly — this IS the LLM reliability finding, not a gap to paper over.
#
# 2. --max-seeds and --n-runs are REDUCED from the literal 200/10 to 40/5.
#    Measured mean LLM call latency on this machine against the real Ollama
#    server: ~20-29s/call (varies with system load). At literal scope
#    (200 seeds x 3 strategies x 10 runs = 6,000 calls, plus ~8% retry
#    overhead), each of A/B/E alone is an estimated ~35-45 HOURS — roughly
#    4-6 days for all three run sequentially, not "overnight". At the reduced
#    40/5 scope (40 x 3 x 5 = 600 calls/run), each of A/B/E is an estimated
#    ~3.5-4.5 hours, so A-F sequentially should complete in one long overnight
#    session (rough estimate: ~11-14 hours total — see the per-run wall-clock
#    this script echoes to stdout/log for the real number on this machine).
#    Below Task 10's literal "target >=200 seed frauds" — report the actual n
#    used in FINDINGS.md and note the resulting Wilson CIs are wider than
#    literal-scope would give, as the explicit, stated tradeoff for finishing
#    in a reportable timeframe. The full-scope commands are included, commented
#    out, at the bottom of this file to run later if more time is available.
#
# 3. Run F (--compare) is an ADDITION beyond TASKS.md's A-E table. FINDINGS.md
#    is required to report "the rules-vs-LLM difference, with a CI on the
#    difference" — that specific paired statistic (same trained model,
#    identical seed frauds, both backends) only exists in this codebase via
#    run_comparison's --compare mode (Task 10's _paired_backend_comparison).
#    Separately aggregating A vs. C (or B vs. D) cannot produce it: those are
#    independent invocations with independent train/test splits per seed, not
#    a seed-level pairing. Given --compare is a single non-repeated run (not
#    n_runs x), it's run at max-seeds 200 — Task 10's literal target — since
#    that only costs ~4h more, not another multi-day commitment.
# ---------------------------------------------------------------------------

set -uo pipefail
cd "$(dirname "$0")"

DATA=data/paysim.csv
OUT=results/reportable
PY=.venv312/bin/python
MAX_SEEDS=40
N_RUNS=5
COMPARE_MAX_SEEDS=200

mkdir -p "$OUT"

if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found. This script needs the real PaySim CSV, not --demo." >&2
    exit 1
fi

run_one() {
    # $1 = label (e.g. "A_xgboost_llm_strict"), $2 = final marker file
    # (results_aggregate.json or results.json) relative to the run's --out,
    # remaining args = the fraudprobe CLI invocation.
    local label="$1" marker="$2"
    shift 2
    local out_dir="$OUT/$label"
    local marker_path="$out_dir/$marker"

    if [ -f "$marker_path" ]; then
        echo "[$label] SKIP — $marker_path already exists."
        return 0
    fi

    echo "[$label] START $(date '+%Y-%m-%d %H:%M:%S')"
    local t0 t1
    t0=$(date +%s)
    "$PY" -m fraudprobe.cli "$@" --out "$out_dir" > "$OUT/$label.log" 2>&1
    local status=$?
    t1=$(date +%s)
    local elapsed=$((t1 - t0))

    if [ $status -eq 0 ] && [ -f "$marker_path" ]; then
        echo "[$label] DONE  $(date '+%Y-%m-%d %H:%M:%S')  (${elapsed}s = $((elapsed/3600))h$((elapsed%3600/60))m) -> $out_dir/"
    else
        echo "[$label] FAILED (exit $status) after ${elapsed}s — see $OUT/$label.log" >&2
    fi
    return $status
}

echo "=== fraudprobe reportable run set — started $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Scope: --max-seeds $MAX_SEEDS --n-runs $N_RUNS for A-E (reduced from TASKS.md's literal"
echo "200/10 to fit an overnight window — see the comment block at the top of this file);"
echo "Run F (--compare) at --max-seeds $COMPARE_MAX_SEEDS (Task 10's literal target; a single"
echo "non-repeated run, so full scope there is cheap)."
echo

# --- Run C, D FIRST (rules backend, fast, no Ollama dependency) -------------
# Rationale: if the LLM runs (A/B/E) fail or the machine dies overnight, the
# rules-backend paired control and baseline numbers are still on disk by
# morning rather than nothing at all.
run_one "C_xgboost_rules_strict" "results_aggregate.json" \
    run --data "$DATA" --model-type xgboost --backend rules \
    --validation strict --max-seeds $MAX_SEEDS --n-runs $N_RUNS --figures

run_one "D_rf_rules_strict" "results_aggregate.json" \
    run --data "$DATA" --model-type rf --backend rules \
    --validation strict --max-seeds $MAX_SEEDS --n-runs $N_RUNS --figures

# --- Run A: PaySim, XGBoost, llm --------------------------------------------
run_one "A_xgboost_llm_strict" "results_aggregate.json" \
    run --data "$DATA" --model-type xgboost --backend llm \
    --validation strict --max-seeds $MAX_SEEDS --n-runs $N_RUNS --figures

# --- Run B: PaySim, Random Forest, llm --------------------------------------
run_one "B_rf_llm_strict" "results_aggregate.json" \
    run --data "$DATA" --model-type rf --backend llm \
    --validation strict --max-seeds $MAX_SEEDS --n-runs $N_RUNS --figures

# --- Run E: PaySim, XGBoost, llm, lenient validation (Task 3 contrast) ------
run_one "E_xgboost_llm_lenient" "results_aggregate.json" \
    run --data "$DATA" --model-type xgboost --backend llm \
    --validation lenient --max-seeds $MAX_SEEDS --n-runs $N_RUNS --figures

# --- Run F: paired rules-vs-LLM comparison (beyond TASKS.md's table — see
#     item 3 in the comment block above; needed for FINDINGS.md's "rules-vs-
#     LLM difference, with a CI on the difference"). NOT seed-level resumable
#     — a crash here re-does the whole comparison, not just the missing part.
run_one "F_xgboost_compare_paired" "results.json" \
    run --data "$DATA" --model-type xgboost --compare \
    --validation strict --max-seeds $COMPARE_MAX_SEEDS --figures

echo
echo "=== fraudprobe reportable run set — finished $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Results under $OUT/. Check $OUT/*.log for any FAILED run above."

exit 0

# ---------------------------------------------------------------------------
# FULL-SCOPE COMMANDS (TASKS.md's literal --max-seeds 200 --n-runs 10),
# commented out — run these individually later if more time is available.
# Estimated wall-clock at this machine's measured ~20-29s/LLM call, ~8% retry
# overhead: ~35-45 hours EACH for A, B, E; minutes for C, D; ~4h for F
# (already run at this scope above, no need to repeat it).
#
# .venv312/bin/python -m fraudprobe.cli run --data data/paysim.csv \
#     --model-type xgboost --backend rules --validation strict \
#     --max-seeds 200 --n-runs 10 --figures --out results/reportable_full/C_xgboost_rules_strict
#
# .venv312/bin/python -m fraudprobe.cli run --data data/paysim.csv \
#     --model-type rf --backend rules --validation strict \
#     --max-seeds 200 --n-runs 10 --figures --out results/reportable_full/D_rf_rules_strict
#
# .venv312/bin/python -m fraudprobe.cli run --data data/paysim.csv \
#     --model-type xgboost --backend llm --validation strict \
#     --max-seeds 200 --n-runs 10 --figures --out results/reportable_full/A_xgboost_llm_strict
#
# .venv312/bin/python -m fraudprobe.cli run --data data/paysim.csv \
#     --model-type rf --backend llm --validation strict \
#     --max-seeds 200 --n-runs 10 --figures --out results/reportable_full/B_rf_llm_strict
#
# .venv312/bin/python -m fraudprobe.cli run --data data/paysim.csv \
#     --model-type xgboost --backend llm --validation lenient \
#     --max-seeds 200 --n-runs 10 --figures --out results/reportable_full/E_xgboost_llm_lenient
# ---------------------------------------------------------------------------
