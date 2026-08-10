"""fraudprobe command-line interface.

Examples
--------
# Zero-setup demo (synthetic data, no downloads, no GPU):
fraudprobe run --demo

# Real research run against PaySim:
fraudprobe run --data paysim.csv --model-type xgboost

# Bring your own classifier:
fraudprobe run --data paysim.csv --model my_model.joblib
"""

from __future__ import annotations

import argparse

from . import __version__
from .adversary import STRATEGIES, LLMRequiredError
from .pipeline import ProbeConfig, run_comparison, run_probe, run_repeated
from .stats import format_aggregate_report


def cmd_run(args) -> int:
    cfg = ProbeConfig(
        data=args.data,
        demo=bool(args.demo or not args.data),
        demo_rows=args.demo_rows,
        sample_legit=args.sample_legit,
        model=args.model,
        model_type=args.model_type,
        tune=not args.no_tune,
        save_model=args.save_model,
        backend=args.backend,
        llm_model=args.llm_model,
        require_llm=args.require_llm,
        validation=args.validation,
        min_value_retention=args.min_value_retention,
        strategies=tuple(args.strategies) if args.strategies else STRATEGIES,
        max_seeds=args.max_seeds,
        out=args.out,
        seed=args.seed,
        explain=not args.no_explain,
        figures=args.figures,
        log=lambda m: print(f"[fraudprobe] {m}"),
    )
    try:
        if args.compare:
            result = run_comparison(cfg)
            for v in result["variants"]:
                print("\n" + v["report_text"])
            pc = result.get("paired_backend_comparison")
            if pc:
                print("\n" + "=" * 62)
                print("  PAIRED COMPARISON (same seed frauds, same trained model)")
                print("=" * 62)
                print(f"  {pc['backend_a']} mean evasion : {pc['mean_evasion_a']*100:5.1f}%")
                print(f"  {pc['backend_b']} mean evasion : {pc['mean_evasion_b']*100:5.1f}%")
                print(f"  Difference ({pc['backend_b']} - {pc['backend_a']}): "
                      f"{pc['mean_difference_b_minus_a']*100:+5.1f} pts  "
                      f"(95% CI [{pc['difference_ci95']['low']*100:+.1f}, "
                      f"{pc['difference_ci95']['high']*100:+.1f}], n={pc['n_paired_groups']} paired groups)")
                sig = pc["paired_test"]
                if sig.get("ttest"):
                    print(f"  t-test p={sig['ttest']['p_value']:.4g}, "
                          f"Cohen's d_z={sig['cohens_d_z']:+.3f}")
                elif sig.get("note"):
                    print(f"  {sig['note']}")
                print("=" * 62)
            else:
                print("\n[fraudprobe] Not enough shared seed+strategy groups across backends "
                      "for a paired comparison.")
            if args.out:
                print(f"\n[fraudprobe] Comparison artefacts written to {args.out}/")
            return 0
        if args.n_runs > 1:
            result = run_repeated(cfg, n_runs=args.n_runs)
            print("\n" + format_aggregate_report(result) + "\n")
            if args.out:
                print(f"[fraudprobe] Aggregate written to {args.out}/results_aggregate.json")
                if args.figures:
                    print(f"[fraudprobe] Aggregate figures/tables/manifest written to {args.out}/ "
                          f"(figures/, results_tables/, MANIFEST.json); per-run artefacts under "
                          f"{args.out}/runs/seed_<n>/")
            return 0
        result = run_probe(cfg)
    except LLMRequiredError as exc:
        print(f"[fraudprobe] --require-llm violated: {exc}", flush=True)
        return 1
    except RuntimeError as exc:
        print(f"[fraudprobe] {exc}", flush=True)
        return 1
    print("\n" + result["report_text"] + "\n")
    exp = result.get("explanation")
    if exp and exp.get("available") and exp.get("evasion_levers"):
        print("  Top SHAP evasion levers (fraud-signal drop, clean -> adversarial):")
        for lever in exp["evasion_levers"][:5]:
            print(f"    {lever['feature']:24s} {lever['fraud_signal_drop']:+.4f}")
        print()
    print(f"[fraudprobe] Artefacts written to {args.out}/ "
          "(adversarial_corpus.csv, results.json, report.txt)")
    return 0


def cmd_serve(args) -> int:
    from .webapp import main as serve_main
    return serve_main(host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fraudprobe",
        description="Test whether a fraud detector can be fooled by AI-mutated transactions.",
    )
    p.add_argument("--version", action="version", version=f"fraudprobe {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the full baseline -> attack -> stress-test pipeline.")
    r.add_argument("--data", type=str, default=None, help="Path to PaySim-shaped CSV.")
    r.add_argument("--demo", action="store_true", help="Use built-in synthetic data.")
    r.add_argument("--demo-rows", type=int, default=60_000)
    r.add_argument("--sample-legit", type=int, default=None,
                   help="Real PaySim only (--data): keep every fraud row plus a stratified "
                        "random sample of this many legitimate rows, instead of all ~6.36M. "
                        "The resulting fraud prevalence is always reported as observed_fraud_rate.")
    r.add_argument("--model", type=str, default=None,
                   help="Path to YOUR fitted classifier (.joblib) to test.")
    r.add_argument("--model-type", choices=["auto", "xgboost", "rf", "gbdt"], default="auto")
    r.add_argument("--no-tune", action="store_true",
                   help="Skip the stratified 5-fold grid-search hyperparameter tuning (default: "
                        "tuning is ON) and use fixed hyperparameters for fast iteration.")
    r.add_argument("--save-model", type=str, default=None)
    r.add_argument("--backend", choices=["rules", "llm"], default="rules",
                   help="'rules' = deterministic mutations; 'llm' = local Ollama model. "
                        "Ignored if --compare is set (runs both).")
    r.add_argument("--compare", action="store_true",
                   help="Attack the SAME trained model + seed frauds with both rules and llm "
                        "backends and report a paired comparison (Task 10) — a statistical test "
                        "of the difference in evasion rate, paired by the shared seed set. "
                        "For a real comparison target --max-seeds 200+; each LLM call is "
                        "~8-12s, so budget wall-clock time accordingly (see --llm-model).")
    r.add_argument("--llm-model", type=str, default="llama3")
    r.add_argument("--require-llm", action="store_true",
                   help="Abort instead of falling back to rules if any LLM mutation fails "
                        "(guarantees a pure-LLM corpus for headline results).")
    r.add_argument("--validation", choices=["lenient", "strict"], default="strict",
                   help="'strict' (default) also requires balances to reconcile "
                        "(newbalanceOrig ~= oldbalanceOrg - amount, etc.); "
                        "'lenient' is fraudprobe's original signs-and-funds-only check.")
    r.add_argument("--min-value-retention", type=float, default=0.90,
                   help="Drop a seed+strategy's mutated rows if together they retain less than "
                        "this fraction of the original amount (default 0.90). An 'evasion' that "
                        "abandons most of the money isn't one.")
    r.add_argument("--strategies", nargs="*", choices=list(STRATEGIES), default=None)
    r.add_argument("--max-seeds", type=int, default=500)
    r.add_argument("--out", type=str, default="fraudprobe_out")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-explain", action="store_true",
                   help="Skip the SHAP evasion attribution.")
    r.add_argument("--figures", action="store_true",
                   help="Write 300dpi PNG + PDF figures to out/figures/ (PR/ROC curves, "
                        "confusion matrices, per-strategy evasion with CIs, threshold sweep, "
                        "SHAP summary and evasion levers), CSV tables to out/results_tables/, "
                        "and out/MANIFEST.json (versions, seeds, timings, git commit, CLI "
                        "invocation) — report-ready artefacts excluded from a dissertation's "
                        "word count, generated properly rather than screenshotted.")
    r.add_argument("--n-runs", type=int, default=1,
                   help="Repeat the full pipeline across this many distinct seeds "
                        "(seed, seed+1, ..., seed+n-1) and write results_aggregate.json "
                        "with means, 95%% CIs, a paired significance test of clean vs. "
                        "adversarial detection, and Wilson CIs per strategy. Default 1 "
                        "(single run); use 10 for reportable results. Individual runs' "
                        "artefacts are not written to disk, only the aggregate. Note this "
                        "multiplies wall-clock time by n_runs, including SHAP if enabled. "
                        "Ignored if --compare is set.")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="Launch the web dashboard.")
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
