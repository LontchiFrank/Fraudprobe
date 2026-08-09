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
from .pipeline import ProbeConfig, run_probe


def cmd_run(args) -> int:
    cfg = ProbeConfig(
        data=args.data,
        demo=bool(args.demo or not args.data),
        demo_rows=args.demo_rows,
        sample_legit=args.sample_legit,
        model=args.model,
        model_type=args.model_type,
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
        log=lambda m: print(f"[fraudprobe] {m}"),
    )
    try:
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
    r.add_argument("--save-model", type=str, default=None)
    r.add_argument("--backend", choices=["rules", "llm"], default="rules",
                   help="'rules' = deterministic mutations; 'llm' = local Ollama model.")
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
