"""Report-ready tabular exports and the run manifest (Task 11).

A dissertation needs figures and tables that are excluded from the word count,
generated properly rather than screenshotted — and a manifest that pins down
exactly what produced them, since "what package versions/seed/commit made this
number" is a question every reviewer eventually asks.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def _git_diff_stat() -> str | None:
    """Only called when the tree is dirty — records what differs from the
    recorded commit hash, since a commit hash alone doesn't tell a reviewer
    that (REVIEW.md item 6)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--stat", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _total_ram_bytes() -> int | None:
    """Best-effort total system RAM without adding a new dependency."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) if out.returncode == 0 else None
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    return None


def _cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _ollama_model_info(model_name: str) -> dict | None:
    try:
        import ollama
        resp = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name and name.split(":")[0] == model_name.split(":")[0]:
                digest = m.get("digest") if isinstance(m, dict) else getattr(m, "digest", None)
                size = m.get("size") if isinstance(m, dict) else getattr(m, "size", None)
                return {"name": name, "digest": digest, "size_bytes": size}
    except Exception:
        return None
    return None


def write_manifest(
    result: dict,
    outdir: Path,
    wall_clock_seconds: float,
    cli_argv: list[str] | None = None,
) -> dict:
    """Everything needed to say what produced a given results.json: package
    versions, interpreter/OS/CPU/RAM, the Ollama model actually used (if any),
    seeds, timings, the git commit, and the full CLI invocation.
    """
    backend = result.get("backend") or "compare"
    llm_model = result.get("llm_model")

    git_dirty = _git_dirty()

    manifest = {
        "fraudprobe_version": result.get("version"),
        "git_commit": _git_commit(),
        "git_dirty": git_dirty,
        "git_diff_stat": _git_diff_stat() if git_dirty else None,
        "cli_invocation": cli_argv if cli_argv is not None else sys.argv,
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {"brand": _cpu_brand(), "logical_cores": os.cpu_count()},
        "ram_bytes": _total_ram_bytes(),
        "package_versions": {
            pkg: _pkg_version(pkg)
            for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost",
                        "imbalanced-learn", "shap", "ollama", "flask", "joblib"]
        },
        "ollama_model": _ollama_model_info(llm_model) if (backend == "llm" or llm_model) else None,
        "seed": result.get("seed") or result.get("base_seed"),
        "seeds": result.get("seeds"),
        "data_source": result.get("data_source"),
        "n_rows": result.get("n_rows"),
        "observed_fraud_rate": result.get("observed_fraud_rate"),
        "wall_clock_seconds": wall_clock_seconds,
        "mode": result.get("mode"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def write_results_tables(result: dict, outdir: Path) -> list[str]:
    """CSV export of every table a report would cite as a table, not prose."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def _write(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            return
        df.to_csv(outdir / f"{name}.csv", index=False)
        written.append(name)

    variants = result.get("variants") if result.get("mode") == "compare" else [result]

    for v in variants:
        suffix = f"_{v['backend']}" if result.get("mode") == "compare" else ""
        stress = v.get("stress", {})

        per_strategy = stress.get("per_strategy", {})
        if per_strategy:
            _write(
                pd.DataFrame([{"strategy": k, **vals} for k, vals in per_strategy.items()]),
                f"per_strategy_evasion{suffix}",
            )

        # Restricted to source == "llm" rows — the pure-LLM result --require-llm
        # was meant to guarantee, reported alongside the all-rows table rather
        # than only measurable by aborting the run on any rules fallback.
        per_strategy_llm_only = stress.get("per_strategy_llm_only", {})
        if per_strategy_llm_only:
            _write(
                pd.DataFrame([{"strategy": k, **vals} for k, vals in per_strategy_llm_only.items()]),
                f"per_strategy_evasion_llm_only{suffix}",
            )

        sweep = stress.get("threshold_sweep", {})
        if sweep.get("grid"):
            _write(pd.DataFrame(sweep["grid"]), f"threshold_sweep{suffix}")
        if sweep.get("fixed_fpr"):
            _write(pd.DataFrame(sweep["fixed_fpr"]), f"threshold_fixed_fpr{suffix}")

        adv = v.get("adversary", {})
        if adv.get("value_retention"):
            _write(
                pd.DataFrame([{"strategy": k, **vals} for k, vals in adv["value_retention"].items()]),
                f"value_retention{suffix}",
            )
        if adv.get("rejection_reasons"):
            _write(pd.DataFrame(list(adv["rejection_reasons"].items()), columns=["reason", "count"]),
                   f"rejection_reasons{suffix}")
        if adv.get("fallback_reasons"):
            _write(pd.DataFrame(list(adv["fallback_reasons"].items()), columns=["reason", "count"]),
                   f"llm_fallback_reasons{suffix}")

        exp = v.get("explanation") or {}
        if exp.get("evasion_levers"):
            _write(pd.DataFrame(exp["evasion_levers"]), f"shap_evasion_levers{suffix}")

    if result.get("mode") == "repeated":
        for key, label in [
            ("aggregate_metrics", "aggregate_metrics"),
            ("per_strategy_wilson_ci", "per_strategy_wilson_ci"),
            ("per_strategy_wilson_ci_llm_only", "per_strategy_wilson_ci_llm_only"),
        ]:
            data = result.get(key, {})
            if data:
                _write(pd.DataFrame([{"metric": k, **v} for k, v in data.items()]), label)
        if result.get("per_run_summary"):
            _write(pd.DataFrame(result["per_run_summary"]), "per_run_summary")

    if result.get("paired_backend_comparison"):
        pc = result["paired_backend_comparison"]
        flat = {k: v for k, v in pc.items() if not isinstance(v, dict)}
        for k, v in pc.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
        _write(pd.DataFrame([flat]), "paired_backend_comparison")

    return written
