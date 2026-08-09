"""fraudprobe web dashboard.

A small Flask app that runs the probe pipeline on demand and renders the resilience
report — baseline vs adversarial detection, the weakness ranking, concrete evading
transactions, and the SHAP attribution of *why* the mutations slipped through.

Run it with::

    fraudprobe serve            # then open http://127.0.0.1:5000

It reuses [pipeline.run_probe](pipeline.py); the browser only ever sees the same
structured result the CLI produces.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import replace

from flask import Flask, jsonify, render_template, request

from .adversary import STRATEGIES
from .pipeline import ProbeConfig, run_comparison, run_probe

# Simple in-process job state (single-user local tool; no DB needed).
_STATE: dict = {"status": "idle", "log": [], "result": None, "error": None}
_LOCK = threading.Lock()


def _run_job(cfg: ProbeConfig, compare: bool) -> None:
    with _LOCK:
        _STATE.update(status="running", log=[], result=None, error=None)

    def log(msg: str) -> None:
        with _LOCK:
            _STATE["log"].append(str(msg))

    cfg = replace(cfg, log=log)
    try:
        result = run_comparison(cfg) if compare else run_probe(cfg)
        with _LOCK:
            _STATE.update(status="done", result=result)
    except Exception as exc:  # surface the failure to the UI
        with _LOCK:
            _STATE.update(status="error", error=f"{exc}\n{traceback.format_exc()}")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True  # pick up template edits without a restart

    @app.get("/")
    def index():
        return render_template("index.html", strategies=list(STRATEGIES))

    @app.post("/api/run")
    def api_run():
        with _LOCK:
            if _STATE["status"] == "running":
                return jsonify({"error": "A run is already in progress."}), 409

        body = request.get_json(force=True, silent=True) or {}
        strategies = body.get("strategies") or list(STRATEGIES)
        compare = bool(body.get("compare", False))
        cfg = ProbeConfig(
            demo=True,
            demo_rows=int(body.get("demo_rows", 60_000)),
            model_type=str(body.get("model_type", "auto")),
            backend=str(body.get("backend", "rules")),
            llm_model=str(body.get("llm_model", "llama3")),
            require_llm=bool(body.get("require_llm", False)),
            validation=str(body.get("validation", "strict")),
            min_value_retention=float(body.get("min_value_retention", 0.90)),
            strategies=tuple(strategies),
            max_seeds=int(body.get("max_seeds", 500)),
            explain=bool(body.get("explain", True)),
            out=body.get("out", "fraudprobe_out"),
            seed=int(body.get("seed", 42)),
        )
        threading.Thread(target=_run_job, args=(cfg, compare), daemon=True).start()
        return jsonify({"status": "started"})

    @app.get("/api/status")
    def api_status():
        with _LOCK:
            return jsonify(
                {
                    "status": _STATE["status"],
                    "log": _STATE["log"][-50:],
                    "result": _STATE["result"],
                    "error": _STATE["error"],
                }
            )

    return app


def main(host: str = "127.0.0.1", port: int = 5000) -> int:
    app = create_app()
    print(f"[fraudprobe] Dashboard running at http://{host}:{port}  (Ctrl-C to stop)")
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
