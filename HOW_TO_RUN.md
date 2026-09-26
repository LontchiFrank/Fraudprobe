<!-- @format -->

How to run fraudprobe
What this is

fraudprobe measures whether a machine learning fraud detection classifier can be evaded by adversarially mutated transactions, and whether a locally hosted large language model generates better evasions than deterministic code. It trains a baseline classifier, generates mutated fraudulent transactions, and re-scores them against the unmodified classifier to measure the drop in detection.

Requirements

Python 3.11 or later. Install the dependencies (they are not bundled, per the submission instructions):

bash
python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -e .

This installs pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, SciPy, SHAP, matplotlib and Flask. No GPU is required and no network access is needed after installation.

Step 1 — verify the installation (about 30 seconds)
bash
pytest -q

Expect 172 passing tests.

Step 2 — run a complete experiment (about 1 minute, no external dependencies)
bash
fraudprobe run --demo --backend rules --max-seeds 10 --out out_demo

This runs the full three-phase pipeline end to end on a built-in synthetic dataset, using the deterministic adversary. It requires no downloaded data and no language model, so it runs anywhere. Results are written to out_demo/: report.txt is human-readable, results.json machine-readable.

Note: the built-in synthetic data is a small fixture for demonstration. Its numbers are not the results reported in the dissertation, and the tool records data_source: synthetic:... in every output so the two cannot be confused.

Step 3 — view the interactive dashboard (optional)
bash
fraudprobe serve --port 5059

Then open http://127.0.0.1:5059. Set the adversary backend to rules unless a local language model is available (see Step 5).

Step 4 — reproduce the dissertation results on real data

The PaySim dataset is excluded from this submission as a public dataset. To reproduce the reported results, download it from https://www.kaggle.com/datasets/ealaxi/paysim1 (free account required), and place PS_20174392719_1491204439457_log.csv at data/paysim.csv. It should contain 6,362,620 rows with 8,213 fraudulent records.

bash
fraudprobe run --data data/paysim.csv --model-type xgboost \
 --backend rules --validation strict \
 --max-seeds 40 --n-runs 5 --figures --out out_paysim

This reproduces condition C from Chapter 5 and takes roughly 30 minutes on a four-core machine.

Step 5 — the language model adversary (optional)

The adversarial backend used for conditions A, B and E requires Ollama running locally with the llama3 model. This is a 4.7 GB download and is not included.

bash
ollama serve &
ollama pull llama3
fraudprobe run --data data/paysim.csv --backend llm --max-seeds 10 --out out_llm

Generation takes several seconds per call on CPU, so a full condition runs for several hours. If Ollama is unavailable the tool does not fail: it falls back to the deterministic adversary and records the fallback reason in results.json under fallback_reasons, so the provenance of every generated row remains visible.

Step 6 — the submitted results

results/reportable/ contains the output of the six experimental conditions reported in Chapter 5. Each condition directory holds results_aggregate.json (means, confidence intervals and significance tests across five runs), results.json, MANIFEST.json (package versions, hardware, seeds, timings and the git commit that ran), results_tables/_.csv and figures/_.png. Every figure in Chapter 5 can be traced to one of these files.

Command reference
Flag Purpose
--demo Use the built-in synthetic fixture instead of a data file
--data PATH Path to the PaySim CSV
--model-type xgboost, rf or gbdt
--backend rules (deterministic) or llm (Ollama)
--validation strict (enforces balance arithmetic) or lenient
--max-seeds N Number of seed frauds to mutate
--n-runs N Independent repetitions for confidence intervals
--figures Export PNG and PDF figures
--no-tune Skip hyperparameter search for a faster run
Troubleshooting
command not found: fraudprobe — the virtual environment is not active, or pip install -e . was not run. Alternatively use python3 -m fraudprobe.cli.
MemoryError on PaySim — add --sample-legit 500000 to work with a subset. Note this raises the apparent fraud prevalence above the natural 0.129% and the reported results do not use it.
XGBoost unavailable — use --model-type rf, which relies only on scikit-learn. 5. Check it runs on a clean machine before submitting

The instruction to ensure it runs on departmental machines is the one most likely to catch you out, because your own machine has Ollama, the PaySim file and an existing virtual environment. Test on a copy that has none of these:

bash
cd /tmp && rm -rf artefact_test
unzip ~/Desktop/FodjoFrank_26708825_artefact.zip -d artefact_test
cd artefact_test
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
fraudprobe run --demo --backend rules --max-seeds 10 --out /tmp/check
cat /tmp/check/report.txt

If all four commands succeed with no Ollama running and no data/ directory present, the artefact will run for the marker.

6. Packaging command
   bash
   cd /path/to/parent-of-fraudprobe
   zip -r FodjoFrank_26708825_artefact.zip fraudprobe \
    -x "_/.venv312/_" "_/data/_" "_/**pycache**/_" "_/.pytest_cache/_" \
    "_/fraudprobe.egg-info/_" "_/.claude/_" "_/.git/_" "_.DS_Store" "_.pyc"

Then confirm what actually went in:

bash
unzip -l FodjoFrank_26708825_artefact.zip | tail -5
unzip -l FodjoFrank_26708825_artefact.zip | grep -cE "venv|paysim|pycache" # expect 0 7. Two points to confirm with the marker
Size limit. Even trimmed, the artefact will be several megabytes. If the submission system caps uploads below that, ask whether results/reportable/ should be reduced to the six results_aggregate.json files alone, or supplied via the repository link instead.
Repository. If the artefact is also published to GitHub, give the URL in both README.md and the report, and make sure results/reportable/ is not excluded by .gitignore — Objective O6 claims a published, reproducible artefact, and an empty repository would not evidence it.
