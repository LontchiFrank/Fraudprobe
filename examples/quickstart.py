"""Minimal end-to-end example using fraudprobe as a library.

Run:  python examples/quickstart.py
"""

from fraudprobe import (
    generate_adversarial_corpus,
    make_demo_data,
    stress_test,
    train_baseline,
)
from fraudprobe.stress import format_report


def main() -> None:
    # 1. Data (swap for load_paysim("paysim.csv") for real results)
    df = make_demo_data(n_rows=40_000, seed=7)

    # 2. Train the stand-in bank classifier
    model, clean_test, baseline = train_baseline(df, model_type="auto", seed=7)
    print(f"Baseline F1 = {baseline['f1']:.3f}")

    # 3. Seed the attack with caught frauds, then mutate them
    clean_fraud = clean_test[clean_test["isFraud"] == 1]
    caught = clean_fraud[model.score_rows(clean_fraud) >= 0.95].head(300)
    corpus, adv_report = generate_adversarial_corpus(caught, seed=7)

    # 4. Stress test and print the report
    stress = stress_test(model, clean_test, corpus)
    print(format_report(baseline, stress, adv_report))


if __name__ == "__main__":
    main()
