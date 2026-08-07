"""fraudprobe: adversarial resilience testing for fraud-detection classifiers.

Point it at a fraud model, and it reports how many AI-mutated frauds slip through
and which mutation strategy is the weak spot.
"""

__version__ = "0.1.0"

from .adversary import generate_adversarial_corpus  # noqa: E402
from .data import engineer_features, load_paysim, make_demo_data  # noqa: E402
from .explain import explain_evasion  # noqa: E402
from .models import ScoredModel, train_baseline  # noqa: E402
from .pipeline import ProbeConfig, run_probe  # noqa: E402
from .stress import stress_test  # noqa: E402

__all__ = [
    "generate_adversarial_corpus",
    "engineer_features",
    "load_paysim",
    "make_demo_data",
    "explain_evasion",
    "ScoredModel",
    "train_baseline",
    "ProbeConfig",
    "run_probe",
    "stress_test",
]
