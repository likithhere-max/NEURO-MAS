"""Phase 3: large-sample check that the point prediction-interval threshold attains its nominal 5% false-positive rate
for iid Gaussian data (vectorised; 2,000,000 trials per n).  Synthetic data only.

    PYTHONPATH=. python -m ml.simulation.p3_pi_large_check [--trials 2000000]
"""
import argparse
import json

import numpy as np
from scipy.stats import t

from app.ml.personalized_anomaly import effective_z_threshold


def run(trials: int = 2_000_000, seed: int = 777) -> dict:
    rng = np.random.default_rng(seed)
    out = {"trials_per_n": trials, "seed": seed, "synthetic": True, "rows": []}
    for n in (3, 5, 15, 30, 31, 40, 60):
        X = rng.standard_normal((trials, n + 1))
        h, x = X[:, :n], X[:, n]
        z = (x - h.mean(1)) / h.std(1, ddof=1)
        thr = effective_z_threshold(n)
        exact = float(t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n))
        rate = float((np.abs(z) >= thr).mean())
        se = float(np.sqrt(0.05 * 0.95 / trials))
        out["rows"].append({"n_history": n, "implemented_threshold": round(thr, 4), "exact_threshold": round(exact, 4),
                            "false_positive_rate": round(rate, 4), "z_vs_0.05": round((rate - 0.05) / se, 2)})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2_000_000)
    print(json.dumps(run(ap.parse_args().trials), indent=1))
