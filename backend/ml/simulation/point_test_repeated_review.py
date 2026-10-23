"""
Repeated-testing study for the Student-t prediction-interval point test (Phase 2, C).

The point test is applied again at EVERY new reading. At a nominal single-test alpha of 0.05 the chance that an
iid-Gaussian (null) series raises the flag at least once over 31 successive reviews (n = 10..40) is far above 5 %.
This script measures it, which is why the SIGNAL level is alpha = 0.001 and alpha = 0.05 is an informational note only.

    cd backend && PYTHONPATH=. python -m ml.simulation.point_test_repeated_review --reps 1000
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from app.ml.personalized_anomaly import detect_personal_anomaly


def run(reps: int, seed: int, N: int = 40, first_review: int = 10, alphas=(0.05, 0.01, 0.001)) -> dict:
    rng = np.random.default_rng(seed)
    series = [rng.normal(size=N) for _ in range(reps)]
    out = {"seed": seed, "reps": reps, "series_length": N, "reviews": f"n={first_review}..{N} ({N - first_review + 1} reviews)",
           "noise": "iid N(0,1) (null: no real change)", "python": platform.python_version(), "numpy": np.__version__,
           "results": {}}
    for a in alphas:
        fw = single = 0
        for x in series:
            hit = False
            for n in range(first_review, N + 1):
                if detect_personal_anomaly(list(x[: n - 1]), float(x[n - 1]), alpha=a)["is_anomaly"]:
                    hit = True
            fw += hit
            single += bool(detect_personal_anomaly(list(x[:19]), float(x[19]), alpha=a)["is_anomaly"])
        out["results"][f"alpha={a}"] = {"single_review_n20_false_alarm": single / reps, "family_wise_false_alarm": fw / reps}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()
    res = run(a.reps, a.seed)
    p = Path(__file__).parent / "results" / "point_test_repeated_review.json"
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
