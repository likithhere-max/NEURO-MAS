"""
Monte-Carlo characterisation of the PRODUCTION longitudinal analysis (`app.ml.longitudinal.analyze_series`), Phase 2 C3.

SIMULATION, not clinical evidence. It measures what the rules do on known synthetic noise so that error rates are
measured rather than assumed. Real patient series need not resemble any scenario.

    cd backend && python -m ml.simulation.anomaly_study --reps 600 --out ml/simulation/results/anomaly_study.json

Null scenarios (mean 10, SD 2): iid Gaussian; AR(1) rho 0.3 / 0.6; iid Student-t(3) rescaled to SD 2.
Alternatives (N=40): step shift of 2 SD / 3 SD at reading 24; linear drift totalling 3 SD over the series.
Per method group: P(signal) at a single review (n = 12, 20, 40); family-wise P(signal at ANY review) when the series is
re-analysed after every new reading from 10 to 40 (repeated testing); and power. Variants of the autocorrelation gate
(`sequential_max_rho`) are compared. Everything is seeded.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.ml.longitudinal import DEFAULT_CONFIG, analyze_series  # noqa: E402

METHODS = {"t_prediction_interval": "point", "cusum": "cusum", "ewma": "ewma", "theil_sen_mann_kendall": "trend"}


def gen(kind: str, n: int, rng) -> np.ndarray:
    if kind == "iid_gaussian":
        z = rng.standard_normal(n)
    elif kind.startswith("ar1_"):
        rho = float(kind.split("_")[1])
        e = rng.standard_normal(n) * np.sqrt(1 - rho * rho)
        z = np.empty(n)
        z[0] = rng.standard_normal()
        for i in range(1, n):
            z[i] = rho * z[i - 1] + e[i]
    elif kind == "iid_t3":
        z = rng.standard_t(3, n) / np.sqrt(3.0)
    else:
        raise ValueError(kind)
    return 10.0 + 2.0 * z


def alternative(kind: str, n: int, rng) -> np.ndarray:
    base = gen("iid_gaussian", n, rng)
    if kind == "step_2sd":
        base[24:] += 4.0
    elif kind == "step_3sd":
        base[24:] += 6.0
    elif kind == "drift_3sd_total":
        base += np.linspace(0, 6.0, n)
    else:
        raise ValueError(kind)
    return base


def review(x: np.ndarray, cfg=DEFAULT_CONFIG) -> dict:
    """One analysis with the autocorrelation gate OFF; records what every gate cap would have done."""
    cfg = replace(cfg, sequential_max_rho=99.0)
    r = analyze_series("tremor_amplitude", np.arange(len(x), dtype=float), x, cfg)
    groups = {METHODS[s["method"]] for s in r.signals}
    out = {"point": "point" in groups, "trend": "trend" in groups}
    for m in ("cusum", "ewma"):
        e = r.statistics.get(m, {})
        out[m + "_alarm"] = e.get("first_alarm_reading_index") is not None
        out[m + "_rho"] = e.get("prefix_lag1_rho")
    return out


def apply_gate(rv: dict, cap: float) -> dict:
    row = {"point": rv["point"], "trend": rv["trend"]}
    for m in ("cusum", "ewma"):
        row[m] = bool(rv[m + "_alarm"] and rv[m + "_rho"] is not None and rv[m + "_rho"] <= cap)
    row["any"] = row["point"] or row["trend"] or row["cusum"] or row["ewma"]
    return row


def rates(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in ("any", "point", "cusum", "ewma", "trend")}


GATES = {"gate_0.2": 0.2, "gate_0.3": 0.3, "gate_off": 99.0}  # production default: 0.3


def _null_scenario(args):
    kind, reps, seed, N = args
    rng = np.random.default_rng(seed)
    series = [gen(kind, N, rng) for _ in range(reps)]
    single = {n: [review(s[:n]) for s in series] for n in (12, 20, 40)}
    fam = [[review(s[:n]) for n in range(10, N + 1)] for s in series]
    entry = {}
    for g, cap in GATES.items():
        entry[g] = {"single_review": {f"n={n}": rates([apply_gate(r, cap) for r in rows]) for n, rows in single.items()},
                    "family_wise_any_review_10_to_40": rates([
                        {k: any(apply_gate(r, cap)[k] for r in per) for k in ("any", "point", "cusum", "ewma", "trend")} for per in fam])}
    return kind, entry


def _alt_scenario(args):
    kind, reps, seed, N = args
    rng = np.random.default_rng(seed)
    rows = [review(alternative(kind, N, rng)) for _ in range(reps)]
    return kind, {g: rates([apply_gate(r, cap) for r in rows]) for g, cap in GATES.items()}


def study(reps: int, seed: int, N: int = 40) -> dict:
    from multiprocessing import Pool

    out = {"seed": seed, "reps": reps, "series_length": N, "python": platform.python_version(), "numpy": np.__version__,
           "default_config": DEFAULT_CONFIG.__dict__, "null": {}, "alternatives": {}}
    nulls = [(k, reps, seed + i, N) for i, k in enumerate(("iid_gaussian", "ar1_0.3", "ar1_0.6", "iid_t3"))]
    alts = [(k, reps, seed + 100 + i, N) for i, k in enumerate(("step_2sd", "step_3sd", "drift_3sd_total"))]
    with Pool() as pool:
        for k, e in pool.map(_null_scenario, nulls):
            out["null"][k] = e
        for k, e in pool.map(_alt_scenario, alts):
            out["alternatives"][k] = e
    out["rejected_fixed_baseline"] = {
        "note": "Exploratory run (not production code): CUSUM/EWMA with textbook limits (h=5, L=3) standardised by a FIXED baseline "
                "of b readings, family-wise false-alarm probability over 35 monitoring steps, iid Gaussian noise, 20000 reps.",
        "median_abs_deviation_scale": {"b=5": 0.65, "b=8": 0.54, "b=12": 0.45, "b=20": 0.34, "b=30": 0.27, "b=50": 0.20},
        "sample_sd_scale": {"b=5": 0.51, "b=8": 0.41, "b=12": 0.32, "b=20": 0.24, "b=30": 0.18, "b=50": 0.13},
        "conclusion": "estimation error in a short baseline makes textbook limits unusable (nominal ~7%); replaced by self-starting Q-statistic charts, whose "
                      "iid-null family-wise false-alarm rate over 40 readings was 6.9% (CUSUM) / 7.0% (EWMA) in a separate 20000-rep run."}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = study(a.reps, a.seed)
    text = json.dumps(r, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
