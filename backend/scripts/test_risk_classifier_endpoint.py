"""
End-to-end verification for the Phase 3 risk / worsening-trend classifier
(ml/training/train_risk_classifier.py, served by app/api/routes/ml_demo.py).
Mirrors scripts/test_ml_demo_endpoint.py: exercises the real HTTP API against
a running `uvicorn app.main:app --reload` server, and deliberately does NOT
touch the patients/workflows endpoints at all -- that absence is itself part
of what this script proves, per the two-track design in
claude/neuro-mas-architecture-v2.md.

WHY THIS SCRIPT DERIVES ITS INPUTS INSTEAD OF HARDCODING THEM
-------------------------------------------------------------
This model takes CHANGE (delta) values relative to a patient's own baseline
visit, not raw measurements -- so there is no such thing as "a real row" that
can simply be copy-pasted out of the dataset the way the disease classifier's
test rows can be. A valid input only exists relative to a specific patient's
first visit. Rather than fabricating plausible-looking deltas, this script
re-runs the training script's exact preprocessing against the same public
dataset and pulls out two genuinely real cases: the single most-worsened
recording and the single most-improved recording in the whole dataset.

WHAT THIS SCRIPT DOES AND DOES NOT PROVE
----------------------------------------
The saved model was fit on ALL rows (train_risk_classifier.py line ~173,
`final_pipeline.fit(X, y)`), which is the correct thing to do once validation
has passed -- but it means every row available here was seen during that final
fit. So this script does NOT and CANNOT measure out-of-sample accuracy. The
honest, leakage-free performance numbers come from the GroupKFold pooled
metrics recorded in the model's metadata (ROC-AUC ~0.64), not from anything
observed here.

What this script DOES prove:
  * the endpoints are reachable, authenticated, and return the documented shape
  * the served model reproduces sane in-sample behavior end-to-end through
    FastAPI, Pydantic aliasing, and the DataFrame column reordering in
    ml_demo.py -- i.e. the 18 delta features survive the HTTP boundary in the
    right order, which is the specific thing most likely to break silently
  * the model RANK-ORDERS a clearly-worsening case above a clearly-improving
    one, which is the only kind of claim a ROC-AUC-validated model supports
  * every response carries the demonstration disclaimer
  * auth is enforced and malformed payloads are rejected

Run with the server already running in another window, from backend/:
    python scripts\\test_risk_classifier_endpoint.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _testcfg  # live-script guard + env-provided password
import sys

import pandas as pd
import requests

BASE = "http://127.0.0.1:8000/api/v1"

# Same source and constants as ml/training/train_risk_classifier.py -- kept
# identical on purpose so this script tests the real contract, not a
# paraphrase of it.
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
TARGET = "total_UPDRS"
WORSENING_THRESHOLD = 3.0

failures = []


def check(label, resp, expect=200):
    ok = resp.status_code == expect
    print(f"{label}: {resp.status_code} (expected {expect}) {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  BODY:", resp.text[:500])
        failures.append(label)
    return resp


def expect_true(label, condition, detail=""):
    print(f"{label}: {'OK' if condition else 'FAIL'}{(' -- ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# 1. Login. Auth is required for these endpoints, but -- unlike every other
#    authenticated route in this project -- nothing about the prediction is
#    tied to this user or any patient record.
# ---------------------------------------------------------------------------
login = check("login", requests.post(f"{BASE}/auth/login", json={
    "email": "apitest@example.com",
    "password": _testcfg.test_password(),
}))
if login.status_code != 200:
    sys.exit("Cannot continue without a token. Is the server running, and does the test user still exist?")
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print()

# ---------------------------------------------------------------------------
# 2. Auth enforcement: the endpoint must reject an unauthenticated call.
# ---------------------------------------------------------------------------
anon = requests.get(f"{BASE}/demo/parkinsons-risk-classifier/info")
expect_true(
    "unauthenticated request is rejected",
    anon.status_code in (401, 403),
    f"got {anon.status_code}, expected 401/403",
)
print()

# ---------------------------------------------------------------------------
# 3. Info endpoint. The route returns the saved metadata verbatim, so this
#    doubles as a check that the metadata schema and the route agree.
# ---------------------------------------------------------------------------
info = check("GET risk model info", requests.get(
    f"{BASE}/demo/parkinsons-risk-classifier/info", headers=headers))
if info.status_code != 200:
    sys.exit("Info endpoint failed -- has ml/training/train_risk_classifier.py been run?")
meta = info.json()

feature_columns = meta["feature_columns_in_order"]
pooled = meta["pooled_metrics"]
gate = meta["validation_threshold"]

print("model:", meta["model"])
print("dataset:", meta["dataset"])
print("pooled_metrics:", pooled)
print("evaluation_strategy:", meta["evaluation_strategy"])
print("threshold_disclaimer:", meta["threshold_disclaimer"])
print()

expect_true("metadata exposes 18 ordered feature columns", len(feature_columns) == 18,
            f"got {len(feature_columns)}")
expect_true("pooled ROC-AUC still clears its own validation gate",
            pooled["roc_auc"] >= gate["min_roc_auc"],
            f"{pooled['roc_auc']:.4f} < {gate['min_roc_auc']}")
expect_true("pooled ROC-AUC is honestly above chance, not inflated",
            gate["chance_level"] < pooled["roc_auc"] < 0.90,
            f"{pooled['roc_auc']:.4f} looks implausible for this dataset -- suspect leakage")
expect_true("metadata keeps the non-clinical threshold disclaimer",
            "NOT a clinically validated" in meta["threshold_disclaimer"])
print()

# ---------------------------------------------------------------------------
# 4. Rebuild the two real cases, using the training script's exact
#    preprocessing. If the dataset can't be reached, say so plainly and stop
#    rather than quietly substituting invented numbers.
# ---------------------------------------------------------------------------
print("Fetching the UCI Telemonitoring dataset to rebuild real delta rows...")
try:
    df = pd.read_csv(URL)
except Exception as exc:  # noqa: BLE001 -- network/proxy failure, report and stop
    print(f"  could not fetch the dataset: {exc}")
    sys.exit(
        "Skipping the prediction checks. This script deliberately has no hardcoded "
        "fallback rows -- fabricated deltas would make a passing run meaningless."
    )

df.columns = [c.strip() for c in df.columns]
df = df.sort_values(["subject#", "test_time"]).reset_index(drop=True)

voice_feature_columns = [
    c for c in df.columns
    if c not in ("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS")
]

baseline = df.groupby("subject#", as_index=False).first().set_index("subject#")
df["baseline_total_UPDRS"] = df["subject#"].map(baseline[TARGET])
for col in voice_feature_columns:
    df[f"baseline_{col}"] = df["subject#"].map(baseline[col])

# Each subject's own first visit is the baseline it is compared against, so it
# carries no information and is dropped -- exactly as in training.
df["is_baseline_row"] = df.groupby("subject#")["test_time"].transform("min") == df["test_time"]
df = df[~df["is_baseline_row"]].reset_index(drop=True)

df["delta_target"] = df[TARGET] - df["baseline_total_UPDRS"]
for col in voice_feature_columns:
    df[f"delta_{col}"] = df[col] - df[f"baseline_{col}"]

expect_true("preprocessing reproduces the row count recorded at training time",
            len(df) == meta["n_recordings_after_dropping_baseline_visits"],
            f"got {len(df)}, metadata says {meta['n_recordings_after_dropping_baseline_visits']}")
expect_true("every feature the model expects was rebuilt",
            all(c in df.columns for c in feature_columns),
            f"missing: {[c for c in feature_columns if c not in df.columns]}")
print()

worsening_row = df.loc[df["delta_target"].idxmax()]
improving_row = df.loc[df["delta_target"].idxmin()]


def describe(name, row):
    print(f"{name}: subject#{int(row['subject#'])} at test_time {row['test_time']:.2f} -- "
          f"total_UPDRS {row['baseline_total_UPDRS']:.2f} -> {row[TARGET]:.2f} "
          f"({row['delta_target']:+.2f} points vs their own baseline)")


describe("most-worsened real recording", worsening_row)
describe("most-improved real recording", improving_row)
expect_true("the worsening case really does cross the demonstration threshold",
            worsening_row["delta_target"] >= WORSENING_THRESHOLD)
expect_true("the improving case really is below the demonstration threshold",
            improving_row["delta_target"] < WORSENING_THRESHOLD)
print()


def payload(row):
    return {c: float(row[c]) for c in feature_columns}


# ---------------------------------------------------------------------------
# 5. Predictions.
# ---------------------------------------------------------------------------
worse = check("POST risk classify (real most-worsened row)", requests.post(
    f"{BASE}/demo/parkinsons-risk-classifier", headers=headers, json=payload(worsening_row),
))
better = check("POST risk classify (real most-improved row)", requests.post(
    f"{BASE}/demo/parkinsons-risk-classifier", headers=headers, json=payload(improving_row),
))
if worse.status_code != 200 or better.status_code != 200:
    sys.exit("Prediction calls failed -- see bodies above.")

worse_result, better_result = worse.json(), better.json()
print("most-worsened row  ->", worse_result["predicted_label"],
      f"(P(worsening)={worse_result['probability_worsening']:.4f})")
print("most-improved row  ->", better_result["predicted_label"],
      f"(P(worsening)={better_result['probability_worsening']:.4f})")
print()

for name, result in (("worsened", worse_result), ("improved", better_result)):
    expect_true(f"{name}: label agrees with predicted_class",
                (result["predicted_class"] == 1) == (result["predicted_label"] == "worsening"))
    expect_true(f"{name}: probabilities form a valid distribution",
                abs(result["probability_worsening"] + result["probability_stable_or_improving"] - 1.0) < 0.01,
                f"{result['probability_worsening']} + {result['probability_stable_or_improving']}")
    expect_true(f"{name}: response carries the demonstration disclaimer",
                "DEMONSTRATION" in result["disclaimer"])
    expect_true(f"{name}: disclaimer admits the modest accuracy",
                "modest" in result["disclaimer"].lower())

# The claim this model actually supports. ROC-AUC measures rank-ordering, so
# rank-ordering is what gets asserted -- not that either individual prediction
# is correct. Both rows were part of the final fit, so a per-row correctness
# assertion would be an in-sample result dressed up as a test.
expect_true(
    "model ranks the most-worsened case above the most-improved case",
    worse_result["probability_worsening"] > better_result["probability_worsening"],
    f"P(worsening) was {worse_result['probability_worsening']:.4f} for a "
    f"{worsening_row['delta_target']:+.2f}-point change but "
    f"{better_result['probability_worsening']:.4f} for a "
    f"{improving_row['delta_target']:+.2f}-point change -- feature order across the "
    f"HTTP boundary is the first thing to suspect",
)
print()

# ---------------------------------------------------------------------------
# 6. Contract enforcement: a payload missing a required delta must be rejected
#    outright, never silently defaulted to zero (which would read as "no change
#    from baseline" and quietly produce a confident, meaningless answer).
# ---------------------------------------------------------------------------
incomplete = payload(worsening_row)
incomplete.pop(feature_columns[0])
check("POST with a missing feature is rejected", requests.post(
    f"{BASE}/demo/parkinsons-risk-classifier", headers=headers, json=incomplete,
), expect=422)
print()

# ---------------------------------------------------------------------------
if failures:
    print(f"=== {len(failures)} CHECK(S) FAILED ===")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("=== ALL CHECKS PASSED ===")
print()
print("Scope reminder: this verifies the HTTP contract and in-sample sanity only.")
print("The honest generalization estimate is the GroupKFold pooled ROC-AUC of")
print(f"{pooled['roc_auc']:.4f} recorded in the model metadata -- not anything measured here.")
print()
print("This script never called patients.py, workflows.py, repository.py, the")
print("orchestrator, or any agent -- the risk classifier stays fully decoupled")
print("from the live patient pipeline, as intended.")
