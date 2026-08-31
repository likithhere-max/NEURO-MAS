"""
Roadmap item 6, Track 2, second deliverable -- RISK / SEVERITY-TREND
prediction (REVISED after the first attempt). Trains and validates a
demonstration regression model on the public UCI Parkinson's Telemonitoring
dataset. Fully decoupled from the live patient pipeline.

WHY THIS IS REVISED FROM THE FIRST ATTEMPT: predicting a patient's ABSOLUTE
total_UPDRS score from voice alone, evaluated on entirely unseen patients via
GroupKFold, gave a NEGATIVE pooled R^2 for every model tried (RandomForest,
HistGradientBoosting, Ridge). That is a real, honest result, not a bug -- the
dominant source of variance in absolute UPDRS across this dataset is
BETWEEN-patient differences (anatomy, comorbidities, individual baseline
severity) that voice biomarkers alone cannot explain for a patient the model
has never seen. Published results near R^2=0.9 on this dataset almost always
use a row-level split, which leaks each patient's own baseline into
training -- exactly the leakage problem this whole revision effort has been
about avoiding.

THE FIX: reframe the target from absolute severity to CHANGE FROM THAT
PATIENT'S OWN BASELINE VISIT (their first recorded session). For every later
visit, this script computes:
    delta_target  = total_UPDRS(visit)  - total_UPDRS(baseline visit)
    delta_features = voice_features(visit) - voice_features(baseline visit)
and predicts delta_target from delta_features (+ elapsed time since
baseline). This cancels out each patient's fixed baseline level -- the
between-patient variance that sank the first attempt -- and asks a more
physiologically plausible question: does a worsening in voice biomarkers
track a worsening in symptom severity for THIS patient over time. This is
also arguably the more clinically useful "risk" framing: is this patient
trending worse, and by how much.

Every patient still contributes only to either train or test in each
GroupKFold fold (grouped by subject#) -- zero patient leakage, same as
before. A naive "assume no change since baseline" baseline is also reported
for honest comparison -- beating that baseline is the real bar to clear, not
an arbitrary R^2 number.

Dataset: Tsanas, A. and Little, M. (2009). Parkinsons Telemonitoring
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZS3N
Raw data: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "total_UPDRS"
MIN_R2_OVER_BASELINE = 0.10  # must beat the naive "no change" baseline by this much R^2
N_FOLDS = 10

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

print("Loading data...")
df = pd.read_csv(URL)
df.columns = [c.strip() for c in df.columns]
df = df.sort_values(["subject#", "test_time"]).reset_index(drop=True)

voice_feature_columns = [
    c for c in df.columns
    if c not in ("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS")
]

# Baseline = each subject's first recorded visit (earliest test_time).
baseline = df.groupby("subject#", as_index=False).first()
baseline = baseline.set_index("subject#")

df["baseline_total_UPDRS"] = df["subject#"].map(baseline[TARGET])
for col in voice_feature_columns:
    df[f"baseline_{col}"] = df["subject#"].map(baseline[col])

# Drop each subject's own baseline row (delta would trivially be 0 for it).
df["is_baseline_row"] = df.groupby("subject#")["test_time"].transform("min") == df["test_time"]
df = df[~df["is_baseline_row"]].reset_index(drop=True)

df["delta_target"] = df[TARGET] - df["baseline_total_UPDRS"]
delta_feature_columns = []
for col in voice_feature_columns:
    delta_col = f"delta_{col}"
    df[delta_col] = df[col] - df[f"baseline_{col}"]
    delta_feature_columns.append(delta_col)

# Features: the deltas themselves, plus elapsed time and baseline severity
# (a patient's starting severity is a legitimate, non-leaky predictor of how
# much their score is likely to move -- it's known at prediction time).
feature_columns = delta_feature_columns + ["test_time", "baseline_total_UPDRS"]
X = df[feature_columns].reset_index(drop=True)
y = df["delta_target"].reset_index(drop=True)
groups = df["subject#"].reset_index(drop=True)
n_subjects = groups.nunique()

print(f"After dropping each subject's own baseline visit: {len(df)} rows, "
      f"{len(feature_columns)} features, {n_subjects} unique subjects.")
print(f"delta_target (change in total_UPDRS from baseline): "
      f"range {y.min():.1f} to {y.max():.1f}, mean {y.mean():.2f}, std {y.std():.2f}.")
print()

CANDIDATES = {
    "RandomForestRegressor": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
    ]),
    "Ridge": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ]),
}

gkf = GroupKFold(n_splits=N_FOLDS)
results = {}

# Naive baseline: "assume no change since baseline visit" -> predicted delta = 0.
naive_pred = np.zeros(len(y), dtype=float)
naive_metrics = {
    "r2": r2_score(y, naive_pred),
    "mae": mean_absolute_error(y, naive_pred),
    "rmse": float(np.sqrt(mean_squared_error(y, naive_pred))),
}
print("--- Naive baseline: predict delta_target=0 (assume no change since baseline visit) ---")
print(f"  r2={naive_metrics['r2']:.4f}  mae={naive_metrics['mae']:.4f}  rmse={naive_metrics['rmse']:.4f}")
print()

for name, factory in CANDIDATES.items():
    print(f"--- GroupKFold ({N_FOLDS} folds, grouped by subject#): {name} ---")
    oof_true = np.zeros(len(y), dtype=float)
    oof_pred = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        pipeline = factory()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[test_idx] = pipeline.predict(X.iloc[test_idx])
        oof_true[test_idx] = y.iloc[test_idx].values

    metrics = {
        "r2": r2_score(oof_true, oof_pred),
        "mae": mean_absolute_error(oof_true, oof_pred),
        "rmse": float(np.sqrt(mean_squared_error(oof_true, oof_pred))),
    }
    results[name] = {"metrics": metrics, "oof_true": oof_true, "oof_pred": oof_pred}
    print(f"  r2={metrics['r2']:.4f}  mae={metrics['mae']:.4f}  rmse={metrics['rmse']:.4f}  "
          f"(vs naive baseline r2={naive_metrics['r2']:.4f})")
    print()

print("=== Comparison (GroupKFold pooled out-of-fold, zero patient leakage) ===")
print(f"{'model':30s} {'r2':>8s} {'mae':>8s} {'rmse':>8s} {'r2_over_baseline':>18s}")
print(f"{'naive (predict 0)':30s} {naive_metrics['r2']:8.4f} {naive_metrics['mae']:8.4f} {naive_metrics['rmse']:8.4f} {'--':>18s}")
for name, r in sorted(results.items(), key=lambda kv: kv[1]["metrics"]["r2"], reverse=True):
    m = r["metrics"]
    over_baseline = m["r2"] - naive_metrics["r2"]
    print(f"{name:30s} {m['r2']:8.4f} {m['mae']:8.4f} {m['rmse']:8.4f} {over_baseline:18.4f}")
print()

best_name = max(results.items(), key=lambda kv: kv[1]["metrics"]["r2"])[0]
best = results[best_name]
best_metrics = best["metrics"]
r2_over_baseline = best_metrics["r2"] - naive_metrics["r2"]
print(f"Best by pooled R^2: {best_name} (r2_over_baseline={r2_over_baseline:.4f})")
print()

passed = r2_over_baseline >= MIN_R2_OVER_BASELINE
print(f"--- Validation gate (best model, pooled out-of-fold): "
      f"r2_over_naive_baseline >= {MIN_R2_OVER_BASELINE} ---")
print(f"model: {best_name}")
print(f"r2_over_baseline {r2_over_baseline:.4f} >= {MIN_R2_OVER_BASELINE}: {passed}")

if passed:
    final_pipeline = CANDIDATES[best_name]()
    final_pipeline.fit(X, y)  # final deployed model uses ALL data
    joblib.dump(final_pipeline, "ml/models/parkinsons_risk_regressor.joblib")
    metadata = {
        "model": f"{best_name} (scikit-learn Pipeline), selected by GroupKFold comparison across "
                 f"{len(CANDIDATES)} candidate model families",
        "candidates_compared": {name: r["metrics"] for name, r in results.items()},
        "naive_baseline_metrics": naive_metrics,
        "dataset": "UCI Parkinsons Telemonitoring (id=189), Tsanas & Little (2009), https://doi.org/10.24432/C5ZS3N",
        "feature_columns_in_order": feature_columns,
        "target_column": "delta_target (total_UPDRS at this visit minus total_UPDRS at the patient's baseline visit)",
        "target_meaning": (
            "Predicted CHANGE in total UPDRS severity score since the patient's first recorded "
            "visit -- positive = symptoms trending worse, negative = trending better, near zero = "
            "stable. This is a change/trend prediction, not an absolute severity score."
        ),
        "why_change_not_absolute_score": (
            "A first attempt at predicting absolute total_UPDRS for entirely unseen patients "
            "(GroupKFold by subject#) gave negative pooled R^2 for every model tried -- the "
            "dominant variance in absolute severity is between-patient differences voice alone "
            "cannot explain for a new patient. Reframing to change-from-own-baseline removes that "
            "between-patient variance and targets a more physiologically plausible, and arguably "
            "more clinically actionable, signal: is this patient trending worse."
        ),
        "n_subjects": int(n_subjects),
        "n_recordings_after_dropping_baseline_visits": len(df),
        "evaluation_strategy": (
            f"GroupKFold cross-validation ({N_FOLDS} folds, grouped by patient subject#) across all "
            f"{n_subjects} subjects. Zero patient leakage between train and test in any fold. "
            "Compared against a naive 'assume no change since baseline' predictor; the reported "
            "gate is R^2 improvement over that naive baseline, not a raw R^2 threshold, since raw "
            "R^2 on a delta target is not directly comparable to R^2 on an absolute-score target."
        ),
        "pooled_metrics": best_metrics,
        "r2_over_naive_baseline": r2_over_baseline,
        "validation_threshold": {"min_r2_over_naive_baseline": MIN_R2_OVER_BASELINE},
        "disclaimer": (
            "Trained on a public research dataset for demonstration purposes only. "
            "Never trained or evaluated on real patient data. Not used anywhere in "
            "the live patient workflow/orchestrator pipeline."
        ),
    }
    with open("ml/models/parkinsons_risk_regressor_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print()
    print(f"PASSED validation gate -- {best_name} (fit on ALL data) saved to ml/models/parkinsons_risk_regressor.joblib")
    print("Metadata (with pooled metrics, naive-baseline comparison, and full candidate comparison) saved.")
else:
    print()
    print("FAILED validation gate -- best candidate still not saved. Do not promote any model.")