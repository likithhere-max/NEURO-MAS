"""
Roadmap item 6, Track 2, second deliverable -- RISK / WORSENING-TREND
classification (REVISED #3, final attempt on this dataset/framing). Trains
and validates a demonstration classifier on the public UCI Parkinson's
Telemonitoring dataset. Fully decoupled from the live patient pipeline.

WHY THIS IS REVISED AGAIN: two prior honest attempts both failed under
patient-level GroupKFold --
  1. Predicting ABSOLUTE total_UPDRS for unseen patients: negative pooled R^2
     for every model (RandomForest, HistGradientBoosting, Ridge).
  2. Predicting the exact CHANGE in total_UPDRS from each patient's own
     baseline visit: still negative R^2 (worse than a naive "assume no
     change" baseline) for every model (RandomForest, Ridge).
Two independent framings, three model families, consistently no signal --
this is a real, replicated finding about this dataset (voice biomarkers
alone do not generalize across UNSEEN patients for UPDRS severity), not a
one-off bad run.

THIS ATTEMPT asks a coarser, lower-bar question: not "predict the exact
point change" (regression, needs exact magnitude) but "is this patient
trending MEANINGFULLY WORSE, yes or no" (binary classification, evaluated by
ROC-AUC, which only needs correct rank-ordering of risk, not exact values).
Label: delta_total_UPDRS (change from the patient's baseline visit) >= 3.0
points = "worsening" (1), else "stable/improving" (0). NOTE: 3.0 points is
an illustrative demonstration threshold for this project, NOT a clinically
validated minimal-detectable-change value -- the metadata says this
explicitly so it is never mistaken for a real clinical cutoff.

If THIS also fails to beat chance level (pooled ROC-AUC ~0.50) under proper
patient-level GroupKFold, no further model attempts should be made on this
dataset/framing -- the honest conclusion becomes the deliverable: voice-only
telemonitoring features do not reliably predict symptom trajectory for a
patient the model has never seen, evaluated properly. That is itself a
legitimate, presentable finding.

Every patient still contributes only to either train or test in each
GroupKFold fold (grouped by subject#) -- zero patient leakage.

Dataset: Tsanas, A. and Little, M. (2009). Parkinsons Telemonitoring
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZS3N
Raw data: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "total_UPDRS"
WORSENING_THRESHOLD = 3.0  # illustrative demonstration cutoff, NOT a validated clinical MCID
MIN_ROC_AUC = 0.60  # modest, explicitly-justified "meaningfully better than chance" bar
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

baseline = df.groupby("subject#", as_index=False).first().set_index("subject#")
df["baseline_total_UPDRS"] = df["subject#"].map(baseline[TARGET])
for col in voice_feature_columns:
    df[f"baseline_{col}"] = df["subject#"].map(baseline[col])

df["is_baseline_row"] = df.groupby("subject#")["test_time"].transform("min") == df["test_time"]
df = df[~df["is_baseline_row"]].reset_index(drop=True)

df["delta_target"] = df[TARGET] - df["baseline_total_UPDRS"]
df["worsening"] = (df["delta_target"] >= WORSENING_THRESHOLD).astype(int)

delta_feature_columns = []
for col in voice_feature_columns:
    delta_col = f"delta_{col}"
    df[delta_col] = df[col] - df[f"baseline_{col}"]
    delta_feature_columns.append(delta_col)

feature_columns = delta_feature_columns + ["test_time", "baseline_total_UPDRS"]
X = df[feature_columns].reset_index(drop=True)
y = df["worsening"].reset_index(drop=True)
groups = df["subject#"].reset_index(drop=True)
n_subjects = groups.nunique()

print(f"After dropping each subject's own baseline visit: {len(df)} rows, "
      f"{len(feature_columns)} features, {n_subjects} unique subjects.")
print(f"Label: worsening = (delta_total_UPDRS >= {WORSENING_THRESHOLD}). "
      f"Positive (worsening) rows: {y.sum()} / {len(y)} ({100*y.mean():.1f}%).")
print()

CANDIDATES = {
    "LogisticRegression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=42)),
    ]),
    "RandomForest": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
}

gkf = GroupKFold(n_splits=N_FOLDS)
results = {}

for name, factory in CANDIDATES.items():
    print(f"--- GroupKFold ({N_FOLDS} folds, grouped by subject#): {name} ---")
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        pipeline = factory()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[test_idx] = pipeline.predict(X.iloc[test_idx])
        oof_proba[test_idx] = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        oof_true[test_idx] = y.iloc[test_idx].values

    metrics = {
        "accuracy": accuracy_score(oof_true, oof_pred),
        "precision": precision_score(oof_true, oof_pred, zero_division=0),
        "recall": recall_score(oof_true, oof_pred, zero_division=0),
        "f1": f1_score(oof_true, oof_pred, zero_division=0),
        "roc_auc": roc_auc_score(oof_true, oof_proba),
    }
    results[name] = {"metrics": metrics, "oof_true": oof_true, "oof_pred": oof_pred, "oof_proba": oof_proba}
    print(f"  accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")
    print()

print("=== Comparison (GroupKFold pooled out-of-fold, zero patient leakage) ===")
print(f"{'model':20s} {'roc_auc':>8s} {'recall':>8s} {'precision':>10s} {'f1':>8s} {'accuracy':>9s}")
for name, r in sorted(results.items(), key=lambda kv: kv[1]["metrics"]["roc_auc"], reverse=True):
    m = r["metrics"]
    print(f"{name:20s} {m['roc_auc']:8.4f} {m['recall']:8.4f} {m['precision']:10.4f} {m['f1']:8.4f} {m['accuracy']:9.4f}")
print()

best_name = max(results.items(), key=lambda kv: kv[1]["metrics"]["roc_auc"])[0]
best = results[best_name]
best_metrics = best["metrics"]
print(f"Best by pooled ROC-AUC: {best_name}")
print()
print("Pooled confusion matrix for the winner (rows=actual, cols=predicted, order=[0,1]):")
print(confusion_matrix(best["oof_true"], best["oof_pred"]))
print()
print("Pooled classification report for the winner:")
print(classification_report(best["oof_true"], best["oof_pred"], target_names=["stable/improving", "worsening"], zero_division=0))

passed = best_metrics["roc_auc"] >= MIN_ROC_AUC
print(f"--- Validation gate (best model, pooled out-of-fold): roc_auc >= {MIN_ROC_AUC} ---")
print(f"model: {best_name}")
print(f"roc_auc {best_metrics['roc_auc']:.4f} >= {MIN_ROC_AUC}: {passed}  (0.50 = pure chance)")

if passed:
    final_pipeline = CANDIDATES[best_name]()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, "ml/models/parkinsons_risk_classifier.joblib")
    metadata = {
        "model": f"{best_name} (scikit-learn Pipeline), selected by GroupKFold comparison across "
                 f"{len(CANDIDATES)} candidate model families",
        "candidates_compared": {name: r["metrics"] for name, r in results.items()},
        "dataset": "UCI Parkinsons Telemonitoring (id=189), Tsanas & Little (2009), https://doi.org/10.24432/C5ZS3N",
        "feature_columns_in_order": feature_columns,
        "target_column": f"worsening (delta_total_UPDRS >= {WORSENING_THRESHOLD} from patient's own baseline visit)",
        "target_meaning": {
            "1": f"total_UPDRS increased by >= {WORSENING_THRESHOLD} points since the patient's first recorded visit",
            "0": "stable or improving relative to baseline",
        },
        "threshold_disclaimer": (
            f"{WORSENING_THRESHOLD} points is an illustrative demonstration threshold chosen for "
            "this project, NOT a clinically validated minimal-detectable-change value. Do not "
            "present this as a clinically established cutoff."
        ),
        "why_this_framing": (
            "Two prior attempts (absolute total_UPDRS regression, and exact delta-from-baseline "
            "regression) both failed patient-level GroupKFold validation with negative R^2 across "
            "5 total model fits. This reframes the same features/target as a coarser binary "
            "'trending worse or not' question, evaluated by ROC-AUC (rank-ordering only, not exact "
            "magnitude), as a lower and more achievable bar."
        ),
        "n_subjects": int(n_subjects),
        "n_recordings_after_dropping_baseline_visits": len(df),
        "positive_class_rate": float(y.mean()),
        "evaluation_strategy": (
            f"GroupKFold cross-validation ({N_FOLDS} folds, grouped by patient subject#) across all "
            f"{n_subjects} subjects. Zero patient leakage between train and test in any fold. All "
            "predictions pooled before computing metrics."
        ),
        "pooled_metrics": best_metrics,
        "validation_threshold": {"min_roc_auc": MIN_ROC_AUC, "chance_level": 0.50},
        "disclaimer": (
            "Trained on a public research dataset for demonstration purposes only. "
            "Never trained or evaluated on real patient data. Not used anywhere in "
            "the live patient workflow/orchestrator pipeline."
        ),
    }
    with open("ml/models/parkinsons_risk_classifier_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print()
    print(f"PASSED validation gate -- {best_name} (fit on ALL data) saved to ml/models/parkinsons_risk_classifier.joblib")
    print("Metadata (with pooled metrics + full candidate comparison) saved.")
else:
    print()
    print("FAILED validation gate -- best candidate still not saved. Do not promote any model.")
    print("This is the honest ceiling for this dataset/framing -- see the comparison table above.")