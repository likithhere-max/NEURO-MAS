"""
Roadmap item 6, Track 2, step 2: train and validate a demonstration
classifier on the public UCI Parkinson'"'"'s voice dataset. This is fully
decoupled from the live patient pipeline -- it never reads real patient
data and is never called from the orchestrator/workflow code. Trained and
evaluated entirely offline; only promoted to ml/models/ if it clears the
validation thresholds below.

Dataset: Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C59C74
Raw data: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Validation thresholds -- the model is only saved to ml/models/ if BOTH are
# met on the held-out test set. Recall matters more than raw accuracy here:
# for anything framed as health-related, missing a real positive case is a
# worse failure than a false alarm.
MIN_ROC_AUC = 0.85
MIN_RECALL_POSITIVE = 0.85

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

print("Loading data...")
df = pd.read_csv(URL)
feature_columns = [c for c in df.columns if c not in ("name", "status")]
X = df[feature_columns]
y = df["status"]
print(f"Loaded {len(df)} rows, {len(feature_columns)} features.")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} rows ({y_train.sum()} positive). Test: {len(X_test)} rows ({y_test.sum()} positive).")
print()

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
])

print("--- 5-fold stratified cross-validation on training set ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
)
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]
    print(f"  {metric}: mean={scores.mean():.4f}  std={scores.std():.4f}  folds={[round(s, 3) for s in scores]}")
print()

print("--- Fitting on full training set, evaluating on held-out test set ---")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_proba)

print(f"Test accuracy:  {test_accuracy:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall:    {test_recall:.4f}")
print(f"Test f1:        {test_f1:.4f}")
print(f"Test roc_auc:   {test_roc_auc:.4f}")
print()
print("Confusion matrix (rows=actual, cols=predicted, order=[0,1]):")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["healthy", "parkinsons"]))

print("--- Feature importances (top 10) ---")
importances = pipeline.named_steps["classifier"].feature_importances_
ranked = sorted(zip(feature_columns, importances), key=lambda t: t[1], reverse=True)
for name, importance in ranked[:10]:
    print(f"  {name}: {importance:.4f}")
print()

passed = test_roc_auc >= MIN_ROC_AUC and test_recall >= MIN_RECALL_POSITIVE
print(f"--- Validation gate: roc_auc >= {MIN_ROC_AUC} and recall >= {MIN_RECALL_POSITIVE} ---")
print(f"roc_auc {test_roc_auc:.4f} >= {MIN_ROC_AUC}: {test_roc_auc >= MIN_ROC_AUC}")
print(f"recall {test_recall:.4f} >= {MIN_RECALL_POSITIVE}: {test_recall >= MIN_RECALL_POSITIVE}")

if passed:
    joblib.dump(pipeline, "ml/models/parkinsons_demo_classifier.joblib")
    metadata = {
        "model": "RandomForestClassifier (scikit-learn Pipeline: StandardScaler + RandomForestClassifier)",
        "dataset": "UCI Parkinsons (id=174), Little M. (2007), https://doi.org/10.24432/C59C74",
        "feature_columns_in_order": feature_columns,
        "target_column": "status",
        "target_meaning": {"0": "healthy", "1": "parkinsons"},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
        },
        "validation_thresholds": {"min_roc_auc": MIN_ROC_AUC, "min_recall_positive": MIN_RECALL_POSITIVE},
        "disclaimer": (
            "Trained on a public research dataset for demonstration purposes only. "
            "Never trained or evaluated on real patient data. Not used anywhere in "
            "the live patient workflow/orchestrator pipeline."
        ),
    }
    with open("ml/models/parkinsons_demo_classifier_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print()
    print("PASSED validation gate -- model saved to ml/models/parkinsons_demo_classifier.joblib")
    print("Metadata saved to ml/models/parkinsons_demo_classifier_metadata.json")
else:
    print()
    print("FAILED validation gate -- model NOT saved. Do not promote this model.")
