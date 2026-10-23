"""Candidate model factories. EVERY data-dependent step (scaling, feature selection) is inside the Pipeline so
that it is re-fitted on each training split only. Hyperparameter 'tuning' is expressed as separate named
candidates, so the nested inner loop chooses among them using training rows only."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# classical voice-perturbation measures: a small, interpretable, literature-motivated baseline feature set
CLASSIC_VOICE = ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR"]


def _lr(C):
    return lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0))])


def _rf(n, depth=None, k=None):
    steps = [("s", StandardScaler())]
    if k:
        steps.append(("k", SelectKBest(f_classif, k=k)))
    steps.append(("c", RandomForestClassifier(n_estimators=n, max_depth=depth, class_weight="balanced", random_state=0, n_jobs=-1)))
    return lambda: Pipeline(steps)


def disease_candidates(fast: bool = False) -> dict:
    n = 50 if fast else 200
    return {
        "LR_C0.1": _lr(0.1), "LR_C1": _lr(1.0),
        "RF_depth3": _rf(n, 3), "RF_full": _rf(n),
        "RF_top10_in_fold": _rf(n, None, 10),
        "HGB_small": lambda: Pipeline([("c", HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.1,
                                                                            class_weight="balanced", random_state=0))]),
    }


def disease_baselines(fast: bool = False) -> dict:
    """Reference procedures evaluated with the SAME nested protocol. 'dummy_prior' = no-skill; 'classic3_LR' = logistic
    regression on three classical perturbation measures (the kind of rule a clinician-engineer would try first)."""
    from sklearn.compose import ColumnTransformer
    return {
        "dummy_prior": lambda: DummyClassifier(strategy="prior"),
        "classic3_LR": lambda: Pipeline([
            ("sel", ColumnTransformer([("keep", "passthrough", CLASSIC_VOICE)], remainder="drop")),
            ("s", StandardScaler()),
            ("c", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0))]),
    }


def risk_candidates(fast: bool = False) -> dict:
    return {"LR_C0.01": _lr(0.01), "LR_C0.1": _lr(0.1), "LR_C1": _lr(1.0),
            "HGB_small": lambda: Pipeline([("c", HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.1,
                                                                                class_weight="balanced", random_state=0))])}
