"""Phase 4A: wording shown to API users must not claim more than the evidence supports (E01) or misdescribe a feature (E02)."""
from app.api.routes import ml_demo
from app.schemas.ml_demo import ParkinsonsRiskFeaturesIn


def test_E01_risk_disclaimer_makes_no_unsupported_performance_claim():
    t = ml_demo.RISK_DISCLAIMER.lower()
    assert "meaningfully better than chance" not in t
    assert "same cross-validation folds" in t and "no confidence interval" in t
    assert "not clinical-grade" in t


def test_E01_disease_disclaimer_does_not_state_a_subject_count_the_publisher_contradicts():
    t = ml_demo.DISCLAIMER
    assert "from 32 people" not in t and "31-32" in t


def test_E02_risk_test_time_is_documented_as_days_since_recruitment_not_since_baseline():
    d = ParkinsonsRiskFeaturesIn.model_fields["test_time"].description.lower()
    assert "recruitment" in d and "not days since the baseline visit" in d
