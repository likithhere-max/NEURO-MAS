"""
Single source of truth for the risk model's input/target conventions (Phase 5, Stage 2).

Why this exists. The risk classifier was trained on the UCI Telemonitoring column `test_time`, which the publisher defines as time
since RECRUITMENT into the study. Different parts of the code base described it differently ("days since baseline visit" in the OOD
bound, "days since recruitment" in the schema after Phase 4A). A text convention that lives in several places drifts, so every place
now imports it from here and a test fails if any serving module re-introduces the other meaning.

What this does NOT do: it does not quantify, or remove, the difference between "since recruitment" and "since the baseline visit".
The model was trained on the former. Removing the mismatch properly means retraining on `test_time - test_time_at_first_visit`
(protocol amendment, needs the data). Until then the serving contract is: supply days since recruitment, exactly as in the dataset.
"""
from __future__ import annotations

TEST_TIME_CONVENTION_ID = "days_since_recruitment_v1"
TEST_TIME_UNIT = "days"
TEST_TIME_SHORT = "Days since RECRUITMENT into the study (the Telemonitoring dataset's own `test_time`)"
TEST_TIME_NOT = "NOT days since the baseline visit"
TEST_TIME_MISMATCH_NOTE = (
    "The model was trained on days since recruitment. The two conventions differ by the baseline visit's own offset from "
    "recruitment, which has not been quantified (it needs the training data). Supply days since recruitment."
)
TEST_TIME_SERVING_DOMAIN_NOTE = (
    "Serving abstains on negative values: this is a conservative engineering bound, not a statement about the training data, whose "
    "minimum `test_time` has not been verified in this project."
)
TEST_TIME_DESCRIPTION = f"{TEST_TIME_SHORT}. It is {TEST_TIME_NOT}. {TEST_TIME_MISMATCH_NOTE}"

RISK_TARGET_DEFINITION = (
    "worsening = 1 if total_UPDRS(t) - total_UPDRS(first visit) >= 3.0 points, else 0. The threshold is a provisional project "
    "parameter, not a validated minimal clinically important difference."
)
RISK_PREDICTION_HORIZON = (
    "CONCURRENT (nowcast): features and label are measured at the same session t. This is not a forecast of future worsening."
)
RISK_BASELINE_ROW_RULE = (
    "Baseline = the first row, in file order, among a patient's rows with the minimum test_time. EVERY row tied at that minimum "
    "test_time is excluded from training; the deltas of all other rows are taken against the baseline row."
)
DISEASE_TARGET_DEFINITION = "status = 1 for a subject with Parkinson's disease, 0 for a healthy subject; constant within a subject."
DISEASE_PREDICTION_HORIZON = "CROSS-SECTIONAL: classifies the voice recording itself; no time dimension."
DISEASE_SUBJECT_ID_RULE = 'subject = the S-code in the recording name, e.g. "phon_R01_S01_1" -> "S01"'

# A model-agnostic statement every artifact carries
DEMO_ONLY_LIMITATION = ("Research demonstration on a small public dataset. Not clinically validated; not for diagnosis, monitoring or "
                        "any decision about a patient.")
