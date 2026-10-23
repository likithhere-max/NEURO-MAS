# Phase 6: Safety Case (hazard log for the research-demonstration scope)

Scope: the two ML components as they exist today (research demonstrations behind demo endpoints, not connected to patient data or to any
agent action). This is an engineering hazard log, not a clinical risk assessment. It defines no clinical thresholds, escalation rules or
treatment advice. Severity and acceptability of residual risk for any clinical use are `REQUIRES_CLINICAL_INPUT`.
The safety argument currently rests on one claim: **the output is not used for any decision**. If that claim fails, most hazards below are unmitigated.

## 1. Claim and evidence

| Claim | Evidence | Status |
|---|---|---|
| C1: no ML output drives an action, message or recommendation | source search: ML routes referenced only by `app/main.py`, `app/schemas/ml_demo.py`, `app/api/routes/ml_demo.py` (Phase 5); disclaimer on every response | `VERIFIED` by code inspection at the Phase 5 commit; re-checked by test `test_p5_serving_contract.py` for the disclaimer only |
| C2: invalid or impossible input never yields a number | 422 on missing/mistyped/NaN/Inf; structured abstention out of domain; 503 when no domain can be derived | tests in `tests/phase5/test_p5_serving_contract.py` |
| C3: only checksum-matching, allow-listed artifacts load | `app/ml/safe_load.py`, tests | tested |
| C4: no document or test asserts clinical validity | `test_no_phase5_document_claims_clinical_validity` and the Phase 6 equivalent | tested |
| C5: performance is clinically acceptable | none | **no evidence; not claimed** |

## 2. Hazard log

Likelihood and severity are not scored (no basis); the table records the cause, existing controls, and what is missing.

| ID | Hazard | Cause | Existing controls | Residual / missing |
|---|---|---|---|---|
| H1 | Disease false positive taken as disease | low historical specificity (0.4375, 8 healthy subjects, unverified); shift | demo-only scope; disclaimer | no calibrated rate; no clinical pathway; needs external cohort |
| H2 | Risk false positive prompts unwarranted concern | threshold 0.5 not tied to any harm trade-off | demo-only scope | acceptable rate `REQUIRES_CLINICAL_INPUT` |
| H3 | Risk false negative taken as reassurance | historical sensitivity 0.598 (unverified); target is concurrent, not a forecast | disclaimer states limits | no safety-net process defined |
| H4 | Disease false negative | model resembles dataset recordings only | demo-only scope | needs external sensitivity estimate |
| H5 | Abstention misread as "normal" | UI treats absence of a score as no finding | structured abstention with reason | UI wording and training not designed |
| H6 | Confident output on shifted data (device, noise, language, stage) | domain check is a coarse range check | abstention for out-of-range values | no drift monitor; no external data |
| H7 | Over-trust of an uncalibrated score | scores displayed like probabilities | disclaimer | calibration unmeasured (needs data); display design out of scope |
| H8 | Target misread (concurrent as forecast; 3.0 points as clinical standard) | naming and UI | intended-use statement; disclaimer | clinical definition of endpoint |
| H9 | Automation bias if consumed by the multi-agent pipeline | architecture permits coupling | no coupling today (C1) | any future coupling needs a new hazard analysis and human sign-off |
| H10 | `test_time` convention mismatch (days since recruitment vs since baseline) | caller supplies a plausible wrong value | documented unit, bound >= 0, single convention scan | undetectable if plausible; skew size `NEEDS_DATA` (D6) |
| H11 | Wrong or tampered artifact served | file replaced | checksum + allow-listed unpickler; provenance status visible | shipped artifacts `INCOMPLETE_DECLARED`; cannot be backfilled truthfully |
| H12 | Patient-level predictions leak | held-out predictions are derivatives of patient data | `protected/` git-ignored output | access control and retention policy not defined |
| H13 | Reporting error presents exploratory or synthetic results as evidence | human or tooling error | claims-checked report generator; evidence labels; `claims_register.json` and test; fail-closed exit codes | registers cover repository documents only, not external communications |

## 3. Conditions that would change the case

Any of: connecting an output to a message, alert, recommendation or record; offering the demo to real users with their own data;
presenting accuracy figures in a UI; a claim of screening, triage or monitoring. Each requires, first: completed
`CLINICAL_INTENDED_USE.md`, an independent cohort result (`CLINICAL_VALIDATION_PROTOCOL.md`), a clinician-led hazard analysis,
human-oversight design (`docs/phase5/SAFETY_AND_HUMAN_OVERSIGHT.md` section 4) and regulatory input.

## 4. What this case does not establish

It does not show the models are safe or effective for any patient. It shows only that, in the demonstration scope, the software
avoids presenting unsupported outputs as findings and fails closed when evidence is missing.
