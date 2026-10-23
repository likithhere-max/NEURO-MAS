# Phase 6: Intended Use

This is an engineering and research scoping document. It is not a regulatory submission, and it does not decide what the
models are for clinically. Where a field needs a clinical or organisational decision it is marked `REQUIRES_CLINICAL_INPUT`
and left empty, deliberately: filling it from engineering preference would be inventing a requirement. No clinical validity is claimed
for either model. Supersedes nothing: `docs/phase5/CLINICAL_INTENDED_USE.md` remains the Phase 5 record.

## 1. Intended use in force today

| Field | Disease classifier | Risk classifier |
|---|---|---|
| Status | research demonstration | research demonstration |
| Population | none (no patient is assessed). Training data: 31-32 people, UCI 174 | none. Training data: 42 patients in an early-stage trial, UCI 189 |
| Setting | developer/reviewer demonstrations; research on public data | same |
| Target | resemblance of a sustained-vowel recording to PD recordings in one public dataset | resemblance of a visit's voice/time profile to visits whose UPDRS is >= 3.0 above that patient's first visit (concurrent) |
| Horizon | none (not a forecast) | none (concurrent, not a forecast) |
| Intended user | developer, researcher, supervised demonstrator | same |
| Action taken on an output | **none.** The output is not used for any decision, message, alert or change of care | **none** |
| Exclusions | any real person's recording presented as an assessment of that person; any patient-care workflow; synthetic or historical figures as evidence | same |

## 2. Candidate clinical intended-use fields (to be completed by qualified people)

A future use is only definable once every row below has an owner and an answer. Engineering cannot complete them.

| Field | Needed answer | Status |
|---|---|---|
| Clinical decision informed | which decision, in which pathway | `REQUIRES_CLINICAL_INPUT` |
| Target population | diagnosis stage, age range, comorbidities, treatment state | `REQUIRES_CLINICAL_INPUT` |
| Setting | clinic, remote home use, telemonitoring programme | `REQUIRES_CLINICAL_INPUT` |
| Target and horizon | diagnosis now, current severity change, forecast over N months; the present risk label is concurrent | `REQUIRES_CLINICAL_INPUT` |
| Minimal important change | whether 3.0 UPDRS points is meaningful for the decision (currently a provisional project parameter) | `REQUIRES_CLINICAL_INPUT` |
| Intended user and training | clinician, nurse, patient, caregiver | `REQUIRES_CLINICAL_INPUT` |
| Action and its owner | what a human does after each output category, including abstention | `REQUIRES_CLINICAL_INPUT` |
| Recording conditions | device, environment, task, language | `REQUIRES_CLINICAL_INPUT` |
| Regulatory classification | applicable regime and obligations | `REQUIRES_REGULATORY_INPUT` |

## 3. Consequences of each output type (analysis frame for the clinicians, not values)

| Output | What goes wrong | Hazard link |
|---|---|---|
| False positive | unnecessary follow-up, anxiety, cost; for the disease model the historical specificity was low (0.4375, on 8 healthy subjects, unverified) | H1, H2 in `SAFETY_CASE.md` |
| False negative | missed change or missed disease; false reassurance (risk-model historical sensitivity 0.598, unverified) | H3, H4 |
| Abstention (out of domain / invalid input) | no information; harm is the delay if the user treats silence as "normal" | H5 |
| Confidently wrong output on shifted data | undetected harm; the domain check is a coarse range check | H6 |

Acceptable rates for each are `REQUIRES_CLINICAL_INPUT`; the sample sizes needed to estimate them are in `CLINICAL_VALIDATION_PROTOCOL.md`.

## 4. Not permitted

Diagnosis, screening, triage, monitoring, risk statements, forecasts of progression, use on a real person's recording as if it applied to them, and any statement of accuracy
without interval, baseline and evidence class (`docs/phase5/CLINICAL_INTENDED_USE.md` section 3 applies unchanged).
