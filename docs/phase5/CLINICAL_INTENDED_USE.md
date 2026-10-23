# Phase 5: Intended Use Statement

This statement describes what the two ML components are for **today**, what they are explicitly not for, and what would have
to be true before any other use could be considered. It is an engineering and research scoping document. It is not a
regulatory submission or a clinical decision, and it contains no clinical thresholds or treatment rules. Defining a clinical
intended use is a decision for qualified clinicians, the responsible organisation and the applicable regulator.

## 1. Current status of both components

| | Disease classifier | Risk classifier |
|---|---|---|
| Status | **Research demonstration** | **Research demonstration** |
| Clinically validated | **No** | **No** |
| Connected to patient data or the NEURO-MAS clinical pipeline | **No** (served only via the demo endpoints; the endpoint text says so) | **No** |
| Trained on | UCI Parkinsons, 195 recordings, 31-32 people | UCI Telemonitoring, 5,801 sessions used, 42 patients |
| What the output means | a score that resembles recordings of people with PD in this dataset | a score that resembles sessions whose UPDRS is >= 3.0 points above that patient's first visit, measured concurrently |

## 2. Permitted use now

* Demonstrating an ML component inside a software system, to developers and reviewers.
* Research and method development on public data, with the registered protocol and honest reporting.
* Supervised demonstrations in which a person explains the limitations and the output is not used for any decision.

## 3. Not permitted / out of scope

* Diagnosis, screening, triage, monitoring or any decision about a real person.
* Use with a person's real recordings as if the output applied to them. The endpoints say they do not assess the submitter.
* Any statement of "risk of worsening" or "likelihood of disease" in a user interface, report or letter.
* Forecasting future progression (the risk label is concurrent, not predictive).
* Any claim of accuracy, sensitivity or specificity without the interval, the baseline, and the evidence class.
* Presenting synthetic-data results, internal cross-validation or historical figures as clinical evidence.

## 4. Questions that an intended use must answer, and who must answer them

Not answered here, deliberately.

| Question | Why it cannot be answered by engineering |
|---|---|
| Which clinical decision would the output inform, and for whom? | clinical pathway design |
| Target population, setting and recording conditions | clinical and operational |
| What action follows a given output, and what is the harm of a false positive / false negative? | clinical risk assessment |
| What minimum performance is acceptable at the decision-relevant prevalence? | clinical, with PPV/NPV (see `STATISTICAL_EVIDENCE_REPORT.md`, section 4) |
| Is a change of 3.0 UPDRS points meaningful for the intended decision? | the threshold is a provisional project parameter, not a validated minimal important difference |
| Regulatory classification and obligations | regulatory |

## 5. Preconditions before the status could change

A change from "research demonstration" requires, at minimum and in addition to the clinical decisions above: a confirmatory
internal run of `p4a-1` on authenticated data; external validation on an independent cohort; prospective or at least
temporally separated evaluation; calibration at the target prevalence; subgroup and fairness analysis; a human-oversight
design (see `SAFETY_AND_HUMAN_OVERSIGHT.md`); and the approvals applicable to the setting. None exists today. Engineering
criteria alone cannot declare clinical readiness.
