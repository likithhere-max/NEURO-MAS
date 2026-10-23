# Phase 6: Model Audit (target, features, label noise, leakage)

Audit of the two shipped models as defined in the code. Everything here is read from source and from shipped metadata; **no
quantity in this document was measured on data**, because the data are unavailable. Items needing the files are marked
`NEEDS_DATA`. No clinical validity is claimed.

## 1. Disease classifier

| Aspect | Finding |
|---|---|
| Target | `status` (Parkinson's vs healthy), a **subject-level** label copied onto each of the subject's recordings (about 6 per subject) |
| Unit of inference | recording; the unit of independence is the **subject** (32 per metadata, 31 per publisher) |
| Features at prediction time | 22 acoustic measures computed from a sustained-vowel recording; all available at prediction time. The shipped pipeline keeps 10 by `SelectKBest` inside the pipeline |
| Population | 31-32 people, 23-24 with PD, 8 healthy. Healthy controls are 25% of subjects; specificity rests on **8 independent units** |
| Label noise | unknown; diagnosis is by the dataset publisher; no inter-rater information. Medication state and disease stage are not in the file |
| Leakage | none direct: scaling and selection are inside each fold. The historical winner was selected on the folds it reports (F5). Subject grouping is by a parsed recording name (parser verified to agree on well-formed names; 31 vs 32 `NEEDS_DATA`) |
| Confounding | recording device, session and room are constant within a subject; a model can learn subject identity or recording condition rather than disease. Subject-grouped CV removes the identity route within the file but not a condition shared by all PD or all healthy recordings. `NEEDS_EXTERNAL_COHORT` |
| Question the target answers | "does this recording resemble a PD recording in this dataset", not "does this person have PD" |

## 2. Risk classifier

| Aspect | Finding |
|---|---|
| Target | `total_UPDRS(t) - total_UPDRS(first visit) >= 3.0`, at the **same visit** `t` (`build_risk_frame`, `ml/validation/data.py`) |
| Meaning | **concurrent** classification: "is this visit at least 3.0 points worse than the first visit". Not a forecast of future worsening. The 3.0-point threshold is a provisional project parameter, not a validated minimal important difference |
| Features | 16 voice deltas relative to the first visit's voice, `test_time`, `baseline_total_UPDRS` |
| Available at prediction time | `test_time`: yes. `baseline_total_UPDRS`: yes, the patient's own first UPDRS. Voice deltas: yes, require the first visit's voice and the current voice. The current UPDRS is the label and is never a feature |
| Structural predictability | `baseline_total_UPDRS` is a term of the label definition and `test_time` correlates with progression, so a model with **no voice** can score well. This is why the registered ablations (`TIME_ONLY`, `TIME_BASELINE`, `TIME_BASELINE_DEMOGRAPHIC`, `VOICE_ONLY`) are mandatory |
| Label noise | The publisher states UPDRS values were linearly interpolated between clinic assessments (publisher description). Labels at interpolated visits are therefore **smoothed functions of few real assessments**, not independent measurements. How many real assessments each patient has is **not verified here** (my recollection is about three; unverified, `NEEDS_DATA`/publisher documentation) |
| Consequence of interpolation | Within-patient rows are highly dependent and the label is monotone-ish in time within a patient by construction. Row-level metrics and row-level CIs are not interpretable; the unit is the **patient** (42). Time alone may predict the label strongly |
| Row selection | The baseline rule drops all rows tied at a patient's minimum `test_time`, not one per patient. 32 extra drops would explain 5,801 vs 5,833 (mechanism proved by test; real count `NEEDS_DATA`) |
| Leakage | no direct feature-label leakage found. Same-visit delta features and the label are measured together: concurrent, not leakage in the technical sense, but it limits what a claim can mean |
| Class balance | 38.2% positive (2,215/5,801); patients differ in whether they ever cross the threshold (`subjects_with_both_classes` is recorded by the loader, `NEEDS_DATA`) |
| Question the target answers | "does the voice/time profile look like a visit 3+ points worse than baseline" |

## 3. Cross-cutting audit findings

1. **Both targets are weaker than their names.** Neither is a diagnosis or a forecast; the user-facing wording already says so (Phase 5 intended-use statement).
2. **Effective sample sizes are tiny.** About 42 patients (risk) and 8 healthy subjects (disease) bound every conclusion; see `sample_size.py` and `STATISTICAL_EVIDENCE_REPORT.md`.
3. **No external data.** All evidence that could ever exist from these files is internal; internal cross-validation cannot detect device, site or population shift.
4. **Historical metrics cannot be tied to a protocol.** They are retained as `HISTORICAL_UNVERIFIED_NOT_REPRODUCED`.
5. **Serving/training parity** is tested for column order, dtype and units (Phase 5) and the provenance contract now records split identifiers and a link to the validation run (`artifact_contract.py` 1.1.0; section 6 of `FINAL_READINESS_REPORT.md`).

## 4. Consequences for the experiments

* Candidate improvements are limited to families with a stated hypothesis (`HYPOTHESES` in `ml/validation/candidates_p6.py`) and are
  compared with the registered baselines on **identical outer folds**; the report pipeline refuses a comparison when fold identity differs.
* All Phase 6 candidate results are labelled `+EXPLORATORY_EXTENSION_P6_1`, never confirmatory. The registered protocol decides the
  confirmatory voice-value verdict.
* Nothing is tuned on any external cohort. If one is ever obtained, the model, preprocessing, threshold and analysis plan are frozen
  before it is opened; recalibration on it is labelled adaptation, not validation.
