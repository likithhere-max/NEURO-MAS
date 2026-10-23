# Phase 5: Generalization and Robustness

**There is no external cohort in this repository and no way to obtain one here. Nothing in this document is external
validation.** Everything described is either an engineering stress test of the pipeline (synthetic or constructed inputs,
labelled as such) or a statement of what has not been tested. Internal cross-validation is never called external.

## 1. What generalization claims are possible today

| Question | Evidence available | Status |
|---|---|---|
| Does the model work on new subjects from the same dataset? | historical LOSO / grouped CV figures, unreproduced | **unverified** |
| ... on a different site, microphone, recording protocol or language? | none | **untested** |
| ... on other populations (age, sex, ethnicity, severity range, comorbidity)? | none; the datasets record age/sex only in the risk file | **untested** |
| ... prospectively, in a care pathway? | none | **untested** |
| ... on data resembling the real world's prevalence? | none; training prevalence is a dataset artefact | **untested** |

The public datasets are small, from single research settings, with 31-32 and 42 individuals. Both models are research
demonstrations (see `CLINICAL_INTENDED_USE.md`).

## 2. Engineering stress tests actually run (`backend/tests/phase5/`)

These test that the *software* refuses, handles or reports abnormal inputs. They say nothing about clinical robustness and use
constructed data only.

| Stress | Outcome tested | Test |
|---|---|---|
| Missing values in the training file | refused, not imputed | `test_missing_values_are_refused_not_imputed` |
| NaN / Inf | refused | `test_non_finite_values_are_refused` |
| Unexpected label category | refused | `test_unexpected_label_category_is_refused` |
| Duplicated records | counts change; refused | `test_duplicated_records_change_the_counts_and_are_refused` |
| Unparseable recording names | refused | `test_unparseable_recording_names_are_refused` |
| Subject with two labels | refused | `test_subject_with_two_labels_is_refused` |
| Changed schema or unit | never authenticates as the training data | `test_a_changed_schema_or_unit_never_authenticates` |
| Row order / float formatting changes | fingerprint unaffected; a tiny value edit is detected | `test_row_order_and_float_formatting_do_not_matter_but_a_tiny_edit_does` |
| Out-of-order and duplicate risk records | deterministic and reported | `test_risk_out_of_order_and_duplicate_records_are_handled_deterministically_and_reported` |
| Temporal shift in `test_time` | changes the fingerprint (shift is visible) | `test_risk_temporal_shift_in_test_time_changes_the_fingerprint` |
| Changed class prevalence | visible in the report, not hidden | `test_changed_class_prevalence_is_visible_in_the_report_not_hidden` |
| Serving: NaN/Inf/missing/mistyped/out-of-range inputs | 422 or abstention, never a default | `test_p5_serving_contract.py` |
| Serving: model unavailable | 503, never a default answer | `test_model_unavailable_is_a_503_not_a_default_answer` |

## 3. Serving-time domain checks (existing, from Phase 2)

`app/ml/ood.py` abstains rather than predicts when an input is invalid or outside the training domain (contract, physical
bounds, algebraic identities, marginal ranges). Honest limits:

* The shipped artifacts only support the **provisional** domain, derived from the scaler's mean and SD (no min/max, no
  covariance); a fitted domain requires the real data (not run).
* The Mahalanobis/joint test is a diagnostic whose false-rejection and missed-OOD rates have not been measured on real data.
* The domain check says an input is not obviously foreign. It does not say the prediction is right.
* The `test_time` lower bound of 0 is a conservative engineering bound, not a clinical rule.

## 4. Planned generalization evidence (blocked on data and cohorts)

1. Run `p4a-1` on authenticated public data (subject-grouped, internal). Label: confirmatory-internal.
2. Leave-one-recording-session-out or site-style splits only where the data carry such structure (the public files do not).
3. External validation requires an independent cohort with its own protocol, ethics approval and data-use agreement; these are
   external dependencies (see `CLINICAL_EVIDENCE_GAPS.md`).
4. Subgroup performance (age, sex, severity) needs enough subjects per subgroup; with 8 healthy subjects it is infeasible for
   the disease task.
