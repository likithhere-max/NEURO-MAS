# Phase 5: Safety and Human Oversight

Scope: what the software does today to avoid unsafe use of the two ML components, and the human-oversight requirements that
must be defined by qualified people before any wider use. **This document defines no clinical thresholds, escalation
rules, treatment recommendations or alert criteria.** Those are clinical decisions and are not invented here.

## 1. Current position

Both components are research demonstrations, served only via demo endpoints, with a response disclaimer stating that they are
not connected to patient data or the clinical pipeline and do not assess the submitter. They are not used to drive any agent
action, notification or recommendation. Source search shows ML routes referenced only by `app/main.py`, `app/schemas/ml_demo.py`
and `app/api/routes/ml_demo.py`.

## 2. Software safeguards in place (tested)

| Safeguard | Behaviour | Evidence |
|---|---|---|
| Restricted artifact loading | only checksum-matching artifacts load; the unpickler allows only a fixed list of class names (three exact additions in Phase 5, others still refused) | `test_E06_*`, existing safe_load tests |
| Input contract | missing, mistyped, NaN and Inf inputs rejected with 422, never defaulted | `test_p5_serving_contract.py` |
| Abstention | physically impossible or out-of-domain input returns a structured abstention, not a number | `app/ml/ood.py`, `test_out_of_range_values_abstain` |
| No silent fallback | model missing or no derivable input domain returns 503 | `test_model_unavailable_is_a_503_not_a_default_answer`, `test_serving_refuses_a_model_it_cannot_derive_an_input_domain_for` |
| Disclaimers | every prediction carries the demonstration disclaimer, including the risk model's limits (no CI, selected on its own folds, not compared with time/baseline-only models) | `app/api/routes/ml_demo.py` |
| Provenance visible | `/info` endpoints return the provenance status; shipped artifacts show `INCOMPLETE_DECLARED` | `test_info_endpoints_expose_provenance_status` |
| Evidence labelling | reports can only print registered claims; unauthenticated or synthetic runs carry explicit labels | `test_p5_workflow.py` |
| Fail-closed validation | no data means `NOT_RUN`, exit 2; CI cannot treat it as success | `test_main_returns_the_nonzero_exit_code_for_ci` |
| Determinism | duplicate requests give identical output | `test_duplicate_requests_are_deterministic` |

## 3. Known hazards (not mitigated by the safeguards above)

1. **Over-trust of a number.** A probability-like score from a model with 0.02 / 0.01 accuracy gain over a trivial baseline can
   be read as authoritative. The disclaimer is the only mitigation; interface design is out of scope.
2. **Uncalibrated scores** displayed as probabilities.
3. **High false-positive rate on healthy recordings** (historical specificity 0.4375, 8 healthy subjects).
4. **Moderate sensitivity for the risk model** (historical 0.598): a negative output is not reassurance.
5. **Distribution shift** (device, noise, language, disease stage) cannot be detected reliably by the provisional domain check.
6. **Mismatch of target** (concurrent worsening mistaken for forecast; provisional 3.0-point threshold mistaken for a clinical standard).
7. **Automation bias** if a downstream agent or interface were to consume the output without human review.
8. **Training/serving skew** in `test_time` if a caller supplies days-since-baseline instead of days-since-recruitment; the
   unit is documented and bounded below at 0, but a plausible wrong value cannot be detected.

## 4. Human-oversight requirements for any use beyond demonstration

To be defined and signed off by clinicians and the responsible organisation; listed as requirements, not as settings:

* A named accountable clinician for any use on real patients.
* The model output is advisory input to a human decision and never triggers an action, message, or change of care on its own.
* The interface shows the evidence class, the uncertainty, the intended-use boundary, and the abstention reason, not only a score.
* A clinician can disregard or override the output without penalty; overrides and abstentions are logged.
* Monitoring plan: input-distribution drift, abstention rate, subgroup performance, and a defined process for withdrawal.
* Incident reporting and a change-control process tied to the experiment manifest (protocol id and hash, artifact checksum,
  provenance status).
* A hazard analysis and risk assessment performed with clinicians and, where applicable, regulatory input.

## 5. What Phase 5 deliberately did not do

* Choose operating thresholds for clinical use or tune any threshold on evaluation data (the protocol uses a fixed 0.5).
* Define alert levels, escalation paths or treatment suggestions.
* Connect either model to the multi-agent clinical pipeline.
* Change the disclaimers' meaning. The risk text is consistent with the protocol audit.
