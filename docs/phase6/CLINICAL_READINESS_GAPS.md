# Phase 6: Clinical Readiness Gaps

Status of each gap that stands between the current research demonstrations and any clinical use. Nothing here is closed by engineering alone.
`docs/phase5/CLINICAL_EVIDENCE_GAPS.md` (21 rows) remains the Phase 5 record; this table updates the state after Phase 6 and adds new rows.
No clinical validity is claimed.

| # | Gap | Phase 5 state | Phase 6 state | Blocking dependency |
|---|---|---|---|---|
| G1 | Historical metrics not reproduced from data | open | open; reproduction pipeline now includes recomputed comprehensive metrics, tested on forged data | D1 (data) |
| G2 | Registered `p4a-1` confirmatory run | open | open; CI job `real-data-validation` added but not yet run on GitHub | D1; push (D10) |
| G3 | Voice added value beyond time and baseline | unknown | unknown; planning shows +0.03 AUC likely unresolvable with 42 patients | D1; possibly more patients |
| G4 | Calibration | unmeasured | metric library, exploratory cross-fitted recalibration implemented; unmeasured | D1 |
| G5 | Operating threshold with clinical rationale | none | none; protocol records `REQUIRES_CLINICAL_INPUT` | D9 |
| G6 | Decision-curve interpretation | none | curves computed on request; interpretation needs a clinical action and threshold range | D9 |
| G7 | External validation | none | `BLOCKED`; protocol drafted (`CLINICAL_VALIDATION_PROTOCOL.md`) | D8 |
| G8 | Prospective / temporal evaluation | none | none | D8, D9 |
| G9 | Subgroup and fairness analysis | none | implemented with minimum-size gates; sex/age/severity only; no race, device or language information exists in the files | D1; richer cohort |
| G10 | Missing-data behaviour | not estimable | not estimable from public data (files have no missing voice records) | external cohort |
| G11 | Label validity (interpolated UPDRS, 3.0-point threshold) | noted | audited (`MODEL_AUDIT.md`); number of real assessments per patient unverified | publisher documentation; D1 |
| G12 | Target meaning (concurrent vs forecast) | noted | unchanged; forecasting target would need a new protocol and data | D9 |
| G13 | Intended use and clinical acceptance criteria | not answered | structured template (`CLINICAL_INTENDED_USE.md`); unanswered | D9 |
| G14 | Hazard analysis | partial | hazard log H1-H13 (`SAFETY_CASE.md`); needs clinician review | D9 |
| G15 | Human-oversight design | requirements listed | unchanged | D9 |
| G16 | Provenance for shipped artifacts | `INCOMPLETE_DECLARED` | contract 1.1.0 adds intended use, split identifiers and validation link; shipped artifacts still cannot be completed truthfully; retraining on authenticated data yields `COMPLETE` | D1, D3 |
| G17 | Training/serving parity | tested | unchanged, tested | none |
| G18 | Regulatory classification | none | none | `REQUIRES_REGULATORY_INPUT` |
| G19 | Monitoring and drift detection | none | none | deployment decision |
| G20 | Calibration plots | none | reliability **tables** only; plots not generated | engineering, low priority, needs data |
| G21 | Remote CI evidence | none | workflow file written and YAML-parsed locally; never executed on GitHub | push (D10) |

Engineering-only gaps (G17, G20, G21) can be closed without clinicians; all others cannot.
