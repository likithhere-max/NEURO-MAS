# NEURO-MAS — ML Validation Report (Deliverable 4)

## 0. Bottom line
**No real-data validation result exists in this report.** The sandbox could not reach UCI,
HuggingFace or GitHub, so neither training dataset could be downloaded. I therefore did **not** re-measure
either model's performance. What I did deliver is (a) an audit of how the shipped numbers were produced,
(b) a validation harness that fixes the methodological problems, and (c) artifact integrity controls.
Please run the harness locally (commands below) and replace section 4 with the real output.

Anything the harness printed during my session came from **synthetic** data I generated only to prove the
code runs. Those numbers are meaningless and must not be quoted.

## 1. Models in the repo
| Artifact | Task | Data | Reported in metadata |
|---|---|---|---|
| `parkinsons_demo_classifier.joblib` | PD vs healthy from sustained-vowel voice features | UCI Parkinsons (Little 2007): 195 recordings, ~31–32 subjects, **~75 % PD** | best of 5 families by LOSO: accuracy 0.774, ROC-AUC 0.794 (RandomForest_Top10Features) |
| `parkinsons_risk_classifier.joblib` | "worsening" = ΔtotalUPDRS ≥ 3 from patient's first visit | UCI Telemonitoring: 42 subjects, 5 801 recordings, 38 % positive | GroupKFold(10) pooled ROC-AUC 0.640 (LogisticRegression) |

## 2. Methodological findings (from reading `ml/training/*.py` and the metadata)
| # | Finding | Why it matters | Status |
|---|---|---|---|
| ML1 | **Selection bias**: the winner among 5 model families is chosen on the same LOSO folds used to report performance. | The reported best AUC is an optimistic estimate; with ≈8 healthy subjects the noise between candidates (0.65–0.79) is larger than the gaps used to choose. | Harness uses **nested** subject-level selection. Result BLOCKED. |
| ML2 | No confidence interval. | With 31 subjects a point AUC is nearly uninformative. | Harness: bootstrap over **subjects**. |
| ML3 | Window/recording-level metrics pooled; subjects with more recordings weigh more. | Not the clinically relevant unit. | Harness: subject-level aggregation. |
| ML4 | No dummy/majority baseline; class imbalance (75 % PD). | An accuracy of 0.75 is achievable by predicting "PD" for everyone. | Harness reports dummy baseline. |
| ML5 | No calibration. | Probabilities are shown to users. | Harness: Brier + ECE. |
| ML6 | Predictive value ignores prevalence. A 75 %-PD training set ≠ a screening population. | PPV in a 1–5 % population is far lower than in the study. | Harness: PPV/NPV at π = 1, 5, 20, 75 %. |
| ML7 | Risk model uses `test_time` and `baseline_total_UPDRS` as features. | The label is "UPDRS rose since baseline"; time-since-baseline and baseline severity (regression to the mean: patients starting high can only drop, those starting low can rise) may be able to predict it with **no voice information at all** (hypothesis — this is exactly what the ablation tests; I have not measured it). | Harness ablations: voice-only, time-only, baseline-only, time+baseline vs FULL. Result BLOCKED. |
| ML8 | Risk label threshold (3.0 UPDRS) is arbitrary (the metadata itself says so). | — | Documented. |
| ML9 | Distribution shift: studio-quality sustained vowels (disease model) and a telemonitoring home device (risk model) vs any phone/browser recording. | Demo endpoint accepts hand-typed feature values; out-of-distribution inputs still return confident outputs (reproduced). | Input NaN/inf rejected; **no OOD detector** — OPEN. |
| ML10 | Artefact reproducibility: trained under scikit-learn 1.9.0 while the repo pinned 1.9.1; no version/hash. | Pickle loading across versions can silently change behaviour. | `MANIFEST.json` (sha256) + pin `>=1.9.0,<1.10`; retrain recommended. |
| ML11 | Pickle deserialisation = code execution. | Supply-chain risk. | SHA-256 check before `joblib.load`; failure ⇒ 503. |
| ML12 | Same-subject near-duplicate recordings (roughly 6 per subject in the disease set). | Random k-fold would leak; LOSO (used) does not. | Verified by reading the code: LOSO by `subject_id` parsed from `phon_R01_S##_#`. Harness re-checks parsing. |

## 3. What was actually executed
* `python ml/validation/validate_models.py artifacts` — **real run**: computes SHA-256 of both joblib files and compares to the manifest (matches by construction; the manifest was generated from these files in this session).
* `disease` / `risk` subcommands — **smoke-tested on synthetic data only** (to prove they run end-to-end and are seeded). No claim about the real models.

## 4. How to obtain the real numbers (do this on your machine)
```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python ml/validation/validate_models.py disease --n-boot 1000 --n-perm 100   # downloads UCI id=174
python ml/validation/validate_models.py risk    --n-boot 1000 --n-perm 100   # downloads UCI id=189
python ml/validation/validate_models.py artifacts
```
Outputs: `ml/validation/results_disease.json`, `results_risk.json`. Seeded (default 20261023).

### Acceptance gates I recommend before the UI keeps showing any ML probability
1. Nested-LOSO subject-level AUC lower 95 % CI bound > 0.5 **and** permutation p < 0.05.
2. Beats the dummy baseline on Brier score.
3. Risk model: FULL must beat `test_time + baseline` ablation by a margin outside the CI; otherwise the "voice" claim is unsupported.
4. Calibration: ECE reported; if > 0.1, show rank/ordinal output instead of a probability.
If gates fail, label the endpoint as a *software demonstration*, not a screening tool (the UI already carries a demo disclaimer).

## 5. Interpretation limits
Even if all gates pass: 31–42 research subjects, one recording protocol, no external cohort, no clinical outcome follow-up. This cannot support clinical use.
