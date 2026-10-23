"""The numbers quoted in docs/phase5 must equal what the shipped artifacts and the committed planning file say.

These are arithmetic/consistency checks on shipped artifacts and planning output.  They are NOT checks that any figure is
reproducible from the real data (which is unavailable)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ml.validation.phase4a import artifact_audit

BACKEND = Path(__file__).resolve().parents[2]
DOCS = BACKEND.parent / "docs" / "phase5"


def _text(name: str) -> str:
    p = DOCS / name
    if not p.exists():
        pytest.fail(f"missing required deliverable docs/phase5/{name}")
    return p.read_text()


@pytest.fixture(scope="module")
def audit():
    return artifact_audit()


def test_all_required_deliverables_exist():
    for n in ("BASELINE_AUDIT.md", "DATA_INTEGRITY_AUDIT.md", "VALIDATION_PROTOCOL_AUDIT.md", "STATISTICAL_EVIDENCE_REPORT.md",
              "GENERALIZATION_AND_ROBUSTNESS.md", "CLINICAL_INTENDED_USE.md", "CLINICAL_EVIDENCE_GAPS.md",
              "SAFETY_AND_HUMAN_OVERSIGHT.md", "FINAL_READINESS_REPORT.md", "REPRODUCIBILITY.md"):
        assert (DOCS / n).exists(), n


def test_majority_baselines_and_gains_match_the_recomputed_audit(audit):
    d, r = audit["models"]["disease"], audit["models"]["risk"]
    assert round(d["reconstructed_confusion_matrices"]["RandomForest_Top10Features"]["majority_class_accuracy"], 4) == 0.7538
    assert round(r["reconstructed_confusion_matrices"]["LogisticRegression"]["majority_class_accuracy"], 4) == 0.6182
    assert round(d["winner_accuracy_minus_majority_accuracy"], 4) == 0.0205
    assert round(r["winner_accuracy_minus_majority_accuracy"], 4) == 0.0122
    t = _text("BASELINE_AUDIT.md")
    for token in ("0.7538", "0.6182", "+0.0205", "+0.0122"):
        assert token in t


def test_confusion_matrices_quoted_in_the_docs_match(audit):
    d = audit["models"]["disease"]["reconstructed_confusion_matrices"]["RandomForest_Top10Features"]
    r = audit["models"]["risk"]["reconstructed_confusion_matrices"]["LogisticRegression"]
    assert (d["TP"], d["FN"], d["FP"], d["TN"]) == (130, 17, 27, 21)
    assert (r["TP"], r["FN"], r["FP"], r["TN"]) == (1325, 890, 1254, 2332)
    t = _text("BASELINE_AUDIT.md")
    for v in ("130", "17", "27", "21", "1,325", "890", "1,254", "2,332"):
        assert v in t


def test_the_untraceable_0_6178_figure_is_nowhere_in_the_code_or_metadata():
    hits = []
    for p in list((BACKEND / "app").rglob("*.py")) + list((BACKEND / "ml").rglob("*.py")) + list((BACKEND / "ml").rglob("*.json")):
        if "results" in p.parts:
            continue
        if "0.6178" in p.read_text(errors="ignore"):
            hits.append(str(p))
    assert hits == []


def test_row_and_subject_counts_in_the_docs_match_the_artifacts(audit):
    d, r = audit["models"]["disease"], audit["models"]["risk"]
    assert (d["training_rows_recorded_in_scaler"], r["training_rows_recorded_in_scaler"]) == (195, 5801)
    assert (d["metadata_n_subjects"], d["publisher_n_subjects"]) == (32, 31)
    assert r["rows_dropped_beyond_first_visit_per_subject"] == 32 and r["expected_rows_if_only_first_visits_dropped"] == 5833
    t = _text("DATA_INTEGRITY_AUDIT.md")
    for token in ("5,801", "5,833", "32", "31"):
        assert token in t


def test_power_planning_numbers_in_the_report_match_the_committed_file():
    p = json.loads((BACKEND / "ml/validation/results/phase5/power_planning.json").read_text())
    assert p["evidence_status"] == "DESIGN_STAGE_PLANNING_NOT_EVIDENCE" and p["clinically_validated"] is False
    t = _text("STATISTICAL_EVIDENCE_REPORT.md")
    a = p["analytic_disease_precision"]
    assert f"{a['se_under_null']:.3f}" in t
    assert f"{a['minimum_detectable_auc_vs_0_5_at_80pct_power']:.3f}"[:4] in t  # 0.835
    for row in p["risk_voice_value_simulation"]["rows"]:
        assert f"{row['mean_auc_difference_full_minus_time_baseline']:+.3f}" in t, row
    for row in p["prevalence_sensitivity"]["operating_points"]["disease_RandomForest_Top10Features"]["by_prevalence"]:
        assert f"{row['ppv']:.3f}" in t and f"{row['npv']:.3f}" in t


def test_no_phase5_document_claims_clinical_validity():
    banned = re.compile(r"\b(clinically (validated|ready|proven)|externally validated|prospectively validated)\b", re.I)
    for p in DOCS.glob("*.md"):
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if banned.search(line):
                if line.lstrip().startswith('* "'):   # the explicit list of statements that must NOT be made
                    continue
                low = line.lower()
                assert any(w in low for w in ("not", "no ", "**no**", "never", "cannot", "neither", "nothing", "without")), f"{p.name}:{ln}: {line}"
