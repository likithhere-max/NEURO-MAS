"""Experiment registry, claims register and the Phase 6 documents.

The registers are derived from / checked against the committed result files, so a status or evidence class cannot be edited upward
without the underlying file saying so.  The documents may quote only numbers that the planning files and code produce.
"""
import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from ml.validation import claims_register as CR
from ml.validation import experiment_registry as ER

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
DOCS = REPO / "docs" / "phase6"
REQUIRED_DOCS = ["DATA_AUTHENTICATION.md", "BASELINE_REPORT.md", "MODEL_AUDIT.md", "MODEL_COMPARISON.md", "ERROR_ANALYSIS.md",
                 "CLINICAL_INTENDED_USE.md", "CLINICAL_VALIDATION_PROTOCOL.md", "SAFETY_CASE.md", "CLINICAL_READINESS_GAPS.md",
                 "ARCHITECTURE_AUDIT.md", "FINAL_READINESS_REPORT.md"]


# --- experiment registry ---------------------------------------------------------------------------------------------------------
def test_committed_registry_matches_the_files_it_describes():
    assert ER.check() == []


def test_registry_lists_failed_blocked_synthetic_planning_and_not_run_runs():
    got = {r["evidence_class"] for r in json.loads(ER.REGISTRY_PATH.read_text())["experiments"]}
    assert {"BLOCKED", "SYNTHETIC_ENGINEERING_TEST", "PLANNING_ESTIMATE", "NOT_RUN"} <= got


def test_no_experiment_is_verified_while_no_authenticated_real_data_result_exists():
    for r in json.loads(ER.REGISTRY_PATH.read_text())["experiments"]:
        assert r["evidence_class"] not in ER.VERIFIED, r["id"]
        assert r["clinically_validated"] is False, r["id"]


def test_derived_class_cannot_be_upgraded_from_the_label():
    assert ER.derive_class({"status": "completed", "evidence_status": "SYNTHETIC_FIXTURE_NOT_EVIDENCE"}, ["SYNTHETIC_ENGINEERING_TEST"]) == "SYNTHETIC_ENGINEERING_TEST"
    assert ER.derive_class({"status": "not_run"}, ["NOT_RUN"]) == "NOT_RUN"
    ok = "REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT"
    assert ER.derive_class({"status": "completed", "evidence_status": ok}, []) == "VERIFIED_INTERNAL_EVIDENCE"
    for suffix in ("+FAST_NOT_REGISTERED_PROTOCOL", "+EXPLORATORY_EXTENSION_P6_1"):
        assert ER.derive_class({"status": "completed", "evidence_status": ok + suffix}, []) == "EXPLORATORY"
    assert ER.derive_class({"status": "completed", "evidence_status": "UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES"}, []) == "EXPLORATORY"


def test_registry_detects_a_changed_output_file(tmp_path, monkeypatch):
    reg = tmp_path / "experiment_registry.json"
    data = json.loads(ER.REGISTRY_PATH.read_text())
    data["experiments"][0]["outputs"][0]["sha256"] = "0" * 64
    reg.write_text(json.dumps(data))
    probs = ER.check(reg)
    assert any("SHA-256 drift" in p for p in probs)


def test_registry_detects_an_upgraded_status_or_class(tmp_path):
    data = json.loads(ER.REGISTRY_PATH.read_text())
    nr = next(e for e in data["experiments"] if e["evidence_class"] == "NOT_RUN")
    nr["evidence_class"] = "VERIFIED_INTERNAL_EVIDENCE"
    reg = tmp_path / "r.json"
    reg.write_text(json.dumps(data))
    probs = ER.check(reg)
    assert any("evidence_class" in p for p in probs) and any("VERIFIED class without" in p for p in probs)


def test_every_registered_output_is_committed_history():
    for r in json.loads(ER.REGISTRY_PATH.read_text())["experiments"]:
        for o in r["outputs"]:
            assert (BACKEND / o["path"]).exists(), o["path"]


# --- claims register ---------------------------------------------------------------------------------------------------------
REPORT = DOCS / "FINAL_READINESS_REPORT.md"


def test_committed_claims_register_is_consistent_and_every_claim_is_in_the_final_report():
    assert CR.check(report=REPORT) == []


def test_no_claim_is_verified():
    for c in json.loads(CR.REGISTER_PATH.read_text())["claims"]:
        assert c["class"] not in ER.VERIFIED, c["id"]


def _with(tmp_path, mutate):
    d = json.loads(CR.REGISTER_PATH.read_text())
    mutate(d)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    return CR.check(p)


def test_a_claim_cannot_be_classed_above_its_evidence(tmp_path):
    def up(d):
        next(c for c in d["claims"] if c["id"] == "CL01")["class"] = "VERIFIED_INTERNAL_EVIDENCE"
    assert any("without a referenced experiment" in p for p in _with(tmp_path, up))

    def mismatch(d):
        next(c for c in d["claims"] if c["id"] == "CL07")["class"] = "NOT_RUN"
    assert any("references experiments of class" in p for p in _with(tmp_path, mismatch))


def test_unknown_class_missing_experiment_missing_file_and_missing_test_are_reported(tmp_path):
    def bad(d):
        d["claims"][0]["class"] = "PROVEN"
        d["claims"][1]["evidence"].append({"type": "experiment", "ref": "E-DOES-NOT-EXIST"})
        d["claims"][2]["evidence"].append({"type": "file", "ref": "docs/phase6/NOPE.md"})
        d["claims"][3]["evidence"].append({"type": "test", "ref": "tests/phase6/test_p6_registers_and_docs.py::test_nope"})
    probs = _with(tmp_path, bad)
    assert any("unknown class" in p for p in probs) and any("not in the experiment registry" in p for p in probs)
    assert any("missing" in p for p in probs) and any("not found" in p for p in probs)


def test_a_claim_asserting_clinical_validity_is_rejected(tmp_path):
    def bad(d):
        d["claims"][0]["claim"] = "The disease model is clinically validated."
    assert any("without negation" in p for p in _with(tmp_path, bad))


def test_a_claim_missing_from_the_final_report_is_reported(tmp_path):
    rep = tmp_path / "FINAL_READINESS_REPORT.md"
    rep.write_text(REPORT.read_text().replace("CL21", "CLXX"))
    assert any("CL21" in p for p in CR.check(report=rep))


# --- documents -------------------------------------------------------------------------------------------------------------------
def test_all_phase6_deliverable_documents_exist():
    for name in REQUIRED_DOCS:
        assert (DOCS / name).exists(), name


def test_no_phase6_document_claims_clinical_validity_without_negation():
    banned = re.compile(r"\b(clinically (validated|ready|proven)|externally validated|prospectively validated)\b", re.I)
    for p in DOCS.glob("*.md"):
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if banned.search(line):
                low = line.lower()
                assert any(w in low for w in CR.NEGATIONS), f"{p.name}:{ln}: {line}"


def test_documents_state_that_real_data_validation_was_not_run():
    for name in ("BASELINE_REPORT.md", "MODEL_COMPARISON.md", "ERROR_ANALYSIS.md", "DATA_AUTHENTICATION.md"):
        t = (DOCS / name).read_text()
        assert "NOT_RUN" in t or "BLOCKED" in t, name


def test_documented_protocol_hashes_are_the_pinned_ones():
    from ml.validation import phase6 as PH6
    t = (DOCS / "MODEL_COMPARISON.md").read_text()
    assert PH6.PROTOCOL_P6_SHA256[:8] in t and PH6.PROTOCOL_P6_SHA256[-6:] in t
    b = (DOCS / "BASELINE_REPORT.md").read_text()
    assert "0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42" in b


def test_model_comparison_voice_rates_equal_the_planning_file():
    rows = json.loads((BACKEND / "ml/validation/results/phase6/voice_sample_size_planning.json").read_text())["simulation"]["rows"]
    t = (DOCS / "MODEL_COMPARISON.md").read_text()
    by = {(r["voice_signal_generator_units"], r["patients"]): r for r in rows}
    for pat in (42, 84, 168):
        pct = f"{by[(0.5, pat)]['detection_rate_interval_above_0'] * 100:.0f}%"
        assert pct in t, (pat, pct)
    for pat in (42,):
        assert f"{by[(1.0, pat)]['detection_rate_interval_above_0'] * 100:.0f}%" in t
    assert f"{by[(0.5, 42)]['mean_auc_difference_full_minus_time_baseline']:.2f}" in t.replace("+", "").replace("about ", "")


def test_the_phase6_no_data_records_are_unchanged_not_run_records():
    for rel in ("baseline_no_data", "extension_no_data"):
        m = json.loads((BACKEND / f"ml/validation/results/phase6/{rel}/experiment_manifest.json").read_text())
        assert m["overall_status"] == "not_run" and m["exit_code"] == 2 and m["clinically_validated"] is False
    r = json.loads((BACKEND / "ml/validation/results/phase6/report_no_data/phase6_NOT_RUN.json").read_text())
    assert r["status"] == "not_run" and r["clinically_validated"] is False
