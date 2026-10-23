import pytest

from app.agents.medication import MedicationSafetyAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import MedicationState, PatientState
from app.services.clinical_rules import (
    canonical_drug_names, check_allergy_conflicts, check_interactions, unassessed_medications,
)


@pytest.mark.parametrize("meds", [
    ["levodopa", "haloperidol"],
    ["Levodopa ", "HALOPERIDOL"],
    ["Sinemet", "haloperidol"],               # brand
    ["levodopa/carbidopa", "Haldol"],         # combination string + brand
    ["Sinemet 25/100 mg", "Haldol 5 mg"],     # dose strings
    ["pyridostigmine", "gentamicin"],         # class member
    ["Mestinon", "tobramycin"],
    ["depakote", "lamictal"],
    ["valproic acid", "lamotrigine"],
    ["Tegretol", "Coumadin"],
])
def test_interactions_found_despite_naming_variants(meds):
    assert len(check_interactions(meds)) == 1, meds


def test_unrelated_drugs_have_no_interaction_but_are_reported_unassessed():
    assert check_interactions(["levodopa", "vitamin_d"]) == []
    assert unassessed_medications(["levodopa", "vitamin_d", "metformin"]) == ["vitamin_d", "metformin"]


@pytest.mark.parametrize("drug,allergy", [
    ("amoxicillin", "penicillin"),
    ("Penicillin V", "penicillin"),
    ("Augmentin", "penicillin"),
    ("PENICILLIN", " penicillin "),
])
def test_allergy_conflicts_handle_class_brand_and_case(drug, allergy):
    assert len(check_allergy_conflicts([drug], [allergy])) == 1


def test_allergy_no_false_conflict():
    assert check_allergy_conflicts(["levodopa"], ["penicillin"]) == []


def test_canonical_names_split_combinations():
    assert {"levodopa"} <= canonical_drug_names("carbidopa/levodopa 25/250")


def _task():
    return AgentTask(workflow_id="w", patient_id="p", agent_name="x", reason="t", input_payload={})


async def test_agent_reports_incomplete_coverage_instead_of_false_safety():
    state = PatientState(patient_id="p", medication_state=MedicationState(active_medications=[{"name": "metformin"}, {"name": "aspirin"}]))
    r = await MedicationSafetyAgent().execute(_task(), state)
    assert r.raw_payload["interactions"] == []
    assert set(r.raw_payload["unassessed_medications"]) == {"metformin", "aspirin"}
    assert r.confidence == 0.0  # the table covers none of them -> no claim of safety
    assert any(f.label == "interaction_check_incomplete" for f in r.findings)


async def test_agent_finds_brand_name_interaction():
    state = PatientState(patient_id="p", medication_state=MedicationState(active_medications=[{"name": "Sinemet"}, {"name": "Haldol"}]))
    r = await MedicationSafetyAgent().execute(_task(), state)
    assert len(r.raw_payload["interactions"]) == 1
    assert r.confidence == 1.0 and r.raw_payload["unassessed_medications"] == []


@pytest.mark.parametrize("value,expect_flag,expect_warning", [
    (0.6, True, False), (0.95, False, False), (60.0, False, True), (-5.0, False, True), (float("nan"), False, True),
])
async def test_adherence_unit_is_validated(value, expect_flag, expect_warning):
    state = PatientState(patient_id="p", medication_state=MedicationState(adherence_pct_30d=value))
    r = await MedicationSafetyAgent().execute(_task(), state)
    assert any(f.label == "adherence_below_threshold" for f in r.findings) is expect_flag
    assert any(w.startswith("adherence_value_invalid") for w in r.warnings) is expect_warning
