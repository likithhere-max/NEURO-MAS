from app.services.clinical_rules import check_allergy_conflicts, check_interactions, detect_emergency


def test_detect_emergency_matches_status_epilepticus():
    is_emergency, matched = detect_emergency(["status epilepticus, seizure not stopping"])
    assert is_emergency is True
    assert matched


def test_detect_emergency_no_match_on_benign_text():
    is_emergency, matched = detect_emergency(["mild tingling in left hand, otherwise feeling fine"])
    assert is_emergency is False
    assert matched == []


def test_known_interaction_detected():
    hits = check_interactions(["levodopa", "haloperidol"])
    assert len(hits) == 1
    assert hits[0]["severity"] == "high"


def test_no_interaction_for_unrelated_drugs():
    hits = check_interactions(["levodopa", "vitamin_d"])
    assert hits == []


def test_allergy_conflict_detected_case_insensitive():
    conflicts = check_allergy_conflicts(["Penicillin"], ["penicillin"])
    assert len(conflicts) == 1
