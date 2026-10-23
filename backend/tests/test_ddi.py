"""DDI source interface: every outcome is explicit and none reads as 'safe' (Phase 2, E1)."""
from datetime import datetime, timezone

import pytest

from app.services.ddi import (
    CuratedTableSource, DDIChecker, DDISource, DDIStatus, SnapshotSource, normalize_name,
)

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def snap(**kw):
    d = {"source_id": "test_db", "version": "v1", "retrieved_at": "2026-09-01T00:00:00Z", "max_age_days": 180,
         "exhaustive_pairs": True, "aliases": {"sinemet": "levodopa"}, "drugs": ["levodopa", "metformin", "aspirin"],
         "interactions": [{"drug_a": "levodopa", "drug_b": "haloperidol", "severity": "high",
                           "description": "antagonism", "reference": "ref-1"}]}
    d.update(kw)
    return SnapshotSource(data=d)


def one(report, a, b):
    return [p for p in report.pairs if {p.drug_a.lower(), p.drug_b.lower()} == {a, b}][0]


def test_known_interaction_with_alias_and_metadata():
    r = DDIChecker([snap()]).check(["Sinemet", "Haloperidol"], NOW)
    p = r.pairs[0]
    assert p.status == DDIStatus.KNOWN_INTERACTION and p.severity == "high"
    a = p.assessments[0]
    assert (a.source_id, a.source_version, a.reference) == ("test_db", "v1", "ref-1")


def test_no_interaction_found_only_from_an_exhaustive_source_and_never_called_safe():
    r = DDIChecker([snap()]).check(["levodopa", "metformin"], NOW)
    assert r.pairs[0].status == DDIStatus.NO_INTERACTION_FOUND_IN_SOURCE
    assert r.complete is True
    assert "absence_of_a_finding_is_not_evidence_of_safety" in r.warnings


def test_curated_table_can_never_answer_no_interaction():
    r = DDIChecker([CuratedTableSource()]).check(["levodopa", "warfarin"], NOW)
    # both drugs appear in rules, but not with each other -> not assessed (NOT "no interaction")
    assert r.pairs[0].status == DDIStatus.COMBINATION_NOT_ASSESSED and r.complete is False
    assert any(w.startswith("ddi_incomplete") for w in r.warnings)


def test_curated_table_finds_its_known_pair():
    r = DDIChecker([CuratedTableSource()]).check(["Sinemet", "Haldol"], NOW)
    assert r.pairs[0].status == DDIStatus.KNOWN_INTERACTION


def test_unrecognized_drug_is_reported_not_ignored():
    r = DDIChecker([snap()]).check(["levodopa", "zzzdrug"], NOW)
    assert r.pairs[0].status == DDIStatus.DRUG_NOT_RECOGNIZED
    assert r.unrecognized_drugs == ["zzzdrug"] and r.complete is False


def test_no_sources_at_all_is_source_unavailable_with_warning():
    r = DDIChecker([]).check(["a", "b"], NOW)
    assert r.pairs[0].status == DDIStatus.SOURCE_UNAVAILABLE and r.complete is False
    assert any(w.startswith("ddi_no_source_available") for w in r.warnings)


def test_missing_or_corrupt_snapshot_is_unavailable_not_empty(tmp_path):
    assert SnapshotSource(path=tmp_path / "nope.json").availability()[0] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    s = SnapshotSource(path=bad)
    ok, why = s.availability()
    assert ok is False and why.startswith("snapshot_unusable")
    r = DDIChecker([s]).check(["a", "b"], NOW)
    assert r.pairs[0].status == DDIStatus.SOURCE_UNAVAILABLE
    assert any("ddi_source_unavailable" in w for w in r.warnings)


def test_snapshot_without_interactions_key_is_rejected():
    assert SnapshotSource(data={"drugs": ["a"]}).availability()[0] is False


def test_outdated_source_does_not_support_absence_but_still_reports_known_interactions():
    old = snap(retrieved_at="2024-01-01T00:00:00Z")
    r = DDIChecker([old]).check(["levodopa", "metformin"], NOW)
    assert r.pairs[0].status == DDIStatus.SOURCE_OUTDATED and r.complete is False
    assert any(w.startswith("ddi_source_outdated") for w in r.warnings)
    k = DDIChecker([old]).check(["levodopa", "haloperidol"], NOW).pairs[0]
    assert k.status == DDIStatus.KNOWN_INTERACTION and k.assessments[0].stale is True


def test_unparseable_retrieved_at_counts_as_outdated():
    r = DDIChecker([snap(retrieved_at="yesterday-ish")]).check(["levodopa", "metformin"], NOW)
    assert r.pairs[0].status == DDIStatus.SOURCE_OUTDATED


class Boom(DDISource):
    source_id, exhaustive_pairs = "boom", True
    def recognizes(self, drug): return True
    def lookup(self, a, b): raise TimeoutError("upstream timeout")


def test_lookup_failure_is_its_own_status_and_never_reads_as_no_interaction():
    r = DDIChecker([Boom()]).check(["a", "b"], NOW)
    assert r.pairs[0].status == DDIStatus.LOOKUP_FAILED and r.complete is False
    assert r.pairs[0].assessments[0].detail == "TimeoutError"


def test_failing_source_does_not_hide_a_hit_from_a_working_one():
    r = DDIChecker([Boom(), snap()]).check(["levodopa", "haloperidol"], NOW)
    assert r.pairs[0].status == DDIStatus.KNOWN_INTERACTION
    assert {a.status for a in r.pairs[0].assessments} == {DDIStatus.LOOKUP_FAILED, DDIStatus.KNOWN_INTERACTION}


def test_highest_severity_wins_across_sources():
    lo = snap(source_id="lo", interactions=[{"drug_a": "levodopa", "drug_b": "haloperidol", "severity": "low", "description": "x"}])
    hi = snap(source_id="hi")
    assert DDIChecker([lo, hi]).check(["levodopa", "haloperidol"], NOW).pairs[0].severity == "high"


def test_pairs_are_symmetric_deduplicated_and_blank_names_ignored():
    r = DDIChecker([snap()]).check(["Levodopa", " levodopa ", "", None, "Haloperidol"], NOW)
    assert len(r.drugs) == 2 and len(r.pairs) == 1
    r2 = DDIChecker([snap()]).check(["Haloperidol", "Levodopa"], NOW)
    assert r2.pairs[0].status == r.pairs[0].status


def test_fewer_than_two_drugs_has_no_pairs():
    assert DDIChecker([snap()]).check(["levodopa"], NOW).pairs == []


def test_large_list_is_truncated_with_a_warning():
    r = DDIChecker([snap()]).check([f"drug{i}" for i in range(60)], NOW)
    assert len(r.drugs) == 40 and any(w.startswith("ddi_truncated") for w in r.warnings)


def test_report_serialises_to_plain_json():
    import json
    json.dumps(DDIChecker([snap(), CuratedTableSource()]).check(["levodopa", "haloperidol", "x"], NOW).as_dict())


def test_normalize_name():
    assert normalize_name("  Carbidopa/Levodopa 25-250 ") == "carbidopa levodopa 25 250"


@pytest.mark.asyncio
async def test_agent_exposes_ddi_report_and_incomplete_warning():
    from app.agents.medication import MedicationSafetyAgent
    from app.schemas.agent import AgentTask
    from app.schemas.patient_state import MedicationState, PatientState
    st = PatientState(patient_id="p", medication_state=MedicationState(active_medications=[{"name": "levodopa"}, {"name": "metformin"}]))
    r = await MedicationSafetyAgent().execute(AgentTask(workflow_id="w", patient_id="p", agent_name="m", reason="t", input_payload={}), st)
    rep = r.raw_payload["ddi_report"]
    assert rep["complete"] is False and "no_interaction_found_in_source" not in rep["status_counts"]
    assert any(w.startswith("ddi_incomplete") for w in r.warnings)


@pytest.mark.asyncio
async def test_agent_with_no_medication_list_is_insufficient_data_not_clean():
    from app.agents.medication import MedicationSafetyAgent
    from app.schemas.agent import AgentTask
    from app.schemas.patient_state import PatientState
    r = await MedicationSafetyAgent().execute(AgentTask(workflow_id="w", patient_id="p", agent_name="m", reason="t", input_payload={}), PatientState(patient_id="p"))
    assert r.status == "insufficient_data" and any("no_active_medication_list" in w for w in r.warnings)
