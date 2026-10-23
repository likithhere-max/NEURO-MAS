"""Phase 3 regression tests: one test group per CONFIRMED defect (IDs match docs/phase3/DEFECTS_AND_FIXES.md).

Each test was written and run against the unmodified code FIRST and failed (output recorded in the defects document);
the fix then made it pass.  Expected values are stated from the mathematical/behavioural requirement, not read back
from the implementation."""
import json
import math

import pytest

from app.agents.diagnostic import NeuroDiagnosticAgent
from app.core import metric_semantics as ms
from app.core.config import get_settings
from app.core.safety import evaluate_action
from app.ml.personalized_anomaly import detect_personal_anomaly
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.ddi import CuratedTableSource, DDIChecker, SnapshotSource, normalize_name
from app.services.knowledge_base import KnowledgeBaseService


# ---------------------------------------------------------------------------------------------------- P3-D01
@pytest.mark.parametrize("field", ["disagreement", "top_confidence", "data_quality"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_D01_non_finite_gate_inputs_never_permit_autonomous_action(field, bad):
    kw = dict(disagreement=0.1, top_confidence=0.9, data_quality=0.9)
    kw[field] = bad
    d = evaluate_action("reminder", is_emergency=False, **kw)
    assert d.permitted_autonomously is False and d.requires_human_approval is True
    assert any("non-finite" in r or "not a finite" in r for r in d.reasons)


def test_D01_finite_inputs_unchanged():
    ok = evaluate_action("reminder", is_emergency=False, disagreement=0.4, top_confidence=0.6, data_quality=0.3)
    assert ok.permitted_autonomously is True                               # all three exactly on their inclusive edges
    assert evaluate_action("reminder", is_emergency=False, disagreement=0.41, top_confidence=0.9, data_quality=0.9).permitted_autonomously is False


def test_D01_emergency_escalation_still_fires_with_garbage_inputs():
    d = evaluate_action("emergency_escalation", is_emergency=True, disagreement=float("nan"), top_confidence=float("nan"), data_quality=float("nan"))
    assert d.permitted_autonomously is True                                # RULE 3 is deliberately independent of consensus state


# ---------------------------------------------------------------------------------------------------- P3-D02
@pytest.fixture
def mcid_file(tmp_path, monkeypatch):
    p = tmp_path / "mcid.json"
    p.write_text(json.dumps({"updrs_total": {"mcid": 0.3, "approved_by": "Dr Test", "source": "unit test"}}))
    s = get_settings()
    monkeypatch.setattr(s, "metric_mcid_path", str(p), raising=False)
    ms._load_mcid.cache_clear()
    yield
    ms._load_mcid.cache_clear()


@pytest.mark.parametrize("base,cur", [(1.1, 1.4), (0.1, 0.4), (2.3, 2.6), (10.1, 10.4), (7.7, 8.0), (3.3, 3.0), (1.4, 1.1)])
def test_D02_change_exactly_equal_to_mcid_is_classified_as_meeting_it(mcid_file, base, cur):
    # |delta| == MCID (0.3) up to IEEE-754 representation error must count (rule is >=); 1.4-1.1 = 0.2999999999999998
    r = ms.classify_change("updrs_total", base, cur)
    assert r["classification"] == ("deteriorating" if cur > base else "improving"), (base, cur, cur - base)


def test_D02_clearly_below_mcid_stays_stable(mcid_file):
    assert ms.classify_change("updrs_total", 10.0, 10.29)["classification"] == "stable"
    assert ms.classify_change("updrs_total", 10.0, 10.299)["classification"] == "stable"


# ---------------------------------------------------------------------------------------------------- P3-D03
@pytest.mark.parametrize("name", [
    "stride_time_variability", "step_length_variability", "gait_speed_variability",      # variability: direction not "lower is worse"
    "tremor_strength",                                                                    # tremor (higher worse) x strength (lower worse)
    "missteps_count",                                                                     # 'steps' is a substring, not the word
    "memory_complaints", "balance_loss_count", "cognitive_impairment_score",
    "seizure_free_days", "pain_free_days", "fall_free_days",                              # '-free' inverts the meaning
    "waterfall_index",                                                                    # 'fall' inside another word
    "pain_medication_adherence",                                                          # conflicting cues
])
def test_D03_ambiguous_or_misleading_names_are_not_given_a_polarity(name):
    assert ms.metric_polarity(name) == "unknown", name


@pytest.mark.parametrize("name,pol", [
    ("tremor_amplitude", "higher_worse"), ("Tremor Amplitude (mm)", "higher_worse"), ("seizure_frequency_per_week", "higher_worse"),
    ("UPDRS_part3", "higher_worse"), ("falls", "higher_worse"), ("fall_count", "higher_worse"), ("sleep_latency", "higher_worse"),
    ("gait_speed_m_s", "lower_worse"), ("Gait Speed", "lower_worse"), ("moca_score", "lower_worse"),
    ("medication_adherence_pct", "lower_worse"), ("grip_strength", "lower_worse"), ("daily_steps", "lower_worse"),
    ("sleep_hours", "lower_worse"), ("stride_length", "lower_worse"), ("balance_score", "lower_worse"),
])
def test_D03_registered_metrics_keep_their_polarity(name, pol):
    assert ms.metric_polarity(name) == pol


def test_D03_ambiguous_metric_change_is_never_called_improving_or_deteriorating():
    r = ms.classify_change("stride_time_variability", 1.0, 1.6)          # +60%
    assert r["classification"] == "changed_direction_unknown"


# ---------------------------------------------------------------------------------------------------- P3-D04
def _src(sid, sev):
    return SnapshotSource(data={"source_id": sid, "exhaustive_pairs": True, "drugs": ["x", "y"],
                                "interactions": [{"drug_a": "x", "drug_b": "y", "severity": sev}]})


@pytest.mark.parametrize("order", [0, 1])
def test_D04_unrecognised_severity_label_is_never_outranked_by_a_known_one(order):
    srcs = [_src("A", "contraindicated"), _src("B", "low")]
    if order:
        srcs.reverse()
    pair = DDIChecker(srcs).check(["x", "y"]).pairs[0]
    assert pair.severity == "contraindicated"            # the more alarming claim must survive, label kept verbatim


def test_D04_known_labels_keep_their_order():
    assert DDIChecker([_src("A", "low"), _src("B", "high"), ]).check(["x", "y"]).pairs[0].severity == "high"
    assert DDIChecker([_src("A", "medium"), _src("B", "low")]).check(["x", "y"]).pairs[0].severity == "medium"


# ---------------------------------------------------------------------------------------------------- P3-D05
def test_D05_non_latin_drug_name_is_reported_not_silently_dropped():
    r = DDIChecker([CuratedTableSource()]).check(["levodopa", "लेवोडोपा", "haloperidol"])
    assert "लेवोडोपा" in r.drugs                          # still part of the checked list
    assert "लेवोडोपा" in r.unrecognized_drugs
    assert r.complete is False and any(w.startswith("ddi_incomplete") for w in r.warnings)


def test_D05_accented_latin_name_is_folded_to_its_ascii_form():
    assert normalize_name("Lévodopa") == "levodopa" and normalize_name("  LEVODOPA!! ") == "levodopa"
    r = DDIChecker([CuratedTableSource()]).check(["Lévodopa", "Haloperidol"])
    assert r.pairs[0].status.value == "known_interaction"


# ---------------------------------------------------------------------------------------------------- P3-D06
def test_D06_stop_words_alone_never_match_a_condition():
    kb = KnowledgeBaseService()
    assert kb.match_symptoms(["the and of in my a to I have it is"]) == []
    assert kb.search("the and of in my a to") == []


def test_D06_score_counts_only_content_words():
    kb = KnowledgeBaseService()
    hits = kb.match_symptoms(["I have a tremor in my hand"])
    assert hits, "tremor/hand must still match"
    for h in hits:
        assert not ({"a", "i", "in", "my", "have"} & set(h["_matched_terms"]))
    # 'tremor' and 'hand' are the only content words: score = overlap / 2
    top = hits[0]
    assert top["_match_score"] == pytest.approx(len(top["_matched_terms"]) / 2, abs=1e-3)


# ---------------------------------------------------------------------------------------------------- P3-D07
def test_D07_float_noise_in_a_constant_history_is_not_a_zero_standard_deviation_cliff():
    # 0.1+0.2 and 0.3 are the same quantity up to representation error (stdev 2.8e-17). A deviation of 1e-7 (3e-7
    # relative) must be judged like the exactly-constant history, i.e. NOT an anomaly (tolerance: 5% of the mean).
    a = detect_personal_anomaly([0.1 + 0.2, 0.3, 0.3, 0.3], 0.3000001)
    b = detect_personal_anomaly([0.3, 0.3, 0.3, 0.3], 0.3000001)
    assert a["is_anomaly"] == b["is_anomaly"] is False and a["method"] == "zero_variance"
    assert detect_personal_anomaly([7.000000000001, 7, 7, 7], 7.01)["is_anomaly"] is False
    assert detect_personal_anomaly([7.000000000001, 7, 7, 7], 7.5)["is_anomaly"] is True       # a real 7% jump still flags


def test_D07_genuinely_small_but_real_variance_still_uses_the_t_interval():
    r = detect_personal_anomaly([1.0, 1.0001, 0.9999, 1.0002, 0.9998], 1.01)
    assert r["method"] == "z_score" and r["is_anomaly"] is True


# ---------------------------------------------------------------------------------------------------- P3-D08
class _Q:
    def __init__(self, rows): self.rows = rows
    def select(self, *_): return self
    def eq(self, *_): return self
    def order(self, *_): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows
        return r


class _FakeClient:
    def __init__(self, tables): self.tables = tables
    def table(self, name): return _Q(self.tables.get(name, []))


def test_D08_adherence_is_averaged_over_ACTIVE_medications_only():
    from app.services.repository import build_patient_state
    meds = [{"name": "A", "active": True, "schedule": "bid", "adherence_pct_30d": 0.9},
            {"name": "B", "active": True, "schedule": "qd", "adherence_pct_30d": 0.7},
            {"name": "OLD", "active": False, "schedule": "qd", "adherence_pct_30d": 0.0}]          # stopped months ago
    st = build_patient_state(_FakeClient({"medications": meds}), {"id": "p"})
    assert st.medication_state.adherence_pct_30d == pytest.approx(0.8)        # (0.9 + 0.7)/2, not (0.9+0.7+0)/3 = 0.533


def test_D08_no_active_medication_means_unknown_not_zero():
    from app.services.repository import build_patient_state
    meds = [{"name": "OLD", "active": False, "schedule": "qd", "adherence_pct_30d": 0.2}]
    assert build_patient_state(_FakeClient({"medications": meds}), {"id": "p"}).medication_state.adherence_pct_30d is None


# ---------------------------------------------------------------------------------------------------- P3-D09
def _task(payload):
    return AgentTask(workflow_id="w", patient_id="p", agent_name="neuro_diagnostic_agent", reason="t", input_payload=payload)


async def test_D09_prior_diagnosis_boost_is_not_counted_as_symptom_evidence(monkeypatch):
    import app.agents.diagnostic as dg
    from app.rag import knowledge_retriever as kr
    cands = [{"disease_id": "d1", "preferred_name": "Parkinson's disease", "matched_terms": [], "icd10_codes": [], "red_flags": [],
              "source_urls": [], "retrieval_score": 0.40, "retrieval_method": "semantic_embedding"}]
    async def fake_match(terms, known, top_k=5):
        return kr._apply_confidence([dict(c) for c in cands], known)
    monkeypatch.setattr(dg.knowledge_retriever, "match_symptoms", fake_match)
    obs = [{"modality": "symptom", "name": "tremor", "free_text": ""}, {"modality": "symptom", "name": "slow", "free_text": ""}]
    known = PatientState(patient_id="p"); known.disease_state.conditions = ["Parkinson's disease"]
    r_known = await NeuroDiagnosticAgent().execute(_task({"observations": obs}), known)
    r_new = await NeuroDiagnosticAgent().execute(_task({"observations": obs}), PatientState(patient_id="p"))
    # the displayed confidence may include the history prior (+0.15, capped) ...
    assert r_known.confidence == pytest.approx(0.45) and r_new.confidence == pytest.approx(0.30)
    # ... but the typed EVIDENCE strength (what the consensus uses) must be the retrieval evidence only: 0.40*0.75 = 0.30
    assert r_known.hypothesis_evidence[0].strength == pytest.approx(0.30)
    assert r_new.hypothesis_evidence[0].strength == pytest.approx(0.30)
    assert "history_prior" in r_known.hypothesis_evidence[0].provenance or r_known.hypothesis_evidence[0].note


# ---------------------------------------------------------------------------------------------------- P3-D10
def test_D10_ml_demo_probabilities_are_flagged_uncalibrated():
    from app.schemas.ml_demo import ParkinsonsPredictionOut, ParkinsonsRiskPredictionOut
    for cls in (ParkinsonsPredictionOut, ParkinsonsRiskPredictionOut):
        o = cls(disclaimer="d")
        assert o.probability_calibrated is False
        assert "class_weight" in o.probability_note or "uncalibrated" in o.probability_note.lower()


def test_D10_balanced_class_weight_inflates_probabilities_relative_to_prevalence():
    """Mathematical basis of D10: with class_weight='balanced' the fitted posterior is for a 50/50 prior, so mean
    predicted P(positive) is far above the true 10% prevalence even for a perfectly specified model."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n = 20000; y = (rng.random(n) < 0.10).astype(int); X = rng.normal(size=(n, 1)) + 1.0 * y[:, None]
    plain = LogisticRegression().fit(X, y).predict_proba(X)[:, 1].mean()
    bal = LogisticRegression(class_weight="balanced").fit(X, y).predict_proba(X)[:, 1].mean()
    assert abs(plain - 0.10) < 0.01 and bal > 0.25


# ---------------------------------------------------------------------------------------------------- P3-D11
async def test_D11_personalization_confidence_is_declared_a_placeholder(monkeypatch):
    from app.agents.personalization import PersonalizationAgent
    async def none(*a, **k): return None
    monkeypatch.setattr("app.agents.personalization._explain_personalization_with_gemini", none)
    r = await PersonalizationAgent().execute(AgentTask(workflow_id="w", patient_id="p", agent_name="personalization_agent", reason="t"),
                                             PatientState(patient_id="p"))
    assert "confidence_is_a_fixed_placeholder_not_a_measure" in r.warnings


# ---------------------------------------------------------------------------------------------------- P3-D12
async def _run_with_intake(monkeypatch, intake_status, intake_quality=None):
    import app.orchestrator.orchestrator as orch
    from app.agents.registry import AGENT_DEPENDENCIES, get_agent
    from app.schemas.agent import AgentResult

    class _S:
        agent_timeout_seconds, workflow_deadline_seconds, max_agent_concurrency = 5.0, 30.0, 4
        consensus_disagreement_threshold, min_confidence_for_autonomous_action = 0.4, 0.6
    monkeypatch.setattr(orch, "get_settings", lambda: _S())
    import app.core.safety as safety
    monkeypatch.setattr(safety, "get_settings", lambda: _S())

    for name in AGENT_DEPENDENCIES:
        async def fake(task, state, _n=name):
            if _n == "data_intake_agent":
                if intake_status == "failed":
                    raise RuntimeError("boom")
                return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.COMPLETED,
                                   raw_payload={"data_quality": intake_quality})
            if _n == "neuro_diagnostic_agent":
                return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.COMPLETED, confidence=0.9)
            if _n == "personalization_agent":
                return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.COMPLETED,
                                   raw_payload={"monitoring_plan": ["Daily gait/balance check-in reminder"], "personalization_reasons": []})
            return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.COMPLETED)
        monkeypatch.setattr(get_agent(name), "execute", fake)
    return await orch.run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                   input_payload={"symptoms": [{"name": "mild headache"}]})


async def test_D12_control_good_data_quality_allows_the_low_risk_action(monkeypatch):
    r = await _run_with_intake(monkeypatch, "completed", 0.8)
    acts = [a for a in r.proposed_actions if a.action_type != "emergency_escalation"]
    assert acts and all(a.requires_human_approval is False for a in acts)         # the gate is not blocking everything


async def test_D12_unknown_data_quality_is_not_assumed_adequate(monkeypatch):
    """If the intake agent failed, data quality is UNKNOWN. The orchestrator used to substitute 0.5, which passes the
    0.3 safety gate, so low-risk actions still ran autonomously on data nobody had assessed."""
    r = await _run_with_intake(monkeypatch, "failed")
    acts = [a for a in r.proposed_actions if a.action_type != "emergency_escalation"]
    assert acts and all(a.requires_human_approval is True for a in acts)
    assert any("data quality" in str(a.safety_reasons).lower() for a in acts)
