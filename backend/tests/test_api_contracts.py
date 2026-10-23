"""API response contracts (Phase 2, H).

1. Shape/type contract of the workflow response the frontend consumes.
2. Vocabulary contract: every status the backend can emit has frontend wording (parsed from frontend/lib/labels.ts),
   so a new backend state can never silently render as 'executed'/neutral.
3. Strict-JSON contract: no NaN/Infinity ever reaches a client.
These run against in-memory fakes; they verify OUR serialisation, not a live deployment."""
import json
import re
from pathlib import Path

import pytest

from app.schemas.agent import AgentStatus
from tests.test_workflow_routes import _post, setup  # noqa: F401  (fixture re-export)

LABELS_TS = (Path(__file__).resolve().parents[2] / "frontend" / "lib" / "labels.ts").read_text()


def _ts_keys(block_name: str) -> set[str]:
    m = re.search(rf"export const {block_name}[^=]*=\s*\{{(.*?)\n\}};", LABELS_TS, re.S)
    assert m, f"{block_name} not found in labels.ts"
    return set(re.findall(r'^\s*"?([a-z_]+)"?\s*:', m.group(1), re.M))


def _strict(text: str):
    def bad(c):
        raise ValueError(f"non-finite JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def test_workflow_response_shape_and_types(setup):  # noqa: F811
    body = _strict(_post("mild headache").text)
    assert isinstance(body["workflow_id"], str) and isinstance(body["patient_id"], str)
    assert isinstance(body["persisted"], bool) and isinstance(body["degraded"], bool)
    assert isinstance(body["failed_agents"], list) and isinstance(body["timed_out_agents"], list)
    assert isinstance(body["is_emergency"], bool) and isinstance(body["emergency_screen"], dict)
    assert isinstance(body["audit_trail"], list) and isinstance(body["proposed_actions"], list)

    c = body["consensus"]
    if c is not None:
        assert set(c) >= {"top_hypothesis", "disagreement", "hypotheses", "margin", "action_permitted", "reason",
                          "score_kind", "calibrated", "probability", "ties", "caveats", "non_voting_agents"}
        assert c["calibrated"] is False and c["probability"] is None, "no probability may be published uncalibrated"
        assert c["score_kind"] == "uncalibrated_support_index"
        for h in c["hypotheses"]:
            assert set(h) >= {"hypothesis", "consensus_score", "contributing_agents", "status", "contradiction_index"}
            assert 0.0 <= h["consensus_score"] <= 1.0

    for name, r in body["agent_results"].items():
        assert set(r) >= {"status", "confidence", "reasoning_summary", "warnings"}, name
        assert r["status"] in {s.value for s in AgentStatus}, (name, r["status"])

    for a in body["proposed_actions"]:
        assert set(a) >= {"action_id", "action_type", "autonomy_level", "risk_level", "status", "reason",
                          "requires_human_approval"}


def test_every_agent_status_has_frontend_styling():
    missing = {s.value for s in AgentStatus} - _ts_keys("AGENT_STATUS_STYLE")
    assert not missing, f"frontend/lib/labels.ts AGENT_STATUS_STYLE lacks: {sorted(missing)}"


def test_every_emittable_action_status_has_frontend_wording():
    # Statuses produced by core/actions.py and the notification outbox mapping.
    from app.services import notification_dispatch as nd
    src = Path(nd.__file__).read_text() + (Path(nd.__file__).parent.parent / "core" / "actions.py").read_text()
    emitted = {"proposed", "pending_approval", "executed", "rejected", "sent_unconfirmed", "delivery_failed", "queued"}
    for s in emitted:
        assert f'"{s}"' in src, f"{s} is no longer emitted anywhere; update this contract"
    missing = emitted - _ts_keys("ACTION_STATUS_LABEL")
    assert not missing, f"frontend/lib/labels.ts ACTION_STATUS_LABEL lacks: {sorted(missing)}"


def test_ml_demo_abstention_contract():
    from app.schemas.ml_demo import ParkinsonsPredictionOut, ParkinsonsRiskPredictionOut
    out = ParkinsonsPredictionOut(abstained=True, domain={"status": "abstain_out_of_domain"}, disclaimer="d")
    d = out.model_dump()
    assert d["predicted_label"] is None and d["probability_parkinsons"] is None and d["abstained"] is True
    r = ParkinsonsRiskPredictionOut(abstained=True, domain={"status": "abstain_invalid_input"}, disclaimer="d").model_dump()
    assert r["predicted_label"] is None and r["probability_worsening"] is None


@pytest.mark.parametrize("field,payload", [
    ("vitals", '{"heart_rate": NaN}'),
    ("vitals", '{"heart_rate": Infinity}'),
    ("wearable", '{"tremor": {"band": [1.0, -Infinity]}}'),
    ("vitals", "{" + ",".join(f'"k{i}": 1' for i in range(300)) + "}"),          # too many keys
    ("vitals", '{"a": {"b": {"c": {"d": {"e": 1}}}}}'),                           # too deep
    ("vitals", '{"note": "' + "x" * 5000 + '"}'),                                  # oversized string
])
def test_untyped_observation_dicts_are_bounded_and_finite(setup, field, payload):  # noqa: F811
    from tests.test_workflow_routes import client
    other = "vitals" if field == "wearable" else "wearable"
    body = '{"trigger_reason":"t","observations":{"%s":%s,"%s":{}}}' % (field, payload, other)
    r = client.post("/api/v1/workflows/neurological-assessment/pat1", content=body,
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, (field, payload[:40], r.status_code)


def test_ordinary_vitals_and_wearable_still_accepted(setup):  # noqa: F811
    from tests.test_workflow_routes import client
    r = client.post("/api/v1/workflows/neurological-assessment/pat1", json={
        "trigger_reason": "t", "observations": {"vitals": {"heart_rate": 72, "bp": {"sys": 120, "dia": 80}, "note": "ok"},
                                                "wearable": {"tremor_amplitude": 0.4, "steps": 3200, "worn": True}}})
    assert r.status_code == 200
