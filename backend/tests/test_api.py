"""
End-to-end API test: register a clinician, create a patient, trigger a
neurological-assessment workflow through the real HTTP layer, and check the
response shape matches what the frontend dashboard depends on.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _register_and_login(role="clinician", email="doc@example.com"):
    client.post("/api/v1/auth/register", json={
        "email": email, "password": "testpass123", "full_name": "Dr Test", "role": role,
    })
    resp = client.post("/api/v1/auth/login", data={"username": email, "password": "testpass123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_health():
    assert client.get("/health").status_code == 200


def test_register_login_and_create_patient_and_run_workflow():
    token = _register_and_login()
    headers = {"Authorization": f"Bearer {token}"}

    patient_resp = client.post("/api/v1/patients", json={
        "full_name": "Test Patient", "age": 45, "sex": "female",
        "conditions": ["Epilepsy (focal and generalized)"],
    }, headers=headers)
    assert patient_resp.status_code == 200, patient_resp.text
    patient_id = patient_resp.json()["id"]

    wf_resp = client.post(f"/api/v1/workflows/neurological-assessment/{patient_id}", json={
        "trigger_reason": "api_test",
        "observations": {
            "symptoms": [{"name": "seizure", "severity": "severe", "free_text": "status epilepticus"}],
        },
    }, headers=headers)
    assert wf_resp.status_code == 200, wf_resp.text
    body = wf_resp.json()
    assert body["is_emergency"] is True
    assert "consensus" in body
    assert len(body["agent_results"]) >= 5
    assert any(a["action_type"] == "emergency_escalation" for a in body["proposed_actions"])

    workflow_id = body["workflow_id"]
    get_resp = client.get(f"/api/v1/workflows/{workflow_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["patient_id"] == patient_id

    explain_resp = client.get(f"/api/v1/patients/{patient_id}/explanations", params={"workflow_id": workflow_id}, headers=headers)
    assert explain_resp.status_code == 200
    assert "which_agents_participated" in explain_resp.json()

    alerts_resp = client.get(f"/api/v1/patients/{patient_id}/alerts", headers=headers)
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()) >= 1


def test_unauthenticated_request_rejected():
    resp = client.get("/api/v1/patients")
    assert resp.status_code == 401


def test_knowledge_search_returns_sourced_results():
    resp = client.post("/api/v1/knowledge/search", json={"query": "tremor bradykinesia", "top_k": 3})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    assert "source_urls" in results[0]
