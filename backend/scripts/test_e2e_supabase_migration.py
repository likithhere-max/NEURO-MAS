"""
End-to-end verification for Roadmap item 4 (Supabase migration of the
remaining 10 tables). Exercises the real HTTP API against a running
`uvicorn app.main:app --reload` server:

  login -> create patient -> add condition/allergy/metric/medications
  -> trigger a real workflow -> read it back -> check patient sub-resources

Run with the server already running in another window:
    python scripts\\test_e2e_supabase_migration.py
"""
import requests

BASE = "http://127.0.0.1:8000/api/v1"


def check(label, resp):
    print(f"{label}: {resp.status_code}")
    if resp.status_code >= 400:
        print("  BODY:", resp.text)
        resp.raise_for_status()
    return resp


# 1. Login (reuses the existing test user)
login = check("login", requests.post(f"{BASE}/auth/login", json={
    "email": "apitest@example.com",
    "password": "TestPassword123!",
}))
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Create patient
patient = check("create patient", requests.post(f"{BASE}/patients", headers=headers, json={
    "full_name": "E2E Test Patient",
    "age": 68,
    "sex": "female",
    "preferred_language": "Hindi",
}))
patient_id = patient.json()["id"]
print("patient_id:", patient_id)

# 3. Add condition / allergy / longitudinal metric / medications
check("add condition", requests.post(f"{BASE}/patients/{patient_id}/conditions", headers=headers, json={
    "disease_id": "NEURO-PD-002",
    "preferred_name": "Parkinson's disease",
}))
check("add allergy", requests.post(f"{BASE}/patients/{patient_id}/allergies", headers=headers, json={
    "substance": "haloperidol",
}))
check("add longitudinal metric", requests.post(f"{BASE}/patients/{patient_id}/longitudinal-metrics", headers=headers, json={
    "metric": "tremor_amplitude",
    "baseline_value": 2.0,
    "current_value": 2.0,
    "window": "30 days",
}))
check("add medication levodopa", requests.post(f"{BASE}/patients/{patient_id}/medications", headers=headers, json={
    "name": "levodopa", "active": True, "adherence_pct_30d": 0.65,
}))
check("add medication haloperidol", requests.post(f"{BASE}/patients/{patient_id}/medications", headers=headers, json={
    "name": "haloperidol", "active": True, "adherence_pct_30d": 0.65,
}))

# 4. Trigger a real workflow -- exercises build_patient_state (should now
# see the condition/allergy/medications/metric just added) and persist_workflow.
wf_payload = {
    "trigger_reason": "manual_check_in",
    "observations": {
        "symptoms": [{"name": "tremor", "severity": "moderate", "free_text": "resting tremor in right hand"}],
        "medication_logs": [],
        "wearable": {"tremor_amplitude": 3.5, "gait_speed_m_s": 0.9},
        "vitals": {},
        "caregiver_notes": None,
    },
}
wf = check("trigger workflow", requests.post(
    f"{BASE}/workflows/neurological-assessment/{patient_id}", headers=headers, json=wf_payload,
))
wf_data = wf.json()
workflow_id = wf_data["workflow_id"]
print("workflow_id:", workflow_id)
print("is_emergency:", wf_data["is_emergency"])
print("consensus:", wf_data["consensus"])
print()
print("--- agent reasoning summaries ---")
for name, r in wf_data["agent_results"].items():
    summary = (r.get("reasoning_summary") or "")[:200]
    print(f"  {name}: status={r['status']}  {summary}")

# 5. Read the persisted workflow back (proves persist_workflow + the Supabase
# reads in get_workflow both work, not just the in-memory response).
readback = check("GET /workflows/{id}", requests.get(f"{BASE}/workflows/{workflow_id}", headers=headers))
rb = readback.json()
print()
print("readback agent_events count:", len(rb["agent_events"]))
print("readback actions count:", len(rb["actions"]))
print("readback audit_trail count:", len(rb["audit_trail"]))

# 6. Patient sub-resource endpoints
alerts = check("GET /patients/{id}/alerts", requests.get(f"{BASE}/patients/{patient_id}/alerts", headers=headers))
print("alerts:", alerts.json())

timeline = check("GET /patients/{id}/timeline", requests.get(f"{BASE}/patients/{patient_id}/timeline", headers=headers))
print("timeline event count:", len(timeline.json()))

risk = check("GET /patients/{id}/risk", requests.get(f"{BASE}/patients/{patient_id}/risk", headers=headers))
print("risk:", risk.json())

explanations = check("GET /patients/{id}/explanations", requests.get(
    f"{BASE}/patients/{patient_id}/explanations", headers=headers, params={"workflow_id": workflow_id},
))
print("explanations.what_happened:", explanations.json().get("what_happened"))

print()
print("=== ALL CHECKS PASSED ===")