"""
End-to-end verification for the 2-day review-sprint features:
  1. POST /workflows/symptom-assessment/{patient_id} -- natural-language
     symptom entry, extraction, and the (unmodified) multi-agent pipeline.
  2. POST /patients/{patient_id}/chat -- context-aware AI Assistant.

Exercises the real HTTP API against a running
`uvicorn app.main:app --reload` server, same pattern as
scripts/test_ml_demo_endpoint.py.

Critically, this proves requirement #5 from the brief directly: two
DIFFERENT natural-language symptom descriptions must produce DIFFERENT
diagnostic candidates and risk levels -- the system must not return the
same canned result regardless of input.

Run with the server already running in another window:
    python scripts\\test_symptom_assessment_and_chat.py
"""
import requests

BASE = "http://127.0.0.1:8000/api/v1"


def check(label, resp):
    print(f"{label}: {resp.status_code}")
    if resp.status_code >= 400:
        print("  BODY:", resp.text)
        resp.raise_for_status()
    return resp


# 1. Login (reuses the existing test user).
login = check("login", requests.post(f"{BASE}/auth/login", json={
    "email": "apitest@example.com",
    "password": "TestPassword123!",
}))
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Create a fresh patient for this test run.
patient = check("create patient", requests.post(f"{BASE}/patients", headers=headers, json={
    "full_name": "Symptom Assessment Test Patient",
    "age": 62,
    "sex": "female",
})).json()
patient_id = patient["id"]
print("patient_id:", patient_id)
print()

# 3. Case A: classic Parkinsonian description in the patient's own words.
case_a = check("POST symptom-assessment (case A: tremor/slowness)", requests.post(
    f"{BASE}/workflows/symptom-assessment/{patient_id}",
    headers=headers,
    json={
        "free_text": (
            "For the last three months I have had hand tremors, mostly when "
            "resting. My movement has become slower and I sometimes feel stiff."
        ),
    },
)).json()
print("Case A extracted symptoms:", [s["name"] for s in case_a["extracted"]["symptoms"]])
print("Case A candidate conditions:", [c["preferred_name"] for c in case_a["workflow"]["agent_results"]["neuro_diagnostic_agent"]["raw_payload"]["candidate_conditions"]])
print("Case A risk bucket:", case_a["workflow"]["agent_results"]["risk_stratification_agent"]["raw_payload"]["overall_bucket"])
print("Case A is_emergency:", case_a["workflow"]["is_emergency"])
print()

# 4. Case B: a completely different, low-severity description.
case_b_patient = check("create patient B", requests.post(f"{BASE}/patients", headers=headers, json={
    "full_name": "Symptom Assessment Test Patient B",
    "age": 34,
    "sex": "male",
})).json()
case_b_id = case_b_patient["id"]

case_b = check("POST symptom-assessment (case B: mild headache)", requests.post(
    f"{BASE}/workflows/symptom-assessment/{case_b_id}",
    headers=headers,
    json={"free_text": "I've had occasional mild headaches for two weeks."},
)).json()
print("Case B extracted symptoms:", [s["name"] for s in case_b["extracted"]["symptoms"]])
print("Case B candidate conditions:", [c["preferred_name"] for c in case_b["workflow"]["agent_results"]["neuro_diagnostic_agent"]["raw_payload"]["candidate_conditions"]])
print("Case B risk bucket:", case_b["workflow"]["agent_results"]["risk_stratification_agent"]["raw_payload"]["overall_bucket"])
print()

# 5. Case C: emergency-pattern description -- must trip the deterministic
# emergency detector regardless of Gemini's paraphrasing, because the raw
# verbatim text is preserved as a symptom's free_text.
case_c_patient = check("create patient C", requests.post(f"{BASE}/patients", headers=headers, json={
    "full_name": "Symptom Assessment Test Patient C",
    "age": 58,
    "sex": "male",
})).json()
case_c_id = case_c_patient["id"]

case_c = check("POST symptom-assessment (case C: stroke-pattern emergency)", requests.post(
    f"{BASE}/workflows/symptom-assessment/{case_c_id}",
    headers=headers,
    json={"free_text": "Sudden weakness on the left side of my body and difficulty speaking."},
)).json()
print("Case C is_emergency:", case_c["workflow"]["is_emergency"])
print()

# 6. Assertions: different input -> different output (requirement #5).
assert case_a["workflow"]["agent_results"]["neuro_diagnostic_agent"]["raw_payload"]["candidate_conditions"] != \
       case_b["workflow"]["agent_results"]["neuro_diagnostic_agent"]["raw_payload"]["candidate_conditions"], \
       "case A and case B must not produce identical candidate conditions"
assert case_c["workflow"]["is_emergency"] is True, \
       "the stroke-pattern description in case C must trip the deterministic emergency detector"

# 7. Chat: ask about case A's actual assessment.
chat1 = check("POST chat (why considered)", requests.post(
    f"{BASE}/patients/{patient_id}/chat", headers=headers,
    json={"message": "Why did the system consider this condition?"},
)).json()
print("Chat answer 1:", chat1["answer"][:300])
print()

chat2 = check("POST chat (risk factors)", requests.post(
    f"{BASE}/patients/{patient_id}/chat", headers=headers,
    json={"message": "What factors increased my risk?"},
)).json()
print("Chat answer 2:", chat2["answer"][:300])
print()

history = check("GET chat history", requests.get(
    f"{BASE}/patients/{patient_id}/chat/history", headers=headers,
)).json()
assert len(history) == 4, f"expected 4 persisted messages (2 user + 2 assistant), got {len(history)}"

# 8. Chat on a patient with NO assessment yet must degrade honestly, not
# fabricate an answer.
no_assessment_patient = check("create patient D (no assessment)", requests.post(
    f"{BASE}/patients", headers=headers, json={"full_name": "No Assessment Patient"},
)).json()
chat_empty = check("POST chat (no assessment yet)", requests.post(
    f"{BASE}/patients/{no_assessment_patient['id']}/chat", headers=headers,
    json={"message": "What's wrong with me?"},
)).json()
assert "no completed symptom assessment" in chat_empty["answer"].lower() or \
       "no assessment" in chat_empty["answer"].lower(), \
       "chat must say plainly that there's no assessment yet, not fabricate one"

print("=== ALL CHECKS PASSED ===")
