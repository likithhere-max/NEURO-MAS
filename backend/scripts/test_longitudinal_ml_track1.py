"""
End-to-end verification for Roadmap item 6, Track 1 (personalized within-
patient anomaly detection). Exercises the real HTTP API against a running
`uvicorn app.main:app --reload` server:

  login -> create patient -> add a longitudinal_metric baseline/current pair
  for tremor_amplitude -> add 5 metric_readings for tremor_amplitude (4
  normal + 1 outlier) -> add a longitudinal_metric pair for
  seizure_frequency (deteriorating, no readings) -> add only 2 readings for
  sleep_disturbance_index (insufficient history, no baseline/current) ->
  trigger a workflow -> check longitudinal_agent's output shows:
    - tremor_amplitude: classification "stable" AND is_personal_anomaly True
      with a large z-score (both signals on one metric)
    - seizure_frequency: classification "deteriorating", no anomaly
    - sleep_disturbance_index: NOT present in personal_anomalies (insufficient
      history degrades gracefully, doesn't fabricate a flag)
    - requires_human_review is True (deteriorating + personal_anomalies)
    - narrative_source is "gemini" (or template_fallback, but not crashed)

Run with the server already running in another window:
    python scripts\\test_longitudinal_ml_track1.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _testcfg  # live-script guard + env-provided password
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
    "password": _testcfg.test_password(),
}))
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Create patient
patient = check("create patient", requests.post(f"{BASE}/patients", headers=headers, json={
    "full_name": "Track1 ML Test Patient",
    "age": 71,
    "sex": "male",
    "preferred_language": "English",
}))
patient_id = patient.json()["id"]
print("patient_id:", patient_id)

# 3a. tremor_amplitude: baseline/current PAIR (stable by that heuristic) ...
check("add longitudinal metric tremor_amplitude", requests.post(
    f"{BASE}/patients/{patient_id}/longitudinal-metrics", headers=headers, json={
        "metric": "tremor_amplitude",
        "baseline_value": 2.0,
        "current_value": 2.0,
        "window": "30 days",
    }))
# ... PLUS a reading history where the latest reading is a big personal outlier.
for value in [2.0, 2.1, 1.9, 2.05, 3.5]:
    check(f"add reading tremor_amplitude={value}", requests.post(
        f"{BASE}/patients/{patient_id}/metric-readings", headers=headers, json={
            "metric": "tremor_amplitude",
            "value": value,
        }))

# 3b. seizure_frequency: baseline/current pair only, no readings -- should
# classify as deteriorating (50% increase) with no anomaly signal at all.
check("add longitudinal metric seizure_frequency", requests.post(
    f"{BASE}/patients/{patient_id}/longitudinal-metrics", headers=headers, json={
        "metric": "seizure_frequency",
        "baseline_value": 2,
        "current_value": 3,
        "window": "30 days",
    }))

# 3c. sleep_disturbance_index: only 2 readings, no longitudinal_metrics row at
# all -- insufficient history, must NOT fabricate an anomaly flag.
for value in [4.0, 4.2]:
    check(f"add reading sleep_disturbance_index={value}", requests.post(
        f"{BASE}/patients/{patient_id}/metric-readings", headers=headers, json={
            "metric": "sleep_disturbance_index",
            "value": value,
        }))

# 4. Trigger a real workflow -- exercises build_patient_state's new
# metric_readings query/grouping and longitudinal_agent's independent
# baseline/current + personal-anomaly evaluation.
wf_payload = {
    "trigger_reason": "manual_check_in",
    "observations": {
        "symptoms": [{"name": "tremor", "severity": "moderate", "free_text": "worse than usual today"}],
        "medication_logs": [],
        "wearable": {},
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

longitudinal_result = wf_data["agent_results"]["longitudinal_agent"]
print()
print("--- longitudinal_agent result ---")
print("status:", longitudinal_result["status"])
print("confidence:", longitudinal_result["confidence"])
print("requires_human_review:", longitudinal_result["requires_human_review"])
print("reasoning_summary:", longitudinal_result["reasoning_summary"])
print("raw_payload:", longitudinal_result["raw_payload"])
print()
print("--- findings ---")
for f in longitudinal_result["findings"]:
    print(" ", f["label"], "|", f["detail"], "| severity:", f["severity"], "| value:", f["value"])

# 5. Assertions -- fail loudly with a clear message if any signal is wrong.
payload = longitudinal_result["raw_payload"]
assert "tremor_amplitude" in payload["stable_metrics"], \
    f"expected tremor_amplitude classified stable, got {payload['stable_metrics']}"
assert "tremor_amplitude" in payload["personal_anomalies"], \
    f"expected tremor_amplitude flagged as personal anomaly, got {payload['personal_anomalies']}"
assert "seizure_frequency" in payload["deteriorating_metrics"], \
    f"expected seizure_frequency deteriorating, got {payload['deteriorating_metrics']}"
assert "seizure_frequency" not in payload["personal_anomalies"], \
    "seizure_frequency has no reading history -- should never appear as a personal anomaly"
assert "sleep_disturbance_index" not in payload["personal_anomalies"], \
    "sleep_disturbance_index only has 2 readings (insufficient history) -- must not fabricate an anomaly"
assert longitudinal_result["requires_human_review"] is True, \
    "requires_human_review should be True (deteriorating metric + personal anomaly)"
assert payload["narrative_source"] in ("gemini", "template_fallback"), \
    f"unexpected narrative_source: {payload['narrative_source']}"

print()
print("=== ALL CHECKS PASSED ===")