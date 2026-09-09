"""
End-to-end verification for Roadmap item 6, Track 2 (UCI Parkinson's
demonstration classifier). Exercises the real HTTP API against a running
`uvicorn app.main:app --reload` server. Deliberately does NOT touch the
patients/workflows endpoints at all -- this proves the demo classifier is
fully decoupled from the live patient pipeline, exactly as required by
claude/neuro-mas-architecture-v2.md's two-track design.

Uses two REAL rows from the UCI Parkinson's dataset (one confirmed
Parkinson's case, one confirmed healthy case) -- not fabricated numbers.

Run with the server already running in another window:
    python scripts\\test_ml_demo_endpoint.py
"""
import requests

BASE = "http://127.0.0.1:8000/api/v1"


def check(label, resp):
    print(f"{label}: {resp.status_code}")
    if resp.status_code >= 400:
        print("  BODY:", resp.text)
        resp.raise_for_status()
    return resp


# 1. Login (reuses the existing test user -- auth is required for this
# endpoint, but nothing about the prediction ties back to this user/patient).
login = check("login", requests.post(f"{BASE}/auth/login", json={
    "email": "apitest@example.com",
    "password": "TestPassword123!",
}))
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Model info endpoint -- confirms the saved metadata (metrics, thresholds,
# dataset citation, disclaimer) is being served correctly.
info = check("GET model info", requests.get(f"{BASE}/demo/parkinsons-classifier/info", headers=headers))
info_data = info.json()
print("model:", info_data["model"])
print("dataset:", info_data["dataset"])
# NOTE: this was "test_metrics" until the LOSO revision replaced the single
# train/test split with pooled leave-one-subject-out metrics. The route returns
# the saved metadata verbatim, so the key follows whatever training wrote.
print("loso_metrics:", info_data["loso_metrics"])
print("evaluation_strategy:", info_data["evaluation_strategy"])
print("disclaimer:", info_data["disclaimer"])
print()

# 3. A REAL confirmed-Parkinson's row from the dataset (status=1).
positive_row = {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 7e-05,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654,
}

# 4. A REAL confirmed-healthy row from the dataset (status=0).
negative_row = {
    "MDVP:Fo(Hz)": 197.076,
    "MDVP:Fhi(Hz)": 206.896,
    "MDVP:Flo(Hz)": 192.055,
    "MDVP:Jitter(%)": 0.00289,
    "MDVP:Jitter(Abs)": 1e-05,
    "MDVP:RAP": 0.00166,
    "MDVP:PPQ": 0.00168,
    "Jitter:DDP": 0.00498,
    "MDVP:Shimmer": 0.01098,
    "MDVP:Shimmer(dB)": 0.097,
    "Shimmer:APQ3": 0.00563,
    "Shimmer:APQ5": 0.0068,
    "MDVP:APQ": 0.00802,
    "Shimmer:DDA": 0.01689,
    "NHR": 0.00339,
    "HNR": 26.775,
    "RPDE": 0.422229,
    "DFA": 0.741367,
    "spread1": -7.3483,
    "spread2": 0.177551,
    "D2": 1.743867,
    "PPE": 0.085569,
}

pos_result = check("POST classify (real positive row)", requests.post(
    f"{BASE}/demo/parkinsons-classifier", headers=headers, json=positive_row,
)).json()
print("positive row prediction:", pos_result)
print()

neg_result = check("POST classify (real negative row)", requests.post(
    f"{BASE}/demo/parkinsons-classifier", headers=headers, json=negative_row,
)).json()
print("negative row prediction:", neg_result)
print()

# 5. Assertions.
assert pos_result["predicted_label"] == "parkinsons", \
    f"expected the real Parkinson's row to predict 'parkinsons', got {pos_result['predicted_label']}"
assert neg_result["predicted_label"] == "healthy", \
    f"expected the real healthy row to predict 'healthy', got {neg_result['predicted_label']}"
assert "DEMONSTRATION" in pos_result["disclaimer"] and "DEMONSTRATION" in neg_result["disclaimer"], \
    "every prediction response must carry the demonstration disclaimer"

print("=== ALL CHECKS PASSED ===")
print()
print("Reminder: this endpoint was never called from patients.py, workflows.py,")
print("repository.py, or any agent -- fully decoupled from the live pipeline,")
print("as intended.")