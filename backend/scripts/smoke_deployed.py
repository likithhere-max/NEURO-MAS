"""
Smoke test for a DEPLOYED (or local) NEURO-MAS API.  Use it before a demo to prove the whole chain works:
host up -> CORS allows your Netlify origin -> Supabase login -> data reads -> ML demo endpoints -> (optional) one real workflow.

Read-only by default. It creates NOTHING unless you pass --workflow PATIENT_ID (then it runs one assessment on that existing patient).
It does not use the guarded test-project scripts in this folder; it only logs in with credentials YOU give it.

    export API_BASE=https://neuro-mas-api.onrender.com/api/v1
    export SMOKE_EMAIL=demo-clinician@example.com
    export SMOKE_PASSWORD='...'                       # never commit this
    export SITE_ORIGIN=https://YOUR-SITE.netlify.app  # the origin whose CORS you want to prove
    python scripts/smoke_deployed.py [--workflow PATIENT_ID]

Exit code 0 only if every check passes.  Needs: pip install requests
"""
import argparse
import os
import sys
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--workflow", metavar="PATIENT_ID", help="also run one neurological-assessment workflow on this existing patient (writes rows)")
ap.add_argument("--timeout", type=int, default=90, help="seconds per request (a sleeping free instance can take ~60 s to wake)")
args = ap.parse_args()

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
ROOT = BASE.rsplit("/api/", 1)[0]
EMAIL, PASSWORD = os.environ.get("SMOKE_EMAIL", ""), os.environ.get("SMOKE_PASSWORD", "")
ORIGIN = os.environ.get("SITE_ORIGIN", "")
T = args.timeout
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if detail and not ok else ""))
    return ok


def get(path, **kw):
    t0 = time.time()
    r = requests.get(f"{BASE}{path}", timeout=T, **kw)
    return r, round(time.time() - t0, 1)


# 1. host is up (and, on a sleeping free instance, wake it)
try:
    r = requests.get(f"{ROOT}/health", timeout=T)
    check("health endpoint answers ok", r.status_code == 200 and r.json().get("status") == "ok", r.text[:120])
except Exception as e:
    check("health endpoint answers ok", False, repr(e))
    sys.exit("API not reachable; fix the host first.")

# 2. CORS for the browser origin
if ORIGIN:
    r = requests.options(f"{BASE}/auth/login", timeout=T, headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    allowed = r.headers.get("access-control-allow-origin")
    check(f"CORS allows {ORIGIN}", r.status_code == 200 and allowed in (ORIGIN, "*"), f"status={r.status_code} allow-origin={allowed}")
else:
    print("SKIP CORS check (set SITE_ORIGIN to your Netlify URL)")

# 3. login
if not (EMAIL and PASSWORD):
    sys.exit("Set SMOKE_EMAIL and SMOKE_PASSWORD to continue past the health/CORS checks.")
r = requests.post(f"{BASE}/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=T)
if not check("login returns a token and role", r.status_code == 200 and "access_token" in r.json(), f"{r.status_code} {r.text[:150]}"):
    sys.exit("Login failed: check the account exists, is email-confirmed, and has a profile row.")
tok = r.json()["access_token"]
H = {"Authorization": f"Bearer {tok}"}
print(f"     role = {r.json().get('role')}")

# 4. data reads
r, s = get("/patients", headers=H)
check("list patients", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
patients = r.json() if r.status_code == 200 else []
print(f"     {len(patients)} patient(s) visible, {s}s")
r, _ = get("/knowledge/diseases", headers=H)
check("knowledge base lists diseases", r.status_code == 200 and len(r.json()) > 0, f"{r.status_code}")
if patients:
    pid = patients[0]["id"]
    for sub in ("timeline", "risk"):
        r, _ = get(f"/patients/{pid}/{sub}", headers=H)
        check(f"patient {sub} loads", r.status_code == 200, f"{r.status_code} {r.text[:120]}")

# 5. ML demo endpoints (model integrity + provenance exposure)
for name in ("parkinsons-classifier", "parkinsons-risk-classifier"):
    r, _ = get(f"/demo/{name}/info", headers=H)
    check(f"ML demo info: {name}", r.status_code == 200, f"{r.status_code} {r.text[:150]}")

# 6. optional: one real workflow (this uses Gemini and writes rows)
if args.workflow:
    body = {"trigger_reason": "manual_check_in", "observations": {
        "symptoms": [{"name": "tremor", "severity": "moderate", "free_text": "resting tremor in right hand"}],
        "medication_logs": [], "wearable": {"tremor_amplitude": 3.5, "gait_speed_m_s": 0.9}, "vitals": {}, "caregiver_notes": None}}
    t0 = time.time()
    r = requests.post(f"{BASE}/workflows/neurological-assessment/{args.workflow}", headers=H, json=body, timeout=max(T, 180))
    ok = r.status_code == 200 and "workflow_id" in r.json()
    check("workflow runs end to end", ok, f"{r.status_code} {r.text[:200]}")
    if ok:
        d = r.json()
        print(f"     {round(time.time() - t0, 1)}s, degraded={d.get('degraded')}, failed_agents={d.get('failed_agents')}")
        r2, _ = get(f"/workflows/{d['workflow_id']}", headers=H)
        check("workflow reads back (persisted)", r2.status_code == 200, f"{r2.status_code}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
