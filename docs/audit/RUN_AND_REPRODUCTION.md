# NEURO-MAS — Run & Reproduction Guide (Deliverable 9)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/RUN_AND_REPRODUCTION.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

## 1. Backend
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # runtime
pip install -r requirements-dev.txt        # pytest, pytest-cov, ruff
```
Environment (`backend/.env`, never commit):
```
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SECRET_KEY=...            # service role; server only
GEMINI_API_KEY=...
ENVIRONMENT=development            # production on any deployed server
CLINICIAN_INVITE_CODE=             # REQUIRED in production for clinician signup
CORS_ORIGINS=["http://localhost:3000"]   # JSON list; pin to your frontend in production
AGENT_TIMEOUT_SECONDS=60
AUTH_RATE_LIMIT_PER_MINUTE=10
```
Run: `uvicorn app.main:app --reload --port 8000` → docs at `http://localhost:8000/docs`.
Note: the old `python -m app.seed.demo_scenarios` step no longer exists (dead SQLAlchemy layer removed). Create demo patients through the UI/API.

## 2. Frontend
```bash
cd frontend
cp .env.local.example .env.local     # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
npm install
npm run dev                          # http://localhost:3000
npx tsc --noEmit && npx next build   # type-check + production build
```
Clinician registration in the UI needs the invite code (or `ENVIRONMENT=development` on the backend).

## 3. Reproduce the audit results
```bash
cd backend
python -m pytest tests -q --cov=app                 # expect: 226 passed, 4 skipped, ~81 % coverage
ruff check --select F,E9 app tests
python ml/validation/validate_models.py artifacts   # sha256 vs ml/models/MANIFEST.json
```
Live tests (only against a **test** Supabase project, never production):
```bash
NEUROMAS_LIVE_TESTS=1 python -m pytest tests/test_api.py -q
```
ML validation on real data (needs internet; this was not possible in my sandbox):
```bash
python ml/validation/validate_models.py disease --n-boot 1000 --n-perm 100
python ml/validation/validate_models.py risk    --n-boot 1000 --n-perm 100
```
Seeds default to 20261023 (`--seed`).

## 4. Regenerate the model manifest after retraining
```bash
python - <<'PY'
import hashlib, json, pathlib
d = pathlib.Path("ml/models"); out = {}
for f in sorted(d.glob("*.joblib")):
    out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(out, indent=2))
PY
```
Copy the hashes into `ml/models/MANIFEST.json` (check its existing structure first) and record the scikit-learn version used for training.

## 5. Quick manual safety checks (backend running)
* `POST /auth/signup` with `role=clinician` and no code in production → 403.
* Submit a symptom text "my face is drooping and I can't speak" → workflow `is_emergency: true`; action status `proposed` with `NOT_DELIVERED` in `safety_reasons` (no provider configured).
* Chat message with an emergency phrase → fixed emergency reply, `is_emergency: true`, no model call.
