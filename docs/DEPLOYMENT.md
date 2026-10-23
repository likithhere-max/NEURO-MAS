# Deploying NEURO-MAS: Netlify (frontend) + Render (backend) + Supabase + Gemini

```
Browser --> Netlify (Next.js frontend) --HTTPS--> Render (FastAPI, Docker) --> Supabase (Auth + Postgres)
                                                                          \--> Google Gemini
```
Netlify hosts only the frontend. The Python API needs its own host (Render below; Railway/Fly work with the same `backend/Dockerfile`).
Do the steps in order. Total time: about 40 minutes the first time.

## What was checked before writing this (in a clean sandbox, no real keys)
- Frontend: type-check clean, 24/24 unit tests, production build OK, 8/8 browser smoke checks (the API is mocked in that test).
- Backend: installs from `requirements.txt` in a fresh environment, boots in `ENVIRONMENT=production`, `/health` answers, CORS allows only the origin you configure, 944 backend tests pass.
- NOT checked (no Supabase/Gemini keys here): login, database reads/writes, live Gemini. That is what `backend/scripts/smoke_deployed.py` is for (step 6). Run it. Do not assume.

## 0. Get the code onto GitHub
The latest work is on branch `phase2/advanced-ml-hardening` (not yet on GitHub). From the project folder on your machine:
```
git push -u origin phase2/advanced-ml-hardening
```
Deploy from that branch (Render: Branch field; Netlify: Production branch). If you would rather deploy `main`, merge the branch into it first; `main` does not contain the Dockerfile or `render.yaml`.

## 1. Supabase (5 min)
1. SQL Editor: run `backend/scripts/migration_phase2_privacy.sql` (adds the patient AI-consent columns). The other Phase 2 migrations are not needed with the default settings.
2. Authentication -> URL Configuration: set **Site URL** to your Netlify URL once you have it (step 4). Until then confirmation emails link to the wrong place.
3. Accounts: signup now sends a confirmation email, and clinician signup needs `CLINICIAN_INVITE_CODE`. For the review, **reuse accounts that already work** or create them in advance (Authentication -> Users -> Add user -> tick Auto Confirm User, then make sure a matching row exists in `profiles`; the app's own signup creates that row, so the easiest route is to sign up through the deployed site once and confirm the email).
4. AI consent (important): the backend blocks every call to Gemini for a patient unless `patients.external_ai_consent = true`. With it false, chat, symptom extraction and narratives fall back to local templates. For demo patients with **fictional** data, run in the SQL Editor:
   ```sql
   update public.patients set external_ai_consent = true, external_ai_consent_at = now();
   ```
   Only do this for fictional patients. Real patient data should never be here.

## 2. Backend on Render (10 min)
1. render.com -> New -> **Blueprint** -> connect the GitHub repo -> branch `phase2/advanced-ml-hardening`. It reads `render.yaml`.
2. Fill the secret variables when asked:

| Variable | Value |
|---|---|
| `SUPABASE_URL` | Supabase -> Project Settings -> API -> Project URL |
| `SUPABASE_PUBLISHABLE_KEY` | `sb_publishable_...` |
| `SUPABASE_SECRET_KEY` | `sb_secret_...` (server only; never in the frontend or Git) |
| `GEMINI_API_KEY` | Google AI Studio key |
| `CLINICIAN_INVITE_CODE` | any long random string you choose |
| `CORS_ORIGINS` | for now `["*"]`; you tighten it in step 5 |

   `ENVIRONMENT=production` and `TRUSTED_PROXY_COUNT=1` are already set by the blueprint.
3. Wait for "Live". Open `https://<your-service>.onrender.com/health` -> `{"status":"ok"}`.
4. The free plan sleeps after about 15 minutes idle; the first request then takes 30-60 s. Open `/health` 5 minutes before the review and keep a tab pinging it, or use the paid Starter plan for that day.

## 3. Frontend on Netlify (10 min)
1. app.netlify.com -> Add new site -> Import from Git -> the repo and branch.
2. Build settings: **Base directory** `frontend`; build command `npm run build`; publish directory `.next` (these match `frontend/netlify.toml`, which also pins Node 22 and the Next.js plugin).
3. Site configuration -> Environment variables: `NEXT_PUBLIC_API_BASE_URL` = `https://<your-service>.onrender.com/api/v1`.
   It is baked in at **build time**: if you change it, trigger a new deploy (Deploys -> Trigger deploy -> Clear cache and deploy site).
4. Deploy. Note the URL, e.g. `https://neuro-mas-demo.netlify.app`.

No Git? From `frontend/` on your machine: `npx netlify-cli deploy --build --prod` (log in when asked, set the same environment variable first with `netlify env:set`).

## 4. Tell Supabase and the backend the Netlify URL (5 min)
- Supabase Site URL = the Netlify URL.
- Render -> Environment: `CORS_ORIGINS` = `["https://neuro-mas-demo.netlify.app"]` (your exact URL, no trailing slash, JSON list). Save; Render redeploys.

## 5. Prove it works (5 min)
```
pip install requests
export API_BASE=https://<your-service>.onrender.com/api/v1
export SITE_ORIGIN=https://neuro-mas-demo.netlify.app
export SMOKE_EMAIL=<a working account>
export SMOKE_PASSWORD='<its password>'
python backend/scripts/smoke_deployed.py                       # read-only
python backend/scripts/smoke_deployed.py --workflow <PATIENT_ID>   # also runs one real assessment (uses Gemini, writes rows)
```
On Windows PowerShell use `$env:API_BASE="..."` instead of `export`. All lines must print PASS. Then open the Netlify URL in a private window and click through `docs/REVIEW_DEMO_GUIDE.md`.

## 6. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Browser console: "blocked by CORS" | `CORS_ORIGINS` is not exactly the Netlify origin (scheme + host, no path, no trailing slash, JSON list). Fix on Render, wait for redeploy. |
| Requests go to `localhost:8000` | `NEXT_PUBLIC_API_BASE_URL` was not set at build time. Set it, then clear cache and redeploy. |
| "Mixed content" error | The API URL is `http://`. It must be `https://`. |
| Login: 401 | wrong password, email not confirmed, or the Supabase keys on Render belong to a different project. |
| Login: 403 "no profile" | the user exists in Auth but has no row in `profiles`. Sign up through the app, or insert the row. |
| Clinician signup: 403 | missing/wrong invite code, or `CLINICIAN_INVITE_CODE` is not set on Render. |
| Everything hangs on the first click | free Render instance is waking up (30-60 s). Retry; see step 2.4. |
| Chat or symptom text looks canned, warning `external_ai_blocked_by_privacy_policy` | patient has no AI consent (step 1.4) or `GEMINI_API_KEY` is wrong. |
| `ML Lab` returns 503 | model artifact integrity check failed; check the Render log for `MANIFEST`. |
| Render deploy fails at build | open the build log; the usual cause is a wrong Dockerfile path: it must be `./backend/Dockerfile` with context `./backend`. |

## 7. Fallback if hosting misbehaves on the day
Run the backend on your laptop and expose it over HTTPS: `cd backend && uvicorn app.main:app --port 8000`, then `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`). Put that HTTPS URL + `/api/v1` into `NEXT_PUBLIC_API_BASE_URL`, redeploy Netlify, and set `CORS_ORIGINS=["*"]` in `backend/.env`. This is the setup you used before, so it is the known quantity.
