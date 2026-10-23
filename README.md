# NEURO-MAS

**A Multi-Agent Collaborative System for Autonomous and Personalized Neurological Healthcare**

A prototype platform where specialized neurological AI agents independently analyze patient data,
communicate structured findings, resolve disagreement through a weighted consensus engine, and
propose bounded, safety-gated autonomous actions (reminders, monitoring, caregiver/clinician
notification, emergency escalation) — never a restricted action like a medication change, which
always requires clinician authorization.

Built for clinicians first: every AI-generated finding is explicitly labeled as needing clinician
review before acting on a real patient. See `docs/SIMULATED_VS_REAL.md` for exactly what's real vs.
prototype in this build, and `docs/ARCHITECTURE.md` for the full system design.

## Quickstart

### Backend (FastAPI)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# configure backend/.env (Supabase + Gemini keys; ENVIRONMENT, CLINICIAN_INVITE_CODE, CORS_ORIGINS) -- see docs/audit/RUN_AND_REPRODUCTION.md
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

Run the test suite (226 passing + 4 live-Supabase tests skipped by default; see docs/audit/TEST_REPORT.md):

```bash
python -m pytest tests -q --cov=app
```

### Frontend (Next.js dashboard)

```bash
cd frontend
npm install
cp .env.local.example .env.local   # points at http://localhost:8000/api/v1 by default
npm run dev
```

Open http://localhost:3000, register as a **clinician** (needs `CLINICIAN_INVITE_CODE` unless the backend runs with `ENVIRONMENT=development`) and create a patient. Click a patient, pick a preset scenario (epilepsy
emergency / Parkinson's deterioration / routine check-in), and click **Trigger assessment** to watch
the agents run, reach (or fail to reach) consensus, and propose safety-gated actions.

## What to look at first

- `backend/app/orchestrator/orchestrator.py` — the closed loop: intake → agents → consensus →
  safety → action → audit
- `backend/app/core/safety.py` — the single authority on what's allowed to happen autonomously
- `backend/app/core/consensus.py` — the weighted-consensus/disagreement math
- `backend/app/agents/` — the 10 specialized agents
- `backend/tests/test_safety.py` — proof the safety engine can't be bypassed by confidence, emergency
  flags, or an active-emergency workflow reclassifying unrelated actions
- `frontend/app/patient/[id]/page.tsx` — the dashboard that makes agent collaboration visible

## Project layout

```
NEURO-MAS/
├── backend/
│   ├── app/
│   │   ├── agents/        # 10 specialized agents + registry
│   │   ├── core/          # messaging schema, consensus engine, safety engine, action engine, config
│   │   ├── orchestrator/  # the conductor
│   │   ├── schemas/       # Pydantic contracts (PatientState, AgentTask/Result, API I/O)
│   │   ├── services/      # knowledge base, clinical rules, LLM provider, auth, repository
│   │   ├── api/routes/    # FastAPI routers
│   ├── ml/                 # demo models, training scripts, validation harness (ml/validation)
│   ├── data/               # curated + raw disease knowledge base (from the uploaded seed pack)
│   └── tests/
├── frontend/                # Next.js 15 + Tailwind dashboard
└── docs/
    ├── ARCHITECTURE.md
    ├── audit/               # 2026-10 engineering audit: findings, maths, security, tests, how to reproduce
    ├── ROADMAP.md
    ├── SIMULATED_VS_REAL.md
    └── RESEARCH_PAPER_OUTLINE.md
```

## The three pillars (title-lock)

Every feature in this codebase is explainable through at least one of:
1. **Multi-agent collaboration** — structured messages, shared state, consensus/disagreement
2. **Autonomous operation** — bounded autonomy levels 0–5, deterministic safety engine
3. **Personalized neurological healthcare** — patient-specific state vector, ΔP-driven plans

Nothing here is general healthcare, a generic chatbot, or a disease-prediction toy — see
`docs/ARCHITECTURE.md` section 8 for the explicit check.
