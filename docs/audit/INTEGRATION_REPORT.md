# NEURO-MAS — Frontend ↔ Backend Integration Report (Deliverable 7)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/INTEGRATION_TEST_REPORT.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

## 1. Contract check (executed)
A script loaded the FastAPI app, read every `request(...)` call in `frontend/lib/api.ts` (30 calls) and matched
method + path against the generated OpenAPI paths: **30/30 matched, 0 missing**. (Match is by path suffix and HTTP
method; it checks existence, not request/response body shapes.)

## 2. Behaviour changes the frontend had to absorb (all implemented, `tsc`/`next build` pass)
| Backend change | Frontend change |
|---|---|
| Signup needs `invite_code` for clinicians | `login/page.tsx` field; `api.register(..., invite_code?)` |
| Actions are `proposed` (not sent) unless a provider delivers | `ActionsPanel.tsx`: status labels ("proposed · not sent"), approve button only for `pending_approval`, error state |
| Workflow may be `degraded` / have `failed_agents` / `persisted:false` | `PatientContext.tsx` stores them; `patient/[id]/page.tsx` shows banners |
| Confidence now = coverage | `AgentActivityPanel.tsx` labels "coverage" vs "score" with tooltips |
| Consensus reports voters/abstained/margin | `ConsensusPanel.tsx` shows them with an explanatory note |
| Chat returns `is_emergency` | `PatientContext.tsx` marks the message |
| KB entries may lack `last_verified` / `review_status` | `evidence/page.tsx` safe fallbacks + search error state |
| 422 body reduced to loc/msg/type | `request()` already surfaces `status: body` text |

## 3. NOT verified
* Not run: any browser session against a running backend + real Supabase (the 4 live tests are skipped). The UI was only type-checked and built.
* Response **body** shapes were not validated against TypeScript types automatically (all `any`).
* Auth token flow, role-based UI gating, and Supabase RLS-driven list filtering are unverified end to end.
* The Evidence tab's citation links were not checked for liveness or correctness.
* The 2000-char chat limit and the error text for rate-limit (HTTP 429) are not surfaced with a friendly message in the UI.

## 4. Known contract smells
* `approveAction` → backend writes status `executed` even though nothing is delivered (kept for DB-constraint safety; see OUTSTANDING M4). The UI therefore may show "executed" after approval — treat as "approved by clinician".
* Frontend `request()` throws raw `status statusText: body` strings; user-facing errors are not localised.
* Hard-coded ngrok header is harmless but unnecessary in production.
