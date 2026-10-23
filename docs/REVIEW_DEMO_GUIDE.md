# Review-day guide: demo flow, honest talking points, likely questions

## Before you walk in (15 min, do it at home)
- [ ] Backend awake: open `https://<api>/health` (free Render sleeps). Keep a tab pinging it.
- [ ] `python backend/scripts/smoke_deployed.py --workflow <PATIENT_ID>` prints all PASS (see `docs/DEPLOYMENT.md` step 5).
- [ ] Log in once on the Netlify URL in the browser you will use. Have the clinician and the patient account details ready.
- [ ] Demo patients are **fictional** and have AI consent switched on (otherwise chat/symptom text is canned).
- [ ] Have the fallback ready: laptop backend + tunnel (`docs/DEPLOYMENT.md` step 7). Screenshots of each screen below on your phone in case the Wi-Fi fails.
- [ ] Never open the Render/Supabase dashboards on the projector (secrets).

## 10-minute demo path
| Min | Screen | What to do | What to say |
|---|---|---|---|
| 0-1 | Login | Sign in as clinician | Roles are enforced by the server and Supabase row-level security; clinician signup needs an invite code. |
| 1-2 | Home | Show the patient list, the agent status | Ten specialist agents behind one orchestrator. |
| 2-4 | Symptom Assessment | Type natural language, e.g. "my right hand shakes at rest and I feel stiff in the mornings" | Gemini extracts structured symptoms; the diagnostic agent matches them against a curated 10-condition knowledge base with cited sources. |
| 4-5 | Analysis / Evidence | Open a ranked condition and its sources | Retrieval, not free generation: only retrieved sources are cited. |
| 5-6 | Next Steps / Risk | Show proposed actions and the risk panel | Actions are only ever **proposed**; a human approves. Nothing is sent to anyone (no notification channel is connected). The risk score is a transparent heuristic, labelled as unvalidated. |
| 6-7 | Emergency (optional) | Enter "face is drooping and I can't speak" | A deterministic rule engine runs first and independently of the AI agents and escalates immediately. |
| 7-8 | Longitudinal | Add two readings of one metric | Compares against the patient's own baseline, not a population. |
| 8-9 | AI Assistant | Ask "what does my last assessment mean?" | Answers only from this patient's stored context; an external-AI consent gate and best-effort redaction sit in front of Gemini. |
| 9-10 | ML Lab | Run a demo prediction, show an abstention | Two research-demonstration models on public data, clearly labelled. They abstain on physically impossible input instead of guessing. |

## Say these things yourself (examiners trust honesty and punish overclaiming)
- It is a **decision-support research prototype**, not a clinical product. Nothing is clinically validated.
- Emergency-detector recall was measured on **our own** test phrases only.
- "Consensus" today is effectively the diagnostic agent's retrieval score; the other agents do not yet emit per-disease support. It is an aggregation framework.
- Agent reliability weights and the risk-score weights are informal engineering defaults.
- The ML models are trained on small public UCI datasets (195 voice recordings, 42 telemonitoring patients). The historical accuracy figures could **not** be independently reproduced because the datasets were unreachable from our build environment; the validation framework (nested subject-grouped cross-validation, pre-registered protocol, fail-closed reporting) is built and tested and will run the moment the files are available. The gain over a majority-class guess in the shipped metadata is small (about 2 and 1 percentage points).
- Patient text goes to Google Gemini only with consent recorded and after best-effort redaction. This is not a compliance claim.
- Data sources that are real: our curated knowledge base (with source links) and PubMed through NCBI. Do **not** mention FDA drug databases or MIMIC-III as integrated.

## Likely questions, honest answers
| Question | Answer |
|---|---|
| How did you validate it? | Software: 944 backend tests and 24 frontend tests, plus a security/safety audit (privilege escalation, fail-open emergency detection and misleading "executed" statuses were found and fixed). Models: validation framework built; real-data results not yet produced (data access). No clinical validation. |
| Why not Alembic / migrations? | The earlier SQLAlchemy layer was removed after the Supabase migration; the schema is versioned as SQL migration files in `backend/scripts/`. |
| What stops a patient from becoming a clinician? | Server-side role check plus invite code for clinician signup, row-level security in Supabase, rate limiting on auth. |
| What happens if Gemini is down or consent is missing? | Deterministic template fallbacks; the response carries a warning instead of failing. |
| What happens if an agent fails? | Per-agent timeouts and an overall deadline; the run is marked degraded and says which agents failed. The emergency pre-scan does not depend on the agents. |
| Is the multi-agent part just a pipeline? | The orchestrator runs agents with declared dependencies, collects results, detects conflicts, applies a safety engine and records an audit trail. Today only one agent votes on the diagnosis, so the benefit is separation of concerns and auditability, not ensemble accuracy. |
| What would you do next? | Run the registered validation on the public datasets, get clinician input on intended use, validate on an independent cohort, add a real notification channel and a consent toggle in the UI. |

## If something breaks live
Say what it is, switch to the screenshots, and move on. Typical causes and fixes are in `docs/DEPLOYMENT.md` section 6.
