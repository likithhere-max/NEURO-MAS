# Research report outline (mapped to this build)

For the academic report, following the section list from the original brief, with a note on what
this build can currently substantiate for each section vs. what needs to be written as future work.

1. **Abstract / Introduction / Problem Statement / Motivation** — write from your own framing;
   ground it in the gap between generic single-model health chatbots and a system that visibly
   coordinates specialized analyses with bounded autonomy.
2. **Existing systems & their limitations** — contrast single-agent LLM health assistants (no
   consensus, no explicit safety gating, no audit trail) against this design.
3. **Proposed system / Objectives** — use `docs/ARCHITECTURE.md` section 1-2 directly.
4. **Multi-agent architecture / Agent responsibilities / Communication** — `docs/ARCHITECTURE.md`
   sections 2, 4; the `AgentMessage` schema in `core/messaging.py` is your communication-protocol
   figure.
5. **Patient state model** — `schemas/patient_state.py`; the P_t vector is directly implemented,
   not just diagrammed.
6. **Neurological knowledge base** — `backend/data/diseases_curated.json` + `sources_and_licenses.txt`;
   be explicit in the report that 10/366 conditions are fully populated and the rest are a named
   registry pending population (this is a completely defensible "current scope" statement, not a
   weakness to hide).
7. **RAG architecture** — `services/knowledge_base.py`; be precise that this is lexical/keyword
   retrieval, not embedding-based semantic search — state that as a limitation with a stated
   upgrade path (`docs/ROADMAP.md` item 2), not as a claim of a full RAG pipeline.
8. **Personalization engine** — `agents/personalization.py`; use the ΔP formulation from the brief's
   section 36, and cite the actual `personalization_reasons` output as your "proof of
   personalization" example (include a real captured JSON trace from a demo run, not a mockup).
9. **Consensus mechanism** — `core/consensus.py`; the original formula is retained as `compute_consensus`, but the pipeline uses the typed-evidence `compute_evidence_consensus` (an uncalibrated index — report it as such; see `docs/phase2/CONSENSUS_DESIGN.md`)
   (`test_consensus.py` gives you a place to pull worked numeric examples from).
10. **Autonomous action engine / Safety architecture** — `core/actions.py`, `core/safety.py`; the
    autonomy-level table and the exhaustive safety tests are strong, concrete evidence for this
    section — screenshot or quote `test_safety.py` results directly.
11. **FHIR integration** — not implemented in this build; report it as designed-for but not yet
    built, and reference `docs/ROADMAP.md` item 4 as the stated plan, rather than claiming it exists.
12. **Database design** — `db/models.py`; include the entity list as your ER diagram basis.
13. **Implementation / UI** — cite the actual dashboard screenshots taken during this session's
    verification pass, and the 31 passing tests as implementation evidence.
14. **Experimental methodology / Evaluation / Results** — this is the section most in need of new
    work: run the two demo scenarios, capture the real numeric consensus/disagreement/confidence
    values, and report those as worked examples. Do **not** report invented sensitivity/specificity/
    AUROC numbers — the brief is explicit that clinical performance claims require real measurement,
    which this prototype has not undergone. Report the *system* metrics that are honestly available:
    e.g. workflow latency, number of agents invoked, disagreement rate across the two demo scenarios,
    action-approval-required rate.
15. **Limitations** — pull directly from `docs/SIMULATED_VS_REAL.md`; it's written precisely for
    this purpose.
16. **Future work** — pull directly from `docs/ROADMAP.md`.
17. **Conclusion** — tie back to the title-lock check in `docs/ARCHITECTURE.md` section 8.

**One general note for the report:** every place this system says "heuristic," "PROTOTYPE," or
"PENDING_VERIFICATION" is a strength to cite, not a weakness to obscure — it's exactly the honesty
the brief itself demanded, and a reviewer will trust the rest of the claims more for it.
