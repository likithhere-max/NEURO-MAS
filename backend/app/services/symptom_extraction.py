"""
Natural-language symptom extraction (new feature, 2-day review sprint).

Turns a patient's free-text symptom description into the SAME structured
`SymptomIn` contract every existing agent already consumes
(app/schemas/api.py: name/severity/onset/free_text). This is deliberately a
translation layer only -- it does not replace, extend, or bypass
data_intake_agent / neuro_diagnostic_agent / risk_stratification_agent /
knowledge_agent, all of which are completely unchanged by this feature.

Uses the existing single Gemini access point (app/ai/gemini_service.py) --
no new model wiring. On any Gemini failure this raises SymptomExtractionError
rather than silently returning an empty/fabricated profile, so the caller
(the workflow route) can surface a real error instead of pretending
extraction happened.
"""
from __future__ import annotations

import asyncio
import logging

from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.api import SymptomIn
from app.schemas.symptom_assessment import ExtractedSymptomProfile, SymptomAssessmentIn

logger = logging.getLogger("neuromas.symptom_extraction")

SYSTEM_INSTRUCTION = (
    "You extract structured clinical features from a patient's own free-text "
    "description of their symptoms, for a neurological decision-support "
    "prototype. You are NOT diagnosing anything and must never name a "
    "disease, condition, or ICD code in your output -- only the symptoms, "
    "timing, and context the patient actually described. Rules: "
    "(1) Only include a symptom, duration, severity, or feature the patient "
    "explicitly stated or clearly implied -- never invent one. "
    "(2) If something is ambiguous or not stated (e.g. exact severity), "
    "leave the field empty/null and add a note to missing_information "
    "instead of guessing. "
    "(3) 'associated_features' means secondary symptoms mentioned alongside "
    "the primary complaint (e.g. stiffness, slowness) -- keep each one short "
    "(2-4 words). "
    "(4) Do not add medical interpretation, severity judgment, or urgency "
    "assessment beyond what the patient's own words support."
)


class SymptomExtractionError(Exception):
    """Raised when Gemini extraction fails after retries. Callers must not
    fabricate a fallback structured profile -- surface this as a real error
    to the user instead (unlike the narrative-only Gemini call sites
    elsewhere in this codebase, there is no safe deterministic template for
    'guess what symptoms this paragraph describes')."""


def _build_prompt(payload: SymptomAssessmentIn) -> str:
    lines = [f'Patient\'s own description: "{payload.free_text.strip()}"']
    follow_ups = {
        "When did it start / has it changed?": payload.onset_answer,
        "Is it getting worse?": payload.worsening_answer,
        "How severe is it?": payload.severity_answer,
        "How often does it occur?": payload.frequency_answer,
        "Where does it occur?": payload.location_answer,
        "Anything that triggers it?": payload.triggers_answer,
        "Any associated symptoms?": payload.associated_symptoms_answer,
    }
    answered = {q: a for q, a in follow_ups.items() if a and a.strip()}
    if answered:
        lines.append("\nAdditional patient-provided answers to guided questions:")
        for q, a in answered.items():
            lines.append(f"- {q} -> {a.strip()}")
    lines.append(
        "\nExtract the structured clinical feature profile described above. "
        "Remember: no diagnosis, no invented facts, flag anything ambiguous "
        "in missing_information instead of guessing."
    )
    return "\n".join(lines)


async def extract_symptom_profile(payload: SymptomAssessmentIn) -> ExtractedSymptomProfile:
    prompt = _build_prompt(payload)
    try:
        return await asyncio.to_thread(
            generate_structured,
            prompt,
            ExtractedSymptomProfile,
            system_instruction=SYSTEM_INSTRUCTION,
        )
    except GeminiServiceError as exc:
        logger.error("symptom_extraction_failed error=%s", repr(exc))
        raise SymptomExtractionError(
            "Could not extract structured symptoms from the description right now "
            "(the language model call failed). Please try again."
        ) from exc


def extracted_profile_to_symptom_ins(
    raw_free_text: str, profile: ExtractedSymptomProfile
) -> list[SymptomIn]:
    """
    Converts the rich extraction into the plain SymptomIn list every
    existing agent already expects. Two important, deliberate choices:

    1. The FIRST entry carries the patient's raw, un-paraphrased free text
       verbatim in `free_text`. data_intake_agent's deterministic emergency
       keyword scanner (app/services/clinical_rules.detect_emergency) scans
       every symptom's free_text -- this guarantees the safety-critical scan
       always sees the patient's exact original wording, never a
       Gemini-paraphrased version of it.
    2. Associated features become their own SymptomIn entries too (not just
       metadata), because neuro_diagnostic_agent only looks at symptoms with
       modality=="symptom" when building its semantic-search query -- so a
       feature like "stiffness" or "slower movement" needs to be a real
       symptom entry to actually influence which conditions get retrieved.
    """
    out: list[SymptomIn] = [
        SymptomIn(
            name="patient-reported symptoms (verbatim)",
            severity=None,
            onset=profile.duration,
            free_text=raw_free_text.strip(),
        )
    ]
    for s in profile.symptoms:
        detail_bits = [b for b in (s.location, s.frequency, s.context) if b]
        out.append(
            SymptomIn(
                name=s.name,
                severity=s.severity,
                onset=s.onset or profile.duration,
                free_text=f"{s.name}" + (f" ({'; '.join(detail_bits)})" if detail_bits else ""),
            )
        )
    for feature in profile.associated_features:
        out.append(
            SymptomIn(
                name=feature,
                severity=None,
                onset=profile.duration,
                free_text=feature,
            )
        )
    return out
