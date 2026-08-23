"""
Deterministic clinical safety rules.

Per spec section 12 ("The LLM must NOT be the final safety authority") and
section 6.11 ("Do NOT make a generative LLM the sole emergency detector"),
every emergency/urgency judgment in NEURO-MAS is made here, with plain
if/else logic over structured fields -- not by asking a model to "decide."

These rules are intentionally simple and clearly not a substitute for
clinical triage protocols; see docs/SIMULATED_VS_REAL.md.
"""
from __future__ import annotations

# Deterministic emergency keyword triggers, one entry per primary condition
# family. Matching is done on normalized (lowercased) observation text.
EMERGENCY_KEYWORDS: dict[str, list[str]] = {
    "epilepsy": [
        "status epilepticus", "seizure not stopping", "seizure over 5 minutes",
        "repeated seizures", "seizure cluster", "not regaining consciousness",
    ],
    "stroke": [
        "face drooping", "facial droop", "arm weakness", "slurred speech",
        "sudden vision loss", "worst headache of my life", "sudden confusion",
        "one sided weakness", "cannot speak",
    ],
    "myasthenia_gravis": [
        "trouble breathing", "difficulty breathing", "myasthenic crisis",
        "can't swallow", "cannot swallow",
    ],
    "guillain_barre": [
        "ascending weakness", "weakness spreading upward", "trouble breathing",
        "rapidly worsening weakness",
    ],
    "general": [
        "unresponsive", "not breathing", "loss of consciousness",
        "severe head injury",
    ],
}

# A small, deliberately conservative, curated interaction/contraindication
# table. This is NOT an exhaustive drug interaction database -- it exists to
# demonstrate the *architecture* of deterministic medication safety checking,
# not to replace a real drug interaction service (e.g. an FDB/First
# Databank-style feed) in production. See docs/SIMULATED_VS_REAL.md.
KNOWN_INTERACTIONS: list[dict] = [
    {
        "drug_a": "levodopa",
        "drug_b": "haloperidol",
        "severity": "high",
        "description": "Typical antipsychotics (e.g. haloperidol) are dopamine "
                        "antagonists and can worsen parkinsonian symptoms / "
                        "counteract levodopa's effect.",
        "source": "General pharmacology principle - clinician verification required",
    },
    {
        "drug_a": "levodopa",
        "drug_b": "risperidone",
        "severity": "medium",
        "description": "Antipsychotics with dopamine-antagonist activity may "
                        "reduce levodopa efficacy and worsen motor symptoms.",
        "source": "General pharmacology principle - clinician verification required",
    },
    {
        "drug_a": "valproate",
        "drug_b": "lamotrigine",
        "severity": "high",
        "description": "Valproate inhibits lamotrigine metabolism, substantially "
                        "raising lamotrigine levels and rash/toxicity risk; "
                        "requires dose adjustment per guideline.",
        "source": "General pharmacology principle - clinician verification required",
    },
    {
        "drug_a": "carbamazepine",
        "drug_b": "warfarin",
        "severity": "medium",
        "description": "Carbamazepine is a strong hepatic enzyme inducer and can "
                        "reduce warfarin's anticoagulant effect.",
        "source": "General pharmacology principle - clinician verification required",
    },
    {
        "drug_a": "pyridostigmine",
        "drug_b": "aminoglycoside",
        "severity": "high",
        "description": "Aminoglycoside antibiotics can worsen neuromuscular "
                        "transmission and precipitate myasthenic crisis.",
        "source": "General pharmacology principle - clinician verification required",
    },
]


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def detect_emergency(observation_texts: list[str]) -> tuple[bool, list[str]]:
    """Deterministic scan for emergency-trigger phrases.

    Returns (is_emergency, matched_phrases). This function is intentionally
    dumb (substring match) so its behavior is fully auditable -- no model
    inference is involved in the safety-critical path.
    """
    matched: list[str] = []
    joined = " | ".join(normalize(t) for t in observation_texts)
    for _condition, phrases in EMERGENCY_KEYWORDS.items():
        for phrase in phrases:
            if phrase in joined:
                matched.append(phrase)
    return (len(matched) > 0, matched)


def check_interactions(active_drug_names: list[str]) -> list[dict]:
    names = {normalize(n) for n in active_drug_names}
    hits = []
    for rule in KNOWN_INTERACTIONS:
        if rule["drug_a"] in names and rule["drug_b"] in names:
            hits.append(rule)
    return hits


def check_allergy_conflicts(active_drug_names: list[str], allergies: list[str]) -> list[dict]:
    conflicts = []
    allergy_set = {normalize(a) for a in allergies}
    for drug in active_drug_names:
        if normalize(drug) in allergy_set:
            conflicts.append({
                "drug": drug,
                "severity": "high",
                "description": f"Patient has a documented allergy to {drug}.",
            })
    return conflicts
