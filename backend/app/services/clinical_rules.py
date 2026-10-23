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

import re
import unicodedata

# Deterministic emergency keyword triggers, one entry per primary condition
# family. Matching is done on normalized (lowercased) observation text.
EMERGENCY_KEYWORDS: dict[str, list[str]] = {
    "epilepsy": [
        "status epilepticus", "seizure not stopping", "seizure over 5 minutes",
        "repeated seizures", "seizure cluster", "not regaining consciousness",
    ],
    "stroke": [
        "face drooping", "facial droop", "face is drooping",
        "arm weakness", "leg weakness",
        "slurred speech", "difficulty speaking", "trouble speaking",
        "cannot speak", "can't speak", "cant speak",
        "sudden vision loss", "trouble seeing", "sudden trouble seeing",
        "worst headache of my life",
        "sudden confusion",
        "one sided weakness", "one-sided weakness",
        "weakness on one side", "weakness on the left side", "weakness on the right side",
        "left side weakness", "right side weakness",
        "left-sided weakness", "right-sided weakness",
        "sudden weakness",
        "sudden numbness", "numbness on one side",
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


# ---------------------------------------------------------------------------
# Text normalisation (audit fix C/H: curly apostrophes, double spaces, hyphens)
# ---------------------------------------------------------------------------
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes, collapse whitespace. Never raises on None."""
    return _WS.sub(" ", (text or "").translate(_APOSTROPHES).strip().lower())


# ---------------------------------------------------------------------------
# Emergency detection
# ---------------------------------------------------------------------------
# Regex triggers for natural phrasings (added after the audit showed 14/15
# realistic emergency phrasings were missed by literal substring matching).
# Design stance: over-triage is acceptable for a safety net, under-triage is
# not. These patterns are NOT a validated triage protocol (see
# docs/SIMULATED_VS_REAL.md); recall/precision on real patient text is
# unmeasured. Each pattern is tagged with a family label for auditability.
_CANT = r"(?:can't|cant|cannot|can not|couldn't|could not|unable to|not able to|no longer able to)"
_NUM_MIN = r"(?:5|6|7|8|9|\d{2,}|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)"

EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    # --- stroke (FAST + other sudden focal deficits) ---
    ("stroke", r"\b(?:face|facial|mouth|smile|lip)\b[^.;,]{0,25}\b(?:droop\w*|drop(?:ped|ping|s)?|crooked|uneven|sagg\w+|lopsided)\b"),
    ("stroke", r"\b(?:droop\w*|crooked|lopsided)\b[^.;,]{0,15}\b(?:face|mouth|smile)\b"),
    ("stroke", r"\b(?:slurr\w*|slurs|slurred)\b"),
    ("stroke", r"\b(?:speech|words?|talking)\b[^.;,]{0,25}\b(?:slur\w*|garbl\w*|jumbl\w*|mumbl\w*|nonsense|gibberish)\b"),
    ("stroke", _CANT + r"[^.;,]{0,10}\b(?:speak|talk|get (?:my|the) words out|form words|find (?:my|the) words|understand (?:what|words?))\b"),
    ("stroke", r"\b(?:difficulty|trouble|hard(?: to)?|struggling(?: to)?)\s+(?:speaking|talking|speak|talk|finding words|getting words out)\b"),
    ("stroke", _CANT + r"[^.;,]{0,12}\b(?:move|lift|raise|feel|use)\b[^.;,]{0,25}\b(?:arm|leg|hand|foot|face|side|limb)s?\b"),
    ("stroke", r"\b(?:arm|leg|hand|foot|face|side|limb)s?\b[^.;,]{0,20}\b(?:went|gone|is|has gone|feels?|became|suddenly)\s+(?:weak|limp|numb|dead|paraly\w+)\b"),
    ("stroke", r"\b(?:paraly\w+|paresis|hemiplegi\w+)\b"),
    ("stroke", r"\b(?:lost|losing|loss of|lose)\b[^.;,]{0,12}\b(?:vision|sight|eyesight)\b"),
    ("stroke", _CANT + r"\s+see\b[^.;,]{0,20}\b(?:out of|with|in)\s+(?:one|my left|my right|either)\b"),
    ("stroke", r"\b(?:thunderclap|worst (?:headache|pain|head pain))\b"),
    ("stroke", r"\bsudden(?:ly)?\b[^.;,]{0,20}\b(?:confus\w+|weak\w*|numb\w*|vision|dizz\w+|loss of balance|trouble (?:walking|seeing|speaking))\b"),
    # --- epilepsy ---
    ("epilepsy", r"\bstatus epilepticus\b"),
    ("epilepsy", r"\b(?:seizures?|convulsions?|convulsing)\b[^.;]{0,40}\b" + _NUM_MIN + r"\s*\+?\s*(?:min|mins|minutes)\b"),
    ("epilepsy", r"\b(?:seizures?|convulsions?)\b[^.;]{0,30}\b(?:won't|will not|not|isn't|hasn't|doesn't)\s+(?:stop\w*|ended|end)\b"),
    ("epilepsy", r"\b(?:seizures?|convulsions?)\b[^.;]{0,30}\b(?:back[- ]to[- ]back|in a row|one after (?:the )?another|without (?:waking|regaining)|cluster)\b"),
    ("epilepsy", r"\b(?:repeated|multiple|several|recurrent|cluster of|serial|continuous)\s+(?:seizures?|convulsions?)\b"),
    ("epilepsy", r"\bkeeps? (?:seizing|convulsing)\b"),
    ("epilepsy", r"\bnot (?:regaining|waking|responding|rousable)\b"),
    # --- breathing / neuromuscular crisis (MG, GBS, bulbar) ---
    ("breathing", r"\b" + _CANT + r"[^.;,]{0,8}\bbreath\w*\b"),
    ("breathing", r"\b(?:trouble|difficulty|hard|struggling|labou?red)\b[^.;,]{0,10}\bbreath\w*\b"),
    ("breathing", r"\b(?:short(?:ness)? of breath|breathless\w*|gasping|choking|turning blue|blue lips)\b"),
    ("breathing", r"\b(?:too weak|unable|can't|cannot)\b[^.;,]{0,10}\b(?:to )?(?:cough|swallow|hold (?:my|his|her|their) head up|lift (?:my|his|her|their) head)\b"),
    ("breathing", r"\b" + _CANT + r"\s+swallow\b"),
    ("breathing", r"\bmyasthenic crisis\b"),
    ("gbs", r"\b(?:ascending|rapidly (?:worsening|progressing)|spreading (?:upward|up))\s+weakness\b|\bweakness spreading\b"),
    # --- consciousness / general ---
    ("general", r"\b(?:passed out|blacked out|black out|fainted|collaps\w+|unconscious|unresponsive|not responding|won't wake|will not wake|not waking|can't wake|cannot wake|loss of consciousness|lost consciousness|not breathing|stopped breathing|severe head injury|head trauma)\b"),
]
_COMPILED_PATTERNS = [(fam, re.compile(pat)) for fam, pat in EMERGENCY_PATTERNS]

# Narrow, explicit negation cues that must appear shortly BEFORE the match in
# the SAME clause. Deliberately minimal: a missed negation costs a false alarm,
# a wrongly-applied negation costs a missed emergency.
_NEGATION_CUE = re.compile(
    r"(?:\bno\b|\bnot\b|\bnone\b|\bnever\b|\bwithout\b|\bdenies\b|\bdenied\b|\bdeny\b|\bnegative for\b|"
    r"\bfree of\b|\bno history of\b|\bno signs? of\b|\bno evidence of\b|\brules? out\b|\bno complaints? of\b)"
    r"[\w\s'/-]{0,25}$"
)
_CLAUSE_SPLIT = re.compile(r"[.;,\n!?]|\bbut\b|\bhowever\b|\balthough\b")


def _is_negated(text: str, start: int) -> bool:
    clause_start = max((m.end() for m in _CLAUSE_SPLIT.finditer(text[:start])), default=0)
    prefix = text[clause_start:start]
    return bool(_NEGATION_CUE.search(prefix))


def detect_emergency_detailed(observation_texts: list[str]) -> dict:
    """Deterministic emergency scan with auditable output.

    Returns {"is_emergency", "matched" (list of matched surface strings),
    "families" (sorted unique), "negated" (matches suppressed by an explicit
    negation cue -- reported, never silently dropped)}.
    """
    matched: list[str] = []
    families: set[str] = set()
    negated: list[str] = []
    for raw in observation_texts or []:
        text = normalize(raw if isinstance(raw, str) else str(raw))
        if not text:
            continue
        for fam, rx in _COMPILED_PATTERNS:
            for m in rx.finditer(text):
                surface = m.group(0).strip()
                if _is_negated(text, m.start()):
                    negated.append(surface)
                else:
                    matched.append(surface)
                    families.add(fam)
        # Legacy literal phrases (kept so no previously-detected phrase regresses).
        for fam, phrases in EMERGENCY_KEYWORDS.items():
            for phrase in phrases:
                idx = text.find(phrase)
                while idx != -1:
                    if _is_negated(text, idx):
                        negated.append(phrase)
                    else:
                        matched.append(phrase)
                        families.add(fam)
                    idx = text.find(phrase, idx + 1)
    seen: set[str] = set()
    uniq = [m for m in matched if not (m in seen or seen.add(m))]
    return {"is_emergency": bool(uniq), "matched": uniq, "families": sorted(families), "negated": negated}


def detect_emergency(observation_texts: list[str]) -> tuple[bool, list[str]]:
    """Back-compatible wrapper: (is_emergency, matched_phrases)."""
    d = detect_emergency_detailed(observation_texts)
    return d["is_emergency"], d["matched"]


# ---------------------------------------------------------------------------
# Drug-name normalisation (brand -> generic, combination splitting, classes)
# ---------------------------------------------------------------------------
# Name normalisation only -- it adds NO new clinical interaction content.
BRAND_TO_GENERIC: dict[str, str] = {
    "sinemet": "levodopa", "madopar": "levodopa", "stalevo": "levodopa", "rytary": "levodopa",
    "duopa": "levodopa", "carbidopa-levodopa": "levodopa", "l-dopa": "levodopa",
    "haldol": "haloperidol", "risperdal": "risperidone",
    "depakote": "valproate", "depakene": "valproate", "epilim": "valproate", "divalproex": "valproate",
    "valproic acid": "valproate", "valproate sodium": "valproate", "sodium valproate": "valproate",
    "lamictal": "lamotrigine", "tegretol": "carbamazepine", "carbatrol": "carbamazepine",
    "coumadin": "warfarin", "jantoven": "warfarin", "mestinon": "pyridostigmine",
    "keppra": "levetiracetam", "amoxil": "amoxicillin", "augmentin": "amoxicillin",
}
DRUG_CLASS_MEMBERS: dict[str, set[str]] = {
    "aminoglycoside": {"gentamicin", "tobramycin", "amikacin", "streptomycin", "neomycin", "kanamycin", "plazomicin"},
    "penicillin": {"penicillin", "amoxicillin", "ampicillin", "piperacillin", "flucloxacillin", "oxacillin", "nafcillin", "dicloxacillin"},
}
_SALT_OR_FORM = re.compile(
    r"\b(?:hydrochloride|hcl|sodium|potassium|sulfate|sulphate|tartrate|mesylate|maleate|er|xr|cr|sr|la|dr|odt|"
    r"tablets?|capsules?|injection|oral|iv|im)\b|\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b|\b\d+(?:\.\d+)?\b")
_SPLIT_COMBO = re.compile(r"\s*(?:/|\+|,|;|\band\b|&)\s*")


def fold_accents(text: str) -> str:
    """Remove Latin diacritics (U+0300-U+036F) so 'L\u00e9vodopa' matches 'levodopa'. Other scripts are left untouched."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not 0x300 <= ord(ch) <= 0x36F)


def canonical_drug_names(name: str) -> set[str]:
    """All generic names and drug classes a free-text drug string maps to."""
    base = normalize(fold_accents(name))
    out: set[str] = set()
    if not base:
        return out
    candidates = {base}
    candidates.update(p for p in _SPLIT_COMBO.split(base) if p)
    for cand in list(candidates):
        stripped = _WS.sub(" ", _SALT_OR_FORM.sub(" ", cand)).strip(" -_")
        if stripped:
            candidates.add(stripped)
    for cand in candidates:
        for variant in {cand, cand.replace("_", " "), cand.replace(" ", "-")}:
            gen = BRAND_TO_GENERIC.get(variant, variant)
            out.add(gen)
    for cls, members in DRUG_CLASS_MEMBERS.items():
        if out & members:
            out.add(cls)
    return {o for o in out if o}


# A small, deliberately conservative, curated interaction/contraindication
# table. This is NOT an exhaustive drug interaction database -- it exists to
# demonstrate the *architecture* of deterministic medication safety checking,
# not to replace a real drug interaction service. See docs/SIMULATED_VS_REAL.md.
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


def covered_drug_terms() -> set[str]:
    """Every drug/class term that appears in at least one interaction rule."""
    return {r["drug_a"] for r in KNOWN_INTERACTIONS} | {r["drug_b"] for r in KNOWN_INTERACTIONS}


def check_interactions(active_drug_names: list[str]) -> list[dict]:
    canon: set[str] = set()
    for n in active_drug_names or []:
        canon |= canonical_drug_names(n)
    return [rule for rule in KNOWN_INTERACTIONS if rule["drug_a"] in canon and rule["drug_b"] in canon]


def unassessed_medications(active_drug_names: list[str]) -> list[str]:
    """Medications that match NO term in the curated rule table -- i.e. the
    interaction check cannot say anything about them. Reported so that an
    empty interaction list is never mistaken for 'checked and safe'."""
    covered = covered_drug_terms()
    return [n for n in (active_drug_names or []) if n and not (canonical_drug_names(n) & covered)]


def check_allergy_conflicts(active_drug_names: list[str], allergies: list[str]) -> list[dict]:
    conflicts = []
    allergy_canon: dict[str, set[str]] = {a: canonical_drug_names(a) for a in (allergies or []) if a}
    for drug in active_drug_names or []:
        dcanon = canonical_drug_names(drug)
        # Allergy matching is intentionally looser than interaction matching
        # (a missed allergy is worse than a false alarm): also match on any
        # whole word of the drug string, e.g. "Penicillin V" -> "penicillin".
        for tok in re.findall(r"[a-z][a-z-]{4,}", normalize(drug)):
            dcanon |= canonical_drug_names(tok)
        for allergen, acanon in allergy_canon.items():
            shared = dcanon & acanon
            if shared:
                conflicts.append({
                    "drug": drug,
                    "allergen": allergen,
                    "matched_on": sorted(shared),
                    "severity": "high",
                    "description": f"Patient has a documented allergy to {allergen}; {drug} matches on {sorted(shared)}.",
                })
                break
    return conflicts
