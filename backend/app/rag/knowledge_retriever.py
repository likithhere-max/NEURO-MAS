"""
Real semantic retrieval over the neurological knowledge base: Gemini
embeddings + Supabase pgvector similarity search (see the
match_knowledge_chunks SQL function from the Step 16 migration).

This supersedes app/services/knowledge_base.py's literal-keyword-overlap
matcher for retrieval QUALITY -- a patient's own phrasing ("hands shake at
rest") now matches textbook terms ("resting tremor") it never would have
matched lexically. That old lexical matcher is NOT deleted: it's kept as an
automatic fallback if the embedding API or the database is ever unreachable,
so diagnostic support degrades gracefully instead of failing outright --
same principle as every other agent in this codebase.
"""
from __future__ import annotations

import asyncio
import logging
import re

from app.ai.gemini_service import embed_text
from app.core.supabase_client import get_supabase_admin_client
from app.services.knowledge_base import STOPWORDS, get_knowledge_base

logger = logging.getLogger("neuromas.rag")

# Same cap as the lexical matcher it supersedes -- semantic similarity is
# still a heuristic signal, not a validated diagnostic probability.
MAX_HEURISTIC_CONFIDENCE = 0.75

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - STOPWORDS


def _semantic_search_sync(query: str, top_k: int) -> list[dict]:
    query_vector = embed_text(query, task_hint="search query")
    admin = get_supabase_admin_client()
    result = admin.rpc(
        "match_knowledge_chunks",
        {"query_embedding": query_vector, "match_count": top_k},
    ).execute()

    query_tokens = _tokenize(query)
    candidates = []
    for row in result.data:
        similarity = float(row["similarity"])
        candidates.append({
            "disease_id": row["disease_id"],
            "preferred_name": row["preferred_name"],
            "matched_terms": sorted(query_tokens & _tokenize(row["content"])),
            "icd10_codes": row.get("icd10_codes", []) or [],
            "red_flags": row.get("red_flags", []) or [],
            "source_urls": row.get("source_urls", []) or [],
            "retrieval_score": round(similarity, 3),
            "retrieval_method": "semantic_embedding",
        })
    return candidates


def _apply_confidence(candidates: list[dict], known_conditions: list[str]) -> list[dict]:
    for c in candidates:
        confidence = min(max(c["retrieval_score"], 0.0), 1.0) * MAX_HEURISTIC_CONFIDENCE
        # Phase 3 (D09): the retrieval-only value is kept separately. The +0.15 below is a PRIOR taken from the patient's
        # record, not evidence from the current symptoms, so it must not be fed to the consensus as symptom evidence.
        c["retrieval_confidence"] = round(confidence, 2)
        c["history_prior_applied"] = False
        if c["preferred_name"] in known_conditions or c["disease_id"] in known_conditions:
            confidence = min(confidence + 0.15, MAX_HEURISTIC_CONFIDENCE)
            c["history_prior_applied"] = True
        c["confidence"] = round(confidence, 2)
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


async def match_symptoms(
    symptom_terms: list[str],
    known_conditions: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Returns candidate conditions ranked by real semantic similarity to the
    given symptom terms. Falls back to the original lexical bag-of-words
    matcher on ANY failure in the embedding/database pipeline -- this
    function must never raise; diagnostic support always returns something
    usable, even in degraded mode.
    """
    query = " ".join(symptom_terms)
    try:
        candidates = await asyncio.to_thread(_semantic_search_sync, query, top_k)
        if candidates:
            return _apply_confidence(candidates, known_conditions)
    except Exception as exc:
        logger.warning("semantic_search_failed_falling_back_to_lexical error=%s", repr(exc))

    # Fallback: original lexical keyword-overlap matcher, unchanged logic.
    kb = get_knowledge_base()
    matches = kb.match_symptoms(symptom_terms, top_k=top_k)
    candidates = [
        {
            "disease_id": m["disease_id"],
            "preferred_name": m["preferred_name"],
            "matched_terms": m["_matched_terms"],
            "icd10_codes": m.get("icd10_codes", []) or [],
            "red_flags": m.get("red_flags", []) or [],
            "source_urls": m.get("source_urls", []) or [],
            "retrieval_score": round(m["_match_score"], 3),
            "retrieval_method": "lexical_fallback",
        }
        for m in matches
    ]
    return _apply_confidence(candidates, known_conditions)