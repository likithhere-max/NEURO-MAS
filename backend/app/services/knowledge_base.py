"""
Neurological disease knowledge base service.

Loads the curated, source-tagged disease records (backend/data/diseases_curated.json)
and provides simple lexical retrieval over them.

HONESTY NOTE: this is a keyword/overlap scoring "RAG-lite" implementation, not
a vector-embedding semantic search. It is explicitly labeled as a retrieval
PROTOTYPE. Swapping in a real embedding model + pgvector/Qdrant is listed as
a "SHOULD HAVE" in the roadmap (see docs/ROADMAP.md) -- doing so would only
change retrieval quality, not the contract this service exposes.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "diseases_curated.json"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Phase 3 (D06): function words carry no symptom information. Before this filter they were counted in BOTH the overlap
# and the denominator of the lexical score, so "the and of in my a to" matched Alzheimer's, essential tremor and meningitis
# with score 0.857 (->confidence 0.64) and "I have a tremor in my hand" gave epilepsy 0.286 from the words "a" and "in" alone.
# Plain English grammar words only -- no medical content, and negations ("no", "not", "without") are deliberately KEPT.
STOPWORDS = frozenset(
    "a an the and or but if so as of in on at to for from by with about into over under again then than too very can will just "
    "i me my mine we us our you your he him his she her it its they them their this that these those is am are was were be been "
    "being have has had do does did having doing there here when where which who whom what how all any both each few more most "
    "other some such own same also would should could may might shall".split())


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - STOPWORDS


class KnowledgeBaseService:
    def __init__(self, path: Path = DATA_PATH):
        raw = json.loads(path.read_text())
        self.meta = raw.get("_meta", {})
        self.diseases: list[dict[str, Any]] = raw.get("diseases", [])
        self._by_id = {d["disease_id"]: d for d in self.diseases}

        # Precompute a bag-of-words index per disease for lexical retrieval.
        self._index: dict[str, set[str]] = {}
        for d in self.diseases:
            blob = " ".join(
                [
                    d.get("preferred_name", ""),
                    " ".join(d.get("synonyms", []) or []),
                    " ".join(d.get("symptoms", []) or []),
                    d.get("definition", ""),
                    " ".join(d.get("red_flags", []) or []),
                ]
            )
            self._index[d["disease_id"]] = _tokenize(blob)

    def get(self, disease_id: str) -> dict[str, Any] | None:
        return self._by_id.get(disease_id)

    def list_all(self) -> list[dict[str, Any]]:
        return self.diseases

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for d in self.diseases:
            idx = self._index[d["disease_id"]]
            overlap = q_tokens & idx
            if not overlap:
                continue
            score = len(overlap) / max(len(q_tokens), 1)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def match_symptoms(self, symptom_terms: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Return diseases ranked by symptom-term overlap, with the overlap
        score attached as `_match_score` and `_matched_terms` for evidence.
        """
        q_tokens: set[str] = set()
        for term in symptom_terms:
            q_tokens |= _tokenize(term)
        if not q_tokens:
            return []
        results = []
        for d in self.diseases:
            idx = self._index[d["disease_id"]]
            overlap = q_tokens & idx
            if not overlap:
                continue
            score = len(overlap) / max(len(q_tokens), 1)
            enriched = dict(d)
            enriched["_match_score"] = round(score, 3)
            enriched["_matched_terms"] = sorted(overlap)
            results.append(enriched)
        results.sort(key=lambda d: d["_match_score"], reverse=True)
        return results[:top_k]


@lru_cache
def get_knowledge_base() -> KnowledgeBaseService:
    return KnowledgeBaseService()
