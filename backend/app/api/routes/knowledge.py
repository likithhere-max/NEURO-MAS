from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.api import KnowledgeSearchIn
from app.services.knowledge_base import get_knowledge_base

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/diseases")
def list_diseases():
    kb = get_knowledge_base()
    return {
        "meta": kb.meta,
        "diseases": [{"disease_id": d["disease_id"], "preferred_name": d["preferred_name"], "icd10_codes": d.get("icd10_codes", [])} for d in kb.list_all()],
    }


@router.get("/diseases/{disease_id}")
def get_disease(disease_id: str):
    kb = get_knowledge_base()
    record = kb.get(disease_id)
    if not record:
        raise HTTPException(404, "disease not found in curated knowledge base")
    return record


@router.post("/search")
def search_knowledge(payload: KnowledgeSearchIn):
    kb = get_knowledge_base()
    return {"query": payload.query, "results": kb.search(payload.query, top_k=payload.top_k)}
