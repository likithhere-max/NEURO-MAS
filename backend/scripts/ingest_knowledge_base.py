"""
Step 17 -- ingestion script: embeds every disease entry from
diseases_curated.json and loads it into Supabase's knowledge_chunks table
for real semantic search. Safe to re-run whenever the JSON file changes --
it clears the table first, then rebuilds it entirely, so knowledge_chunks
is always a derived copy of the JSON file, never a second source of truth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.gemini_service import embed_text
from app.core import privacy

privacy._ctx.set(privacy.ProcessingContext(purpose="kb_ingestion", system=True))  # curated KB text, no patient data
from app.core.supabase_client import get_supabase_admin_client
from app.services.knowledge_base import get_knowledge_base

kb = get_knowledge_base()
admin = get_supabase_admin_client()

print("Clearing existing knowledge_chunks rows...")
admin.table("knowledge_chunks").delete().neq("disease_id", "").execute()

rows = []
for d in kb.list_all():
    content = " | ".join(filter(None, [
        d.get("preferred_name", ""),
        "Synonyms: " + ", ".join(d.get("synonyms", []) or []) if d.get("synonyms") else "",
        "Symptoms: " + ", ".join(d.get("symptoms", []) or []),
        "Definition: " + d.get("definition", "") if d.get("definition") else "",
        "Red flags: " + ", ".join(d.get("red_flags", []) or []) if d.get("red_flags") else "",
    ]))
    print(f"Embedding: {d['preferred_name']}...")
    vector = embed_text(content, task_hint="search document")
    rows.append({
        "disease_id": d["disease_id"],
        "preferred_name": d["preferred_name"],
        "content": content,
        "icd10_codes": d.get("icd10_codes", []),
        "red_flags": d.get("red_flags", []),
        "source_urls": d.get("source_urls", []),
        "embedding": vector,
    })

print(f"Inserting {len(rows)} knowledge chunks into Supabase...")
result = admin.table("knowledge_chunks").insert(rows).execute()
print(f"Done. Inserted {len(result.data)} rows.")