"""
Step 17 verification -- proves real semantic search works by querying with
phrasing that WOULDN'T match the old lexical matcher at all (no shared
words with the knowledge base text), and confirming it still finds the
right condition via embedding similarity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.gemini_service import embed_text
from app.core.supabase_client import get_supabase_admin_client

admin = get_supabase_admin_client()

query = "my hands shake a lot when I'm just sitting still, and I move really slowly now"
query_vector = embed_text(query, task_hint="search query")

result = admin.rpc("match_knowledge_chunks", {
    "query_embedding": query_vector,
    "match_count": 3,
}).execute()

for row in result.data:
    print(f"{row['preferred_name']} -- similarity={row['similarity']:.3f}")