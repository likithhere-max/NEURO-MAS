"""
Verifies the 6 newly added conditions are retrievable via semantic search,
using rephrased (non-textbook) descriptions for each.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.gemini_service import embed_text
from app.core.supabase_client import get_supabase_admin_client

admin = get_supabase_admin_client()

test_queries = [
    "I get sudden electric shock-like pain on one side of my face when I chew or touch it",
    "one side of my face suddenly drooped and I can't close my eye properly",
    "I had numbness in my arm and slurred speech for a few minutes but it went away completely",
    "my hand shakes when I try to write or hold a cup, but not when it's just resting",
    "I have a really bad headache with a stiff neck and I feel feverish and confused",
]

for query in test_queries:
    vector = embed_text(query, task_hint="search query")
    result = admin.rpc("match_knowledge_chunks", {"query_embedding": vector, "match_count": 1}).execute()
    top = result.data[0] if result.data else None
    print(f"Query: {query}")
    print(f"  -> {top['preferred_name']} (similarity={top['similarity']:.3f})" if top else "  -> no match")