"""
Step 16 test script -- proves embed_text() works and returns a vector of
the expected dimensionality, before we build anything on top of it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.gemini_service import embed_text
from app.core.config import get_settings

settings = get_settings()

vector = embed_text(
    "Resting tremor, bradykinesia, and rigidity in an elderly patient.",
    task_hint="search document",
)

print("Expected dimensions:", settings.gemini_embedding_dimensions)
print("Actual length:", len(vector))
print("First 5 values:", vector[:5])