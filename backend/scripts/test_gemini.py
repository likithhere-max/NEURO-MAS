"""
Step 12 test script -- proves the Gemini API key + google-genai SDK work,
completely independent of the rest of the app.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai

from app.core.config import get_settings

settings = get_settings()

if not settings.gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not set -- check backend/.env")

client = genai.Client(api_key=settings.gemini_api_key)

response = client.models.generate_content(
    model=settings.gemini_default_model,
    contents="In one short sentence, what is a neurological symptom?",
)

print("Model used:", settings.gemini_default_model)
print("Response text:")
print(response.text)