"""
Step 13 test script -- proves ai/gemini_service.py works through a realistic
Pydantic schema, not just plain text like Step 12's test did.
"""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from pydantic import BaseModel

from app.ai.gemini_service import generate_structured


class SymptomSummary(BaseModel):
    summary: str
    likely_urgency: str  # "low" | "medium" | "high" -- just for this smoke test


result = generate_structured(
    "Patient reports a mild headache for 2 days, no other symptoms, no history of migraines.",
    SymptomSummary,
    system_instruction=(
        "You summarize patient-reported symptoms in one sentence and label "
        "likely urgency as low, medium, or high. This is a demo system, not "
        "medical advice."
    ),
)

print("Type:", type(result))
print("Result:", result)