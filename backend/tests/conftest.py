import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Route the test suite at its own throwaway SQLite file, set BEFORE any
# app module is imported (app.core.config / app.db.session read this at
# import time), so tests never touch the dev database used by the demo
# seed script or a locally running server.
_TEST_DB = Path(__file__).resolve().parent / "test_neuromas.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ["SECRET_KEY"] = "test-only-secret"
