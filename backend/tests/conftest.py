import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test may call Gemini. The low-level call is replaced by an immediate failure, which exercises the
    deterministic / template-fallback paths (the 'Gemini narrates, never decides' design) and removes the
    3x retry sleeps. Tests that need a model answer monkeypatch generate_structured explicitly."""
    import app.ai.gemini_service as gs

    def _boom(*a, **k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(gs, "_call_gemini", _boom)
    monkeypatch.setattr(gs, "_call_embed", _boom)
    yield


@pytest.fixture(autouse=True)
def _reset_notifier_and_limiter():
    from app.core import ratelimit
    from app.services import notifier

    original = notifier.get_provider()
    ratelimit.reset()
    yield
    notifier.set_provider(original)
    ratelimit.reset()
