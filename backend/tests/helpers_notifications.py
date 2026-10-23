"""TEST-ONLY mock notification provider. Lives under tests/ on purpose: the application ships no mock provider."""
import threading
import time

from app.services.notification_dispatch import NotificationProviderV2, ProviderResponse


class MockProvider(NotificationProviderV2):
    channel = "mock"

    def __init__(self, script=None, delay=0.0):
        self.script = list(script or [])        # outcomes consumed in order; last one repeats
        self.calls, self.delay, self.lock = [], delay, threading.Lock()
        self.receipts = {}

    def send(self, request):
        with self.lock:
            self.calls.append(request)
            step = self.script.pop(0) if len(self.script) > 1 else (self.script[0] if self.script else ("sent", ""))
        if self.delay:
            time.sleep(self.delay)
        if isinstance(step, Exception):
            raise step
        outcome, detail = step
        return ProviderResponse(outcome, f"mid-{len(self.calls)}" if outcome in ("sent", "delivered") else None, detail)

    def poll(self, provider_message_id):
        return self.receipts.get(provider_message_id)
