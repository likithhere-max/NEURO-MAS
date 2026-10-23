"""'executed' may only be claimed when delivery is confirmed (audit: it was set for emergency escalation with no sender)."""
from app.core.actions import build_action
from app.services import notifier
from app.services.notifier import DeliveryResult


def _emergency_action():
    return build_action(patient_id="p", workflow_id="w", plan_item="emergency escalation", action_type="emergency_escalation",
                        reason="matched", evidence=[], is_emergency=True, disagreement=1.0, top_confidence=0.0,
                        data_quality=0.0, clinician_restricted_actions=[])


def test_default_provider_does_not_claim_delivery():
    a = _emergency_action()
    assert a.status == "proposed"
    assert any(r.startswith("NOT_DELIVERED") for r in a.safety_reasons)


def test_confirmed_delivery_is_marked_executed():
    class Fake:
        def deliver(self, **kw):
            return DeliveryResult(True, "fake-sms", "ok")
    notifier.set_provider(Fake())
    a = _emergency_action()
    assert a.status == "executed" and "delivered_via:fake-sms" in a.safety_reasons


def test_failing_provider_never_fakes_success_or_crashes():
    class Broken:
        def deliver(self, **kw):
            raise ConnectionError("smtp down")
    notifier.set_provider(Broken())
    a = _emergency_action()
    assert a.status == "proposed" and any("delivery_failed" in r for r in a.safety_reasons)


def test_actions_needing_approval_never_call_the_provider():
    calls = []
    class Spy:
        def deliver(self, **kw):
            calls.append(kw); return DeliveryResult(True, "spy", "")
    notifier.set_provider(Spy())
    a = build_action(patient_id="p", workflow_id="w", plan_item="prescription change", action_type="prescription_modification",
                     reason="r", evidence=[], is_emergency=False, disagreement=0.0, top_confidence=1.0, data_quality=1.0,
                     clinician_restricted_actions=[])
    assert a.status == "pending_approval" and calls == []
